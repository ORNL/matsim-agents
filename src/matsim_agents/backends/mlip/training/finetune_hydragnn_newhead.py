"""The ONLY HydraGNN fine-tune: keep pretrained branch-7 (OMat24) as the single head.

matsim-agents uses only HydraGNN branch-7 (OMat24). The other 15 heads are
discarded and pretrained branch-7 is re-keyed as ``branch-0`` on top of the
pretrained backbone, then trained. The head surgery reuses
``ORNL/HydraGNN_GFM_FineTuning4Materials`` (``utils/update_model.py``).

Strategies:

* ``unfrozen`` -- train backbone + branch-7 head.
* ``frozen``   -- freeze the message-passing backbone and train only the head.

The objective is per-atom energy MSE + force MSE (forces via ``autograd``);
with ``num_branches == 1`` HydraGNN's decoder uses ``branch-0`` directly.

Saved artifacts (in ``output_dir``):

* ``ft_model.pk``   -- the fine-tuned single-head checkpoint.
* ``config.json``   -- the **original** GFM (16-branch) config, so eval can
  rebuild the backbone and re-apply the head surgery for a byte-identical model.
* ``newhead.json``  -- the fine-tune head config passed to ``update_architecture``
  (plus the strategy), consumed by the single-head calculator path.
* ``cost.json``     -- wall time / GPU-hours / peak memory / parameter counts.

Must run in the **hydragnn** venv (needs ``torch_scatter``/``hydragnn``).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from matsim_agents.backends.mlip.config import (
    HYDRAGNN_DATASET_HEADS,
    resolve_hydragnn_inference_head,
)
from matsim_agents.backends.mlip.devices import ACCELERATOR_TYPES, default_torch_device
from matsim_agents.backends.mlip.hydragnn_checkpoint import load_hydragnn_state_dict
from matsim_agents.backends.mlip.hydragnn_newhead import (
    apply_newhead_surgery,
    load_update_model,
)
from matsim_agents.backends.mlip.training.cost import CostReport, count_parameters, track_cost
from matsim_agents.backends.mlip.training.hydragnn_data import (
    _atoms_to_data,
    _resolve_hydragnn_paths,
    _split,
    load_gfm_state_dict,
)
from matsim_agents.backends.mlip.training.train_loop import train_keep_best
from matsim_agents.core.seeding import seed_process

log = logging.getLogger(__name__)

STRATEGIES = ("unfrozen", "frozen")

# Map our user-facing strategy -> (load_pretrained, ORNL freeze_mode).
_STRATEGY_SPEC = {
    "unfrozen": (True, "none"),
    "frozen": (True, "message_passing"),
}

GFM_HEAD_DATASETS = HYDRAGNN_DATASET_HEADS


def _resolve_head_index(head=None) -> int:
    """Always 7 (OMat24); any other head raises."""
    return resolve_hydragnn_inference_head(head)


# --------------------------------------------------------------------------- #
# Fine-tune head config                                                        #
# --------------------------------------------------------------------------- #


def build_newhead_ft_config(
    gfm_cfg: dict,
    *,
    freeze_mode: str,
    head_arch: dict | None = None,
) -> dict:
    """Build the fine-tune config consumed by ORNL ``update_model``.

    A single ``branch-0`` graph head is created. By default its architecture
    matches the GFM's existing branch-0 head (self-consistent sizing); pass
    ``head_arch`` to override ``num_sharedlayers``/``dim_sharedlayers``/
    ``num_headlayers``/``dim_headlayers``.
    """
    arch = gfm_cfg["NeuralNetwork"]["Architecture"]
    base_branch = copy.deepcopy(arch["output_heads"]["graph"][0])
    branch0 = {"type": "branch-0", "architecture": head_arch or base_branch["architecture"]}
    return {
        "NeuralNetwork": {
            "Architecture": {
                "output_heads": {"graph": [branch0]},
                "output_dim": [1],
                "output_type": ["graph"],
                "task_weights": [1.0],
            },
            "Training": {"freeze_mode": freeze_mode, "keep_pretrained_decoder": False},
        }
    }


def _keep_pretrained_branch(decoder_target, branch_index: int) -> None:
    """Prune the pretrained decoder to the *selected* branch, keyed as ``branch-0``.

    Generalises ORNL ``_keep_pretrained_branch_zero`` (which only ever keeps
    ``branch-0``) to any of the 16 pretrained branches. The chosen branch's
    trained ``graph_shared`` and per-head ``heads_NN`` sub-modules are kept **by
    reference** -- so their pretrained weights are preserved -- and re-keyed as
    ``branch-0`` so HydraGNN's single-branch decoder path uses them directly.
    """
    import torch.nn as nn

    src = f"branch-{branch_index}"
    if src not in decoder_target.graph_shared:
        raise ValueError(
            f"pretrained graph_shared has no {src!r} "
            f"(available: {list(decoder_target.graph_shared.keys())})"
        )

    graph_shared = nn.ModuleDict({"branch-0": decoder_target.graph_shared[src]})
    heads_nn = nn.ModuleList()
    for head_module in decoder_target.heads_NN:
        if src not in head_module:
            raise ValueError(f"pretrained heads_NN entry has no {src!r}")
        heads_nn.append(nn.ModuleDict({"branch-0": head_module[src]}))

    decoder_target.graph_shared = graph_shared
    decoder_target.heads_NN = heads_nn
    decoder_target.num_branches = 1

    config_heads = getattr(decoder_target, "config_heads", None)
    if isinstance(config_heads, dict) and "graph" in config_heads:
        specs = config_heads["graph"]
        chosen = dict(specs[branch_index] if branch_index < len(specs) else specs[0])
        chosen["type"] = "branch-0"
        config_heads["graph"] = [chosen]


def apply_pretrained_head_surgery(
    model, branch_index: int, *, ft_repo=None, freeze_mode: str = "none"
):
    """Keep the *selected* pretrained branch as the single ``branch-0`` head.

    Unlike :func:`apply_newhead_surgery` (which discards all 16 pretrained
    readouts and grows a randomly-initialised head), this preserves the chosen
    head's pretrained weights -- the MACE/UMA-style approach of adapting an
    existing dataset-specific readout instead of bootstrapping one from scratch.
    Requires the pretrained decoder to already be loaded into ``model``. Returns
    the updated model.
    """
    um = load_update_model(ft_repo)
    decoder_target = um._resolve_decoder_target(model)
    _keep_pretrained_branch(decoder_target, int(branch_index))
    um._drop_wrapper_decoder_modules(model, decoder_target)
    if freeze_mode and freeze_mode != "none":
        model = um.apply_freeze_mode(
            model, {"NeuralNetwork": {"Training": {"freeze_mode": freeze_mode}}}
        )
    return model


def _resume_single_head_model(model, state_dict, ft_config, *, ft_repo, freeze_mode):
    model = apply_newhead_surgery(model, ft_config, ft_repo=ft_repo, freeze=False)
    load_hydragnn_state_dict(model, state_dict)
    if freeze_mode != "none":
        model = load_update_model(ft_repo).apply_freeze_mode(
            model, {"NeuralNetwork": {"Training": {"freeze_mode": freeze_mode}}}
        )
    return model


# --------------------------------------------------------------------------- #
# Training-label scales                                                       #
# --------------------------------------------------------------------------- #


def _target_scales(train_graphs) -> tuple[float, float]:
    """RMS force scale and per-atom formation-energy scale over the train split.

    Dividing the force and energy residuals by these scales makes the training
    loss ``O(1)`` regardless of the material. Without it the force term is
    ``force_weight * mean(F^2)`` (``~95 * ~19 ~ 1800`` for the Cantor HEA), which
    sits at the ``predict-zero-forces`` value and a gentle step cannot escape it,
    collapsing the new head into a constant predictor. Scales are floored so a
    single-composition or near-equilibrium split cannot amplify noise.
    """

    sq = 0.0
    n = 0
    peratom = []
    for g in train_graphs:
        f = g.forces.reshape(-1)
        sq += float((f.double() ** 2).sum())
        n += f.numel()
        na = int(g.atomic_numbers.numel())
        peratom.append(float(g.energy.item()) / max(na, 1))
    f_scale = (sq / max(n, 1)) ** 0.5
    f_scale = f_scale if f_scale > 1e-6 else 1.0
    e_scale = float(np.std(peratom)) if len(peratom) > 1 else 0.0
    e_scale = max(e_scale, 1e-2)  # eV/atom floor
    return f_scale, e_scale


# --------------------------------------------------------------------------- #
# Single-branch MLIP loss                                                      #
# --------------------------------------------------------------------------- #


def _batch_loss_single_head(
    model, batch, e_w, f_w, *, f_scale=1.0, e_scale=1.0, autocast_dtype=None
):
    from matsim_agents.backends.mlip.hydragnn_precision import hydragnn_autocast

    with hydragnn_autocast(batch.pos.device, autocast_dtype):
        return _batch_loss_single_head_impl(
            model, batch, e_w, f_w, f_scale=f_scale, e_scale=e_scale
        )


def _batch_loss_single_head_impl(model, batch, e_w, f_w, *, f_scale=1.0, e_scale=1.0):
    """Energy (per-atom) + force MSE for the single ``branch-0`` head.

    ``f_scale``/``e_scale`` normalise the residuals to ``O(1)`` so the objective
    is well-conditioned and balanced (see :func:`_target_scales`); with the
    scales at ``1.0`` this reduces to the raw eV / eV-per-atom loss.
    """
    import torch
    import torch.nn.functional as F

    num_graphs = int(batch.num_graphs)
    batch.pos.requires_grad_(True)
    node_counts = torch.bincount(batch.batch, minlength=num_graphs)

    pred = model(batch)
    energy = pred[0].squeeze(-1) if isinstance(pred, (list, tuple)) else pred.squeeze(-1)

    forces_pred = -torch.autograd.grad(
        energy,
        batch.pos,
        grad_outputs=torch.ones_like(energy),
        create_graph=True,
    )[0]

    # Labels already contain total-cell formation energies; normalize only.
    e_target = batch.energy.view(num_graphs)
    e_loss = F.mse_loss((energy / node_counts) / e_scale, (e_target / node_counts) / e_scale)
    f_loss = F.mse_loss(forces_pred / f_scale, batch.forces / f_scale)
    return e_w * e_loss + f_w * f_loss


@dataclass(frozen=True)
class _LossSpec:
    """Loss weights, normalisation scales, and autocast dtype for one fine-tune run."""

    e_w: float
    f_w: float
    f_scale: float
    e_scale: float
    autocast_dtype: Any

    def __call__(self, model, batch):
        return _batch_loss_single_head(
            model,
            batch,
            self.e_w,
            self.f_w,
            f_scale=self.f_scale,
            e_scale=self.e_scale,
            autocast_dtype=self.autocast_dtype,
        )


# --------------------------------------------------------------------------- #
# Fine-tune stages                                                            #
# --------------------------------------------------------------------------- #


def _finetune_protocol(
    strategy: str, head_index: int | None, freeze_mode: str, *, resumed_from: Path | None
) -> dict:
    head_init = "resumed" if resumed_from is not None else "pretrained"
    protocol = {
        "backend": "hydragnn",
        "protocol": f"newhead-finetune-{strategy}",
        "head_init": head_init,
        "pretrained_heads": [GFM_HEAD_DATASETS[head_index]] if head_index is not None else [],
        "freeze_mode": freeze_mode,
    }
    if resumed_from is not None:
        protocol["resumed_from"] = str(resumed_from)
    return protocol


def _load_resume_metadata(
    gfm_logdir: Path,
    gfm_checkpoint: str | None,
    *,
    head_arch: dict | None,
    head_index: int | None,
) -> tuple[dict, str | None, int]:
    """Verify a previous single-head run and return its metadata, checkpoint, and head."""
    from ase.calculators.calculator import Calculator

    from matsim_agents.backends.mlip.hydragnn_energy import restore_hydragnn_total_energy

    restore_hydragnn_total_energy(Calculator(), gfm_logdir, gfm_checkpoint)
    previous = json.loads((gfm_logdir / "energy-convention.json").read_text())
    gfm_checkpoint = gfm_checkpoint or previous["checkpoint"]
    resume_metadata = json.loads((gfm_logdir / "newhead.json").read_text())
    if resume_metadata.get("head_index") != head_index:
        raise ValueError(
            f"Cannot resume {gfm_logdir}: head_index={resume_metadata.get('head_index')!r}; "
            "only models fine-tuned from branch-7 (OMat24) are allowed"
        )
    if head_arch is not None:
        raise ValueError("Resuming a single-head checkpoint cannot replace its head architecture")
    return resume_metadata, gfm_checkpoint, resume_metadata["head_index"]


def _loss_weights(
    arch: dict, normalize: bool, energy_weight: float | None, force_weight: float | None
) -> tuple[float, float]:
    # With the normalised loss the residuals are already O(1), so forces and
    # energy are balanced at unit weight; the raw config force_weight (~95) is
    # only appropriate for the un-normalised eV^2 objective.
    _w_default = 1.0 if normalize else arch.get("force_weight", 1.0)
    e_w = float(
        energy_weight
        if energy_weight is not None
        else (1.0 if normalize else arch.get("energy_weight", 1.0))
    )
    f_w = float(force_weight if force_weight is not None else _w_default)
    return e_w, f_w


def _build_single_head_model(
    hcfg: dict,
    *,
    param_dtype: Any,
    dev: Any,
    load_pretrained: bool,
    gfm_logdir: Path,
    gfm_checkpoint: str | None,
    resume_metadata: dict | None,
    freeze_mode: str,
    head_arch: dict | None,
    head_index: int | None,
    ft_repo: str | Path | None,
) -> tuple[Any, dict]:
    """Build the GFM, load its backbone, and reduce the decoder to one ``branch-0`` head."""
    import torch
    from hydragnn.models.create import create_model_config

    model = create_model_config(config=hcfg["NeuralNetwork"], verbosity=0)
    torch.set_default_dtype(param_dtype)
    model = model.to(dtype=param_dtype, device=dev)
    if not load_pretrained:
        raise ValueError("HydraGNN fine-tuning must start from pretrained branch-7 (OMat24)")
    ckpt_path, state_dict = load_gfm_state_dict(gfm_logdir, gfm_checkpoint, dev)
    if resume_metadata is None:
        load_hydragnn_state_dict(model, state_dict, strict=False)
    log.info("Loaded pretrained backbone from %s", ckpt_path.name)

    # The recorded ``ft_config`` always describes a single branch-0 head with the
    # GFM's head architecture, so the eval/calculator path rebuilds a matching
    # shape and loads the fine-tuned checkpoint over it regardless of how the
    # head was initialised here.
    ft_config = build_newhead_ft_config(hcfg, freeze_mode=freeze_mode, head_arch=head_arch)
    if resume_metadata is not None:
        ft_config = resume_metadata["ft_config"]
        model = _resume_single_head_model(
            model, state_dict, ft_config, ft_repo=ft_repo, freeze_mode=freeze_mode
        )
        log.info("Resumed fine-tuned backbone and single head without reinitializing readout")
    else:
        model = apply_pretrained_head_surgery(
            model, head_index, ft_repo=ft_repo, freeze_mode=freeze_mode
        )
        log.info(
            "Kept pretrained head branch-%d (%s) as branch-0",
            head_index,
            GFM_HEAD_DATASETS[head_index],
        )
    # update_architecture builds new Linear layers with the default dtype; make
    # sure the whole model (backbone + new head) is on the training precision.
    torch.set_default_dtype(param_dtype)
    model = model.to(dtype=param_dtype, device=dev)
    return model, ft_config


def _warmup_cosine_scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_frac: float):
    """Linear warmup then cosine decay over the full training run.

    A schedule (plus gradient clipping) lets us use a larger effective step
    without the constant-predictor collapse that a flat high LR caused.
    """
    import math

    import torch

    steps_per_epoch = max(steps_per_epoch, 1)
    total_steps = max(steps_per_epoch * int(epochs), 1)
    warmup_steps = max(int(total_steps * float(warmup_frac)), 1)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


def _save_single_head_logdir(
    model, output_dir: Path, *, config_path: Path, checkpoint_name: str, metadata: dict
) -> Path:
    """Write the checkpoint, the ORIGINAL GFM config, and ``newhead.json``."""
    import torch

    ckpt_out = output_dir / checkpoint_name
    torch.save({"model_state_dict": model.state_dict()}, ckpt_out)
    # The original GFM config lets eval rebuild the backbone and re-apply surgery.
    shutil.copyfile(config_path, output_dir / "config.json")
    (output_dir / "newhead.json").write_text(json.dumps(metadata, indent=2))
    return ckpt_out


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def finetune_hydragnn_newhead(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    gfm_logdir: str | Path,
    strategy: str = "unfrozen",
    head: str | int = "OMat24",
    gfm_checkpoint: str | None = None,
    head_arch: dict | None = None,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 4,
    val_fraction: float = 0.1,
    seed: int,
    energy_weight: float | None = None,
    force_weight: float | None = None,
    normalize: bool = True,
    grad_clip: float = 10.0,
    warmup_frac: float = 0.05,
    charge: float = 0.0,
    spin: float = 0.0,
    hydragnn_root: str | Path | None = None,
    ft_repo: str | Path | None = None,
    device: str | None = None,
    checkpoint_name: str = "ft_model.pk",
    run: bool = True,
    elemental_reference_manifest: str | Path | None = None,
    validation_dataset: str | Path | None = None,
) -> Path:
    """Keep pretrained branch-7 (OMat24) as the single head and fine-tune it.

    ``strategy`` is ``"unfrozen"`` or ``"frozen"``; ``head`` must be branch-7.
    Returns the output logdir (``config.json`` + ``newhead.json`` +
    ``checkpoint_name``). With ``run=False`` only one forward/loss step runs.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}; got {strategy!r}")
    load_pretrained, freeze_mode = _STRATEGY_SPEC[strategy]
    head_index = _resolve_head_index(head)

    from matsim_agents.data.formation_training import (
        prepare_hydragnn_training_frames,
        prepare_hydragnn_validation_frames,
        save_hydragnn_energy_convention,
    )

    seed_process(seed, "training")
    output_dir = Path(output_dir).expanduser().resolve()
    frames = prepare_hydragnn_training_frames(
        dataset_path, output_dir, elemental_reference_manifest
    )
    # A dataset split's validation partition replaces the random hold-out.
    validation_frames = (
        prepare_hydragnn_validation_frames(
            validation_dataset, output_dir, elemental_reference_manifest
        )
        if validation_dataset is not None
        else None
    )
    root, example_dir = _resolve_hydragnn_paths(hydragnn_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    gfm_logdir = Path(gfm_logdir).expanduser().resolve()
    resume_metadata = None
    if (gfm_logdir / "newhead.json").exists():
        resume_metadata, gfm_checkpoint, head_index = _load_resume_metadata(
            gfm_logdir, gfm_checkpoint, head_arch=head_arch, head_index=head_index
        )

    import torch
    from hydragnn.preprocess.graph_samples_checks_and_updates import get_radius_graph_pbc
    from hydragnn.train.train_validate_test import resolve_precision
    from torch_geometric.loader import DataLoader

    # --- config + precision ---
    config_path = gfm_logdir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"GFM config.json not found in {gfm_logdir}")
    with open(config_path) as fh:
        hcfg = json.load(fh)
    arch = hcfg["NeuralNetwork"]["Architecture"]
    precision_str = hcfg["NeuralNetwork"]["Training"].get("precision", "fp32")
    _, param_dtype, autocast_dtype = resolve_precision(precision_str)
    torch.set_default_dtype(param_dtype)

    dev = torch.device(device or default_torch_device())
    radius = float(arch["radius"])
    max_neighbours = int(arch["max_neighbours"])
    e_w, f_w = _loss_weights(arch, normalize, energy_weight, force_weight)

    # --- read dataset ---
    if not frames:
        raise ValueError(f"No frames read from {dataset_path}")

    model, ft_config = _build_single_head_model(
        hcfg,
        param_dtype=param_dtype,
        dev=dev,
        load_pretrained=load_pretrained,
        gfm_logdir=gfm_logdir,
        gfm_checkpoint=gfm_checkpoint,
        resume_metadata=resume_metadata,
        freeze_mode=freeze_mode,
        head_arch=head_arch,
        head_index=head_index,
        ft_repo=ft_repo,
    )
    # When the backbone is trainable (unfrozen) it must run in train()
    # mode so its norm layers update; a frozen backbone stays in eval() to keep
    # its pretrained running stats fixed. Running a trainable backbone in eval()
    # was a key reason the new-head fine-tune failed to move off zero-shot.
    backbone_trainable = freeze_mode == "none"
    model.train() if backbone_trainable else model.eval()

    trainable_params, total_params = count_parameters(model)
    log.info(
        "New-head fine-tune [%s]: num_branches=%d, %d / %d params trainable (%.3f%%)",
        strategy,
        int(getattr(model, "num_branches", 1)),
        trainable_params,
        total_params,
        100.0 * trainable_params / max(total_params, 1),
    )

    # --- build graphs ---
    add_edges_pbc = get_radius_graph_pbc(radius=radius, max_neighbours=max_neighbours)
    graphs = [_atoms_to_data(a, add_edges_pbc, param_dtype, charge, spin) for a in frames]
    graphs = [g for g in graphs if g is not None]
    if len(graphs) < 2:
        raise ValueError(f"Only {len(graphs)} usable frames (need >=2 with energy+forces).")
    if validation_frames is not None:
        train_graphs = graphs
        val_graphs = [
            g
            for g in (
                _atoms_to_data(a, add_edges_pbc, param_dtype, charge, spin)
                for a in validation_frames
            )
            if g is not None
        ]
    else:
        train_graphs, val_graphs = _split(graphs, val_fraction, seed)
    log.info("Graphs: %d train / %d val", len(train_graphs), len(val_graphs))

    # Force/energy normalisation scales (train split only); 1.0 disables it.
    if normalize:
        f_scale, e_scale = _target_scales(train_graphs)
    else:
        f_scale, e_scale = 1.0, 1.0
    log.info(
        "Loss normalisation: normalize=%s f_scale=%.4f eV/A e_scale=%.4f eV/atom "
        "(e_w=%.3f f_w=%.3f)",
        normalize,
        f_scale,
        e_scale,
        e_w,
        f_w,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    num_gpus = 1 if dev.type in ACCELERATOR_TYPES else 0
    report = CostReport(
        model_backend="hydragnn",
        dataset_label=Path(dataset_path).parent.name or Path(dataset_path).stem,
        base_model=str(gfm_logdir.name),
        dataset_path=str(Path(dataset_path).resolve()),
        n_train_frames=len(train_graphs),
        n_val_frames=len(val_graphs),
        epochs=int(epochs),
        num_gpus=num_gpus,
        device=str(dev),
        trainable_params=trainable_params,
        total_params=total_params,
        frozen_params=total_params - trainable_params,
        extra={
            "strategy": strategy,
            "head": GFM_HEAD_DATASETS[head_index],
            "head_index": head_index,
            "freeze_mode": freeze_mode,
            "load_pretrained": load_pretrained,
            "energy_weight": e_w,
            "force_weight": f_w,
            "normalize": normalize,
            "f_scale": f_scale,
            "e_scale": e_scale,
            "protocol": f"newhead-finetune-{strategy}",
        },
    )

    loss_spec = _LossSpec(e_w, f_w, f_scale, e_scale, autocast_dtype)
    if not run:
        loader = DataLoader(train_graphs, batch_size=min(batch_size, len(train_graphs)))
        batch = next(iter(loader)).to(dev)
        loss = loss_spec(model, batch)
        log.info("Dry-run forward OK; loss=%.6f", float(loss.detach()))
        return output_dir

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size) if val_graphs else None
    scheduler = _warmup_cosine_scheduler(optimizer, len(train_loader), epochs, warmup_frac)
    with track_cost(report):
        train_keep_best(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_spec,
            dev=dev,
            epochs=epochs,
            n_train=len(train_graphs),
            n_val=len(val_graphs),
            train_mode=backbone_trainable,
            scheduler=scheduler,
            grad_clip=grad_clip,
        )

    ckpt_out = _save_single_head_logdir(
        model,
        output_dir,
        config_path=config_path,
        checkpoint_name=checkpoint_name,
        metadata={
            "strategy": strategy,
            "head": GFM_HEAD_DATASETS[head_index],
            "head_index": head_index,
            "freeze_mode": freeze_mode,
            "load_pretrained": load_pretrained,
            "checkpoint": checkpoint_name,
            "ft_config": ft_config,
            "normalize": normalize,
            "f_scale": f_scale,
            "e_scale": e_scale,
            "energy_convention": "dft:declared_elemental_formation_energy",
        },
    )
    save_hydragnn_energy_convention(
        output_dir,
        ckpt_out,
        inference_metadata={
            "finetune": _finetune_protocol(
                strategy,
                head_index,
                freeze_mode,
                resumed_from=gfm_logdir if resume_metadata else None,
            )
        },
    )
    report.write(output_dir / "cost.json")
    log.info(
        "New-head fine-tune complete [%s] -> %s (%.1fs, %.4f GPU-h)",
        strategy,
        output_dir,
        report.wall_time_s,
        report.gpu_hours,
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="AL-collected extxyz dataset.")
    parser.add_argument("--elemental-reference-manifest", required=True)
    parser.add_argument("--output-dir", required=True, help="Output logdir.")
    parser.add_argument("--gfm-logdir", required=True, help="GFM dir (config.json + .pk).")
    parser.add_argument("--strategy", default="unfrozen", choices=list(STRATEGIES))
    parser.add_argument(
        "--head", default="OMat24", choices=["OMat24", "7", "branch-7"], help="Always branch-7."
    )
    parser.add_argument("--gfm-checkpoint", default=None, help="GFM .pk filename/path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--validation-dataset",
        default=None,
        help="Validation partition for keeping the best epoch (replaces --val-fraction).",
    )
    parser.add_argument("--seed", type=int, required=True, help="Seed of every random stream.")
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--force-weight", type=float, default=None)
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable force/energy loss normalisation (use raw eV objective).",
    )
    parser.set_defaults(normalize=True)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--charge", type=float, default=0.0)
    parser.add_argument("--spin", type=float, default=0.0)
    parser.add_argument("--hydragnn-root", default=None)
    parser.add_argument(
        "--ft-repo", default=None, help="ORNL HydraGNN_GFM_FineTuning4Materials path."
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-name", default="ft_model.pk")
    parser.add_argument("--dry-run", action="store_true", help="Single forward/loss only.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    finetune_hydragnn_newhead(
        args.dataset,
        args.output_dir,
        gfm_logdir=args.gfm_logdir,
        strategy=args.strategy,
        head=args.head,
        gfm_checkpoint=args.gfm_checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        validation_dataset=args.validation_dataset,
        seed=args.seed,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        normalize=args.normalize,
        grad_clip=args.grad_clip,
        warmup_frac=args.warmup_frac,
        charge=args.charge,
        spin=args.spin,
        hydragnn_root=args.hydragnn_root,
        ft_repo=args.ft_repo,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
        run=not args.dry_run,
        elemental_reference_manifest=args.elemental_reference_manifest,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
