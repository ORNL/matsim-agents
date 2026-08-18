"""Alternative HydraGNN fine-tune: drop all 16 branch heads, grow ONE new head.

This is the second HydraGNN fine-tuning option (the first being the routed-head
fine-tune in :mod:`matsim_agents.active_learning.finetune_hydragnn`). Instead of
keeping the multi-branch decoder and adapting the routed head(s), we discard the
entire 16-branch decoder and create a single fresh ``branch-0`` head on top of
the pretrained backbone, then train it (with the backbone either trainable or
frozen). This mirrors the capability in
``ORNL/HydraGNN_GFM_FineTuning4Materials`` (``utils/update_model.py``): we reuse
that repository's ``update_architecture`` / ``apply_freeze_mode`` directly so
the head surgery matches the reference implementation exactly.

Three strategies are supported:

* ``unfrozen``   -- load the pretrained backbone, replace the 16 heads with one
  fresh (randomly-initialised) ``branch-0`` head, train the **whole** model.
* ``frozen``     -- same head replacement, but **freeze the message-passing
  backbone** (``freeze_mode="message_passing"``) and train only the new head.
* ``scratch``    -- do **not** load any pretrained weights; the backbone and the
  new head are both randomly initialised (a from-scratch baseline).

The objective is the same energy+force MLIP loss used by the routed fine-tune
(per-atom energy MSE + force MSE, forces via ``autograd``), but with a single
branch there is no branch-weighting MLP -- when ``num_branches == 1`` HydraGNN's
decoder uses ``branch-0`` directly (see ``hydragnn/models/Base.py``).

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
import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from matsim_agents.active_learning.cost import CostReport, count_parameters, track_cost
from matsim_agents.active_learning.finetune_hydragnn import (
    _atoms_to_data,
    _find_gfm_checkpoint,
    _resolve_hydragnn_paths,
    _split,
)

log = logging.getLogger(__name__)

STRATEGIES = ("unfrozen", "frozen", "scratch")
_DEFAULT_FT_REPO = "/global/cfs/projectdirs/m5216/mlupopa/HydraGNN_GFM_FineTuning4Materials"

# Map our user-facing strategy -> (load_pretrained, ORNL freeze_mode).
_STRATEGY_SPEC = {
    "unfrozen": (True, "none"),
    "frozen": (True, "message_passing"),
    "scratch": (False, "none"),
}

# The GFM's 16 pretrained branch heads, in decoder order, each trained on a
# specific upstream dataset. Selecting one of these (instead of growing a fresh
# random head) lets us fine-tune a readout that already knows the right physics
# -- e.g. ``MPTrj``/``Alexandria``/``OMat24`` for bulk (transition-metal) solids.
GFM_HEAD_DATASETS = (
    "Alexandria",          # 0
    "ANI1x",               # 1
    "MPTrj",               # 2
    "OC2020",              # 3
    "OC2022",              # 4
    "OC25",                # 5
    "ODAC23",              # 6
    "OMat24",              # 7
    "OMol25",              # 8
    "OMol25-neutral",      # 9
    "OMol25-non-neutral",  # 10
    "OPoly2026",           # 11
    "Nabla2DFT",           # 12
    "QCML",                # 13
    "QM7X",                # 14
    "transition1x",        # 15
)


def _resolve_head_index(head) -> int | None:
    """Map a head selector (``None`` | int | dataset name) to a branch index.

    ``None`` -> ``None`` (grow a fresh, randomly-initialised ``branch-0``, the
    default behaviour). An int (or digit string) is validated against the 16
    pretrained branches. A name is matched case-insensitively against
    :data:`GFM_HEAD_DATASETS` (e.g. ``"MPTrj"`` -> ``2``).
    """
    if head is None:
        return None
    if isinstance(head, bool):  # guard: bool is an int subclass
        raise TypeError(f"invalid head selector {head!r}")
    if isinstance(head, int) or (isinstance(head, str) and head.strip().lstrip("-").isdigit()):
        idx = int(head)
        if not 0 <= idx < len(GFM_HEAD_DATASETS):
            raise ValueError(
                f"head index {idx} out of range 0..{len(GFM_HEAD_DATASETS) - 1}"
            )
        return idx
    key = str(head).strip().lower()
    for i, name in enumerate(GFM_HEAD_DATASETS):
        if name.lower() == key:
            return i
    raise ValueError(
        f"unknown head {head!r}; expected an index 0..{len(GFM_HEAD_DATASETS) - 1} "
        f"or one of {GFM_HEAD_DATASETS}"
    )


# --------------------------------------------------------------------------- #
# ORNL reference-repo loading + head config                                   #
# --------------------------------------------------------------------------- #


def load_update_model(ft_repo: str | Path | None = None):
    """Import ORNL's ``utils/update_model.py`` by file path.

    The module name ``utils`` collides with HydraGNN's own ``utils`` on
    ``sys.path``, so we load it directly from the repo file to avoid shadowing.
    """
    repo = Path(
        ft_repo or os.environ.get("HYDRAGNN_FT_REPO") or _DEFAULT_FT_REPO
    ).expanduser().resolve()
    path = repo / "utils" / "update_model.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"ORNL update_model.py not found at {path}. Clone "
            "ORNL/HydraGNN_GFM_FineTuning4Materials or set HYDRAGNN_FT_REPO."
        )
    spec = importlib.util.spec_from_file_location("ornl_update_model", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def apply_newhead_surgery(model, ft_config: dict, *, ft_repo=None, freeze: bool = True):
    """Replace the decoder with a single fresh ``branch-0`` head (+ optional freeze).

    Uses ORNL ``update_architecture`` (and ``apply_freeze_mode`` when ``freeze``)
    directly so the surgery matches the reference implementation. Returns the
    updated model.
    """
    um = load_update_model(ft_repo)
    model = um.update_architecture(model, ft_config)
    if freeze:
        model = um.apply_freeze_mode(model, ft_config)
    return model


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


# --------------------------------------------------------------------------- #
# Per-element linear reference energies                                        #
# --------------------------------------------------------------------------- #


def _fit_reference_energies(graphs) -> dict[int, float]:
    """Least-squares per-element linear reference energies ``e_ref[Z]``.

    Solves ``E_total_i approx sum_Z n_{i,Z} * e_ref[Z]`` over ``graphs`` (each
    carrying a TOTAL DFT energy in eV). Subtracting the per-structure reference
    sum from the DFT total leaves a small formation-scale target that matches
    the formation energy HydraGNN's head predicts, which is what keeps the
    fine-tune from collapsing onto the large raw-energy offset. Fit on the train
    split only to avoid leakage.
    """
    zs = sorted({int(z) for g in graphs for z in g.atomic_numbers.tolist()})
    col = {z: j for j, z in enumerate(zs)}
    a = np.zeros((len(graphs), len(zs)), dtype=np.float64)
    b = np.zeros(len(graphs), dtype=np.float64)
    for i, g in enumerate(graphs):
        for z in g.atomic_numbers.tolist():
            a[i, col[int(z)]] += 1.0
        b[i] = float(g.energy.item())
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    return {z: float(coef[col[z]]) for z in zs}


def _attach_reference_energies(graphs, e_ref_map: dict[int, float], dtype) -> None:
    """Attach the per-structure reference sum (TOTAL eV) as ``g.e_ref`` in place."""
    import torch

    for g in graphs:
        ref = sum(e_ref_map.get(int(z), 0.0) for z in g.atomic_numbers.tolist())
        g.e_ref = torch.tensor([ref], dtype=dtype)


def _target_scales(train_graphs, e_ref_map: dict[int, float]) -> tuple[float, float]:
    """RMS force scale and per-atom formation-energy scale over the train split.

    Dividing the force and energy residuals by these scales makes the training
    loss ``O(1)`` regardless of the material. Without it the force term is
    ``force_weight * mean(F^2)`` (``~95 * ~19 ~ 1800`` for the Cantor HEA), which
    sits at the ``predict-zero-forces`` value and a gentle step cannot escape it,
    collapsing the new head into a constant predictor. Scales are floored so a
    single-composition or near-equilibrium split cannot amplify noise.
    """
    import numpy as np

    sq = 0.0
    n = 0
    peratom = []
    for g in train_graphs:
        f = g.forces.reshape(-1)
        sq += float((f.double() ** 2).sum())
        n += f.numel()
        na = int(g.atomic_numbers.numel())
        ref = sum(e_ref_map.get(int(z), 0.0) for z in g.atomic_numbers.tolist())
        peratom.append((float(g.energy.item()) - ref) / max(na, 1))
    f_scale = (sq / max(n, 1)) ** 0.5
    f_scale = f_scale if f_scale > 1e-6 else 1.0
    e_scale = float(np.std(peratom)) if len(peratom) > 1 else 0.0
    e_scale = max(e_scale, 1e-2)  # eV/atom floor
    return f_scale, e_scale


# --------------------------------------------------------------------------- #
# Single-branch MLIP loss                                                      #
# --------------------------------------------------------------------------- #


def _batch_loss_single_head(model, batch, e_w, f_w, *, f_scale=1.0, e_scale=1.0):
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

    # HydraGNN predicts a TOTAL formation energy (eV); the DFT labels are TOTAL
    # energies (eV) on the raw reference. Subtract the per-structure linear
    # reference sum so the target is formation-scale and matches the head's
    # output, then normalise the loss to eV/atom (divide by node_counts) and by
    # the dataset scales so forces and energy contribute on the same footing.
    e_target = batch.energy.view(num_graphs)
    e_ref = (
        batch.e_ref.view(num_graphs)
        if hasattr(batch, "e_ref") and batch.e_ref is not None
        else torch.zeros_like(e_target)
    )
    e_form_target = e_target - e_ref
    e_loss = F.mse_loss((energy / node_counts) / e_scale, (e_form_target / node_counts) / e_scale)
    f_loss = F.mse_loss(forces_pred / f_scale, batch.forces / f_scale)
    return e_w * e_loss + f_w * f_loss


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def finetune_hydragnn_newhead(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    gfm_logdir: str | Path,
    strategy: str = "unfrozen",
    head: str | int | None = None,
    gfm_checkpoint: str | None = None,
    head_arch: dict | None = None,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 4,
    val_fraction: float = 0.1,
    seed: int = 0,
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
) -> Path:
    """Drop all 16 heads, grow one new ``branch-0`` head, and fine-tune it.

    ``strategy`` is one of ``{"unfrozen", "frozen", "scratch"}``. When ``head``
    is given (a branch index ``0..15`` or a dataset name from
    :data:`GFM_HEAD_DATASETS`, e.g. ``"MPTrj"``), the *selected pretrained* head
    is kept and fine-tuned instead of growing a randomly-initialised one -- the
    MACE/UMA-style readout adaptation. Head selection requires a pretrained
    backbone (``strategy`` ``unfrozen`` or ``frozen``); it is invalid with
    ``scratch``. Returns the output logdir (``config.json`` + ``newhead.json`` +
    ``checkpoint_name``). With ``run=False`` only a single forward/loss step runs
    (dry-run wiring check).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}; got {strategy!r}")
    load_pretrained, freeze_mode = _STRATEGY_SPEC[strategy]
    head_index = _resolve_head_index(head)
    if head_index is not None and not load_pretrained:
        raise ValueError(
            f"head selection ({head!r}) requires a pretrained backbone; "
            f"strategy={strategy!r} does not load pretrained weights"
        )

    root, example_dir = _resolve_hydragnn_paths(hydragnn_root)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gfm_logdir = Path(gfm_logdir).expanduser().resolve()

    import torch
    from torch_geometric.loader import DataLoader

    from hydragnn.models.create import create_model_config
    from hydragnn.preprocess.graph_samples_checks_and_updates import get_radius_graph_pbc
    from hydragnn.train.train_validate_test import resolve_precision

    # --- config + precision ---
    config_path = gfm_logdir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"GFM config.json not found in {gfm_logdir}")
    with open(config_path) as fh:
        hcfg = json.load(fh)
    arch = hcfg["NeuralNetwork"]["Architecture"]
    precision_str = hcfg["NeuralNetwork"]["Training"].get("precision", "fp32")
    _, param_dtype, _ = resolve_precision(precision_str)
    torch.set_default_dtype(param_dtype)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    radius = float(arch["radius"])
    max_neighbours = int(arch["max_neighbours"])
    # With the normalised loss the residuals are already O(1), so forces and
    # energy are balanced at unit weight; the raw config force_weight (~95) is
    # only appropriate for the un-normalised eV^2 objective.
    _w_default = 1.0 if normalize else arch.get("force_weight", 1.0)
    e_w = float(energy_weight if energy_weight is not None else (1.0 if normalize else arch.get("energy_weight", 1.0)))
    f_w = float(force_weight if force_weight is not None else _w_default)

    # --- read dataset ---
    raw = ase_read(str(dataset_path), index=":")
    frames = [raw] if isinstance(raw, Atoms) else list(raw)
    if not frames:
        raise ValueError(f"No frames read from {dataset_path}")

    # --- build model (+ optionally load pretrained backbone) ---
    model = create_model_config(config=hcfg["NeuralNetwork"], verbosity=0)
    torch.set_default_dtype(param_dtype)
    model = model.to(dtype=param_dtype, device=dev)
    if load_pretrained:
        ckpt_path = _find_gfm_checkpoint(gfm_logdir, gfm_checkpoint)
        state = torch.load(ckpt_path, map_location=dev)
        state_dict = state.get("model_state_dict", state)
        state_dict = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)
        log.info("Loaded pretrained backbone from %s", ckpt_path.name)
    else:
        log.info("strategy=scratch: skipping pretrained checkpoint load")

    # --- head surgery: keep ONE branch-0 head (+ optional freeze) ---
    # The recorded ``ft_config`` always describes a single branch-0 head with the
    # GFM's head architecture, so the eval/calculator path rebuilds a matching
    # shape and loads the fine-tuned checkpoint over it regardless of how the
    # head was initialised here (fresh-random vs. kept-pretrained).
    ft_config = build_newhead_ft_config(hcfg, freeze_mode=freeze_mode, head_arch=head_arch)
    if head_index is not None:
        # Keep the SELECTED pretrained readout (its trained weights) and re-key
        # it as branch-0, instead of growing a random head that must bootstrap
        # force learning from scratch.
        model = apply_pretrained_head_surgery(
            model, head_index, ft_repo=ft_repo, freeze_mode=freeze_mode
        )
        log.info(
            "Kept pretrained head branch-%d (%s) as branch-0",
            head_index, GFM_HEAD_DATASETS[head_index],
        )
    else:
        model = apply_newhead_surgery(
            model, ft_config, ft_repo=ft_repo, freeze=(freeze_mode != "none")
        )
    # update_architecture builds new Linear layers with the default dtype; make
    # sure the whole model (backbone + new head) is on the training precision.
    torch.set_default_dtype(param_dtype)
    model = model.to(dtype=param_dtype, device=dev)
    # When the backbone is trainable (unfrozen/scratch) it must run in train()
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
    train_graphs, val_graphs = _split(graphs, val_fraction, seed)
    log.info("Graphs: %d train / %d val", len(train_graphs), len(val_graphs))

    # Fit per-element linear reference energies on the TRAIN split and subtract
    # them so the head regresses formation-scale energies (see loss docstring).
    e_ref_map = _fit_reference_energies(train_graphs)
    _attach_reference_energies(train_graphs, e_ref_map, param_dtype)
    _attach_reference_energies(val_graphs, e_ref_map, param_dtype)
    log.info(
        "Fitted per-element reference energies (%d elements): %s",
        len(e_ref_map),
        {int(k): round(v, 3) for k, v in e_ref_map.items()},
    )

    # Force/energy normalisation scales (train split only); 1.0 disables it.
    if normalize:
        f_scale, e_scale = _target_scales(train_graphs, e_ref_map)
    else:
        f_scale, e_scale = 1.0, 1.0
    log.info(
        "Loss normalisation: normalize=%s f_scale=%.4f eV/A e_scale=%.4f eV/atom "
        "(e_w=%.3f f_w=%.3f)",
        normalize, f_scale, e_scale, e_w, f_w,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    num_gpus = 1 if dev.type == "cuda" else 0
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
            "head": (GFM_HEAD_DATASETS[head_index] if head_index is not None else None),
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

    if not run:
        loader = DataLoader(train_graphs, batch_size=min(batch_size, len(train_graphs)))
        batch = next(iter(loader)).to(dev)
        loss = _batch_loss_single_head(model, batch, e_w, f_w, f_scale=f_scale, e_scale=e_scale)
        log.info("Dry-run forward OK; loss=%.6f", float(loss.detach()))
        return output_dir

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size) if val_graphs else None

    # Linear warmup then cosine decay over the full training run. A schedule
    # (plus gradient clipping below) lets us use a larger effective step without
    # the constant-predictor collapse that a flat high LR caused.
    import math

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(steps_per_epoch * int(epochs), 1)
    warmup_steps = max(int(total_steps * float(warmup_frac)), 1)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # Keep the checkpoint with the lowest validation loss, not the final epoch:
    # on these small AL datasets the loss overfits well before the last epoch.
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    with track_cost(report):
        for epoch in range(epochs):
            model.train() if backbone_trainable else model.eval()
            running = 0.0
            for batch in train_loader:
                batch = batch.to(dev)
                optimizer.zero_grad(set_to_none=True)
                loss = _batch_loss_single_head(model, batch, e_w, f_w, f_scale=f_scale, e_scale=e_scale)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip
                    )
                optimizer.step()
                scheduler.step()
                running += float(loss.detach()) * batch.num_graphs
            train_mse = running / max(len(train_graphs), 1)
            val_msg = ""
            selection_mse = train_mse
            if val_loader is not None:
                model.eval()
                vrun = 0.0
                for batch in val_loader:
                    batch = batch.to(dev)
                    vloss = _batch_loss_single_head(model, batch, e_w, f_w, f_scale=f_scale, e_scale=e_scale)
                    vrun += float(vloss.detach()) * batch.num_graphs
                selection_mse = vrun / max(len(val_graphs), 1)
                val_msg = f" val={selection_mse:.6f}"
            log.info("Epoch %d/%d: train=%.6f%s", epoch + 1, epochs, train_mse, val_msg)
            if selection_mse < best_val:
                best_val = selection_mse
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Restore the best-validation weights before saving.
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Best epoch %d/%d: val=%.6f (restored for save)", best_epoch, epochs, best_val)

    # --- save logdir ---
    ckpt_out = output_dir / checkpoint_name
    torch.save({"model_state_dict": model.state_dict()}, ckpt_out)
    # Save the ORIGINAL GFM config so eval can rebuild backbone + re-apply surgery.
    shutil.copyfile(config_path, output_dir / "config.json")
    (output_dir / "newhead.json").write_text(
        json.dumps(
            {
                "strategy": strategy,
                "head": (GFM_HEAD_DATASETS[head_index] if head_index is not None else None),
                "head_index": head_index,
                "freeze_mode": freeze_mode,
                "load_pretrained": load_pretrained,
                "checkpoint": checkpoint_name,
                "ft_config": ft_config,
                "normalize": normalize,
                "f_scale": f_scale,
                "e_scale": e_scale,
                "reference_energies": {str(k): v for k, v in e_ref_map.items()},
            },
            indent=2,
        )
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
    parser.add_argument("--output-dir", required=True, help="Output logdir.")
    parser.add_argument("--gfm-logdir", required=True, help="GFM dir (config.json + .pk).")
    parser.add_argument("--strategy", default="unfrozen", choices=list(STRATEGIES))
    parser.add_argument(
        "--head", default=None,
        help="Fine-tune a PRETRAINED head instead of a fresh random one: a branch "
        "index 0..15 or a dataset name (e.g. MPTrj, Alexandria, OMat24). "
        "Requires --strategy unfrozen|frozen. Default: grow a random head.",
    )
    parser.add_argument("--gfm-checkpoint", default=None, help="GFM .pk filename/path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--force-weight", type=float, default=None)
    parser.add_argument(
        "--no-normalize", dest="normalize", action="store_false",
        help="Disable force/energy loss normalisation (use raw eV objective).",
    )
    parser.set_defaults(normalize=True)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--charge", type=float, default=0.0)
    parser.add_argument("--spin", type=float, default=0.0)
    parser.add_argument("--hydragnn-root", default=None)
    parser.add_argument("--ft-repo", default=None, help="ORNL HydraGNN_GFM_FineTuning4Materials path.")
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
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
