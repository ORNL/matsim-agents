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


# --------------------------------------------------------------------------- #
# Single-branch MLIP loss                                                      #
# --------------------------------------------------------------------------- #


def _batch_loss_single_head(model, batch, e_w, f_w):
    """Energy (per-atom) + force MSE for the single ``branch-0`` head."""
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

    e_target = batch.energy.view(num_graphs)
    e_loss = F.mse_loss(energy / node_counts, e_target / node_counts)
    f_loss = F.mse_loss(forces_pred, batch.forces)
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
    gfm_checkpoint: str | None = None,
    head_arch: dict | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 4,
    val_fraction: float = 0.1,
    seed: int = 0,
    energy_weight: float | None = None,
    force_weight: float | None = None,
    charge: float = 0.0,
    spin: float = 0.0,
    hydragnn_root: str | Path | None = None,
    ft_repo: str | Path | None = None,
    device: str | None = None,
    checkpoint_name: str = "ft_model.pk",
    run: bool = True,
) -> Path:
    """Drop all 16 heads, grow one new ``branch-0`` head, and fine-tune it.

    ``strategy`` is one of ``{"unfrozen", "frozen", "scratch"}``. Returns the
    output logdir (``config.json`` + ``newhead.json`` + ``checkpoint_name``).
    With ``run=False`` only a single forward/loss step runs (dry-run wiring
    check).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}; got {strategy!r}")
    load_pretrained, freeze_mode = _STRATEGY_SPEC[strategy]

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
    e_w = float(energy_weight if energy_weight is not None else arch.get("energy_weight", 1.0))
    f_w = float(force_weight if force_weight is not None else arch.get("force_weight", 1.0))

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

    # --- head surgery: drop 16 heads -> single fresh branch-0 (+ freeze) ---
    ft_config = build_newhead_ft_config(hcfg, freeze_mode=freeze_mode, head_arch=head_arch)
    model = apply_newhead_surgery(
        model, ft_config, ft_repo=ft_repo, freeze=(freeze_mode != "none")
    )
    # update_architecture builds new Linear layers with the default dtype; make
    # sure the whole model (backbone + new head) is on the training precision.
    torch.set_default_dtype(param_dtype)
    model = model.to(dtype=param_dtype, device=dev)
    model.eval()  # keeps any frozen backbone norm stats fixed; heads have no BN

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
            "freeze_mode": freeze_mode,
            "load_pretrained": load_pretrained,
            "energy_weight": e_w,
            "force_weight": f_w,
            "protocol": f"newhead-finetune-{strategy}",
        },
    )

    if not run:
        loader = DataLoader(train_graphs, batch_size=min(batch_size, len(train_graphs)))
        batch = next(iter(loader)).to(dev)
        loss = _batch_loss_single_head(model, batch, e_w, f_w)
        log.info("Dry-run forward OK; loss=%.6f", float(loss.detach()))
        return output_dir

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size) if val_graphs else None

    with track_cost(report):
        for epoch in range(epochs):
            model.eval()
            running = 0.0
            for batch in train_loader:
                batch = batch.to(dev)
                optimizer.zero_grad(set_to_none=True)
                loss = _batch_loss_single_head(model, batch, e_w, f_w)
                loss.backward()
                optimizer.step()
                running += float(loss.detach()) * batch.num_graphs
            train_mse = running / max(len(train_graphs), 1)
            val_msg = ""
            if val_loader is not None:
                vrun = 0.0
                for batch in val_loader:
                    batch = batch.to(dev)
                    vloss = _batch_loss_single_head(model, batch, e_w, f_w)
                    vrun += float(vloss.detach()) * batch.num_graphs
                val_msg = f" val={vrun / max(len(val_graphs), 1):.6f}"
            log.info("Epoch %d/%d: train=%.6f%s", epoch + 1, epochs, train_mse, val_msg)

    # --- save logdir ---
    ckpt_out = output_dir / checkpoint_name
    torch.save({"model_state_dict": model.state_dict()}, ckpt_out)
    # Save the ORIGINAL GFM config so eval can rebuild backbone + re-apply surgery.
    shutil.copyfile(config_path, output_dir / "config.json")
    (output_dir / "newhead.json").write_text(
        json.dumps(
            {
                "strategy": strategy,
                "freeze_mode": freeze_mode,
                "load_pretrained": load_pretrained,
                "checkpoint": checkpoint_name,
                "ft_config": ft_config,
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
    parser.add_argument("--gfm-checkpoint", default=None, help="GFM .pk filename/path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--force-weight", type=float, default=None)
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
        gfm_checkpoint=args.gfm_checkpoint,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
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
