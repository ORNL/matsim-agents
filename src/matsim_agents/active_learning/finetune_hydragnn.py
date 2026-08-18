"""Fine-tune the SC26 multi-branch HydraGNN foundation model on an AL dataset.

Per the project decision (option "a"): keep the shared backbone
(``graph_convs`` + ``graph_shared``) **and** the composition-conditioned
branch-weighting MLP **frozen**, and fine-tune only the branch head(s) that the
branch-MLP actually routes this dataset's chemistry to (see
:mod:`matsim_agents.active_learning.branch_routing`). This adapts the expert
readout the model uses at inference while preserving the foundation model's
general representations -- appropriate for the small AL datasets and avoiding
catastrophic forgetting.

Head parameters are named ``model.heads_NN.<i>.branch-<b>.<layer>.{weight,bias}``;
the backbone is ``model.graph_convs`` + ``model.graph_shared``. Freezing keeps
``requires_grad=True`` only for the routed branch heads.

The training objective mirrors inference (``inference_fused.run_fused_inference``):
a branch-weighted average of per-branch energy/forces, using the frozen
branch-MLP's softmax weights restricted+renormalised to the routed heads (the
non-routed heads carry ~0 weight for these compositions, so this matches the
full fused prediction).

The output is a logdir (``config.json`` + a ``.pk`` checkpoint) consumable by
:func:`matsim_agents.active_learning.calculator.build_hydragnn_calculator`,
plus a ``cost.json`` recording wall time / GPU-hours / peak memory / parameter
counts.

Must run in the **hydragnn** venv (needs ``torch_scatter``/``hydragnn``), not
the fairchem venv.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from matsim_agents.active_learning.branch_routing import (
    BRANCH_DATASETS,
    branch_weights_for_frames,
    composition_vector,
    load_branch_mlp,
)
from matsim_agents.active_learning.cost import CostReport, count_parameters, track_cost

log = logging.getLogger(__name__)

_HEAD_PARAM_RE = re.compile(r"heads_NN\.\d+\.branch-(\d+)\.")


# --------------------------------------------------------------------------- #
# Environment / path setup                                                    #
# --------------------------------------------------------------------------- #


def _resolve_hydragnn_paths(hydragnn_root: str | Path | None) -> tuple[Path, Path]:
    """Return ``(hydragnn_root, example_dir)`` and put both on ``sys.path``."""
    root = (
        Path(
            hydragnn_root
            or os.environ.get("HYDRAGNN_ROOT")
            or "/global/cfs/projectdirs/m5216/mlupopa/HydraGNN"
        )
        .expanduser()
        .resolve()
    )
    if not (root / "hydragnn").is_dir():
        raise FileNotFoundError(f"hydragnn package not found under {root}")
    example_dir = root / "examples" / "multidataset_hpo_sc26"
    for p in (str(root), str(example_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root, example_dir


# --------------------------------------------------------------------------- #
# Dataset -> HydraGNN graphs                                                  #
# --------------------------------------------------------------------------- #


def _reference_energy(atoms: Atoms) -> float | None:
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    try:
        return float(atoms.get_potential_energy())
    except Exception:  # noqa: BLE001
        return None


def _reference_forces(atoms: Atoms) -> np.ndarray | None:
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"], dtype=float)
    try:
        return np.asarray(atoms.get_forces(), dtype=float)
    except Exception:  # noqa: BLE001
        return None


def _atoms_to_data(atoms: Atoms, add_edges_pbc, dtype, charge: float, spin: float):
    """Convert an ASE ``Atoms`` (with energy+forces) to a HydraGNN ``Data``."""
    import torch
    from torch_geometric.data import Data

    z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    hist = composition_vector(atoms)  # length-118 float32

    energy = _reference_energy(atoms)
    forces = _reference_forces(atoms)
    if energy is None or forces is None:
        return None

    data = Data(
        x=torch.tensor(z, dtype=dtype).unsqueeze(1),
        atomic_numbers=torch.tensor(z, dtype=torch.long),
        pos=torch.tensor(pos, dtype=dtype),
        chemical_composition=torch.tensor(hist, dtype=torch.float32).unsqueeze(1),
        graph_attr=torch.tensor([[float(charge), float(spin)]], dtype=torch.float32),
        natoms=torch.tensor([len(z)], dtype=torch.long),
        cell=torch.tensor(cell, dtype=dtype),
        pbc=torch.tensor(pbc, dtype=torch.bool),
        energy=torch.tensor([float(energy)], dtype=dtype),
        forces=torch.tensor(forces, dtype=dtype),
    )
    return add_edges_pbc(data)


# --------------------------------------------------------------------------- #
# Per-element linear reference energies                                        #
# --------------------------------------------------------------------------- #


def _fit_reference_energies(graphs) -> dict[int, float]:
    """Least-squares per-element linear reference energies ``e_ref[Z]``.

    Solves ``E_total_i approx sum_Z n_{i,Z} * e_ref[Z]`` over ``graphs`` (each
    carrying a TOTAL DFT energy in eV). Subtracting the per-structure reference
    sum from the DFT total leaves a small formation-scale target that matches
    the formation energy HydraGNN's head predicts, which keeps the fine-tune
    from regressing the large raw-energy offset. Fit on the train split only to
    avoid leakage.
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


def _reshape_composition(data, num_graphs: int):
    """Return composition as ``[num_graphs, 118]`` (mirrors inference_fused)."""
    comp = data.chemical_composition
    if comp.dim() == 1:
        comp = comp.unsqueeze(0)
    if comp.dim() == 2:
        if comp.size(0) == num_graphs:
            return comp
        if comp.size(1) == num_graphs:
            return comp.t()
        if comp.size(0) % num_graphs == 0:
            return comp.view(num_graphs, -1)
    if comp.dim() == 3 and comp.size(0) == num_graphs:
        return comp.view(num_graphs, -1)
    raise ValueError(f"Unsupported chemical_composition shape {tuple(comp.shape)}")


# --------------------------------------------------------------------------- #
# Freezing                                                                    #
# --------------------------------------------------------------------------- #


def _apply_freezing(model, routed: set[int], *, unfreeze_backbone: bool) -> None:
    """Freeze everything except the routed branch heads (and optionally backbone)."""
    for name, p in model.named_parameters():
        m = _HEAD_PARAM_RE.search(name)
        if m is not None:
            p.requires_grad_(int(m.group(1)) in routed)
        elif ("graph_convs" in name or "graph_shared" in name) and unfreeze_backbone:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


# --------------------------------------------------------------------------- #
# Routing                                                                     #
# --------------------------------------------------------------------------- #


def _select_routed_branches(
    branch_mlp_path: str | Path,
    frames: list[Atoms],
    *,
    weight_threshold: float,
) -> tuple[list[int], np.ndarray]:
    """Return routed branch indices (weight >= threshold, always >=1) + weights."""
    mlp = load_branch_mlp(branch_mlp_path)
    mean_w = branch_weights_for_frames(mlp, frames)
    routed = [i for i in range(len(mean_w)) if mean_w[i] >= weight_threshold]
    if not routed:
        routed = [int(np.argmax(mean_w))]
    return routed, mean_w


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def finetune_hydragnn(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    gfm_logdir: str | Path,
    branch_mlp_path: str | Path,
    gfm_checkpoint: str | None = None,
    routed_branches: list[int] | None = None,
    weight_threshold: float = 0.1,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 4,
    val_fraction: float = 0.1,
    seed: int = 0,
    energy_weight: float | None = None,
    force_weight: float | None = None,
    unfreeze_backbone: bool = False,
    charge: float = 0.0,
    spin: float = 0.0,
    hydragnn_root: str | Path | None = None,
    device: str | None = None,
    checkpoint_name: str = "ft_model.pk",
    run: bool = True,
) -> Path:
    """Fine-tune the routed HydraGNN head(s) on ``dataset_path``.

    Returns the output logdir (containing ``config.json`` + ``checkpoint_name``).
    With ``run=False`` the pipeline is built and a single forward/loss step is
    executed (dry-run validation) but no optimisation/epoch loop runs.
    """
    root, example_dir = _resolve_hydragnn_paths(hydragnn_root)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gfm_logdir = Path(gfm_logdir).expanduser().resolve()

    import torch
    from hydragnn.models.create import create_model_config
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
    precision, param_dtype, _ = resolve_precision(precision_str)
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

    # --- routing ---
    if routed_branches is None:
        routed, mean_w = _select_routed_branches(
            branch_mlp_path, frames, weight_threshold=weight_threshold
        )
    else:
        routed = sorted(set(routed_branches))
        mean_w = np.zeros(len(BRANCH_DATASETS), dtype=float)
    log.info(
        "Routed branches: %s (%s)",
        routed,
        ", ".join(f"branch-{b}[{BRANCH_DATASETS[b]}]" for b in routed),
    )

    # --- build + load model ---
    model = create_model_config(config=hcfg["NeuralNetwork"], verbosity=0)
    torch.set_default_dtype(param_dtype)
    model = model.to(dtype=param_dtype, device=dev)
    ckpt_path = _find_gfm_checkpoint(gfm_logdir, gfm_checkpoint)
    state = torch.load(ckpt_path, map_location=dev)
    state_dict = state.get("model_state_dict", state)
    state_dict = {
        (k[len("module.") :] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # no BN/dropout in the heads; keeps frozen backbone stats fixed
    _apply_freezing(model, set(routed), unfreeze_backbone=unfreeze_backbone)
    trainable_params, total_params = count_parameters(model)
    log.info(
        "HydraGNN fine-tune: %d / %d params trainable (%.3f%%)",
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

    e_ref_map = _fit_reference_energies(train_graphs)
    _attach_reference_energies(train_graphs, e_ref_map, param_dtype)
    _attach_reference_energies(val_graphs, e_ref_map, param_dtype)
    log.info(
        "Fitted per-element reference energies (%d elements): %s",
        len(e_ref_map),
        {int(k): round(v, 3) for k, v in e_ref_map.items()},
    )

    branch_mlp = load_branch_mlp(branch_mlp_path).to(device=dev, dtype=param_dtype)
    routed_t = torch.tensor(routed, dtype=torch.long, device=dev)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

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
            "routed_branches": routed,
            "routed_datasets": [BRANCH_DATASETS[b] for b in routed],
            "mean_branch_weights": mean_w.tolist(),
            "unfreeze_backbone": unfreeze_backbone,
            "energy_weight": e_w,
            "force_weight": f_w,
            "protocol": "routed-head-finetune",
        },
    )

    if not run:
        # Dry-run: single forward/loss step on one small batch to validate wiring.
        loader = DataLoader(train_graphs, batch_size=min(batch_size, len(train_graphs)))
        batch = next(iter(loader)).to(dev)
        loss = _batch_loss(model, branch_mlp, batch, routed_t, e_w, f_w, param_dtype)
        log.info("Dry-run forward OK; loss=%.6f", float(loss.detach()))
        return output_dir

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size) if val_graphs else None

    # Keep the checkpoint with the lowest validation loss, not the final epoch:
    # on these small AL datasets the loss overfits well before the last epoch.
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    with track_cost(report):
        for epoch in range(epochs):
            model.eval()
            running = 0.0
            for batch in train_loader:
                batch = batch.to(dev)
                optimizer.zero_grad(set_to_none=True)
                loss = _batch_loss(model, branch_mlp, batch, routed_t, e_w, f_w, param_dtype)
                loss.backward()
                optimizer.step()
                running += float(loss.detach()) * batch.num_graphs
            train_mse = running / max(len(train_graphs), 1)
            val_msg = ""
            selection_mse = train_mse
            if val_loader is not None:
                vrun = 0.0
                for batch in val_loader:
                    batch = batch.to(dev)
                    vloss = _batch_loss(model, branch_mlp, batch, routed_t, e_w, f_w, param_dtype)
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
    shutil.copyfile(config_path, output_dir / "config.json")
    report.write(output_dir / "cost.json")
    (output_dir / "routing.json").write_text(
        json.dumps(
            {
                "routed_branches": routed,
                "routed_datasets": [BRANCH_DATASETS[b] for b in routed],
                "mean_branch_weights": mean_w.tolist(),
                "checkpoint": checkpoint_name,
                "reference_energies": {str(k): v for k, v in e_ref_map.items()},
            },
            indent=2,
        )
    )
    log.info(
        "HydraGNN fine-tune complete -> %s (%.1fs, %.4f GPU-h)",
        output_dir,
        report.wall_time_s,
        report.gpu_hours,
    )
    return output_dir


def _batch_loss(model, branch_mlp, batch, routed_t, e_w, f_w, param_dtype):
    """Weighted-average energy+force MSE over the routed branch heads."""
    import torch
    import torch.nn.functional as F

    num_graphs = int(batch.num_graphs)
    batch.pos.requires_grad_(True)

    comp = _reshape_composition(batch, num_graphs).to(device=batch.pos.device, dtype=param_dtype)
    logits = branch_mlp(comp)  # [G, 16]
    weights_all = F.softmax(logits, dim=-1)
    w_routed = weights_all.index_select(1, routed_t)  # [G, R]
    w_routed = w_routed / w_routed.sum(dim=1, keepdim=True).clamp_min(1e-12)

    node_counts = torch.bincount(batch.batch, minlength=num_graphs)

    weighted_energy = batch.pos.new_zeros(num_graphs)
    original = getattr(batch, "dataset_name", None)
    for j, b in enumerate(routed_t.tolist()):
        batch.dataset_name = torch.full(
            (num_graphs, 1), b, dtype=torch.long, device=batch.pos.device
        )
        pred = model(batch)
        energy = pred[0].squeeze(-1) if isinstance(pred, (list, tuple)) else pred.squeeze(-1)
        weighted_energy = weighted_energy + w_routed[:, j] * energy
    if original is None and hasattr(batch, "dataset_name"):
        delattr(batch, "dataset_name")
    elif original is not None:
        batch.dataset_name = original

    # Forces = -dE/dx (double-backprop so head params get force gradients).
    forces_pred = -torch.autograd.grad(
        weighted_energy,
        batch.pos,
        grad_outputs=torch.ones_like(weighted_energy),
        create_graph=True,
    )[0]

    # HydraGNN predicts a TOTAL formation energy (eV) while the DFT labels are
    # TOTAL energies (eV) on the raw reference. Subtract the per-structure linear
    # reference sum so the target is formation-scale and matches the head's
    # output, then normalise the loss to eV/atom (divide by node_counts).
    e_target = batch.energy.view(num_graphs)
    e_ref = (
        batch.e_ref.view(num_graphs)
        if hasattr(batch, "e_ref") and batch.e_ref is not None
        else torch.zeros_like(e_target)
    )
    e_form_target = e_target - e_ref
    e_loss = F.mse_loss(weighted_energy / node_counts, e_form_target / node_counts)
    f_loss = F.mse_loss(forces_pred, batch.forces)
    return e_w * e_loss + f_w * f_loss


def _find_gfm_checkpoint(logdir: Path, checkpoint: str | None) -> Path:
    if checkpoint:
        p = Path(checkpoint)
        return p if p.is_absolute() else (logdir / checkpoint)
    cands = sorted(logdir.glob("*.pk"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No .pk checkpoint in {logdir}")
    return cands[-1]


def _split(graphs, val_fraction, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(graphs))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(graphs) * val_fraction))) if len(graphs) > 1 else 0
    val_idx = set(idx[:n_val].tolist())
    train = [g for i, g in enumerate(graphs) if i not in val_idx]
    val = [g for i, g in enumerate(graphs) if i in val_idx]
    return train, val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="AL-collected extxyz dataset.")
    parser.add_argument("--output-dir", required=True, help="Output logdir.")
    parser.add_argument("--gfm-logdir", required=True, help="GFM dir (config.json + .pk).")
    parser.add_argument("--branch-mlp", required=True, help="mlp_branch_weights.pt path.")
    parser.add_argument("--gfm-checkpoint", default=None, help="GFM .pk filename/path.")
    parser.add_argument(
        "--routed-branches",
        default=None,
        help="Comma-separated branch indices; default = data-driven via branch-MLP.",
    )
    parser.add_argument("--weight-threshold", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--force-weight", type=float, default=None)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--charge", type=float, default=0.0)
    parser.add_argument("--spin", type=float, default=0.0)
    parser.add_argument("--hydragnn-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-name", default="ft_model.pk")
    parser.add_argument("--dry-run", action="store_true", help="Single forward/loss only.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    routed = None
    if args.routed_branches:
        routed = [int(x) for x in args.routed_branches.split(",") if x.strip()]

    finetune_hydragnn(
        args.dataset,
        args.output_dir,
        gfm_logdir=args.gfm_logdir,
        branch_mlp_path=args.branch_mlp,
        gfm_checkpoint=args.gfm_checkpoint,
        routed_branches=routed,
        weight_threshold=args.weight_threshold,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        unfreeze_backbone=args.unfreeze_backbone,
        charge=args.charge,
        spin=args.spin,
        hydragnn_root=args.hydragnn_root,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
        run=not args.dry_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
