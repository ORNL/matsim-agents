"""HydraGNN fine-tuning data helpers: paths, ASE -> graph conversion, checkpoints, splits."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from matsim_agents.core.seeding import numpy_rng

_COMP_DIM = 118


def _resolve_hydragnn_paths(hydragnn_root: str | Path | None) -> tuple[Path, Path]:
    """Return ``(hydragnn_root, example_dir)`` and put both on ``sys.path``."""
    source = hydragnn_root or os.environ.get("HYDRAGNN_ROOT")
    if not source:
        raise ValueError("Set hydragnn_root or HYDRAGNN_ROOT to the HydraGNN checkout")
    root = Path(source).expanduser().resolve()
    if not (root / "hydragnn").is_dir():
        raise FileNotFoundError(f"hydragnn package not found under {root}")
    example_dir = root / "examples" / "multidataset_hpo_sc26"
    for p in (str(root), str(example_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root, example_dir


def composition_vector(atoms: Atoms) -> np.ndarray:
    """Length-118 per-element atom-count histogram (Z = 1..118)."""
    z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    hist, _ = np.histogram(z, bins=range(1, _COMP_DIM + 2))
    return hist.astype(np.float32)


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
        return np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
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
    hist = composition_vector(atoms)

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


def _find_gfm_checkpoint(logdir: Path, checkpoint: str | None) -> Path:
    if checkpoint:
        p = Path(checkpoint)
        return p if p.is_absolute() else (logdir / checkpoint)
    cands = sorted(logdir.glob("*.pk"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No .pk checkpoint in {logdir}")
    return cands[-1]


def load_gfm_state_dict(gfm_logdir: Path, checkpoint: str | None, dev: Any) -> tuple[Path, dict]:
    """Load a GFM checkpoint's state dict with any DDP ``module.`` prefixes stripped."""
    import torch

    ckpt_path = _find_gfm_checkpoint(gfm_logdir, checkpoint)
    state = torch.load(ckpt_path, map_location=dev)
    state_dict = state.get("model_state_dict", state)
    state_dict = {
        (k[len("module.") :] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
    return ckpt_path, state_dict


def _split(graphs, val_fraction, seed):
    rng = numpy_rng(seed)
    idx = np.arange(len(graphs))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(graphs) * val_fraction))) if len(graphs) > 1 else 0
    val_idx = set(idx[:n_val].tolist())
    train = [g for i, g in enumerate(graphs) if i not in val_idx]
    val = [g for i, g in enumerate(graphs) if i in val_idx]
    return train, val
