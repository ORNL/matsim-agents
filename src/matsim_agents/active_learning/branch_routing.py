"""Map AL datasets to HydraGNN GFM branch heads via the branch-weighting MLP.

The SC26 multidataset HydraGNN foundation model has 16 branch heads, one per
training dataset. A small composition-conditioned MLP
(``mlp_branch_weights.pt``) predicts, for a given structure, a softmax weight
over those 16 branches. We reuse it here to pick -- per fine-tuning dataset --
the single branch head whose training chemistry the model considers most
similar, so HydraGNN fine-tuning can update only that head.

The MLP consumes ``chemical_composition``: a length-118 vector of per-element
atom counts (``np.histogram(Z, bins=range(1, 120))``), matching
``multidataset_hpo_sc26/structure_optimization_ASE.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

log = logging.getLogger(__name__)

# Branch index -> training dataset alias (order of --multi_model_list /
# datadir0..15 in the SC26 GFM job scripts).
BRANCH_DATASETS: tuple[str, ...] = (
    "Alexandria",       # 0  inorganic bulk (DFT-PBE)
    "ANI1x",            # 1  organic molecules
    "MPTrj",            # 2  inorganic bulk (Materials Project trajectories)
    "OC2020",           # 3  catalysis surfaces/adsorbates
    "OC2022",           # 4  catalysis surfaces/adsorbates
    "OC25",             # 5  catalysis
    "ODAC23",           # 6  MOFs / direct-air-capture
    "OMat24",           # 7  inorganic bulk materials
    "OMol25",           # 8  molecules
    "OMol25-neutral",   # 9  molecules (neutral)
    "OMol25-non-neutral",  # 10 molecules (charged)
    "OPoly2026",        # 11 polymers
    "Nabla2DFT",        # 12 molecules
    "QCML",             # 13 molecules (quantum-chemistry ML)
    "QM7X",             # 14 small organic molecules
    "transition1x",     # 15 organic reaction transition states
)
NUM_BRANCHES = len(BRANCH_DATASETS)
_HIDDEN_DIMS: tuple[int, ...] = (128, 64)
_COMP_DIM = 118


def _build_branch_mlp(num_branches: int, hidden_dims: tuple[int, ...]):
    """Replicate ``multidataset_hpo_sc26.branch_weighting_mlp.BranchWeightMLP``.

    Kept local so we avoid importing the training script (which pulls in
    mpi4py/hydragnn at module load). Architecture: LazyLinear -> ReLU ->
    (Linear -> ReLU)* -> Linear(num_branches).
    """
    import torch.nn as nn

    layers: list = [nn.LazyLinear(hidden_dims[0]), nn.ReLU()]
    in_dim = hidden_dims[0]
    for h in hidden_dims[1:]:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, num_branches))
    return nn.Sequential(*layers)


def load_branch_mlp(
    checkpoint: str | Path,
    *,
    num_branches: int = NUM_BRANCHES,
    hidden_dims: tuple[int, ...] = _HIDDEN_DIMS,
):
    """Load the composition->branch-weight MLP from ``mlp_branch_weights.pt``."""
    import torch

    mlp = _build_branch_mlp(num_branches, hidden_dims)
    # LazyLinear materialises its weight on first forward; run a dummy pass so
    # load_state_dict(strict=True) has concrete parameters to match.
    mlp(torch.zeros(1, _COMP_DIM))
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt.get("mlp_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Original module stored the Sequential under ``self.net`` -> strip prefix
    # so keys line up with our bare nn.Sequential.
    state = {k[len("net."):] if k.startswith("net.") else k: v for k, v in state.items()}
    mlp.load_state_dict(state, strict=True)
    mlp.eval()
    return mlp


def composition_vector(atoms: Atoms) -> np.ndarray:
    """Length-118 per-element atom-count histogram (Z = 1..118)."""
    z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    hist, _ = np.histogram(z, bins=range(1, _COMP_DIM + 2))
    return hist.astype(np.float32)


def branch_weights_for_frames(mlp, frames: list[Atoms]) -> np.ndarray:
    """Mean softmax branch weight over ``frames`` (shape ``[num_branches]``)."""
    import torch
    import torch.nn.functional as F

    param = next(mlp.parameters())
    comps = np.stack([composition_vector(a) for a in frames], axis=0)
    with torch.no_grad():
        x = torch.from_numpy(comps).to(device=param.device, dtype=param.dtype)
        logits = mlp(x)
        weights = F.softmax(logits, dim=-1)
    return weights.mean(dim=0).cpu().numpy()


def route_dataset(mlp, dataset_path: str | Path) -> dict:
    """Return the dominant branch for ``dataset_path`` and the weight profile."""
    raw = ase_read(str(dataset_path), index=":")
    frames = [raw] if isinstance(raw, Atoms) else list(raw)
    if not frames:
        raise ValueError(f"No frames read from {dataset_path}")
    mean_w = branch_weights_for_frames(mlp, frames)
    dominant = int(np.argmax(mean_w))
    order = np.argsort(mean_w)[::-1]
    return {
        "dataset_path": str(dataset_path),
        "n_frames": len(frames),
        "dominant_branch": dominant,
        "dominant_dataset": BRANCH_DATASETS[dominant],
        "dominant_weight": float(mean_w[dominant]),
        "top3": [(int(i), BRANCH_DATASETS[i], float(mean_w[i])) for i in order[:3]],
        "mean_weights": mean_w.tolist(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--branch-mlp", required=True, help="mlp_branch_weights.pt path.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="PATH",
        help="AL dataset extxyz (repeatable).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    mlp = load_branch_mlp(args.branch_mlp)
    for ds in args.dataset:
        r = route_dataset(mlp, ds)
        top3 = "  ".join(f"{name}({w:.3f})" for _, name, w in r["top3"])
        print(
            f"{Path(ds).parent.name or ds}: branch-{r['dominant_branch']} "
            f"[{r['dominant_dataset']}] w={r['dominant_weight']:.3f}  "
            f"(n={r['n_frames']})  top3: {top3}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
