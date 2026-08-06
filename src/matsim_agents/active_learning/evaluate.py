"""Evaluate an (optionally fine-tuned) MLIP against a held-out DFT test set.

This is the accuracy counterpart to the active-learning loop: given a trained
model and a fixed set of DFT-labelled structures, it computes energy/force
error metrics and parity data so we can quantify how HydraGNN or UMA *behave
after* being fine-tuned on the AL-collected labels.

The held-out test set is an extended-XYZ file in the same schema written by
:func:`matsim_agents.active_learning.trainer.append_frames_to_extxyz`
(reference energy in ``atoms.info['energy']``, forces in
``atoms.arrays['forces']``). Standard ASE/extxyz calculator results are used
as a fallback when those keys are absent.

The model is described by an :class:`~matsim_agents.active_learning.config.ALConfig`
YAML (its ``mlip`` block selects the backend). ``--model-path`` overrides the
active checkpoint so the *same* config can be pointed at each AL iteration's
fine-tuned model (HydraGNN logdir or UMA model name/checkpoint dir).

Example::

    python -m matsim_agents.active_learning.evaluate \\
        --al-config examples/paper_cases/al_zn_formate_uma.yaml \\
        --test-set runs/al-zn-formate/test_set.extxyz \\
        --model-path runs/al-zn-formate/iter2_model \\
        --iteration 2 \\
        --out-json runs/al-zn-formate/eval/iter2.json \\
        --parity-npz runs/al-zn-formate/eval/iter2_parity.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from matsim_agents.active_learning.config import ALConfig, MLIPConfig

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reference-label extraction                                                  #
# --------------------------------------------------------------------------- #


def _reference_energy(atoms: Atoms) -> float | None:
    """Reference total energy (eV): prefer our ``info['energy']`` schema."""
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    try:
        return float(atoms.get_potential_energy())
    except Exception:  # noqa: BLE001 — no reference available
        return None


def _reference_forces(atoms: Atoms) -> np.ndarray | None:
    """Reference forces (eV/A), shape (N, 3): prefer our ``arrays['forces']``."""
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"], dtype=float)
    try:
        return np.asarray(atoms.get_forces(), dtype=float)
    except Exception:  # noqa: BLE001 — no reference available
        return None


# --------------------------------------------------------------------------- #
# Metrics container                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class EvalMetrics:
    """Scalar accuracy metrics for one model on one test set."""

    backend: str
    model_path: str
    iteration: int | None
    test_set: str
    n_frames_total: int
    n_frames_evaluated: int
    n_atoms_total: int

    # Energy (per structure).
    energy_mae_eV: float
    energy_rmse_eV: float
    # Energy (per atom) — the standard size-intensive metric.
    energy_mae_eV_per_atom: float
    energy_rmse_eV_per_atom: float
    # Energy (per atom) after removing a per-element *linear* reference from the
    # (pred-ref) energy difference (E_lin = sum_Z n_Z c_Z, fit by least squares).
    # MLIP backends (esp. UMA) use their own per-element energy zero, so this
    # composition-dependent shift -- the same linear-reference trick the
    # fine-tune training applies -- isolates the *relative* error and reduces to
    # the old constant offset when every frame shares one composition.
    energy_mae_eV_per_atom_shifted: float
    energy_rmse_eV_per_atom_shifted: float
    energy_mean_offset_eV_per_atom: float

    # Forces (per component, offset-invariant — the robust headline metric).
    force_mae_eV_per_A: float
    force_rmse_eV_per_A: float

    failures: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Core evaluation                                                             #
# --------------------------------------------------------------------------- #


def evaluate_frames(
    mlip_cfg: MLIPConfig,
    frames: list[Atoms],
    *,
    iteration: int | None = None,
    model_path: str | None = None,
    test_set_label: str = "",
) -> tuple[EvalMetrics, dict[str, np.ndarray]]:
    """Run single-points with ``mlip_cfg`` and score them against references.

    Returns ``(metrics, parity)`` where ``parity`` holds the raw arrays for
    scatter plots: per-atom reference/predicted energies and flattened
    reference/predicted force components.
    """
    from matsim_agents.active_learning.calculator import make_mlip_calculator

    calc = make_mlip_calculator(mlip_cfg)

    e_ref_pa: list[float] = []
    e_pred_pa: list[float] = []
    e_ref_tot: list[float] = []
    e_pred_tot: list[float] = []
    e_natoms: list[int] = []
    e_numbers: list[np.ndarray] = []
    f_ref_all: list[np.ndarray] = []
    f_pred_all: list[np.ndarray] = []
    n_atoms_total = 0
    failures: list[str] = []

    for i, atoms in enumerate(frames):
        e_ref = _reference_energy(atoms)
        f_ref = _reference_forces(atoms)
        if e_ref is None and f_ref is None:
            failures.append(f"frame {i}: no reference energy or forces")
            continue
        try:
            probe = atoms.copy()
            probe.calc = calc
            e_pred = float(probe.get_potential_energy())
            f_pred = np.asarray(probe.get_forces(), dtype=float)
        except Exception as exc:  # noqa: BLE001 — record and skip bad frames
            failures.append(f"frame {i}: prediction failed: {exc}")
            continue

        n = len(atoms)
        n_atoms_total += n
        if e_ref is not None:
            e_ref_tot.append(e_ref)
            e_pred_tot.append(e_pred)
            e_ref_pa.append(e_ref / n)
            e_pred_pa.append(e_pred / n)
            e_natoms.append(n)
            e_numbers.append(np.asarray(atoms.get_atomic_numbers(), dtype=int))
        if f_ref is not None and f_ref.shape == f_pred.shape:
            f_ref_all.append(f_ref.reshape(-1))
            f_pred_all.append(f_pred.reshape(-1))

    n_eval = max(len(e_ref_tot), len(f_ref_all))
    if n_eval == 0:
        raise RuntimeError(
            f"No frames could be evaluated ({len(failures)} failures). "
            "Check the test set labels and the model path."
        )

    e_ref_tot_a = np.asarray(e_ref_tot)
    e_pred_tot_a = np.asarray(e_pred_tot)
    e_ref_pa_a = np.asarray(e_ref_pa)
    e_pred_pa_a = np.asarray(e_pred_pa)

    de_tot = e_pred_tot_a - e_ref_tot_a
    de_pa = e_pred_pa_a - e_ref_pa_a
    offset_pa = float(np.mean(de_pa)) if de_pa.size else 0.0
    # Remove a per-element linear reference (E_lin = sum_Z n_Z c_Z) from the
    # pred-ref total-energy difference, mirroring the reference-energy
    # subtraction the fine-tune training applies. This generalises the single
    # constant shift to a composition-dependent offset, so varying-stoichiometry
    # frames are scored on the shape of the energy surface rather than a
    # per-element reference mismatch; for a fixed composition it reduces to the
    # old constant shift. Applied identically to zero-shot and fine-tuned evals.
    if de_tot.size:
        zs = sorted({int(z) for nums in e_numbers for z in nums.tolist()})
        col = {z: j for j, z in enumerate(zs)}
        comp = np.zeros((len(e_numbers), len(zs)), dtype=float)
        for i_row, nums in enumerate(e_numbers):
            for z in nums.tolist():
                comp[i_row, col[int(z)]] += 1.0
        coef, *_ = np.linalg.lstsq(comp, de_tot, rcond=None)
        n_arr = np.asarray(e_natoms, dtype=float)
        de_pa_shifted = (de_tot - comp @ coef) / n_arr
    else:
        de_pa_shifted = de_pa

    if f_ref_all:
        f_ref_a = np.concatenate(f_ref_all)
        f_pred_a = np.concatenate(f_pred_all)
        df = f_pred_a - f_ref_a
        force_mae = float(np.mean(np.abs(df)))
        force_rmse = float(np.sqrt(np.mean(df**2)))
    else:
        f_ref_a = np.empty(0)
        f_pred_a = np.empty(0)
        force_mae = float("nan")
        force_rmse = float("nan")

    def _mae(x: np.ndarray) -> float:
        return float(np.mean(np.abs(x))) if x.size else float("nan")

    def _rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")

    metrics = EvalMetrics(
        backend=mlip_cfg.backend,
        model_path=model_path or "",
        iteration=iteration,
        test_set=test_set_label,
        n_frames_total=len(frames),
        n_frames_evaluated=n_eval,
        n_atoms_total=n_atoms_total,
        energy_mae_eV=_mae(de_tot),
        energy_rmse_eV=_rmse(de_tot),
        energy_mae_eV_per_atom=_mae(de_pa),
        energy_rmse_eV_per_atom=_rmse(de_pa),
        energy_mae_eV_per_atom_shifted=_mae(de_pa_shifted),
        energy_rmse_eV_per_atom_shifted=_rmse(de_pa_shifted),
        energy_mean_offset_eV_per_atom=offset_pa,
        force_mae_eV_per_A=force_mae,
        force_rmse_eV_per_A=force_rmse,
        failures=failures,
    )
    parity = {
        "e_ref_eV_per_atom": e_ref_pa_a,
        "e_pred_eV_per_atom": e_pred_pa_a,
        "f_ref_eV_per_A": f_ref_a,
        "f_pred_eV_per_A": f_pred_a,
    }
    return metrics, parity


def _apply_model_override(cfg: ALConfig, model_path: str | None) -> None:
    """Point the active backend at ``model_path`` (in place)."""
    if model_path is None:
        return
    if cfg.mlip.backend == "hydragnn" and cfg.mlip.hydragnn is not None:
        cfg.mlip.hydragnn.logdir = Path(model_path)
    elif cfg.mlip.backend == "uma" and cfg.mlip.uma is not None:
        cfg.mlip.uma.model_name = model_path
    else:  # pragma: no cover — guarded by MLIPConfig validator
        raise ValueError(f"Cannot apply model override for backend {cfg.mlip.backend!r}")


def _subsample(parity: dict[str, np.ndarray], max_points: int) -> dict[str, np.ndarray]:
    """Cap the force parity arrays for compact plotting (energies are per-frame)."""
    f_ref = parity["f_ref_eV_per_A"]
    if max_points > 0 and f_ref.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(f_ref.size, size=max_points, replace=False)
        idx.sort()
        parity = dict(parity)
        parity["f_ref_eV_per_A"] = f_ref[idx]
        parity["f_pred_eV_per_A"] = parity["f_pred_eV_per_A"][idx]
    return parity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--al-config", required=True, help="AL config YAML (selects backend).")
    parser.add_argument("--test-set", required=True, help="Held-out DFT extxyz test set.")
    parser.add_argument("--out-json", required=True, help="Output metrics JSON path.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override the active checkpoint: HydraGNN logdir or UMA model name/dir.",
    )
    parser.add_argument("--iteration", type=int, default=None, help="AL iteration (metadata).")
    parser.add_argument("--parity-npz", default=None, help="Optional parity arrays output (.npz).")
    parser.add_argument(
        "--max-parity-points",
        type=int,
        default=20000,
        help="Cap on force parity points written (0 = keep all).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = ALConfig.from_yaml(args.al_config)
    _apply_model_override(cfg, args.model_path)

    frames = ase_read(args.test_set, index=":")
    if isinstance(frames, Atoms):
        frames = [frames]
    log.info("Loaded %d test frames from %s", len(frames), args.test_set)

    metrics, parity = evaluate_frames(
        cfg.mlip,
        frames,
        iteration=args.iteration,
        model_path=args.model_path,
        test_set_label=str(args.test_set),
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(asdict(metrics), indent=2))
    log.info(
        "iter=%s  E_MAE=%.4f eV/atom (shifted %.4f)  F_MAE=%.4f eV/A  -> %s",
        metrics.iteration,
        metrics.energy_mae_eV_per_atom,
        metrics.energy_mae_eV_per_atom_shifted,
        metrics.force_mae_eV_per_A,
        out_json,
    )

    if args.parity_npz:
        parity_path = Path(args.parity_npz)
        parity_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(parity_path, **_subsample(parity, args.max_parity_points))
        log.info("Wrote parity arrays -> %s", parity_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
