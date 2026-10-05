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
        --elemental-reference-manifest runs/al-zn-formate/elemental_references.json \\
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

from matsim_agents.active_learning.config import ALConfig, MLIPConfig, TrainerConfig
from matsim_agents.active_learning.dataset_governance import structure_identity
from matsim_agents.discovery.energy_references import (
    load_elemental_reference_manifest,
    predict_elemental_references,
    validate_dataset_reference_method,
)

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
    n_energy_frames_evaluated: int
    n_force_frames_evaluated: int
    n_atoms_total: int

    # Energy (per structure).
    energy_mae_eV: float
    energy_rmse_eV: float
    # Energy (per atom) — the standard size-intensive metric.
    energy_mae_eV_per_atom: float
    energy_rmse_eV_per_atom: float
    # Deprecated output aliases for formation-energy errors; no fitted shifts.
    energy_mae_eV_per_atom_shifted: float
    energy_rmse_eV_per_atom_shifted: float
    energy_mean_offset_eV_per_atom: float

    # Forces (per component, offset-invariant — the robust headline metric).
    force_mae_eV_per_A: float
    force_rmse_eV_per_A: float

    failures: list[str] = field(default_factory=list)
    formation_energy_mae_eV_per_atom: float = float("nan")
    formation_energy_rmse_eV_per_atom: float = float("nan")
    elemental_reference_provenance: dict[str, object] = field(default_factory=dict)


@dataclass
class PromotionDecision:
    """Auditable held-out comparison between a candidate and incumbent model."""

    approved: bool
    reasons: list[str]
    candidate_metrics: dict[str, object]
    incumbent_metrics: dict[str, object]


def assess_promotion(
    candidate: EvalMetrics,
    incumbent: EvalMetrics,
    trainer: TrainerConfig,
) -> PromotionDecision:
    """Apply absolute accuracy and incumbent-regression promotion gates."""
    reasons: list[str] = []
    minimum = trainer.promotion_min_evaluated_frames
    for model_name, metrics in (("candidate", candidate), ("incumbent", incumbent)):
        for label, count in (
            ("energy", metrics.n_energy_frames_evaluated),
            ("force", metrics.n_force_frames_evaluated),
        ):
            if count < minimum:
                reasons.append(
                    f"{model_name} evaluated {count} {label}-labelled frames; minimum is {minimum}"
                )
    metrics = (
        (
            "formation_energy_mae_eV_per_atom",
            candidate.formation_energy_mae_eV_per_atom,
            incumbent.formation_energy_mae_eV_per_atom,
            trainer.promotion_max_energy_mae_eV_per_atom,
        ),
        (
            "force_mae_eV_per_A",
            candidate.force_mae_eV_per_A,
            incumbent.force_mae_eV_per_A,
            trainer.promotion_max_force_mae_eV_per_A,
        ),
    )
    for name, candidate_value, incumbent_value, absolute_limit in metrics:
        if not np.isfinite(candidate_value):
            reasons.append(f"candidate {name} is not finite")
            continue
        if candidate_value > absolute_limit:
            reasons.append(
                f"candidate {name}={candidate_value:.6g} exceeds limit {absolute_limit:.6g}"
            )
        if not np.isfinite(incumbent_value):
            reasons.append(f"incumbent {name} is not finite")
            continue
        regression_limit = incumbent_value * (1.0 + trainer.promotion_max_relative_regression)
        if candidate_value > regression_limit + 1e-12:
            reasons.append(
                f"candidate {name}={candidate_value:.6g} exceeds incumbent regression "
                f"limit {regression_limit:.6g}"
            )
    return PromotionDecision(
        approved=not reasons,
        reasons=reasons,
        candidate_metrics=asdict(candidate),
        incumbent_metrics=asdict(incumbent),
    )


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
    ref_frames: list[Atoms] | None = None,
    elemental_reference_manifest: Path | None = None,
) -> tuple[EvalMetrics, dict[str, np.ndarray]]:
    """Run single-points with ``mlip_cfg`` and score them against references.

    Returns ``(metrics, parity)`` where ``parity`` holds the raw arrays for
    scatter plots: per-atom reference/predicted energies and flattened
    reference/predicted force components.

    Energy comparisons require a DFT-labelled pure-element manifest. The model
    evaluates those fixed geometries first, then both methods subtract their
    own elemental baselines. No offsets are fitted to training or test data.
    ``ref_frames`` is retained only to reject obsolete fitted-offset callers.
    """
    from matsim_agents.active_learning.calculator import make_mlip_calculator

    if ref_frames is not None:
        raise ValueError(
            "ref_frames energy-offset fitting is unsupported; "
            "provide an elemental reference manifest"
        )
    has_energy_labels = any(_reference_energy(frame) is not None for frame in frames)
    if has_energy_labels and elemental_reference_manifest is None:
        raise ValueError("energy evaluation requires a DFT-labelled elemental reference manifest")
    if test_set_label and elemental_reference_manifest is not None:
        manifest, _, _ = load_elemental_reference_manifest(
            elemental_reference_manifest, required_elements=set()
        )
        validate_dataset_reference_method(
            test_set_label,
            reference_backend=manifest["backend"],
            reference_method_signature=manifest["method_signature"],
        )
    calc = make_mlip_calculator(mlip_cfg)
    references = (
        predict_elemental_references(
            elemental_reference_manifest,
            calc,
            required_elements={
                symbol for frame in frames for symbol in frame.get_chemical_symbols()
            },
        )
        if elemental_reference_manifest is not None
        else None
    )

    e_ref_pa: list[float] = []
    e_pred_pa: list[float] = []
    e_ref_tot: list[float] = []
    e_pred_tot: list[float] = []
    f_ref_all: list[np.ndarray] = []
    f_pred_all: list[np.ndarray] = []
    n_atoms_total = 0
    failures: list[str] = []
    formation_ref: list[float] = []
    formation_pred: list[float] = []

    for i, atoms in enumerate(frames):
        e_ref = _reference_energy(atoms)
        f_ref = _reference_forces(atoms)
        if e_ref is None and f_ref is None:
            failures.append(f"frame {i}: no reference energy or forces")
            continue
        try:
            probe = atoms.copy()
            probe.calc = calc
            e_pred = float(probe.get_potential_energy()) if e_ref is not None else None
            f_pred = np.asarray(probe.get_forces(), dtype=float)
        except Exception as exc:  # noqa: BLE001 — record and skip bad frames
            failures.append(f"frame {i}: prediction failed: {exc}")
            continue

        n = len(atoms)
        n_atoms_total += n
        if e_ref is not None:
            if references is None or e_pred is None:
                raise ValueError("energy comparison requires elemental references and predictions")
            formation_ref.append(references.formation_energy(atoms, e_ref, model=False))
            formation_pred.append(references.formation_energy(atoms, e_pred, model=True))
            e_ref_tot.append(e_ref)
            e_pred_tot.append(e_pred)
            e_ref_pa.append(e_ref / n)
            e_pred_pa.append(e_pred / n)
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
    formation_ref_a = np.asarray(formation_ref)
    formation_pred_a = np.asarray(formation_pred)
    de_pa_shifted = formation_pred_a - formation_ref_a

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
        n_energy_frames_evaluated=len(e_ref_tot),
        n_force_frames_evaluated=len(f_ref_all),
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
        formation_energy_mae_eV_per_atom=_mae(de_pa_shifted),
        formation_energy_rmse_eV_per_atom=_rmse(de_pa_shifted),
        elemental_reference_provenance=references.provenance if references is not None else {},
    )
    parity = {
        "e_ref_eV_per_atom": e_ref_pa_a,
        "e_pred_eV_per_atom": e_pred_pa_a,
        "formation_ref_eV_per_atom": formation_ref_a,
        "formation_pred_eV_per_atom": formation_pred_a,
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
        cfg.mlip.hydragnn.checkpoint = None
    elif cfg.mlip.backend == "uma" and cfg.mlip.uma is not None:
        cfg.mlip.uma.model_name = model_path
    elif cfg.mlip.backend == "mace" and cfg.mlip.mace is not None:
        cfg.mlip.mace.family = "checkpoint"
        cfg.mlip.mace.model = model_path
    else:  # pragma: no cover — guarded by MLIPConfig validator
        raise ValueError(f"Cannot apply model override for backend {cfg.mlip.backend!r}")


def evaluate_promotion_candidate(
    cfg: ALConfig,
    candidate_model_path: str,
    *,
    iteration: int,
    training_set: Path,
) -> PromotionDecision:
    """Evaluate incumbent and candidate models on the configured held-out set."""
    validation_set = cfg.trainer.validation_set
    if validation_set is None:
        raise ValueError("model promotion requires trainer.validation_set")
    training_path = Path(training_set).resolve()
    validation_path = Path(validation_set).resolve()
    if validation_path == training_path:
        raise ValueError("trainer.validation_set must be held out from the training set")
    reference_path = cfg.trainer.validation_reference_set
    if reference_path is not None:
        resolved_reference = Path(reference_path).resolve()
        if resolved_reference == training_path:
            raise ValueError(
                "trainer.validation_reference_set must be held out from the training set"
            )
        if resolved_reference == validation_path:
            raise ValueError(
                "trainer.validation_reference_set must differ from trainer.validation_set"
            )
    validation_frames = list(ase_read(validation_set, index=":"))
    if training_path.is_file():
        training_frames = list(ase_read(training_path, index=":"))
        training_identities = {structure_identity(frame) for frame in training_frames}
        overlap = sum(
            structure_identity(frame) in training_identities for frame in validation_frames
        )
        if overlap:
            raise ValueError(
                "trainer.validation_set must be held out from the training set; "
                f"found {overlap} overlapping geometries"
            )
    if reference_path is None:
        raise ValueError("model comparison requires a DFT-labelled elemental reference manifest")
    manifest, _, _ = load_elemental_reference_manifest(reference_path, required_elements=set())
    if training_path.is_file():
        validate_dataset_reference_method(
            training_path,
            reference_backend=manifest["backend"],
            reference_method_signature=manifest["method_signature"],
        )

    incumbent_cfg = cfg.mlip.model_copy(deep=True)
    candidate_cfg = cfg.mlip.model_copy(deep=True)
    candidate_wrapper = cfg.model_copy(deep=True)
    candidate_wrapper.mlip = candidate_cfg
    _apply_model_override(candidate_wrapper, candidate_model_path)
    incumbent_metrics, _ = evaluate_frames(
        incumbent_cfg,
        validation_frames,
        iteration=iteration,
        test_set_label=str(validation_set),
        elemental_reference_manifest=reference_path,
    )
    candidate_metrics, _ = evaluate_frames(
        candidate_cfg,
        validation_frames,
        iteration=iteration,
        model_path=candidate_model_path,
        test_set_label=str(validation_set),
        elemental_reference_manifest=reference_path,
    )
    return assess_promotion(candidate_metrics, incumbent_metrics, cfg.trainer)


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
    parser.add_argument(
        "--elemental-reference-manifest",
        type=Path,
        help="Required for energy comparisons; unnecessary for force-only datasets.",
    )
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
        elemental_reference_manifest=args.elemental_reference_manifest,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(asdict(metrics), indent=2))
    log.info(
        "iter=%s  raw E_MAE=%.4f eV/atom (formation %.4f)  F_MAE=%.4f eV/A  -> %s",
        metrics.iteration,
        metrics.energy_mae_eV_per_atom,
        metrics.formation_energy_mae_eV_per_atom,
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
