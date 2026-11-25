"""Acquisition / uncertainty quantification for the active-learning loop.

Three strategies are implemented:

* ``ensemble``         — Variance over forces predicted by N independently
                          trained HydraGNN models (deep ensemble).
* ``mc_dropout``       — Predictive variance from K stochastic forward passes
                          with dropout enabled at inference time. For backends
                          without native dropout (e.g. UMA) dropout is injected
                          at load time (see :func:`inject_inference_dropout`).
* ``random``           — Baseline: uniformly subsample candidates.
* ``ensemble_then_dropout`` — Pre-screen with ensemble (cheap, single batch
                          per model) then refine the top 4·n_select with
                          MC-Dropout. Useful when the ensemble agrees on the
                          easy 95% of candidates.

All scorers return ``per-candidate score in eV/Å`` (force RMS-disagreement),
larger = more uncertain.

A simple greedy-farthest-point diversity filter on (composition fingerprint
+ score) is provided to avoid spending VASP on near-duplicate frames.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from matsim_agents.backends.mlip.dropout import enable_dropout
from matsim_agents.core.seeding import numpy_rng, seeded_torch
from matsim_agents.workflows.active_learning.candidates import Candidate
from matsim_agents.workflows.active_learning.config import AcquisitionConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathwayUncertainty:
    """Uncertainty of forward and reverse barriers evaluated on one fixed path."""

    forward_barrier_eV: float
    reverse_barrier_eV: float

    @property
    def pathway_eV(self) -> float:
        return max(self.forward_barrier_eV, self.reverse_barrier_eV)


# --------------------------------------------------------------------------- #
# Score implementations                                                       #
# --------------------------------------------------------------------------- #


def _forces_from(calc, atoms: Atoms) -> np.ndarray:
    a = atoms.copy()
    a.calc = calc
    return np.asarray(a.get_forces(), dtype=np.float64)


def score_ensemble(candidates: Sequence[Candidate], calculators: Sequence) -> np.ndarray:
    """Per-candidate force-disagreement (RMS over atoms of std-over-models)."""
    if len(calculators) < 2:
        raise ValueError("ensemble scoring needs >=2 calculators")
    out = np.zeros(len(candidates), dtype=np.float64)
    for i, cand in enumerate(candidates):
        per_model = np.stack([_forces_from(c, cand.atoms) for c in calculators], axis=0)
        # std over models, per atom & component, then RMS over (atom, component)
        std = per_model.std(axis=0)
        out[i] = float(np.sqrt(np.mean(std**2)))
    return out


def score_ensemble_pathway(images: Sequence[Atoms], calculators: Sequence) -> PathwayUncertainty:
    """Score barrier uncertainty from correlated per-model energies on a fixed path."""
    if len(images) < 2:
        raise ValueError("pathway scoring needs at least two images")
    if len(calculators) < 2:
        raise ValueError("ensemble pathway scoring needs >=2 calculators")

    forward, reverse = [], []
    for calculator in calculators:
        energies = []
        for image in images:
            atoms = image.copy()
            atoms.calc = calculator
            energies.append(float(atoms.get_potential_energy()))
        if not np.isfinite(energies).all():
            raise ValueError("ensemble returned non-finite pathway energies")
        peak = max(energies)
        forward.append(peak - energies[0])
        reverse.append(peak - energies[-1])
    return PathwayUncertainty(
        forward_barrier_eV=float(np.std(forward)),
        reverse_barrier_eV=float(np.std(reverse)),
    )


def score_mc_dropout_pathway(
    images: Sequence[Atoms],
    calculator,
    passes: int,
    p: float,
    *,
    seed: int,
) -> PathwayUncertainty:
    """Score fixed-path barrier uncertainty with a shared dropout draw per pass."""
    if len(images) < 2:
        raise ValueError("pathway scoring needs at least two images")
    if passes < 2:
        raise ValueError("MC-dropout pathway scoring needs at least two passes")
    if not 0.0 < p < 1.0:
        raise ValueError("MC-dropout probability must be between 0 and 1")
    if not hasattr(calculator, "model"):
        raise AttributeError(
            "MC-dropout pathway scoring requires a calculator exposing `.model`; got "
            + type(calculator).__name__
        )

    import torch
    import torch.nn as nn

    model = calculator.model
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.p = float(p)
    devices = {parameter.device for parameter in model.parameters()}
    forward, reverse = [], []
    try:
        for pass_index in range(passes):
            with seeded_torch(seed, "mc-dropout-pathway", pass_index, devices=devices):
                cpu_state = torch.random.get_rng_state()
                cuda_devices = [device for device in devices if device.type == "cuda"]
                cuda_states = {device: torch.cuda.get_rng_state(device) for device in cuda_devices}
                energies = []
                for image in images:
                    torch.random.set_rng_state(cpu_state)
                    for device, state in cuda_states.items():
                        torch.cuda.set_rng_state(state, device)
                    enable_dropout(model)
                    reset = getattr(calculator, "reset", None)
                    if callable(reset):
                        reset()
                    atoms = image.copy()
                    atoms.calc = calculator
                    energies.append(float(atoms.get_potential_energy()))
            if not np.isfinite(energies).all():
                raise ValueError("MC-dropout returned non-finite pathway energies")
            peak = max(energies)
            forward.append(peak - energies[0])
            reverse.append(peak - energies[-1])
    finally:
        model.eval()
    return PathwayUncertainty(
        forward_barrier_eV=float(np.std(forward)),
        reverse_barrier_eV=float(np.std(reverse)),
    )


def score_mc_dropout(
    candidates: Sequence[Candidate], calculator, passes: int, p: float, *, seed: int
) -> np.ndarray:
    """K stochastic forward passes; score = RMS std of forces across passes.

    The dropout masks come from the ``"mc-dropout"`` stream of *seed*.
    """
    # Most ASE-style HydraGNN calculators expose the underlying torch model
    # via ``calculator.model``. We toggle dropout layers into train() mode
    # while keeping the rest of the model in eval() (no BatchNorm updates).
    if not hasattr(calculator, "model"):
        raise AttributeError(
            "score_mc_dropout requires the calculator to expose `.model` "
            "(the underlying torch module). Got: " + type(calculator).__name__
        )

    # Override existing dropout p if desired
    import torch.nn as nn

    if p > 0:
        for m in calculator.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                m.p = float(p)

    out = np.zeros(len(candidates), dtype=np.float64)
    devices = {parameter.device for parameter in calculator.model.parameters()}
    with seeded_torch(seed, "mc-dropout", devices=devices):
        for i, cand in enumerate(candidates):
            per_pass = []
            for _ in range(passes):
                enable_dropout(calculator.model)
                reset = getattr(calculator, "reset", None)
                if callable(reset):
                    reset()
                per_pass.append(_forces_from(calculator, cand.atoms))
            stack = np.stack(per_pass, axis=0)
            out[i] = float(np.sqrt(np.mean(stack.std(axis=0) ** 2)))
    # Restore eval()
    calculator.model.eval()
    return out


def score_random(candidates: Sequence[Candidate], rng: np.random.Generator) -> np.ndarray:
    return rng.random(len(candidates))


# --------------------------------------------------------------------------- #
# Diversity filter                                                            #
# --------------------------------------------------------------------------- #


def _composition_fingerprint(atoms: Atoms) -> np.ndarray:
    """118-dim element-count fingerprint (cheap composition descriptor)."""
    z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    fp = np.zeros(118, dtype=np.float64)
    for zi in z:
        if 1 <= zi <= 118:
            fp[zi - 1] += 1.0
    n = float(np.sum(fp))
    return fp / n if n > 0 else fp


def greedy_farthest_point(
    candidates: Sequence[Candidate],
    scores: np.ndarray,
    n_select: int,
    fingerprint_weight: float = 1.0,
    score_weight: float = 1.0,
    *,
    group_labels: Sequence[str | None] | None = None,
    max_per_group: int | None = None,
) -> list[int]:
    """Pick ``n_select`` indices that are diverse in composition AND high-score.

    Uses farthest-point sampling on the joint (fp, score) feature, seeded by
    the highest-score candidate. Prefer to ``argsort(scores)[-n:]`` when you
    expect many near-duplicates.
    """
    if max_per_group is None and n_select >= len(candidates):
        return list(range(len(candidates)))
    if max_per_group is not None:
        if max_per_group < 1:
            raise ValueError("max_per_group must be positive")
        if group_labels is None or len(group_labels) != len(candidates):
            raise ValueError("group_labels must match candidates when max_per_group is set")

    fps = np.stack([_composition_fingerprint(c.atoms) for c in candidates], axis=0)
    fps *= float(fingerprint_weight)
    s = scores.reshape(-1, 1) * float(score_weight)
    feat = np.concatenate([fps, s], axis=1)

    if max_per_group is None:
        chosen = [int(np.argmax(scores))]
        dists = np.linalg.norm(feat - feat[chosen[0]], axis=1)
        for _ in range(n_select - 1):
            nxt = int(np.argmax(dists))
            chosen.append(nxt)
            dists = np.minimum(dists, np.linalg.norm(feat - feat[nxt], axis=1))
        return chosen

    counts: dict[str, int] = {}
    selected_mask = np.zeros(len(candidates), dtype=bool)
    chosen: list[int] = []
    dists = np.full(len(candidates), np.inf)
    for _ in range(min(n_select, len(candidates))):
        eligible = np.array(
            [
                not selected_mask[index] and (label is None or counts.get(label, 0) < max_per_group)
                for index, label in enumerate(group_labels)
            ],
            dtype=bool,
        )
        if not np.any(eligible):
            break
        metric = scores.copy() if not chosen else dists.copy()
        metric[~eligible] = -np.inf
        nxt = int(np.argmax(metric))
        chosen.append(nxt)
        selected_mask[nxt] = True
        label = group_labels[nxt]
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
        dists = np.minimum(dists, np.linalg.norm(feat - feat[nxt], axis=1))
    return chosen


# --------------------------------------------------------------------------- #
# Top-level acquisition                                                       #
# --------------------------------------------------------------------------- #


def select_candidates(
    candidates: Sequence[Candidate],
    cfg: AcquisitionConfig,
    primary_calculator,
    ensemble_calculators: Sequence | None = None,
    *,
    seed: int,
) -> tuple[list[Candidate], np.ndarray]:
    """Score and down-select candidates.

    *seed* drives every random choice: the random-strategy scores (its root
    stream) and the MC-dropout masks.

    Returns (selected_candidates, full_score_array). ``full_score_array`` has
    one entry per *input* candidate so downstream code can log distributions.
    """
    if not candidates:
        return [], np.zeros(0, dtype=np.float64)

    if cfg.strategy == "random":
        scores = score_random(candidates, numpy_rng(seed))
    elif cfg.strategy == "ensemble":
        if not ensemble_calculators or len(ensemble_calculators) < 2:
            raise ValueError("ensemble strategy requires >=2 ensemble_calculators")
        scores = score_ensemble(candidates, ensemble_calculators)
    elif cfg.strategy == "mc_dropout":
        scores = score_mc_dropout(
            candidates, primary_calculator, cfg.mc_dropout_passes, cfg.mc_dropout_p, seed=seed
        )
    elif cfg.strategy == "ensemble_then_dropout":
        if not ensemble_calculators or len(ensemble_calculators) < 2:
            raise ValueError("ensemble_then_dropout requires >=2 ensemble_calculators")
        coarse = score_ensemble(candidates, ensemble_calculators)
        # Refine the top 4*n_select by MC-Dropout
        keep_n = min(len(candidates), max(cfg.n_select * 4, cfg.n_select))
        top = np.argsort(coarse)[-keep_n:]
        refined = score_mc_dropout(
            [candidates[i] for i in top],
            primary_calculator,
            cfg.mc_dropout_passes,
            cfg.mc_dropout_p,
            seed=seed,
        )
        scores = np.full(len(candidates), -np.inf, dtype=np.float64)
        scores[top] = refined
    else:
        raise ValueError(f"Unknown acquisition strategy: {cfg.strategy!r}")

    # Threshold on raw uncertainty (ignored for random)
    keep_mask = np.ones_like(scores, dtype=bool)
    if cfg.strategy != "random" and cfg.min_uncertainty_eV_per_A > 0:
        keep_mask = scores >= cfg.min_uncertainty_eV_per_A
        if not np.any(keep_mask):
            log.warning(
                "All %d candidates fall below min_uncertainty_eV_per_A=%g; "
                "selecting top-N anyway to keep the loop progressing.",
                len(candidates),
                cfg.min_uncertainty_eV_per_A,
            )
            keep_mask = np.ones_like(scores, dtype=bool)

    eligible_idx = np.where(keep_mask)[0]
    eligible_scores = scores[eligible_idx]
    eligible_cands = [candidates[i] for i in eligible_idx]

    if cfg.max_per_neb_pathway is not None:
        groups = [
            (
                str(
                    candidate.provenance.get("transition_id")
                    or candidate.parent_id
                    or candidate.sampling_run_id
                    or candidate.candidate_id
                )
                if candidate.sampling_method == "neb"
                else None
            )
            for candidate in eligible_cands
        ]
    else:
        groups = None

    if cfg.diversity_filter:
        chosen_local = greedy_farthest_point(
            eligible_cands,
            eligible_scores,
            cfg.n_select,
            group_labels=groups,
            max_per_group=cfg.max_per_neb_pathway,
        )
    else:
        order = np.argsort(eligible_scores)[::-1]
        chosen_local = []
        counts = {}
        for index in order:
            group = None if groups is None else groups[int(index)]
            if group is not None and counts.get(group, 0) >= cfg.max_per_neb_pathway:
                continue
            chosen_local.append(int(index))
            if group is not None:
                counts[group] = counts.get(group, 0) + 1
            if len(chosen_local) >= cfg.n_select:
                break

    selected = [eligible_cands[i] for i in chosen_local]
    return selected, scores


# --------------------------------------------------------------------------- #
# Empirical calibration on validation data                                    #
# --------------------------------------------------------------------------- #


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):  # average ranks of ties
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def calibrate_uncertainty(
    scores: np.ndarray,
    errors: np.ndarray,
    *,
    error_threshold: float,
    partition: str,
    n_bins: int = 5,
) -> dict:
    """Relate acquisition scores to actual errors on labelled validation frames.

    ``scores`` are ensemble or MC-dropout disagreements and ``errors`` the
    model's actual errors on the same frames (e.g. per-frame force MAE against
    DFT). Returns the Spearman rank correlation, the mean error and the fraction
    above ``error_threshold`` per score quantile bin, and the lowest score at or
    above which most frames exceed the threshold (with the precision and recall
    of flagging them), or no screening threshold when no such upper tail exists.
    Calibrate on ``validation`` (or a dedicated calibration subset),
    never on test data. The result is an empirical relation on these frames, not
    a calibrated probability of error.
    """
    if partition == "test":
        raise ValueError("calibrate uncertainty on validation data, never on test data")
    scores = np.asarray(scores, dtype=float)
    errors = np.asarray(errors, dtype=float)
    keep = np.isfinite(scores) & np.isfinite(errors)
    scores, errors = scores[keep], errors[keep]
    if len(scores) < max(3, n_bins):
        raise ValueError(f"{len(scores)} frames are too few to calibrate {n_bins} bins")
    rank_s, rank_e = _ranks(scores), _ranks(errors)
    spearman = (
        float(np.corrcoef(rank_s, rank_e)[0, 1]) if rank_s.std() and rank_e.std() else float("nan")
    )
    edges = np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1))
    bins = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        # Only the last bin includes its upper edge, so tied edges never count a
        # frame twice.
        upper = scores <= high if index == n_bins - 1 else scores < high
        inside = (scores >= low) & upper
        if not inside.any():
            continue
        bins.append(
            {
                "score_range": [float(low), float(high)],
                "n_frames": int(inside.sum()),
                "mean_score": float(scores[inside].mean()),
                "mean_error": float(errors[inside].mean()),
                "fraction_above_threshold": float((errors[inside] > error_threshold).mean()),
            }
        )
    above = errors > error_threshold
    # The lowest score from which most of the flagged frames (all scores at or
    # above it) exceed the error threshold; none if no upper tail does.
    screening = None
    for threshold in np.unique(scores):
        flagged = scores >= threshold
        precision = float(above[flagged].mean())
        if precision >= 0.5:
            screening = {
                "score_threshold": float(threshold),
                "n_flagged": int(flagged.sum()),
                "precision": precision,
                "recall": float(flagged[above].mean()) if above.any() else None,
            }
            break
    return {
        "partition": partition,
        "n_frames": int(len(scores)),
        "error_threshold": error_threshold,
        "spearman_score_vs_error": spearman,
        "bins": bins,
        "screening": screening,
        "note": (
            "Empirical relation between disagreement and error on these validation frames; "
            "not a calibrated error probability. Check it again on independent data before "
            "relying on it."
        ),
    }


def validation_uncertainty_report(
    frames: Sequence[Atoms],
    primary_calculator,
    *,
    error_threshold: float,
    ensemble_calculators: Sequence | None = None,
    mc_dropout_calculator=None,
    mc_dropout_passes: int = 8,
    mc_dropout_p: float = 0.1,
    partition: str = "validation",
    seed: int,
) -> dict:
    """Score labelled frames, compute the primary model's force errors, and calibrate."""
    candidates = [
        Candidate(candidate_id=str(i), atoms=atoms, seed_path="", md_step=0)
        for i, atoms in enumerate(frames)
    ]
    if ensemble_calculators:
        scores = score_ensemble(candidates, [primary_calculator, *ensemble_calculators])
        method = "ensemble"
    elif mc_dropout_calculator is not None:
        scores = score_mc_dropout(
            candidates, mc_dropout_calculator, mc_dropout_passes, mc_dropout_p, seed=seed
        )
        method = "mc_dropout"
    else:
        raise ValueError("give ensemble_calculators or an mc_dropout_calculator")
    errors = []
    for atoms in frames:
        reference = atoms.arrays.get("forces")
        if reference is None:
            reference = atoms.get_forces()
        predicted = _forces_from(primary_calculator, atoms)
        errors.append(float(np.mean(np.abs(predicted - np.asarray(reference, dtype=float)))))
    report = calibrate_uncertainty(
        scores, np.asarray(errors), error_threshold=error_threshold, partition=partition
    )
    report["method"] = method
    report["error"] = "per-frame force MAE (eV/A) of the primary model against DFT"
    return report
