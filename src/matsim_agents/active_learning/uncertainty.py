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

import numpy as np
from ase import Atoms

from matsim_agents.active_learning.candidates import Candidate
from matsim_agents.active_learning.config import AcquisitionConfig

log = logging.getLogger(__name__)


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


def _enable_dropout(model) -> None:
    """Set every Dropout layer in a torch ``nn.Module`` to train() while keeping BN in eval."""
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def inject_inference_dropout(
    model,
    p: float = 0.1,
    target_layers: str = "linear",
    max_layers: int | None = None,
) -> int:
    """Insert test-time dropout into a model that was trained without it.

    For backends such as UMA there are no native ``nn.Dropout`` layers, so
    MC-Dropout would measure zero variance. This attaches an ``nn.Dropout``
    child to each target layer and registers a forward hook that passes the
    layer's output through it. The injected modules start in ``eval()`` mode
    (identity), so deterministic prediction/relaxation is unaffected; they only
    become stochastic when :func:`score_mc_dropout` toggles dropout layers into
    ``train()`` mode.

    Idempotent: layers already carrying an injected dropout are skipped.

    Returns the number of layers instrumented.

    Note: this is a *heuristic* uncertainty (uncalibrated), not a Bayesian
    posterior — the model was not trained with dropout in these positions.
    """
    import torch
    import torch.nn as nn

    if target_layers == "linear":
        types: tuple[type, ...] = (nn.Linear,)
    elif target_layers == "all":
        types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    else:
        raise ValueError(f"target_layers must be 'linear' or 'all', got {target_layers!r}")

    # Snapshot targets first: we mutate ._modules during the loop, which would
    # otherwise invalidate the model.modules() generator.
    targets = [
        m
        for m in model.modules()
        if isinstance(m, types) and not getattr(m, "_mc_dropout_injected", False)
    ]

    def _make_hook(drop: nn.Module):
        def _hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                return drop(output)
            return output

        return _hook

    n = 0
    for module in targets:
        drop = nn.Dropout(p=float(p))
        drop.eval()  # dormant until score_mc_dropout flips it to train()
        module.add_module("_mc_injected_dropout", drop)
        module.register_forward_hook(_make_hook(drop))
        module._mc_dropout_injected = True  # type: ignore[attr-defined]
        n += 1
        if max_layers is not None and n >= max_layers:
            break
    return n


def score_mc_dropout(
    candidates: Sequence[Candidate], calculator, passes: int, p: float
) -> np.ndarray:
    """K stochastic forward passes; score = RMS std of forces across passes."""
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
    for i, cand in enumerate(candidates):
        per_pass = []
        for _ in range(passes):
            _enable_dropout(calculator.model)
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
) -> list[int]:
    """Pick ``n_select`` indices that are diverse in composition AND high-score.

    Uses farthest-point sampling on the joint (fp, score) feature, seeded by
    the highest-score candidate. Prefer to ``argsort(scores)[-n:]`` when you
    expect many near-duplicates.
    """
    if n_select >= len(candidates):
        return list(range(len(candidates)))

    fps = np.stack([_composition_fingerprint(c.atoms) for c in candidates], axis=0)
    fps *= float(fingerprint_weight)
    s = scores.reshape(-1, 1) * float(score_weight)
    feat = np.concatenate([fps, s], axis=1)

    chosen = [int(np.argmax(scores))]
    dists = np.linalg.norm(feat - feat[chosen[0]], axis=1)
    for _ in range(n_select - 1):
        nxt = int(np.argmax(dists))
        chosen.append(nxt)
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
    seed: int = 0,
) -> tuple[list[Candidate], np.ndarray]:
    """Score and down-select candidates.

    Returns (selected_candidates, full_score_array). ``full_score_array`` has
    one entry per *input* candidate so downstream code can log distributions.
    """
    if not candidates:
        return [], np.zeros(0, dtype=np.float64)

    rng = np.random.default_rng(seed)

    if cfg.strategy == "random":
        scores = score_random(candidates, rng)
    elif cfg.strategy == "ensemble":
        if not ensemble_calculators or len(ensemble_calculators) < 2:
            raise ValueError("ensemble strategy requires >=2 ensemble_calculators")
        scores = score_ensemble(candidates, ensemble_calculators)
    elif cfg.strategy == "mc_dropout":
        scores = score_mc_dropout(
            candidates, primary_calculator, cfg.mc_dropout_passes, cfg.mc_dropout_p
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
        )
        scores = np.full(len(candidates), -np.inf, dtype=np.float64)
        scores[top] = refined
    else:
        raise ValueError(f"Unknown acquisition strategy: {cfg.strategy!r}")

    # Threshold on raw uncertainty (ignored for random)
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
    else:
        keep_mask = np.ones_like(scores, dtype=bool)

    eligible_idx = np.where(keep_mask)[0]
    eligible_scores = scores[eligible_idx]
    eligible_cands = [candidates[i] for i in eligible_idx]

    if cfg.diversity_filter:
        chosen_local = greedy_farthest_point(eligible_cands, eligible_scores, cfg.n_select)
    else:
        order = np.argsort(eligible_scores)[::-1][: cfg.n_select]
        chosen_local = list(order)

    selected = [eligible_cands[i] for i in chosen_local]
    return selected, scores
