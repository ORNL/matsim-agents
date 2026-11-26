"""Weighted ensemble statistics of per-cascade surviving-defect counts.

Each realization contributes one value (for example its Frenkel-pair count) and a nonnegative
weight. The mean uses reliability weights, so scaling every weight by a constant changes nothing.
Intervals assume independent realizations: the Student-t interval uses the Kish effective sample
size, and the bootstrap resamples (value, weight) pairs. Neither says anything about whether the
cascades themselves are physically right, and an interval from few realizations is wide, not
precise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

Method = str  # "t" or "bootstrap"


@dataclass(frozen=True)
class EnsembleEstimate:
    mean: float
    standard_error: float
    ci_low: float
    ci_high: float
    confidence: float
    method: str
    n: int
    effective_n: float
    seed: int | None  # bootstrap only


def _prepare(values: Sequence[float], weights: Sequence[float] | None):
    x = np.asarray(values, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.shape != x.shape:
        raise ValueError("values and weights must be one-dimensional and the same length")
    if len(x) < 2:
        raise ValueError("at least two realizations are required")
    if not (np.isfinite(x).all() and np.isfinite(w).all()):
        raise ValueError("values and weights must be finite")
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be nonnegative with a positive sum")
    return x, w


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def weighted_estimate(
    values: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    confidence: float = 0.95,
    method: Method = "t",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> EnsembleEstimate:
    """Weighted mean with a confidence interval over independent realizations."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if method not in ("t", "bootstrap"):
        raise ValueError(f"method must be 't' or 'bootstrap', got {method!r}")
    x, w = _prepare(values, weights)
    v1, v2 = float(w.sum()), float(np.sum(w**2))
    mean = _weighted_mean(x, w)
    effective_n = v1**2 / v2
    if effective_n <= 1.0 + 1e-12:
        raise ValueError("weights are concentrated on one realization; no interval is defined")
    # Unbiased weighted variance for reliability weights; equals the sample variance if equal.
    variance = float(np.sum(w * (x - mean) ** 2) / (v1 - v2 / v1))
    standard_error = float(np.sqrt(variance * v2 / v1**2))

    if method == "t":
        half = float(stats.t.ppf(0.5 + confidence / 2, effective_n - 1.0)) * standard_error
        low, high, used_seed = mean - half, mean + half, None
    else:
        if n_bootstrap < 100:
            raise ValueError("n_bootstrap must be at least 100")
        rng = np.random.default_rng(seed)
        support = np.flatnonzero(w > 0)  # zero-weight draws would give 0/0
        draws = support[rng.integers(0, len(support), size=(n_bootstrap, len(support)))]
        means = np.sum(w[draws] * x[draws], axis=1) / np.sum(w[draws], axis=1)
        low, high = np.quantile(means, [0.5 - confidence / 2, 0.5 + confidence / 2])
        low, high, used_seed = float(low), float(high), seed
    return EnsembleEstimate(
        mean=mean,
        standard_error=standard_error,
        ci_low=float(low),
        ci_high=float(high),
        confidence=confidence,
        method=method,
        n=len(x),
        effective_n=effective_n,
        seed=used_seed,
    )


def convergence_curve(
    values: Sequence[float],
    sizes: Sequence[int],
    weights: Sequence[float] | None = None,
    *,
    confidence: float = 0.95,
) -> tuple[EnsembleEstimate, ...]:
    """Estimates from the first ``n`` realizations for each ``n`` in ``sizes``.

    Realizations are used in the given order, so shuffle them beforehand if their order carries
    information; a half-width that stops shrinking with ``n`` is the signal of non-convergence.
    """
    x, w = _prepare(values, weights)
    if not sizes or any(n < 2 or n > len(x) for n in sizes):
        raise ValueError(f"sizes must be nonempty and within [2, {len(x)}]")
    return tuple(weighted_estimate(x[:n], w[:n], confidence=confidence) for n in sizes)


__all__ = ["EnsembleEstimate", "convergence_curve", "weighted_estimate"]
