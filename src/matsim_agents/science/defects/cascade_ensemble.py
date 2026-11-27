"""Stratified cascade ensembles with sampling probabilities, weights and intervals (WP3.8).

A stratum is a cell of the sampled factors (recoil energy bin, species, composition,
temperature, ...) with its physical probability. Realizations are independent cascades with their
own seed. The stratified estimator is ``sum_h P_h * mean_h`` with variance
``sum_h P_h^2 s_h^2 / n_h``; the interval uses Welch-Satterthwaite degrees of freedom. When a
stratum is oversampled, the weight is carried by ``P_h``, not by the sample count. This is a
software statistic: it says nothing about whether the cascades are physically right.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from matsim_agents.core.seeding import stream_seed


class EnsembleError(ValueError):
    pass


@dataclass(frozen=True)
class Stratum:
    name: str
    probability: float  # physical probability of the stratum, summing to 1 over strata
    factors: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Realization:
    realization_id: str  # one independent cascade
    stratum: str
    seed: int
    value: float


def plan_stratified(
    strata: Sequence[Stratum], n_per_stratum: Mapping[str, int], base_seed: int
) -> list[tuple[str, int, int]]:
    """(stratum, index, seed) for every planned cascade, with reproducible per-cascade seeds."""
    names = [s.name for s in strata]
    if len(set(names)) != len(names):
        raise EnsembleError("stratum names must be unique")
    total = sum(s.probability for s in strata)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise EnsembleError(f"stratum probabilities must sum to 1, got {total}")
    if any(s.probability < 0 for s in strata):
        raise EnsembleError("stratum probabilities must be nonnegative")
    plan = []
    for s in strata:
        n = int(n_per_stratum.get(s.name, 0))
        if s.probability > 0 and n < 2:
            raise EnsembleError(
                f"stratum {s.name!r} has probability {s.probability} but only {n} planned "
                "cascades; at least 2 are needed to estimate its variance"
            )
        plan.extend(
            (s.name, i, stream_seed(base_seed, "cascade-ensemble", s.name, i)) for i in range(n)
        )
    return plan


def one_value_per_cascade(
    frames: Sequence[tuple[str, float, float]],
) -> tuple[dict[str, float], int]:
    """Reduce (cascade_id, time, value) frames to the last frame of each cascade.

    Frames of one cascade are correlated, so they never enter the ensemble as separate
    realizations. Returns the per-cascade values and the number of frames discarded.
    """
    last: dict[str, tuple[float, float]] = {}
    for cid, t, v in frames:
        if cid not in last or t >= last[cid][0]:
            last[cid] = (t, v)
    return {c: v for c, (_, v) in last.items()}, len(frames) - len(last)


@dataclass(frozen=True)
class StratifiedEstimate:
    mean: float
    standard_error: float
    ci_low: float
    ci_high: float
    confidence: float
    dof: float
    n_realizations: int
    per_stratum: dict[str, tuple[int, float, float]]  # n, mean, sample variance
    unsampled_probability: float
    seeds: tuple[int, ...]
    statement: str


def stratified_estimate(
    strata: Sequence[Stratum],
    realizations: Sequence[Realization],
    *,
    confidence: float = 0.95,
) -> StratifiedEstimate:
    if not 0.0 < confidence < 1.0:
        raise EnsembleError(f"confidence must be in (0, 1), got {confidence}")
    by_name = {s.name: s for s in strata}
    if len(by_name) != len(strata):
        raise EnsembleError("stratum names must be unique")
    total = sum(s.probability for s in strata)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise EnsembleError(f"stratum probabilities must sum to 1, got {total}")
    if any(s.probability < 0 for s in strata):
        raise EnsembleError("stratum probabilities must be nonnegative")
    ids = [r.realization_id for r in realizations]
    if len(set(ids)) != len(ids):
        raise EnsembleError("duplicate realization_id: frames of one cascade are not independent")
    seeds = [(r.stratum, r.seed) for r in realizations]
    if len(set(seeds)) != len(seeds):
        raise EnsembleError("two realizations in one stratum share a seed")
    groups: dict[str, list[float]] = defaultdict(list)
    for r in realizations:
        if r.stratum not in by_name:
            raise EnsembleError(
                f"realization {r.realization_id!r} has unknown stratum {r.stratum!r}"
            )
        if not math.isfinite(r.value):
            raise EnsembleError("realization values must be finite")
        groups[r.stratum].append(r.value)
    mean = var = 0.0
    dof_den = 0.0
    per: dict[str, tuple[int, float, float]] = {}
    for s in strata:
        vals = groups.get(s.name, [])
        if s.probability > 0 and len(vals) < 2:
            raise EnsembleError(
                f"stratum {s.name!r} (probability {s.probability}) has {len(vals)} realizations; "
                "at least 2 are needed"
            )
        if not vals or s.probability == 0:
            continue
        n = len(vals)
        m = float(np.mean(vals))
        s2 = float(np.var(vals, ddof=1))
        per[s.name] = (n, m, s2)
        mean += s.probability * m
        term = s.probability**2 * s2 / n
        var += term
        dof_den += term**2 / (n - 1)
    se = math.sqrt(var)
    if var == 0.0:
        dof = float(sum(n - 1 for n, _, _ in per.values()))
        half = 0.0
    else:
        dof = var**2 / dof_den
        half = float(stats.t.ppf(0.5 + confidence / 2, dof)) * se
    return StratifiedEstimate(
        mean=mean,
        standard_error=se,
        ci_low=mean - half,
        ci_high=mean + half,
        confidence=confidence,
        dof=dof,
        n_realizations=len(realizations),
        per_stratum=per,
        unsampled_probability=float(sum(s.probability for s in strata if s.name not in per)),
        seeds=tuple(r.seed for r in realizations),
        statement=(
            "stratified estimate over independent cascades; the interval assumes independent "
            "realizations and normal stratum means, and says nothing about physical correctness"
        ),
    )


__all__ = [
    "EnsembleError",
    "Realization",
    "StratifiedEstimate",
    "Stratum",
    "one_value_per_cascade",
    "plan_stratified",
    "stratified_estimate",
]
