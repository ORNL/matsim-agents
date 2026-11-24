"""Quantitative coverage and statistical-convergence reports.

Statistical uncertainty (finite sampling) is reported separately from physical uncertainty
(model/functional choice) and numerical uncertainty (convergence settings). Only the first is
computed here; the other two are carried as explicit "not assessed" fields, never as zero.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from matsim_agents.seeding import python_rng

Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion; (0, 1) when ``n`` is zero."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("require 0 <= successes <= n")
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean_interval(
    values: Sequence[float], *, seed: int, n_boot: int = 1000, level: float = 0.95
) -> tuple[float, float] | None:
    """Percentile bootstrap interval of the mean, deterministic for ``seed``; None for n < 2."""
    if len(values) < 2:
        return None
    rng = python_rng(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(((1 - level) / 2) * (n_boot - 1))]
    hi = means[int((1 - (1 - level) / 2) * (n_boot - 1))]
    return lo, hi


@dataclass(frozen=True)
class StratumCoverage:
    stratum: str
    population: int
    sampled: int
    target: int
    fraction: float
    fraction_ci: tuple[float, float]
    met: bool


@dataclass(frozen=True)
class CoverageReport:
    strata: tuple[StratumCoverage, ...]
    unmet: tuple[str, ...]
    unrepresented: tuple[str, ...]  # present in the population, never sampled

    @property
    def complete(self) -> bool:
        return not self.unmet


def coverage_report(
    population: Mapping[str, str],
    sampled_ids: Sequence[str],
    targets: Mapping[str, int],
) -> CoverageReport:
    """Per-stratum coverage of ``sampled_ids`` against ``targets`` (minimum counts).

    ``population`` maps candidate id to stratum. A target for a stratum with no population is
    reported as unmet. Sampled ids outside the population are an error.
    """
    unknown = [c for c in sampled_ids if c not in population]
    if unknown:
        raise ValueError(f"{len(unknown)} sampled ids are not in the population")
    pop = Counter(population.values())
    got = Counter(population[c] for c in set(sampled_ids))
    rows = []
    for stratum in sorted(set(pop) | set(targets)):
        n_pop, n_got, target = pop.get(stratum, 0), got.get(stratum, 0), targets.get(stratum, 0)
        rows.append(
            StratumCoverage(
                stratum,
                n_pop,
                n_got,
                target,
                n_got / n_pop if n_pop else 0.0,
                wilson_interval(n_got, n_pop) if n_pop else (0.0, 1.0),
                n_got >= target,
            )
        )
    return CoverageReport(
        tuple(rows),
        tuple(r.stratum for r in rows if not r.met),
        tuple(r.stratum for r in rows if r.population > 0 and r.sampled == 0),
    )


@dataclass(frozen=True)
class ConvergenceReport:
    quantity: str
    n: int
    mean: float | None
    statistical_ci: tuple[float, float] | None
    half_width: float | None
    target_half_width: float
    converged: bool
    physical_uncertainty: str = "not assessed (functional and pseudopotential choice)"
    numerical_uncertainty: str = "not assessed (k-points, cutoff, smearing)"
    note: str = ""


def convergence_report(
    quantity: str,
    values: Sequence[float],
    target_half_width: float,
    *,
    seed: int,
    min_n: int = 5,
) -> ConvergenceReport:
    """Statistical convergence of a sampled mean. Fewer than ``min_n`` values is never converged."""
    n = len(values)
    if n < min_n:
        return ConvergenceReport(
            quantity, n, None, None, None, target_half_width, False,
            note=f"fewer than {min_n} samples",
        )  # fmt: skip
    interval = bootstrap_mean_interval(values, seed=seed)
    assert interval is not None
    half = (interval[1] - interval[0]) / 2
    return ConvergenceReport(
        quantity,
        n,
        sum(values) / n,
        interval,
        half,
        target_half_width,
        half <= target_half_width,
        note="statistical (sampling) uncertainty only",
    )
