"""KMC ensembles: independent streams, first passage, censoring, outcome probabilities."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from matsim_agents.kinetics.graph import TransitionNetwork
from matsim_agents.kinetics.kmc import run_kmc
from matsim_agents.kinetics.rates import Require

Z95 = 1.959963984540054


class OutcomeEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    probability: float
    wilson_low: float
    wilson_high: float


class EnsembleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_trajectories: int
    seed: int
    temperature_K: float
    horizon_s: float | None
    max_events: int
    terminations: dict[str, int]
    outcomes: dict[str, OutcomeEstimate]
    n_censored: int
    censored_fraction: float
    first_passage_times_s: tuple[float, ...]
    censoring_times_s: tuple[float, ...]
    mean_first_passage_s: float | None
    median_first_passage_s: float | None
    first_passage_biased: bool
    mean_events: float
    warnings: tuple[str, ...]


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def run_ensemble(
    network: TransitionNetwork,
    initial_state_id: str,
    temperature_K: float,
    *,
    n_trajectories: int,
    seed: int,
    horizon_s: float = math.inf,
    max_events: int = 100_000,
    require: Require = "qualified",
    fallback_attempt_frequency_hz: float | None = None,
) -> EnsembleResult:
    """Independent trajectories on streams ``("ensemble", i)``.

    Outcome probabilities are fractions of all trajectories, with censored and stuck runs
    kept in the denominator, so a short horizon lowers them instead of being ignored.
    First-passage statistics use absorbed runs only and are flagged biased when any run was
    censored.
    """
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be at least 1")

    terminations: dict[str, int] = {}
    kinds: dict[str, int] = {}
    passage: list[float] = []
    censor: list[float] = []
    events = 0
    for i in range(n_trajectories):
        traj = run_kmc(
            network,
            initial_state_id,
            temperature_K,
            seed=seed,
            stream=("ensemble", i),
            horizon_s=horizon_s,
            max_events=max_events,
            require=require,
            fallback_attempt_frequency_hz=fallback_attempt_frequency_hz,
            record_steps=False,
        )
        terminations[traj.termination] = terminations.get(traj.termination, 0) + 1
        events += traj.n_events
        if traj.termination == "absorbed":
            passage.append(traj.final_time_s)
            kinds[traj.absorbing_kind or "unspecified"] = (
                kinds.get(traj.absorbing_kind or "unspecified", 0) + 1
            )
        else:
            censor.append(traj.final_time_s)

    outcomes = {}
    for kind, count in sorted(kinds.items()):
        low, high = wilson_interval(count, n_trajectories)
        outcomes[kind] = OutcomeEstimate(
            count=count,
            probability=count / n_trajectories,
            wilson_low=low,
            wilson_high=high,
        )

    warnings: list[str] = []
    if censor:
        warnings.append(
            f"{len(censor)} of {n_trajectories} trajectories ended without absorption; "
            "first-passage statistics cover absorbed runs only and are biased low"
        )
    return EnsembleResult(
        n_trajectories=n_trajectories,
        seed=seed,
        temperature_K=temperature_K,
        horizon_s=None if math.isinf(horizon_s) else horizon_s,
        max_events=max_events,
        terminations=terminations,
        outcomes=outcomes,
        n_censored=len(censor),
        censored_fraction=len(censor) / n_trajectories,
        first_passage_times_s=tuple(passage),
        censoring_times_s=tuple(censor),
        mean_first_passage_s=math.fsum(passage) / len(passage) if passage else None,
        median_first_passage_s=_median(passage),
        first_passage_biased=bool(censor),
        mean_events=events / n_trajectories,
        warnings=tuple(warnings),
    )


__all__ = ["EnsembleResult", "OutcomeEstimate", "run_ensemble", "wilson_interval"]
