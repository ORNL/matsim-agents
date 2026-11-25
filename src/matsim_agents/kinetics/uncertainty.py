"""Parameter uncertainty and sensitivity of KMC outcomes.

A saddle-energy shift is applied to a pathway (an event and its reverse together), which keeps
``E_f - E_r`` consistent. Barrier errors share a common component with correlation ``rho``,
which the caller must state: independence is not assumed. Missing channels are modelled by
dropping events or adding explicitly labelled hypothetical ones.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping

from pydantic import BaseModel, ConfigDict

from matsim_agents.core.seeding import numpy_rng
from matsim_agents.kinetics.contracts import TransitionEvent
from matsim_agents.kinetics.ensemble import run_ensemble
from matsim_agents.kinetics.graph import TransitionNetwork
from matsim_agents.kinetics.rates import Require

PathwayKey = tuple[str, str, str]


def pathway_key(event: TransitionEvent) -> PathwayKey:
    a, b = sorted((event.source_id, event.target_id))
    return a, b, event.mechanism


def pathway_keys(network: TransitionNetwork) -> tuple[PathwayKey, ...]:
    return tuple(sorted({pathway_key(e) for e in network.events if e.evidence is not None}))


def perturb_network(
    network: TransitionNetwork,
    *,
    barrier_shift_eV: Mapping[PathwayKey, float] | None = None,
    prefactor_scale: Mapping[PathwayKey, float] | None = None,
    drop: Collection[PathwayKey] = (),
) -> TransitionNetwork:
    """A copy with shifted saddle energies, scaled attempt frequencies, dropped pathways."""
    shifts, scales, dropped = barrier_shift_eV or {}, prefactor_scale or {}, set(drop)
    out = TransitionNetwork(network.provenance)
    for state in network.states:
        out.add_state(state)
    for event in network.events:
        key = pathway_key(event)
        if key in dropped:
            continue
        ev = event.evidence
        if ev is not None and key in shifts:
            delta = max(shifts[key], -min(ev.forward_barrier_eV, ev.reverse_barrier_eV))
            ev = ev.model_copy(
                update={
                    "forward_barrier_eV": ev.forward_barrier_eV + delta,
                    "reverse_barrier_eV": ev.reverse_barrier_eV + delta,
                }
            )
        nu = event.attempt_frequency_hz
        if nu is not None and key in scales:
            nu = nu * scales[key]
        out.add_transition(
            TransitionEvent.model_validate(
                event.model_copy(update={"evidence": ev, "attempt_frequency_hz": nu}).model_dump()
            )
        )
    return out


class SensitivityEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    pathway: PathwayKey
    outcome: str
    baseline_probability: float
    probability_minus: float
    probability_plus: float
    slope_per_eV: float


def barrier_sensitivity(
    network: TransitionNetwork,
    initial_state_id: str,
    temperature_K: float,
    outcome: str,
    *,
    delta_eV: float = 0.05,
    n_trajectories: int,
    seed: int,
    horizon_s: float = math.inf,
    max_events: int = 100_000,
    require: Require = "qualified",
    fallback_attempt_frequency_hz: float | None = None,
) -> tuple[SensitivityEntry, ...]:
    """One-at-a-time central difference of an outcome probability w.r.t. each pathway barrier.

    Every run reuses the same seed and streams (common random numbers), which cuts the
    noise of the difference. The slope is only as good as ``n_trajectories`` allows.
    """
    if delta_eV <= 0:
        raise ValueError("delta_eV must be positive")

    def prob(net: TransitionNetwork) -> float:
        res = run_ensemble(
            net,
            initial_state_id,
            temperature_K,
            n_trajectories=n_trajectories,
            seed=seed,
            horizon_s=horizon_s,
            max_events=max_events,
            require=require,
            fallback_attempt_frequency_hz=fallback_attempt_frequency_hz,
        )
        return res.outcomes[outcome].probability if outcome in res.outcomes else 0.0

    base = prob(network)
    rows = []
    for key in pathway_keys(network):
        minus = prob(perturb_network(network, barrier_shift_eV={key: -delta_eV}))
        plus = prob(perturb_network(network, barrier_shift_eV={key: delta_eV}))
        rows.append(
            SensitivityEntry(
                pathway=key,
                outcome=outcome,
                baseline_probability=base,
                probability_minus=minus,
                probability_plus=plus,
                slope_per_eV=(plus - minus) / (2 * delta_eV),
            )
        )
    return tuple(rows)


class UncertaintyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: str
    n_samples: int
    barrier_sigma_eV: float
    barrier_correlation: float
    log10_prefactor_sigma: float
    temperature_sigma_K: float
    probabilities: tuple[float, ...]
    mean: float
    std: float
    p05: float
    p95: float
    notes: tuple[str, ...]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def propagate_uncertainty(
    network: TransitionNetwork,
    initial_state_id: str,
    temperature_K: float,
    outcome: str,
    *,
    barrier_sigma_eV: float,
    barrier_correlation: float,
    n_samples: int,
    n_trajectories: int,
    seed: int,
    log10_prefactor_sigma: float = 0.0,
    temperature_sigma_K: float = 0.0,
    drop_probability: float = 0.0,
    horizon_s: float = math.inf,
    max_events: int = 100_000,
    require: Require = "qualified",
    fallback_attempt_frequency_hz: float | None = None,
) -> UncertaintyResult:
    """Monte Carlo over barrier, prefactor and temperature errors.

    ``barrier_correlation`` (rho in [0, 1]) is the pairwise correlation of barrier errors
    across pathways: error_i = sigma * (sqrt(rho) z_common + sqrt(1 - rho) z_i). It has no
    default. ``drop_probability`` removes each pathway at random to represent channels that
    a different method would not have found (or that are missing).
    """
    if not 0.0 <= barrier_correlation <= 1.0:
        raise ValueError("barrier_correlation must lie in [0, 1]")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if min(barrier_sigma_eV, log10_prefactor_sigma, temperature_sigma_K) < 0:
        raise ValueError("sigmas must be nonnegative")
    if not 0.0 <= drop_probability < 1.0:
        raise ValueError("drop_probability must lie in [0, 1)")

    keys = pathway_keys(network)
    shared, own = math.sqrt(barrier_correlation), math.sqrt(1.0 - barrier_correlation)
    values: list[float] = []
    for j in range(n_samples):
        rng = numpy_rng(seed, "uncertainty", j)
        z_common = float(rng.standard_normal())
        shifts = {
            k: barrier_sigma_eV * (shared * z_common + own * float(rng.standard_normal()))
            for k in keys
        }
        scales = {k: 10.0 ** (log10_prefactor_sigma * float(rng.standard_normal())) for k in keys}
        drops = {k for k in keys if drop_probability and float(rng.random()) < drop_probability}
        temp = max(1.0, temperature_K + temperature_sigma_K * float(rng.standard_normal()))
        sample = perturb_network(
            network, barrier_shift_eV=shifts, prefactor_scale=scales, drop=drops
        )
        res = run_ensemble(
            sample,
            initial_state_id,
            temp,
            n_trajectories=n_trajectories,
            seed=seed,
            horizon_s=horizon_s,
            max_events=max_events,
            require=require,
            fallback_attempt_frequency_hz=fallback_attempt_frequency_hz,
        )
        values.append(res.outcomes[outcome].probability if outcome in res.outcomes else 0.0)

    mean = math.fsum(values) / n_samples
    var = math.fsum((v - mean) ** 2 for v in values) / (n_samples - 1)
    return UncertaintyResult(
        outcome=outcome,
        n_samples=n_samples,
        barrier_sigma_eV=barrier_sigma_eV,
        barrier_correlation=barrier_correlation,
        log10_prefactor_sigma=log10_prefactor_sigma,
        temperature_sigma_K=temperature_sigma_K,
        probabilities=tuple(values),
        mean=mean,
        std=math.sqrt(var),
        p05=_quantile(values, 0.05),
        p95=_quantile(values, 0.95),
        notes=(
            "errors are assumed Gaussian with the stated sigmas; they are inputs, not measured",
            f"pairwise barrier-error correlation = {barrier_correlation}",
            "spread includes the sampling noise of each finite ensemble",
        ),
    )


__all__ = [
    "PathwayKey",
    "SensitivityEntry",
    "UncertaintyResult",
    "barrier_sensitivity",
    "pathway_key",
    "pathway_keys",
    "perturb_network",
    "propagate_uncertainty",
]
