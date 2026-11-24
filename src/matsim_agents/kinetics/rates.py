"""Harmonic transition-state-theory rates (C2.1).

``rate = g * nu * exp(-Ea / (kB * T))``. The attempt frequency ``nu`` is never defaulted
silently: it comes from the event or from an explicit, recorded fallback. An event without
barrier evidence has no rate (it is not a zero-barrier event).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

from matsim_agents.kinetics.contracts import TransitionEvent, TransitionRate
from matsim_agents.kinetics.graph import TransitionNetwork

KB_EV_PER_K = 8.617333262e-5

Require = Literal["qualified", "exploratory"]


class RateError(ValueError):
    """No defensible rate exists for this event under the requested policy."""


def htst_rate(
    barrier_eV: float, temperature_K: float, attempt_frequency_hz: float, multiplicity: int = 1
) -> float:
    """``g * nu * exp(-Ea/kT)``; underflow returns 0.0, never NaN."""
    if not math.isfinite(barrier_eV) or barrier_eV < 0:
        raise RateError("barrier must be finite and nonnegative")
    if not math.isfinite(temperature_K) or temperature_K <= 0:
        raise RateError("temperature must be finite and positive")
    if not math.isfinite(attempt_frequency_hz) or attempt_frequency_hz <= 0:
        raise RateError("attempt frequency must be finite and positive")
    if multiplicity < 1:
        raise RateError("multiplicity must be at least 1")
    log_rate = (
        math.log(multiplicity)
        + math.log(attempt_frequency_hz)
        - barrier_eV / (KB_EV_PER_K * temperature_K)
    )
    if log_rate < math.log(5e-324):
        return 0.0
    rate = math.exp(log_rate)
    if not math.isfinite(rate):
        raise RateError("rate overflows a double")
    return rate


def event_rate(
    event: TransitionEvent,
    temperature_K: float,
    *,
    require: Require = "qualified",
    fallback_attempt_frequency_hz: float | None = None,
) -> TransitionRate:
    """Rate of one event. ``require='qualified'`` rejects anything not qualified."""
    barrier = event.barrier_eV
    if barrier is None:
        raise RateError(f"{event.transition_id}: no barrier evidence")
    if require == "qualified" and event.qualification != "qualified":
        raise RateError(f"{event.transition_id}: evidence is {event.qualification}, not qualified")
    assumptions: list[str] = [f"HTST with g={event.multiplicity}"]
    nu = event.attempt_frequency_hz
    if nu is None:
        if require == "qualified" or fallback_attempt_frequency_hz is None:
            raise RateError(f"{event.transition_id}: no explicit attempt frequency")
        nu = fallback_attempt_frequency_hz
        assumptions.append(f"fallback attempt frequency {nu:g} Hz (not computed)")
    elif event.prefactor_note:
        assumptions.append(event.prefactor_note)
    if event.qualification != "qualified":
        assumptions.append(f"evidence is {event.qualification}; exploratory only")
    if event.evidence is not None and event.evidence.synthetic:
        assumptions.append("synthetic evidence")
    return TransitionRate(
        transition_id=event.transition_id,
        temperature_K=temperature_K,
        rate_per_s=htst_rate(barrier, temperature_K, nu, event.multiplicity),
        qualification=event.qualification,
        assumptions=tuple(assumptions),
    )


def state_rates(
    network: TransitionNetwork,
    state_id: str,
    temperature_K: float,
    *,
    require: Require = "qualified",
    fallback_attempt_frequency_hz: float | None = None,
) -> tuple[TransitionRate, ...]:
    """Rates of every outgoing event; raises if any lacks usable evidence."""
    return tuple(
        event_rate(
            e,
            temperature_K,
            require=require,
            fallback_attempt_frequency_hz=fallback_attempt_frequency_hz,
        )
        for e in network.outgoing(state_id)
    )


def detailed_balance_violations(
    network: TransitionNetwork,
    state_energies_eV: Mapping[str, float],
    temperature_K: float,
    *,
    require: Require = "exploratory",
    fallback_attempt_frequency_hz: float | None = None,
    tolerance: float = 1e-6,
) -> tuple[str, ...]:
    """Pairs whose ``ln(k_f/k_r)`` differs from ``-(E_target - E_source)/kT``.

    Only evaluated on request; needs state energies and both directions with evidence.
    """
    bad: list[str] = []
    for event in network.events:
        back = network.reverse_of(event)
        if back is None or event.transition_id > back.transition_id:
            continue
        if event.source_id not in state_energies_eV or event.target_id not in state_energies_eV:
            raise RateError(f"missing state energy for {event.transition_id}")
        fwd = event_rate(
            event,
            temperature_K,
            require=require,
            fallback_attempt_frequency_hz=fallback_attempt_frequency_hz,
        )
        rev = event_rate(
            back,
            temperature_K,
            require=require,
            fallback_attempt_frequency_hz=fallback_attempt_frequency_hz,
        )
        if fwd.rate_per_s == 0 or rev.rate_per_s == 0:
            continue
        delta = state_energies_eV[event.target_id] - state_energies_eV[event.source_id]
        expected = -delta / (KB_EV_PER_K * temperature_K)
        if abs(math.log(fwd.rate_per_s / rev.rate_per_s) - expected) > tolerance:
            bad.append(event.transition_id)
    return tuple(bad)


__all__ = [
    "KB_EV_PER_K",
    "RateError",
    "detailed_balance_violations",
    "event_rate",
    "htst_rate",
    "state_rates",
]
