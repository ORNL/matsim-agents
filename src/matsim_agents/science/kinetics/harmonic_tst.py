"""Harmonic transition-state-theory rates from supplied vibrational modes.

Adds to :mod:`matsim_agents.science.kinetics.rates` what a constant attempt frequency cannot
give: a temperature-dependent prefactor and a vibrational free-energy correction to the NEB
barrier. Frequencies are inputs (Hz); none are computed or defaulted here. The saddle point
must be given with its N-1 real modes, the imaginary reaction-coordinate mode removed by the
caller. Tunnelling, anharmonicity and the electronic free energy are not included.

``rate = g * (kB T / h) * exp(-(Ea + F_saddle - F_min) / kB T)`` with
``F = sum_i [h nu_i / 2 + kB T ln(1 - exp(-h nu_i / kB T))]``. At high temperature this
reduces to the Vineyard form ``g * prod(nu_min) / prod(nu_saddle) * exp(-Ea / kB T)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from matsim_agents.science.kinetics.contracts import TransitionEvent, TransitionRate
from matsim_agents.science.kinetics.rates import KB_EV_PER_K, RateError, Require

H_EV_S = 4.135667696e-15


def _check_modes(freqs_hz: Sequence[float], label: str) -> None:
    if len(freqs_hz) == 0:
        raise RateError(f"{label}: no vibrational modes given")
    if any((not math.isfinite(f)) or f <= 0 for f in freqs_hz):
        raise RateError(f"{label}: frequencies must be finite and positive")


def _check_temperature(temperature_K: float) -> None:
    if not math.isfinite(temperature_K) or temperature_K <= 0:
        raise RateError("temperature must be finite and positive")


def vibrational_free_energy_eV(
    freqs_hz: Sequence[float], temperature_K: float, *, include_zpe: bool = True
) -> float:
    """Harmonic ``F_vib`` in eV (quantum oscillators; zero-point energy optional)."""
    _check_modes(freqs_hz, "vibrational free energy")
    _check_temperature(temperature_K)
    kt = KB_EV_PER_K * temperature_K
    total = 0.0
    for f in freqs_hz:
        x = H_EV_S * f / kt
        total += kt * math.log(-math.expm1(-x)) + (0.5 * H_EV_S * f if include_zpe else 0.0)
    return total


def vineyard_prefactor_hz(
    minimum_freqs_hz: Sequence[float], saddle_freqs_hz: Sequence[float]
) -> float:
    """``prod(nu_min) / prod(nu_saddle)``: the high-temperature limit of the prefactor."""
    _check_modes(minimum_freqs_hz, "minimum")
    _check_modes(saddle_freqs_hz, "saddle")
    if len(saddle_freqs_hz) != len(minimum_freqs_hz) - 1:
        raise RateError("the saddle needs exactly one mode fewer than the minimum")
    log_ratio = sum(math.log(f) for f in minimum_freqs_hz) - sum(
        math.log(f) for f in saddle_freqs_hz
    )
    return math.exp(log_ratio)


@dataclass(frozen=True)
class HarmonicRate:
    rate_per_s: float
    activation_free_energy_eV: float
    effective_prefactor_hz: float  # rate * exp(Ea / kT) / g, so it carries the T dependence
    log_effective_prefactor: float  # natural log of effective_prefactor_hz; finite when it is inf
    vibrational_correction_eV: float
    vineyard_prefactor_hz: float
    log_rate_sigma: float | None
    assumptions: tuple[str, ...]


def harmonic_tst_rate(
    barrier_eV: float,
    minimum_freqs_hz: Sequence[float],
    saddle_freqs_hz: Sequence[float],
    temperature_K: float,
    *,
    multiplicity: int = 1,
    barrier_sigma_eV: float | None = None,
    prefactor_log_sigma: float | None = None,
) -> HarmonicRate:
    """Rate with a vibrational free-energy correction; uncertainty only if sigmas are given.

    ``log_rate_sigma`` assumes independent barrier and prefactor errors and a linearised
    exponent; it is ``None`` unless at least one sigma is supplied, never zero by default.
    """
    if not math.isfinite(barrier_eV) or barrier_eV < 0:
        raise RateError("barrier must be finite and nonnegative")
    if multiplicity < 1:
        raise RateError("multiplicity must be at least 1")
    for name, sigma in (
        ("barrier_sigma_eV", barrier_sigma_eV),
        ("prefactor_log_sigma", prefactor_log_sigma),
    ):
        if sigma is not None and (not math.isfinite(sigma) or sigma < 0):
            raise RateError(f"{name} must be finite and nonnegative")
    _check_temperature(temperature_K)
    vineyard = vineyard_prefactor_hz(minimum_freqs_hz, saddle_freqs_hz)
    correction = vibrational_free_energy_eV(saddle_freqs_hz, temperature_K) - (
        vibrational_free_energy_eV(minimum_freqs_hz, temperature_K)
    )
    dg = barrier_eV + correction
    kt = KB_EV_PER_K * temperature_K
    log_rate = math.log(multiplicity) + math.log(kt / H_EV_S) - dg / kt
    rate = math.exp(log_rate) if log_rate > -700.0 else 0.0
    log_pref = log_rate + barrier_eV / kt - math.log(multiplicity)
    sigma = None
    if barrier_sigma_eV is not None or prefactor_log_sigma is not None:
        sigma = math.hypot((barrier_sigma_eV or 0.0) / kt, prefactor_log_sigma or 0.0)
    return HarmonicRate(
        rate_per_s=rate,
        activation_free_energy_eV=dg,
        effective_prefactor_hz=math.exp(log_pref) if log_pref < 709.0 else math.inf,
        log_effective_prefactor=log_pref,
        vibrational_correction_eV=correction,
        vineyard_prefactor_hz=vineyard,
        log_rate_sigma=sigma,
        assumptions=(
            f"harmonic TST with g={multiplicity}, quantum oscillators incl. zero-point energy",
            "no tunnelling, anharmonicity or electronic free energy",
            "frequencies supplied by the caller, not computed here",
        ),
    )


def harmonic_event_rate(
    event: TransitionEvent,
    minimum_freqs_hz: Sequence[float],
    saddle_freqs_hz: Sequence[float],
    temperature_K: float,
    *,
    require: Require = "qualified",
    barrier_sigma_eV: float | None = None,
) -> tuple[TransitionRate, HarmonicRate]:
    """Rate of a network event with its prefactor from modes instead of a constant.

    Same evidence gate as :func:`event_rate`: no barrier, or non-qualified evidence under
    ``require='qualified'``, gives no rate. A constant attempt frequency on the event is
    replaced and recorded.
    """
    barrier = event.barrier_eV
    if barrier is None:
        raise RateError(f"{event.transition_id}: no barrier evidence")
    if require == "qualified" and event.qualification != "qualified":
        raise RateError(f"{event.transition_id}: evidence is {event.qualification}, not qualified")
    harmonic = harmonic_tst_rate(
        barrier,
        minimum_freqs_hz,
        saddle_freqs_hz,
        temperature_K,
        multiplicity=event.multiplicity,
        barrier_sigma_eV=barrier_sigma_eV,
    )
    notes = list(harmonic.assumptions)
    if event.attempt_frequency_hz is not None:
        notes.append(f"constant attempt frequency {event.attempt_frequency_hz:g} Hz replaced")
    if event.qualification != "qualified":
        notes.append(f"evidence is {event.qualification}; exploratory only")
    if event.evidence is not None and event.evidence.synthetic:
        notes.append("synthetic evidence")
    return (
        TransitionRate(
            transition_id=event.transition_id,
            temperature_K=temperature_K,
            rate_per_s=harmonic.rate_per_s,
            qualification=event.qualification,
            assumptions=tuple(notes),
        ),
        harmonic,
    )


def reverse_barrier_eV(forward_barrier_eV: float, reaction_energy_eV: float) -> float:
    """``Ea_reverse = Ea_forward - (E_target - E_source)``; negative results are an error."""
    rev = forward_barrier_eV - reaction_energy_eV
    if not math.isfinite(rev) or rev < 0:
        raise RateError(
            f"inconsistent barriers: forward {forward_barrier_eV} eV, "
            f"reaction energy {reaction_energy_eV} eV give a negative reverse barrier"
        )
    return rev


__all__ = [
    "H_EV_S",
    "HarmonicRate",
    "harmonic_event_rate",
    "harmonic_tst_rate",
    "reverse_barrier_eV",
    "vibrational_free_energy_eV",
    "vineyard_prefactor_hz",
]
