"""Magnetic consistency checks for formation energies, NEB barriers and CT-KMC rates (M8).

Energies are only differenced when they come from one numerical method and one magnetic
treatment. The checks return verdicts with reasons; they never adjust energies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from matsim_agents.discovery.magnetic_results import MagneticResult, classify_moments
from matsim_agents.discovery.magnetism import MagneticConfiguration

K_B_EV_PER_K = 8.617333262e-5
Verdict = Literal["consistent", "inconsistent", "insufficient_evidence"]


@dataclass(frozen=True)
class MagneticEvidence:
    """What one energy was computed with."""

    label: str
    method_hash: str
    config: MagneticConfiguration | None
    result: MagneticResult | None
    qualification: str = "unqualified"  # unqualified | software_tested | numerically_validated

    @property
    def spin_polarized(self) -> bool:
        return bool(self.config is not None and self.config.spin_polarized)

    @property
    def classification(self) -> str:
        return classify_moments(self.result) if self.result is not None else "insufficient_evidence"


@dataclass(frozen=True)
class ConsistencyReport:
    verdict: Verdict
    reasons: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.verdict == "consistent"


def _compare(a: MagneticEvidence, b: MagneticEvidence, *, require_same_class: bool) -> list[str]:
    reasons: list[str] = []
    if a.method_hash != b.method_hash:
        reasons.append(f"{a.label} and {b.label} use different numerical methods")
    if a.spin_polarized != b.spin_polarized:
        reasons.append(f"{a.label} and {b.label} differ in spin polarization")
    if (
        a.config is not None
        and b.config is not None
        and a.config.magnetism_model != b.config.magnetism_model
    ):
        reasons.append(f"{a.label} and {b.label} use different magnetism models")
    if require_same_class and a.classification != b.classification:
        reasons.append(
            f"{a.label} is {a.classification} but {b.label} is {b.classification}; "
            "the energy difference mixes magnetic states"
        )
    return reasons


def _missing(items: Sequence[MagneticEvidence]) -> list[str]:
    out = []
    for e in items:
        if e.config is None:
            out.append(f"{e.label}: no magnetic configuration recorded")
        if e.result is None or not e.result.converged:
            out.append(f"{e.label}: no converged magnetic result")
        elif not e.result.local_moments_muB and e.spin_polarized:
            out.append(f"{e.label}: no local moments to classify")
    return out


def check_formation_energy(
    defect: MagneticEvidence,
    reference: MagneticEvidence,
    *,
    require_same_class: bool = True,
) -> ConsistencyReport:
    """A defect formation energy needs defect and pristine cells treated identically."""
    missing = _missing([defect, reference])
    if missing:
        return ConsistencyReport("insufficient_evidence", tuple(missing))
    reasons = _compare(defect, reference, require_same_class=require_same_class)
    return ConsistencyReport("inconsistent" if reasons else "consistent", tuple(reasons))


def check_barrier(
    initial: MagneticEvidence,
    saddle_or_images: Sequence[MagneticEvidence],
    final: MagneticEvidence,
    *,
    path_flags: Sequence[str] = (),
) -> ConsistencyReport:
    """A barrier needs endpoints and images from one method and one magnetic state, plus a
    path without collapse or switching."""
    everything = [initial, *saddle_or_images, final]
    missing = _missing(everything)
    if missing:
        return ConsistencyReport("insufficient_evidence", tuple(missing))
    reasons = list(path_flags)
    for other in everything[1:]:
        reasons += _compare(initial, other, require_same_class=True)
    return ConsistencyReport("inconsistent" if reasons else "consistent", tuple(reasons))


@dataclass(frozen=True)
class RateSensitivity:
    temperature_K: float
    rates_per_s: Mapping[str, float]
    max_ratio: float
    sensitive: bool


def arrhenius_rate(barrier_eV: float, temperature_K: float, prefactor_per_s: float) -> float:
    if temperature_K <= 0 or prefactor_per_s <= 0:
        raise ValueError("temperature and prefactor must be positive")
    return prefactor_per_s * math.exp(-barrier_eV / (K_B_EV_PER_K * temperature_K))


def rate_sensitivity(
    barriers_eV: Mapping[str, float],
    temperature_K: float,
    *,
    prefactor_per_s: float = 1.0e13,
    ratio_threshold: float = 2.0,
) -> RateSensitivity:
    """Rates of one event for different magnetic states; ``sensitive`` when they span more than
    ``ratio_threshold`` (default a factor of two)."""
    if len(barriers_eV) < 2:
        raise ValueError("need barriers for at least two magnetic states")
    rates = {k: arrhenius_rate(v, temperature_K, prefactor_per_s) for k, v in barriers_eV.items()}
    ratio = max(rates.values()) / min(rates.values())
    return RateSensitivity(temperature_K, rates, ratio, ratio > ratio_threshold)


@dataclass(frozen=True)
class KMCGate:
    allowed: bool
    reasons: tuple[str, ...]
    flags: tuple[str, ...] = ()


def gate_transition_for_kmc(
    barrier_report: ConsistencyReport,
    evidence: Sequence[MagneticEvidence],
    *,
    sensitivity: RateSensitivity | None = None,
    allow_unqualified: bool = False,
) -> KMCGate:
    """Admit a magnetic transition to CT-KMC only with consistent, qualified evidence.

    ``allow_unqualified`` admits software-only evidence but always flags it.
    """
    reasons: list[str] = []
    flags: list[str] = []
    if not barrier_report.usable:
        reasons.append(f"barrier magnetic consistency is {barrier_report.verdict}")
        reasons += list(barrier_report.reasons)
    unqualified = [e.label for e in evidence if e.qualification != "numerically_validated"]
    if unqualified:
        message = f"evidence not numerically validated: {unqualified}"
        (flags if allow_unqualified else reasons).append(message)
    if sensitivity is not None and sensitivity.sensitive:
        flags.append(
            f"rate depends on magnetic state (ratio {sensitivity.max_ratio:.1f} at "
            f"{sensitivity.temperature_K:.0f} K)"
        )
    return KMCGate(not reasons, tuple(reasons), tuple(flags))
