"""MLIP radiation-qualification record and level resolution.

A level is granted only from stated evidence checks, each with a criterion and a domain, and
only inside that domain: chemistry, temperature range, recoil-energy range, simulation-size
range, precision and short-range model. Levels nest (each needs the lower ones) but never
transfer between domains: a pure-W record says nothing about a W alloy. This module records
and resolves evidence; it does not run the calculations that produce it, and a check passing
here means the supplied numbers met the supplied criterion, not that the model is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Level(IntEnum):
    UNASSESSED = 0
    EQUILIBRIUM_QUALIFIED = 1
    DEFECT_QUALIFIED = 2
    SHORT_RANGE_QUALIFIED = 3
    CASCADE_QUALIFIED = 4


REQUIRED_CHECKS: dict[Level, tuple[str, ...]] = {
    Level.EQUILIBRIUM_QUALIFIED: ("lattice", "elastic", "nve_energy_conservation"),
    Level.DEFECT_QUALIFIED: (
        "vacancy_formation",
        "interstitial_formation",
        "defect_binding",
        "high_temperature",
    ),
    Level.SHORT_RANGE_QUALIFIED: (
        "pair_curves",
        "threshold_displacement",
        "directional_thresholds",
        "distorted_configurations",
    ),
    Level.CASCADE_QUALIFIED: ("cascade_defect_formation", "surviving_defect_statistics"),
}


@dataclass(frozen=True)
class Domain:
    elements: frozenset[str]
    temperature_K: tuple[float, float]
    recoil_energy_eV: tuple[float, float]
    n_atoms: tuple[int, int]
    precision: str
    short_range_model: str

    def __post_init__(self) -> None:
        for lo, hi in (self.temperature_K, self.recoil_energy_eV, self.n_atoms):
            if lo > hi:
                raise ValueError("a range must have low <= high")
        if not self.elements:
            raise ValueError("a domain needs at least one element")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    criterion: str
    value: float | None
    dataset: str


@dataclass(frozen=True)
class QualificationRecord:
    checkpoint: str
    domain: Domain
    checks: tuple[Check, ...]


@dataclass(frozen=True)
class Query:
    elements: frozenset[str]
    temperature_K: float
    recoil_energy_eV: float
    n_atoms: int
    precision: str
    short_range_model: str


@dataclass(frozen=True)
class Resolution:
    level: Level
    reasons: tuple[str, ...]
    missing: dict[str, tuple[str, ...]]


def _outside(domain: Domain, q: Query) -> list[str]:
    out = []
    if not q.elements <= domain.elements:
        out.append(f"elements {sorted(q.elements - domain.elements)} are outside the domain")
    if not domain.temperature_K[0] <= q.temperature_K <= domain.temperature_K[1]:
        out.append("temperature is outside the qualified range")
    if not domain.recoil_energy_eV[0] <= q.recoil_energy_eV <= domain.recoil_energy_eV[1]:
        out.append("recoil energy is outside the qualified range")
    if not domain.n_atoms[0] <= q.n_atoms <= domain.n_atoms[1]:
        out.append("simulation size is outside the qualified range")
    if q.precision != domain.precision:
        out.append(f"precision {q.precision!r} differs from {domain.precision!r}")
    if q.short_range_model != domain.short_range_model:
        out.append("short-range model differs from the qualified one")
    return out


def resolve_level(record: QualificationRecord, query: Query, *, checkpoint: str) -> Resolution:
    """Highest level whose required checks pass, or UNASSESSED outside the domain."""
    if record.checkpoint != checkpoint:
        return Resolution(Level.UNASSESSED, ("record is for a different checkpoint",), {})
    reasons = _outside(record.domain, query)
    if reasons:
        return Resolution(Level.UNASSESSED, tuple(reasons), {})
    failed = {c.name for c in record.checks if not c.passed}
    # A failed result for a name blocks it even if another result for that name passed.
    passed = {c.name for c in record.checks if c.passed} - failed
    level = Level.UNASSESSED
    missing: dict[str, tuple[str, ...]] = {}
    notes: list[str] = []
    for lv in sorted(REQUIRED_CHECKS):
        need = REQUIRED_CHECKS[lv]
        lacking = tuple(n for n in need if n not in passed)
        if lacking:
            missing[lv.name] = lacking
            notes.append(f"{lv.name}: missing or failed {list(lacking)}")
            break
        level = lv
    bad = sorted(failed & {n for v in REQUIRED_CHECKS.values() for n in v})
    if bad:
        notes.append(f"failed checks: {bad}")
    return Resolution(level, tuple(notes), missing)


def require_level(resolution: Resolution, minimum: Level) -> None:
    if resolution.level < minimum:
        raise PermissionError(
            f"model is {resolution.level.name}, {minimum.name} required: "
            + "; ".join(resolution.reasons or ("no evidence",))
        )


__all__ = [
    "REQUIRED_CHECKS",
    "Check",
    "Domain",
    "Level",
    "QualificationRecord",
    "Query",
    "Resolution",
    "require_level",
    "resolve_level",
]
