"""Observables, acceptance criteria and convergence analysis for ``pw.x`` numerical scans.

A scan varies one numerical parameter (cutoff, k-mesh, smearing width, ``conv_thr``) over an
ordered series of points. Every point holds the same cases (structures). The most demanding
point is the reference; a point is accepted only when it and every more demanding point agree
with the reference within the acceptance criteria, so a non-monotonic series is not accepted
early. The cheapest accepted point is selected.

Scheduler completion is not numerical success: a point is usable only when ``pw.x`` printed
``JOB DONE``, the SCF converged, and all requested observables were parsed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from matsim_agents.backends.dft.qe_relax import parse_pw_stdout

KBAR_PER_GPA = 10.0

_RE_PRESSURE = re.compile(r"total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(-?\d+\.\d+)")
_RE_SCF_OK = re.compile(r"convergence has been achieved in\s+\d+\s+iterations")
_RE_TOTAL_MAG = re.compile(r"total magnetization\s*=\s*(-?\d+\.\d+)")
_RE_ABS_MAG = re.compile(r"absolute magnetization\s*=\s*(-?\d+\.\d+)")
_RE_ENERGY = re.compile(r"^!\s+total energy\s+=\s*(-?\d+\.\d+)\s+Ry", re.MULTILINE)
_RY_TO_EV = 13.605693122994


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Numerical agreement required between a point and the reference point."""

    energy_per_atom_meV: float = 1.0
    derived_energy_eV: float = 0.02  # defect formation, binding and barrier energies
    force_eV_per_A: float = 0.005
    force_defect_eV_per_A: float = 0.01
    stress_GPa: float = 0.1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> AcceptanceCriteria:
        return cls(**dict(data or {}))


@dataclass
class PointObservables:
    """What one ``pw.x`` run yields for one case."""

    natoms: int
    energy_eV: float
    forces_eV_per_A: list[list[float]] | None
    stress_kbar: float | None  # pressure (trace/3) as printed
    total_magnetization: float | None
    absolute_magnetization: float | None
    scf_converged: bool
    job_done: bool
    wall_time_sec: float | None
    derived: dict[str, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.scf_converged and self.job_done and math.isfinite(self.energy_eV)

    @property
    def convergence_usable(self) -> bool:
        forces = self.forces_eV_per_A
        return (
            self.usable
            and self.natoms > 0
            and forces is not None
            and len(forces) == self.natoms
            and all(len(row) == 3 and all(math.isfinite(x) for x in row) for row in forces)
            and self.stress_kbar is not None
            and math.isfinite(self.stress_kbar)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "natoms": self.natoms,
            "energy_eV": self.energy_eV,
            "forces_eV_per_A": self.forces_eV_per_A,
            "stress_kbar": self.stress_kbar,
            "total_magnetization": self.total_magnetization,
            "absolute_magnetization": self.absolute_magnetization,
            "scf_converged": self.scf_converged,
            "job_done": self.job_done,
            "wall_time_sec": self.wall_time_sec,
            "derived": dict(self.derived),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PointObservables:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


def parse_observables(
    stdout_path: str | Path, natoms: int, return_code: int = 0
) -> PointObservables:
    """Read energy, forces, pressure and magnetization from a ``pw.x`` stdout.

    An unreadable or empty file yields an unusable result (``energy_eV`` NaN), never an
    exception, so a failed run is recorded rather than lost.
    """
    path = Path(stdout_path)
    if not path.is_file():
        return PointObservables(natoms, math.nan, None, None, None, None, False, False, None)
    text = path.read_text(errors="replace")
    energies = _RE_ENERGY.findall(text)
    energy = float(energies[-1]) * _RY_TO_EV if energies else math.nan
    pw = parse_pw_stdout(str(path), return_code)
    pressures = _RE_PRESSURE.findall(text)
    total_mag = _RE_TOTAL_MAG.findall(text)
    abs_mag = _RE_ABS_MAG.findall(text)
    return PointObservables(
        natoms=natoms,
        energy_eV=energy,
        forces_eV_per_A=pw.final_forces_eV_per_A,
        stress_kbar=float(pressures[-1]) if pressures else None,
        total_magnetization=float(total_mag[-1]) if total_mag else None,
        absolute_magnetization=float(abs_mag[-1]) if abs_mag else None,
        scf_converged=bool(_RE_SCF_OK.search(text)) and "convergence NOT achieved" not in text,
        job_done=pw.job_done,
        wall_time_sec=pw.wall_time_sec,
    )


def apply_derived(
    cases: Mapping[str, PointObservables], definitions: Sequence[Mapping[str, Any]]
) -> None:
    """Fill ``derived`` energies as linear combinations ``sum(coefficient * energy_eV)``.

    A definition is ``{"name": ..., "terms": {case_name: coefficient}}``. A definition whose
    cases are missing or unusable is skipped, which makes the point incomplete downstream.
    """
    for definition in definitions:
        terms = definition["terms"]
        members = [cases.get(case) for case in terms]
        if any(member is None or not member.usable for member in members):
            continue
        value = sum(float(c) * cases[case].energy_eV for case, c in terms.items())
        owner = cases[next(iter(terms))]
        owner.derived[definition["name"]] = value


@dataclass
class ScanPoint:
    value: Any
    cases: dict[str, PointObservables]
    cost: float  # core-hours or wall seconds; a larger cost is more demanding
    cell_ids: list[str] = field(default_factory=list)
    expected_cases: frozenset[str] = field(default_factory=frozenset)
    required_derived: tuple[tuple[str, str], ...] = ()
    diagnostic_only: bool = False


@dataclass
class PointComparison:
    value: Any
    complete: bool
    energy_per_atom_meV: float | None = None
    derived_eV: float | None = None
    force_eV_per_A: float | None = None
    stress_GPa: float | None = None
    magnetization_changed: bool = False
    accepted: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _max_force_difference(a: PointObservables, b: PointObservables) -> float | None:
    if a.forces_eV_per_A is None or b.forces_eV_per_A is None:
        return None
    if len(a.forces_eV_per_A) != len(b.forces_eV_per_A):
        return math.inf
    return max(
        (
            abs(x - y)
            for fa, fb in zip(a.forces_eV_per_A, b.forces_eV_per_A, strict=True)
            for x, y in zip(fa, fb, strict=True)
        ),
        default=0.0,
    )


def _magnetization_differs(obs: PointObservables, ref: PointObservables) -> bool:
    for value, reference in (
        (obs.total_magnetization, ref.total_magnetization),
        (obs.absolute_magnetization, ref.absolute_magnetization),
    ):
        if value is None or reference is None or not math.isfinite(value + reference):
            return True
        if abs(value - reference) > 0.1 * max(1.0, abs(reference)):
            return True
    return False


def compare_point(
    point: ScanPoint,
    reference: ScanPoint,
    criteria: AcceptanceCriteria,
    case_kinds: Mapping[str, str] | None = None,
    magnetic_cases: Sequence[str] = (),
) -> PointComparison:
    """Differences between ``point`` and ``reference`` over all shared cases."""
    case_kinds = case_kinds or {}
    out = PointComparison(value=point.value, complete=True)
    expected = reference.expected_cases or frozenset(reference.cases)
    missing = sorted(
        name
        for name in expected
        if name not in reference.cases
        or name not in point.cases
        or not point.cases[name].convergence_usable
        or not reference.cases[name].convergence_usable
    )
    if missing:
        out.complete = False
        out.reasons.append(f"unusable or missing cases: {missing}")
        return out
    for owner, name in reference.required_derived:
        for cases in (point.cases, reference.cases):
            value = cases[owner].derived.get(name) if owner in cases else None
            if value is None or not math.isfinite(value):
                out.complete = False
                out.reasons.append(f"derived energy {name!r} missing or nonfinite")
                return out
    for name in magnetic_cases:
        if name not in expected:
            continue
        for obs in (point.cases[name], reference.cases[name]):
            if any(
                value is None or not math.isfinite(value)
                for value in (obs.total_magnetization, obs.absolute_magnetization)
            ):
                out.complete = False
                out.reasons.append(f"{name}: missing or nonfinite magnetization")
                return out
    de, dd, df, ds = [], [], [], []
    for name, ref in reference.cases.items():
        obs = point.cases[name]
        de.append(abs(obs.energy_eV - ref.energy_eV) / ref.natoms * 1000.0)
        for key, ref_value in ref.derived.items():
            if key not in obs.derived:
                out.complete = False
                out.reasons.append(f"derived energy {key!r} missing")
            else:
                dd.append(abs(obs.derived[key] - ref_value))
        diff = _max_force_difference(obs, ref)
        if diff is not None:
            tol = (
                criteria.force_defect_eV_per_A
                if case_kinds.get(name) in _DEFECT_KINDS
                else criteria.force_eV_per_A
            )
            df.append(diff)
            if diff > tol:
                out.reasons.append(f"{name}: force difference {diff:.4f} eV/A > {tol}")
        if obs.stress_kbar is not None and ref.stress_kbar is not None:
            ds.append(abs(obs.stress_kbar - ref.stress_kbar) / KBAR_PER_GPA)
        if name in magnetic_cases and _magnetization_differs(obs, ref):
            out.magnetization_changed = True
    out.energy_per_atom_meV = max(de) if de else None
    out.derived_eV = max(dd) if dd else None
    out.force_eV_per_A = max(df) if df else None
    out.stress_GPa = max(ds) if ds else None
    if (
        out.energy_per_atom_meV is not None
        and out.energy_per_atom_meV > criteria.energy_per_atom_meV
    ):
        out.reasons.append(
            f"energy {out.energy_per_atom_meV:.3f} meV/atom > {criteria.energy_per_atom_meV}"
        )
    if out.derived_eV is not None and out.derived_eV > criteria.derived_energy_eV:
        out.reasons.append(f"derived energy {out.derived_eV:.4f} eV > {criteria.derived_energy_eV}")
    if out.stress_GPa is not None and out.stress_GPa > criteria.stress_GPa:
        out.reasons.append(f"stress {out.stress_GPa:.4f} GPa > {criteria.stress_GPa}")
    if out.magnetization_changed:
        out.reasons.append("magnetic state differs from the reference")
    if point.diagnostic_only:
        out.reasons.append("below pseudopotential recommendation; diagnostic only")
    out.accepted = out.complete and not out.reasons
    return out


_DEFECT_KINDS = frozenset({"vacancy", "substitution", "interstitial", "neb_image", "defect"})


@dataclass
class ScanOutcome:
    parameter: str
    reference_value: Any
    comparisons: list[PointComparison]
    selected_value: Any | None
    status: str  # "qualified" | "not_converged" | "incomplete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "reference_value": self.reference_value,
            "selected_value": self.selected_value,
            "status": self.status,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


def analyse_scan(
    parameter: str,
    points: Sequence[ScanPoint],
    criteria: AcceptanceCriteria,
    case_kinds: Mapping[str, str] | None = None,
    magnetic_cases: Sequence[str] = (),
) -> ScanOutcome:
    """Select the cheapest point that agrees with the most demanding one.

    ``points`` are ordered from least to most demanding. The reference must itself be
    complete; otherwise the scan is ``incomplete`` and nothing is selected. A scan whose
    only accepted point is the reference is ``not_converged``: it shows no plateau.
    """
    if len(points) < 3:
        raise ValueError(f"scan {parameter!r} needs at least three points to show a plateau")
    reference = points[-1]
    if (
        not reference.cases
        or reference.diagnostic_only
        or not all(o.convergence_usable for o in reference.cases.values())
        or not reference.expected_cases.issubset(reference.cases)
    ):
        return ScanOutcome(parameter, reference.value, [], None, "incomplete")
    comparisons = [
        compare_point(p, reference, criteria, case_kinds, magnetic_cases) for p in points
    ]
    if not comparisons[-1].complete:
        return ScanOutcome(parameter, reference.value, comparisons, None, "incomplete")
    selected = None
    for index in range(len(points) - 1):
        if all(c.accepted for c in comparisons[index:]):
            selected = points[index].value
            break
    if selected is None:
        status = "incomplete" if any(not c.complete for c in comparisons[:-1]) else "not_converged"
    else:
        status = "qualified"
    return ScanOutcome(parameter, reference.value, comparisons, selected, status)


__all__ = [
    "AcceptanceCriteria",
    "PointComparison",
    "PointObservables",
    "ScanOutcome",
    "ScanPoint",
    "analyse_scan",
    "apply_derived",
    "compare_point",
    "parse_observables",
]
