"""Per-material readiness records tied to the qualification matrix.

A record states what is fixed (composition and its source), what the current workflow cannot
represent, and what the matrix still lacks. It never raises a status: ``qualified`` is read
from the matrix row and is only true if the matrix says so.

Composition rules:

* the basis (``wt`` or ``at``) is explicit and a wt% input is converted to at% with
  standard atomic weights; the conversion is part of the record;
* at most one species may be left as ``None`` (the balance); the rest must be positive and
  sum to at most 100;
* a composition with no cited ``source`` is *not fixed*. Only compositions defined by the
  material's own designation (V-4Cr-4Ti, SiC stoichiometry) are supplied as defaults; W-rich,
  Fe-Cr RAFM and ODS compositions must come from the caller with a source.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from matsim_agents.workflows.qualification_matrix import (
    QualificationMatrix,
    QualificationStatus,
)

Basis = Literal["wt", "at"]
SUM_TOLERANCE = 1e-6

# Standard atomic weights (IUPAC abridged), g/mol.
ATOMIC_WEIGHT: dict[str, float] = {
    "C": 12.011,
    "O": 15.999,
    "Si": 28.085,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938,
    "Fe": 55.845,
    "Y": 88.906,
    "Ta": 180.948,
    "W": 183.84,
    "Re": 186.207,
}


class MaterialError(ValueError):
    """A composition or material record is not usable."""


def _resolve(composition: Mapping[str, float | None]) -> dict[str, float]:
    """Fill the single balance species and validate; values are percent on the given basis."""
    unknown = sorted(set(composition) - set(ATOMIC_WEIGHT))
    if unknown:
        raise MaterialError(f"no atomic weight for: {', '.join(unknown)}")
    balance = [k for k, v in composition.items() if v is None]
    if len(balance) > 1:
        raise MaterialError("at most one balance species may be None")
    given = {k: float(v) for k, v in composition.items() if v is not None}
    if any(not math.isfinite(v) or v <= 0 for v in given.values()):
        raise MaterialError("percentages must be positive and finite")
    total = math.fsum(given.values())
    if total > 100 + SUM_TOLERANCE:
        raise MaterialError(f"percentages sum to {total}, above 100")
    if balance:
        rest = 100.0 - total
        if rest <= SUM_TOLERANCE:
            raise MaterialError("no room left for the balance species")
        given[balance[0]] = rest
    elif abs(total - 100.0) > SUM_TOLERANCE:
        raise MaterialError(f"percentages sum to {total}, not 100")
    return given


def to_atomic_fractions(composition: Mapping[str, float | None], basis: Basis) -> dict[str, float]:
    """Atomic fractions summing to 1. ``wt`` input is converted with ``ATOMIC_WEIGHT``."""
    if basis not in ("wt", "at"):
        raise MaterialError(f"basis must be 'wt' or 'at', not {basis!r}")
    pct = _resolve(composition)
    moles = pct if basis == "at" else {k: v / ATOMIC_WEIGHT[k] for k, v in pct.items()}
    total = math.fsum(moles.values())
    return {k: v / total for k, v in moles.items()}


def to_weight_fractions(composition_at: Mapping[str, float]) -> dict[str, float]:
    """Inverse conversion, from atomic fractions to weight fractions (for round-trip checks)."""
    pct = _resolve({k: 100.0 * v for k, v in composition_at.items()})
    mass = {k: v * ATOMIC_WEIGHT[k] for k, v in pct.items()}
    total = math.fsum(mass.values())
    return {k: v / total for k, v in mass.items()}


@dataclass(frozen=True)
class MaterialSpec:
    row_id: str
    name: str
    composition: Mapping[str, float | None] | None
    basis: Basis
    source: str | None
    scope_limits: tuple[str, ...]


@dataclass(frozen=True)
class MaterialReadiness:
    row_id: str
    name: str
    composition_at: dict[str, float] | None
    basis: Basis
    source: str | None
    composition_fixed: bool
    matrix_status: str
    qualified: bool
    missing_criteria: tuple[str, ...]
    scope_limits: tuple[str, ...]
    blockers: tuple[str, ...]
    ready_for_calculation: bool


def default_material_specs() -> dict[str, MaterialSpec]:
    """One spec per material row. Only designation-defined compositions are pre-filled."""
    return {
        "material-w-rich": MaterialSpec(
            "material-w-rich",
            "W-rich alloys",
            None,
            "at",
            None,
            ("random bcc solid solution only; precipitates and grain boundaries not modelled",),
        ),
        "material-fecr-rafm": MaterialSpec(
            "material-fecr-rafm",
            "Fe-Cr RAFM steels",
            None,
            "wt",
            None,
            (
                "carbides, martensitic microstructure and magnetism are not modelled",
                "a simplified bcc random alloy stands in for the steel",
            ),
        ),
        "material-v-4cr-4ti": MaterialSpec(
            "material-v-4cr-4ti",
            "V-4Cr-4Ti (wt%)",
            {"V": None, "Cr": 4.0, "Ti": 4.0},
            "wt",
            "alloy designation V-4Cr-4Ti (nominal 4 wt% Cr, 4 wt% Ti, balance V)",
            ("interstitial O, C, N impurities are not modelled",),
        ),
        "material-ods": MaterialSpec(
            "material-ods",
            "ODS steels",
            None,
            "wt",
            None,
            (
                "oxide dispersoids and the matrix-oxide interface cannot be represented",
                "only the steel matrix could be studied, which is not the ODS material",
            ),
        ),
        "material-sic-sic": MaterialSpec(
            "material-sic-sic",
            "SiC/SiC composites",
            {"Si": 50.0, "C": 50.0},
            "at",
            "SiC stoichiometry (1:1 by formula)",
            (
                "polytype is unspecified and needs a source",
                "fibre, matrix and interphase of the composite are not representable",
                "the random-alloy generator does not describe an ordered compound",
            ),
        ),
    }


def assess_material(matrix: QualificationMatrix, spec: MaterialSpec) -> MaterialReadiness:
    """Combine a spec with its matrix row. Raises ``QualificationError`` for an unknown row."""
    row = matrix.row(spec.row_id)
    blockers: list[str] = []
    composition_at: dict[str, float] | None = None
    if spec.composition is None:
        blockers.append("no composition supplied")
    else:
        composition_at = to_atomic_fractions(spec.composition, spec.basis)
    source_ok = bool(spec.source and spec.source.strip())
    if composition_at is not None and not source_ok:
        blockers.append("composition has no cited source")
    fixed = composition_at is not None and source_ok
    if spec.scope_limits:
        blockers.append("workflow scope limits apply: " + "; ".join(spec.scope_limits))
    missing = tuple(row.missing())
    if missing:
        blockers.append("matrix criteria without measured evidence: " + ", ".join(missing))
    qualified = row.status is QualificationStatus.QUALIFIED
    return MaterialReadiness(
        row_id=spec.row_id,
        name=spec.name,
        composition_at=composition_at,
        basis=spec.basis,
        source=spec.source if source_ok else None,
        composition_fixed=fixed,
        matrix_status=row.status.value,
        qualified=qualified,
        missing_criteria=missing,
        scope_limits=spec.scope_limits,
        blockers=tuple(blockers),
        ready_for_calculation=fixed,
    )


def assess_all(
    matrix: QualificationMatrix, specs: Mapping[str, MaterialSpec] | None = None
) -> tuple[MaterialReadiness, ...]:
    """Readiness of every material row in the matrix."""
    chosen = default_material_specs() if specs is None else specs
    return tuple(
        assess_material(matrix, chosen[r.row_id]) for r in matrix.rows if r.dimension == "material"
    )


__all__ = [
    "ATOMIC_WEIGHT",
    "MaterialError",
    "MaterialReadiness",
    "MaterialSpec",
    "assess_all",
    "assess_material",
    "default_material_specs",
    "to_atomic_fractions",
    "to_weight_fractions",
]
