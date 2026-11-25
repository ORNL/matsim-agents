"""Optional interfaces for transport solvers (EPW, Kubo-Greenwood, ShengBTE, almaBTE).

MATSIM-AGENTS embeds none of these solvers. This module provides what surrounds
them: availability and readiness reporting, cost estimates from a measured
reference, ingestion of existing results in a validated format, and the documented
combinations of transport evidence:

- ``kappa_total = kappa_e + kappa_ph``, only for matching structure and conditions;
- Wiedemann-Franz ``kappa_e = L sigma T``, only from an evidenced ``sigma``, a
  *sourced* Lorenz number, and a recorded temperature, always labelled
  ``APPROXIMATED``.

Nothing here substitutes a smaller supercell, zero-fills a missing value, or
treats ``sigma/tau`` as a conductivity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from matsim_agents.execution.contracts import PROPERTY_UNITS, EvidenceLevel, at_least
from matsim_agents.execution.properties import (
    PROPERTY_FAMILY,
    PropertyConditions,
    PropertyEvidence,
    PropertyStatus,
    convert_to_canonical,
)

INGEST_FORMAT = "matsim-transport-v1"
TRANSPORT_PROPERTIES = frozenset(
    n for n, f in PROPERTY_FAMILY.items() if f in {"thermal", "electrical"}
)
# pi^2/3 (k_B/e)^2 with CODATA-2018 k_B and e: the free-electron (Sommerfeld) limit.
_KB, _E = 1.380649e-23, 1.602176634e-19
SOMMERFELD_LORENZ = (
    math.pi**2 / 3 * (_KB / _E) ** 2,
    "Sommerfeld free-electron value, CODATA 2018",
)


class TransportStatus(StrEnum):
    AVAILABLE = "available"
    NOT_CONFIGURED = "not_configured"
    METHOD_UNSUPPORTED = "METHOD_UNSUPPORTED"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"


@dataclass(frozen=True)
class TransportBackend:
    name: str
    executable: str
    produces: tuple[str, ...]
    requires_ordered_crystal: bool
    # Assumed asymptotic cost scaling in the number of atoms; scales a MEASURED reference.
    cost_exponent: float
    scattering: str


BACKENDS: dict[str, TransportBackend] = {
    "epw": TransportBackend(
        "epw",
        "epw.x",
        (
            "electrical_conductivity",
            "electrical_conductivity_tensor",
            "electrical_conductivity_over_relaxation_time",
            "electronic_scattering_rate",
            "thermal_conductivity_electronic",
        ),
        True,
        3.0,
        "electron-phonon",
    ),
    "kubo_greenwood": TransportBackend(
        "kubo_greenwood",
        "",
        ("electrical_conductivity", "thermal_conductivity_electronic"),
        False,
        3.0,
        "static-disorder snapshots (no inelastic scattering)",
    ),
    "shengbte": TransportBackend(
        "shengbte",
        "ShengBTE",
        ("thermal_conductivity_lattice", "thermal_conductivity_tensor"),
        True,
        3.0,
        "three-phonon and isotope",
    ),
    "almabte": TransportBackend(
        "almabte",
        "",
        ("thermal_conductivity_lattice", "thermal_conductivity_tensor"),
        False,
        3.0,
        "three-phonon, isotope, and boundary",
    ),
}


@dataclass
class TransportReadiness:
    backend: str
    property: str
    status: TransportStatus
    reasons: list[str]
    estimated_node_hours: float | None = None

    @property
    def ready(self) -> bool:
        return self.status == TransportStatus.AVAILABLE


def check_backend(
    backend: str,
    prop: str,
    *,
    executable: str | None = None,
    n_atoms: int | None = None,
    ordered_crystal: bool | None = None,
    reference_node_hours: float | None = None,
    reference_atoms: int | None = None,
    remaining_node_hours: float | None = None,
    n_runs: int = 1,
) -> TransportReadiness:
    """Whether ``backend`` can evaluate ``prop`` for this cell; never runs anything."""
    spec = BACKENDS[backend]
    if prop not in spec.produces:
        return TransportReadiness(
            backend,
            prop,
            TransportStatus.METHOD_UNSUPPORTED,
            [f"{backend} does not evaluate {prop}"],
        )
    if spec.requires_ordered_crystal and ordered_crystal is not True:
        return TransportReadiness(
            backend,
            prop,
            TransportStatus.METHOD_UNSUPPORTED,
            [
                f"{backend} needs an ordered crystal; a chemically disordered cell needs a "
                "validated disorder treatment that is not provided here"
            ],
        )
    path = executable or (shutil.which(spec.executable) if spec.executable else None)
    if not path or not os.access(path, os.X_OK):
        need = spec.executable or "an explicit executable path"
        return TransportReadiness(
            backend, prop, TransportStatus.NOT_CONFIGURED, [f"executable not found ({need})"]
        )
    cost = None
    if reference_node_hours is not None and reference_atoms and n_atoms:
        cost = n_runs * reference_node_hours * (n_atoms / reference_atoms) ** spec.cost_exponent
    reasons: list[str] = []
    if cost is None:
        reasons.append(
            "no measured reference timing: run one smaller calculation and pass "
            "reference_node_hours and reference_atoms"
        )
    elif remaining_node_hours is not None and cost > remaining_node_hours:
        reasons.append(
            f"estimated {cost:.1f} node-hours (assumed N^{spec.cost_exponent:g} scaling from "
            f"the reference) exceed the remaining {remaining_node_hours:.1f}"
        )
    if reasons:
        return TransportReadiness(backend, prop, TransportStatus.RESOURCE_BLOCKED, reasons, cost)
    return TransportReadiness(backend, prop, TransportStatus.AVAILABLE, [], cost)


class IngestedProperty(BaseModel):
    value: float | list[list[float]]
    unit: str
    uncertainty: float | list[list[float]]


class TransportRecord(BaseModel):
    """The validated file format for transport results computed or measured elsewhere."""

    format: str
    structure_id: str
    composition: str
    method: str
    backend: str
    backend_version: str | None = None
    evidence_level: EvidenceLevel
    converged: bool
    measured: bool = False
    conditions: PropertyConditions
    properties: dict[str, IngestedProperty] = Field(min_length=1)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid(self) -> TransportRecord:
        if self.format != INGEST_FORMAT:
            raise ValueError(f"format must be {INGEST_FORMAT!r}")
        bad = sorted(set(self.properties) - TRANSPORT_PROPERTIES)
        if bad:
            raise ValueError(f"not transport properties: {bad}")
        return self


def _convert_value(prop: str, item: IngestedProperty) -> tuple[Any, Any]:
    def convert(x: Any) -> Any:
        if isinstance(x, list):
            return [[convert_to_canonical(prop, v, item.unit) for v in row] for row in x]
        return convert_to_canonical(prop, x, item.unit)

    return convert(item.value), convert(item.uncertainty)


def ingest_transport_record(path: Path) -> list[PropertyEvidence]:
    """Validated PropertyEvidence from a ``matsim-transport-v1`` file.

    The file's SHA-256 becomes the evidence identifier. A converged calculation
    must carry DFT-level evidence; an unconverged one is kept as UNCONVERGED.
    """
    raw = Path(path).read_bytes()
    record = TransportRecord.model_validate(json.loads(raw))
    evidence_id = f"file-{hashlib.sha256(raw).hexdigest()[:24]}"
    out = []
    for prop, item in record.properties.items():
        value, uncertainty = _convert_value(prop, item)
        if record.measured:
            status = PropertyStatus.MEASURED
        elif not record.converged:
            status = PropertyStatus.UNCONVERGED
        elif at_least(record.evidence_level, EvidenceLevel.LOW_FIDELITY_DFT):
            status = PropertyStatus.CALCULATED
        else:
            status = PropertyStatus.SURROGATE_PREDICTED
        extra: dict[str, Any] = {}
        if status == PropertyStatus.UNCONVERGED:
            extra["reason"] = "ingested calculation is marked unconverged"
        else:
            extra["uncertainty"] = uncertainty
            extra["evidence_ids"] = [evidence_id]
        out.append(
            PropertyEvidence(
                structure_id=record.structure_id,
                composition=record.composition,
                property=prop,
                status=status,
                value=value,
                unit=PROPERTY_UNITS[prop],
                conditions=record.conditions,
                method=record.method,
                backend=record.backend,
                backend_version=record.backend_version,
                input_sha256=record.input_sha256,
                output_sha256={path.name: hashlib.sha256(raw).hexdigest()},
                evidence_level=record.evidence_level,
                converged=record.converged,
                limitations=record.limitations,
                **extra,
            )
        )
    return out


def _scalar(e: PropertyEvidence, what: str) -> float:
    if not e.is_scoreable() or not isinstance(e.value, float | int):
        raise ValueError(f"{what} needs a scalar value with evidence")
    if e.uncertainty is None:
        raise ValueError(f"{what} needs an uncertainty")
    return float(e.value)


def wiedemann_franz(
    sigma: PropertyEvidence,
    *,
    lorenz_number: float,
    lorenz_source: str,
    lorenz_uncertainty: float = 0.0,
    temperature_K: float,
) -> PropertyEvidence:
    """Electronic thermal conductivity ``L sigma T`` from an evidenced conductivity.

    An approximation: it assumes elastic electron scattering, so it fails where inelastic
    scattering matters and in strongly disordered or correlated metals.
    """
    if sigma.property != "electrical_conductivity":
        raise ValueError("Wiedemann-Franz needs electrical_conductivity evidence (not sigma/tau)")
    if not lorenz_source.strip():
        raise ValueError("the Lorenz number needs a source")
    if not (math.isfinite(lorenz_number) and lorenz_number > 0):
        raise ValueError("the Lorenz number must be positive and finite")
    if lorenz_uncertainty < 0:
        raise ValueError("the Lorenz uncertainty cannot be negative")
    if temperature_K <= 0:
        raise ValueError("a temperature must be recorded")
    if sigma.conditions.temperature_K != temperature_K:
        raise ValueError(
            f"sigma was evidenced at {sigma.conditions.temperature_K} K, not {temperature_K} K"
        )
    value = _scalar(sigma, "sigma")
    rel = math.hypot(
        float(sigma.uncertainty) / value if value else math.inf,  # type: ignore[arg-type]
        lorenz_uncertainty / lorenz_number,
    )
    kappa = lorenz_number * value * temperature_K
    return PropertyEvidence(
        structure_id=sigma.structure_id,
        composition=sigma.composition,
        chemical_ordering=sigma.chemical_ordering,
        property="thermal_conductivity_electronic",
        status=PropertyStatus.APPROXIMATED,
        value=kappa,
        unit="W/(m K)",
        conditions=sigma.conditions,
        method="wiedemann_franz",
        evidence_ids=sorted(set(sigma.evidence_ids) | {sigma.property_id()}),
        evidence_level=sigma.evidence_level,
        converged=sigma.converged,
        uncertainty=kappa * rel,
        derived_from=["electrical_conductivity"],
        approximation=(
            f"kappa_e = L sigma T with L = {lorenz_number:.4e} W Ohm/K^2 ({lorenz_source}), "
            f"T = {temperature_K} K"
        ),
        limitations=["Wiedemann-Franz assumes elastic scattering; L is not universal."],
    )


_LEVEL_ORDER = [
    EvidenceLevel.HYPOTHESIS,
    EvidenceLevel.MLIP_PREDICTION,
    EvidenceLevel.LOW_FIDELITY_DFT,
    EvidenceLevel.CONVERGED_DFT,
    EvidenceLevel.EXPERIMENTAL,
]


def combine_total_thermal_conductivity(
    kappa_e: PropertyEvidence, kappa_ph: PropertyEvidence
) -> PropertyEvidence:
    """``kappa = kappa_e + kappa_ph`` for one structure at identical conditions."""
    if kappa_e.property != "thermal_conductivity_electronic":
        raise ValueError("kappa_e must be thermal_conductivity_electronic evidence")
    if kappa_ph.property != "thermal_conductivity_lattice":
        raise ValueError("kappa_ph must be thermal_conductivity_lattice evidence")
    if (kappa_e.structure_id, kappa_e.composition) != (kappa_ph.structure_id, kappa_ph.composition):
        raise ValueError("kappa_e and kappa_ph describe different structures")
    if kappa_e.conditions != kappa_ph.conditions:
        raise ValueError("kappa_e and kappa_ph were evaluated at different conditions")
    ve, vp = _scalar(kappa_e, "kappa_e"), _scalar(kappa_ph, "kappa_ph")
    ue, up = float(kappa_e.uncertainty), float(kappa_ph.uncertainty)  # type: ignore[arg-type]
    parts = (kappa_e, kappa_ph)
    approximated = any(p.status == PropertyStatus.APPROXIMATED for p in parts)
    level = min((p.evidence_level for p in parts), key=_LEVEL_ORDER.index)
    surrogate = any(p.status == PropertyStatus.SURROGATE_PREDICTED for p in parts)
    approximation = None
    if approximated:
        status = PropertyStatus.APPROXIMATED
        approximation = "; ".join(p.approximation for p in parts if p.approximation)
    elif surrogate:
        status = PropertyStatus.SURROGATE_PREDICTED
    elif all(p.status == PropertyStatus.CALCULATED for p in parts):
        status = PropertyStatus.CALCULATED
    else:
        status = PropertyStatus.APPROXIMATED
        approximation = "sum of a measured and a calculated component"
    if status == PropertyStatus.SURROGATE_PREDICTED and at_least(
        level, EvidenceLevel.LOW_FIDELITY_DFT
    ):
        level = EvidenceLevel.MLIP_PREDICTION
    ids = sorted({*kappa_e.evidence_ids, *kappa_ph.evidence_ids})
    return PropertyEvidence(
        structure_id=kappa_e.structure_id,
        composition=kappa_e.composition,
        chemical_ordering=kappa_e.chemical_ordering,
        property="thermal_conductivity_total",
        status=status,
        value=ve + vp,
        unit="W/(m K)",
        conditions=kappa_e.conditions,
        method="kappa_e + kappa_ph",
        evidence_ids=ids,
        evidence_level=level,
        converged=all(p.converged is True for p in parts)
        if status == PropertyStatus.CALCULATED
        else None,
        uncertainty=math.hypot(ue, up),
        derived_from=["thermal_conductivity_electronic", "thermal_conductivity_lattice"],
        approximation=approximation,
        limitations=sorted({*kappa_e.limitations, *kappa_ph.limitations}),
    )


def sigma_over_tau_is_not_sigma() -> str:
    """The explicit refusal used where an absolute conductivity is requested from sigma/tau."""
    return (
        "sigma/tau is reported as sigma/tau; an absolute conductivity needs an evidenced "
        "relaxation time or a scattering calculation"
    )


__all__ = [
    "BACKENDS",
    "INGEST_FORMAT",
    "SOMMERFELD_LORENZ",
    "TransportBackend",
    "TransportReadiness",
    "TransportRecord",
    "TransportStatus",
    "check_backend",
    "combine_total_thermal_conductivity",
    "ingest_transport_record",
    "sigma_over_tau_is_not_sigma",
    "wiedemann_franz",
]
