"""Material-property evidence: one validated record per property of a candidate.

A :class:`PropertyEvidence` carries a thermal, electrical, mechanical, or future
structural property with its units, operating conditions, method, provenance,
convergence, uncertainty, and qualification. Each :class:`PropertyStatus` is a
distinct state: a missing, unsupported, or unconverged property has no value
and is never replaced by zero or by a related quantity. Elastic stiffness never
stands in for strength, toughness, creep, ductility, swelling, or
embrittlement, and a surrogate prediction or documented approximation is never
qualified. Records map onto the append-only evidence ledger
(:func:`to_ledger_record`).
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from matsim_agents.execution.contracts import (
    DFT_EVIDENCE,
    PROPERTY_UNITS,
    EvidenceLevel,
    at_least,
)
from matsim_agents.execution.ledger import (
    EvidenceOutcome,
    EvidenceRecord,
    LedgerQuantity,
    MethodRecord,
)

PropertyFamily = Literal[
    "thermal",
    "electrical",
    "mechanical",
    "structural_performance",
    "plasma_interaction",
    "phase_stability",
    "nuclear",
]

PROPERTY_FAMILY: dict[str, PropertyFamily] = {
    "thermal_conductivity_total": "thermal",
    "thermal_conductivity_electronic": "thermal",
    "thermal_conductivity_lattice": "thermal",
    "thermal_conductivity_tensor": "thermal",
    "thermal_conductivity_degradation": "thermal",
    "electrical_conductivity": "electrical",
    "electrical_conductivity_tensor": "electrical",
    "electrical_resistivity": "electrical",
    "electrical_conductivity_over_relaxation_time": "electrical",
    "electronic_scattering_rate": "electrical",
    "elastic_stiffness_tensor": "mechanical",
    "bulk_modulus": "mechanical",
    "shear_modulus": "mechanical",
    "youngs_modulus": "mechanical",
    "poissons_ratio": "mechanical",
    "elastic_anisotropy": "mechanical",
    "elastic_stability_min_eigenvalue": "mechanical",
    "yield_strength": "structural_performance",
    "fracture_toughness": "structural_performance",
    "creep_rate": "structural_performance",
    "ductility_elongation": "structural_performance",
    "irradiation_swelling": "structural_performance",
    "embrittlement_dbtt_shift": "structural_performance",
    "defect_recovery_fraction": "structural_performance",
    "thermal_shock_damage_threshold": "structural_performance",
    "thermal_fatigue_cycles_to_failure": "structural_performance",
    "thermal_expansion_coefficient": "thermal",
    "sputtering_yield": "plasma_interaction",
    "hydrogen_retention_fraction_H": "plasma_interaction",
    "hydrogen_retention_fraction_D": "plasma_interaction",
    "hydrogen_retention_fraction_T": "plasma_interaction",
    "helium_retention_fraction": "plasma_interaction",
    "max_phase_stable_temperature": "phase_stability",
    "recrystallization_temperature": "phase_stability",
    "shutdown_dose_rate": "nuclear",
    "specific_activity": "nuclear",
    "decay_heat": "nuclear",
}
assert set(PROPERTY_FAMILY) == set(PROPERTY_UNITS)

PROPERTY_SHAPE: dict[str, tuple[int, int]] = {
    "thermal_conductivity_tensor": (3, 3),
    "electrical_conductivity_tensor": (3, 3),
    "elastic_stiffness_tensor": (6, 6),
}
# Beyond small-strain elasticity: needs its own model or measurement.
FUTURE_PROPERTIES = frozenset(
    n for n, f in PROPERTY_FAMILY.items() if f == "structural_performance"
)
ELASTIC_PROPERTIES = frozenset(n for n, f in PROPERTY_FAMILY.items() if f == "mechanical")
# Transport and rate-like properties are meaningless without a temperature.
_NEEDS_TEMPERATURE = (
    frozenset(
        n
        for n, f in PROPERTY_FAMILY.items()
        if f in {"thermal", "electrical", "plasma_interaction"}
    )
    | FUTURE_PROPERTIES
)
# Activation quantities exist only for a named irradiation history.
_NEEDS_IRRADIATION = frozenset(n for n, f in PROPERTY_FAMILY.items() if f == "nuclear")

# multiply a value in the key unit to get the canonical unit
_UNIT_FACTORS: dict[str, dict[str, float]] = {
    "W/(m K)": {"W/(m K)": 1.0, "W/m/K": 1.0, "mW/(cm K)": 0.1, "W/(cm K)": 100.0},
    "S/m": {"S/m": 1.0, "MS/m": 1e6, "S/cm": 100.0, "%IACS": 5.8e5 * 0.01},
    "ohm m": {"ohm m": 1.0, "uohm cm": 1e-8, "nohm m": 1e-9},
    "S/(m s)": {"S/(m s)": 1.0},
    "1/s": {"1/s": 1.0, "1/fs": 1e15, "1/ps": 1e12},
    "GPa": {"GPa": 1.0, "MPa": 1e-3, "Pa": 1e-9, "kbar": 0.1, "Mbar": 100.0},
    "MPa": {"MPa": 1.0, "GPa": 1e3, "Pa": 1e-6},
    "MPa m^0.5": {"MPa m^0.5": 1.0, "MPa sqrt(m)": 1.0},
    "1": {"1": 1.0, "%": 0.01},
    "K": {"K": 1.0},
    "1/K": {"1/K": 1.0, "1e-6/K": 1e-6},
    "MW/m^2": {"MW/m^2": 1.0, "GW/m^2": 1e3, "W/m^2": 1e-6},
    "atoms/ion": {"atoms/ion": 1.0},
    "Sv/h": {"Sv/h": 1.0, "mSv/h": 1e-3, "uSv/h": 1e-6},
    "Bq/kg": {"Bq/kg": 1.0, "Bq/g": 1e3, "TBq/kg": 1e12},
    "W/kg": {"W/kg": 1.0, "mW/kg": 1e-3, "W/g": 1e3},
}


def convert_to_canonical(property_name: str, value: float, unit: str) -> float:
    """Convert ``value`` in ``unit`` to the property's canonical unit; unknown units are errors."""
    canonical = PROPERTY_UNITS[property_name]
    factor = _UNIT_FACTORS.get(canonical, {}).get(unit)
    if factor is None:
        raise ValueError(f"cannot convert {unit!r} to {canonical!r} for {property_name}")
    return value * factor


class PropertyStatus(StrEnum):
    """Provenance of a property value; each is a different evidential state."""

    MEASURED = "measured"
    CALCULATED = "calculated"
    SURROGATE_PREDICTED = "surrogate_predicted"
    APPROXIMATED = "approximated"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    UNCONVERGED = "unconverged"


NO_VALUE_STATUSES = frozenset({PropertyStatus.MISSING, PropertyStatus.UNSUPPORTED})
VALUE_STATUSES = frozenset(
    {
        PropertyStatus.MEASURED,
        PropertyStatus.CALCULATED,
        PropertyStatus.SURROGATE_PREDICTED,
        PropertyStatus.APPROXIMATED,
    }
)


class PropertyQualification(StrEnum):
    UNVALIDATED = "unvalidated"
    NUMERICALLY_VALIDATED = "numerically_validated"
    SCIENTIFICALLY_QUALIFIED = "scientifically_qualified"


class PropertyConditions(BaseModel):
    """Operating state under which a property holds."""

    temperature_K: float | None = Field(None, ge=0)
    pressure_GPa: float | None = None
    loading: str | None = Field(None, description="Strain/stress state or loading protocol.")
    irradiation: str | None = Field(None, description="Dose/damage state, if irradiated.")
    magnetic_state: str | None = None
    defect_state: str | None = None
    transmutation_state: str | None = None


Scalar = float
Matrix = list[list[float]]


def _finite(value: Scalar | Matrix, shape: tuple[int, int] | None, what: str) -> None:
    if shape is None:
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{what} must be one finite number")
        return
    rows, cols = shape
    if (
        not isinstance(value, list)
        or len(value) != rows
        or any(not isinstance(r, list) or len(r) != cols for r in value)
    ):
        raise ValueError(f"{what} must be a {rows}x{cols} matrix")
    if not all(math.isfinite(x) for row in value for x in row):
        raise ValueError(f"{what} must be finite")


class PropertyEvidence(BaseModel):
    """A property of one candidate structure, with everything needed to judge it."""

    schema_version: Literal[1] = 1
    structure_id: str = Field(min_length=1)
    composition: str = Field(min_length=1)
    chemical_ordering: str | None = Field(None, description="Ordering label or description.")
    sro: dict[str, float] | None = Field(None, description="Warren-Cowley parameters by pair.")
    parent_structure: str | None = None
    property: str
    status: PropertyStatus
    value: Scalar | Matrix | None = None
    unit: str
    conditions: PropertyConditions = Field(default_factory=PropertyConditions)
    method: str | None = Field(None, description="Numerical or experimental method.")
    backend: str | None = None
    backend_version: str | None = None
    input_sha256: dict[str, str] = Field(default_factory=dict)
    output_sha256: dict[str, str] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_level: EvidenceLevel = EvidenceLevel.HYPOTHESIS
    converged: bool | None = None
    convergence: dict[str, Any] = Field(default_factory=dict)
    uncertainty: Scalar | Matrix | None = Field(
        None, description="One standard deviation in the property's unit."
    )
    qualification: PropertyQualification = PropertyQualification.UNVALIDATED
    qualification_basis: list[str] = Field(default_factory=list)
    derived_from: list[str] = Field(
        default_factory=list, description="Other properties this value was computed from."
    )
    approximation: str | None = Field(None, description="Documented approximation, if any.")
    reason: str | None = Field(None, description="Why a property is missing or not usable.")
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid(self) -> PropertyEvidence:
        if self.property not in PROPERTY_UNITS:
            raise ValueError(f"unknown property {self.property!r}")
        expected = PROPERTY_UNITS[self.property]
        if self.unit != expected:
            raise ValueError(
                f"property {self.property!r} is reported in {expected!r}, not {self.unit!r}"
            )
        shape = PROPERTY_SHAPE.get(self.property)
        t = self.conditions.temperature_K
        if self.property in _NEEDS_TEMPERATURE and not (t is not None and t > 0):
            raise ValueError(f"{self.property} needs conditions.temperature_K > 0")
        if self.property == "thermal_conductivity_degradation" and not (
            self.conditions.irradiation or self.conditions.defect_state
        ):
            raise ValueError("a conductivity degradation needs an irradiation or defect state")
        if self.property in _NEEDS_IRRADIATION and not self.conditions.irradiation:
            raise ValueError(f"{self.property} needs conditions.irradiation (scenario and cooling)")
        if self.status in NO_VALUE_STATUSES:
            if self.value is not None or self.uncertainty is not None:
                raise ValueError(f"a {self.status.value} property has no value or uncertainty")
            if not self.reason:
                raise ValueError(f"a {self.status.value} property must say what is missing")
            if self.qualification != PropertyQualification.UNVALIDATED:
                raise ValueError(f"a {self.status.value} property cannot be qualified")
            return self
        if self.status == PropertyStatus.UNCONVERGED:
            if not self.reason:
                raise ValueError("an unconverged property needs a reason")
            if self.converged:
                raise ValueError("an unconverged property cannot be marked converged")
            if self.qualification != PropertyQualification.UNVALIDATED:
                raise ValueError("an unconverged property cannot be qualified")
        if self.value is None:
            if self.status != PropertyStatus.UNCONVERGED:
                raise ValueError(f"a {self.status.value} property needs a value")
            return self
        _finite(self.value, shape, f"{self.property} value")
        if self.status != PropertyStatus.UNCONVERGED:
            if self.uncertainty is None:
                raise ValueError(f"a {self.status.value} property needs an uncertainty")
            if not self.evidence_ids:
                raise ValueError(f"a {self.status.value} property must cite evidence_ids")
        if self.uncertainty is not None:
            _finite(self.uncertainty, shape, f"{self.property} uncertainty")
            flat = (
                [self.uncertainty]
                if shape is None
                else [x for row in self.uncertainty for x in row]  # type: ignore[union-attr]
            )
            if any(u < 0 for u in flat):
                raise ValueError("uncertainty cannot be negative")
        self._check_status_rules()
        return self

    def _check_status_rules(self) -> None:
        status, level = self.status, self.evidence_level
        if status == PropertyStatus.MEASURED and level != EvidenceLevel.EXPERIMENTAL:
            raise ValueError("a measured property has experimental evidence")
        if status == PropertyStatus.CALCULATED:
            if level not in DFT_EVIDENCE:
                raise ValueError(
                    "a calculated property needs DFT-level evidence; a surrogate "
                    "or MLIP estimate is surrogate_predicted"
                )
            if self.converged is not True:
                raise ValueError(
                    "a calculated property needs converged=True; otherwise it is unconverged"
                )
        if status == PropertyStatus.SURROGATE_PREDICTED and at_least(
            level, EvidenceLevel.LOW_FIDELITY_DFT
        ):
            raise ValueError("a surrogate prediction cannot carry DFT or experimental evidence")
        if status == PropertyStatus.APPROXIMATED and not self.approximation:
            raise ValueError("an approximated property must document its approximation")
        if self.property in FUTURE_PROPERTIES and ELASTIC_PROPERTIES & set(self.derived_from):
            raise ValueError(f"{self.property} cannot be derived from elastic properties")
        q = self.qualification
        if q == PropertyQualification.UNVALIDATED:
            return
        if status not in {PropertyStatus.MEASURED, PropertyStatus.CALCULATED}:
            raise ValueError(f"a {status.value} property cannot be validated or qualified")
        if self.converged is not True and status == PropertyStatus.CALCULATED:
            raise ValueError("validation needs converged numerical output")
        if q == PropertyQualification.SCIENTIFICALLY_QUALIFIED and not self.qualification_basis:
            raise ValueError("scientific qualification needs a qualification_basis")
        if (
            q == PropertyQualification.SCIENTIFICALLY_QUALIFIED
            and status == (PropertyStatus.CALCULATED)
            and not at_least(self.evidence_level, EvidenceLevel.CONVERGED_DFT)
        ):
            raise ValueError("scientific qualification needs converged DFT or better")

    def property_id(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return f"pe-{hashlib.sha256(payload.encode()).hexdigest()[:24]}"

    @property
    def family(self) -> PropertyFamily:
        return PROPERTY_FAMILY[self.property]

    def is_scoreable(self) -> bool:
        """A value exists and is converged or documented; unconverged values are not scored."""
        return self.status in VALUE_STATUSES and self.value is not None


def to_ledger_record(
    evidence: PropertyEvidence, *, campaign_id: str | None = None
) -> EvidenceRecord:
    """The append-only ledger record of a property; tensors are stored as JSON text."""
    values: list[LedgerQuantity] = []
    uncertainties: dict[str, float] = {}
    if evidence.value is not None:
        stored = (
            evidence.value
            if isinstance(evidence.value, float | int)
            else json.dumps(evidence.value)
        )
        values.append(
            LedgerQuantity(
                name=evidence.property,
                value=stored,
                unit=evidence.unit,
                structure=evidence.structure_id,
            )
        )
    if evidence.uncertainty is not None:
        u = evidence.uncertainty
        uncertainties[evidence.property] = (
            float(u) if isinstance(u, float | int) else max(x for row in u for x in row)
        )
    outcome = {
        PropertyStatus.UNCONVERGED: EvidenceOutcome.NOT_CONVERGED,
        PropertyStatus.MISSING: EvidenceOutcome.FAILED_TO_COMPUTE,
        PropertyStatus.UNSUPPORTED: EvidenceOutcome.FAILED_TO_COMPUTE,
    }.get(evidence.status, EvidenceOutcome.MEASURED)
    assumptions = [
        *evidence.limitations,
        *([f"approximation: {evidence.approximation}"] if evidence.approximation else []),
        f"status: {evidence.status.value}",
        f"conditions: {evidence.conditions.model_dump_json(exclude_none=True)}",
    ]
    return EvidenceRecord(
        kind="property",
        subject=evidence.composition,
        campaign_id=campaign_id,
        method=MethodRecord(
            backend=evidence.backend or "unspecified",
            backend_version=evidence.backend_version,
            method_signature=evidence.method,
            settings={"structure_id": evidence.structure_id},
        ),
        evidence_level=evidence.evidence_level,
        outcome=outcome,
        values=values,
        numerical_convergence={"converged": evidence.converged, **evidence.convergence},
        input_sha256=evidence.input_sha256,
        output_artifact_sha256=evidence.output_sha256,
        physical_assumptions=assumptions,
        uncertainties=uncertainties,
        failure_reason=evidence.reason if outcome != EvidenceOutcome.MEASURED else None,
        parent_record_ids=evidence.evidence_ids,
    )


__all__ = [
    "ELASTIC_PROPERTIES",
    "FUTURE_PROPERTIES",
    "PROPERTY_FAMILY",
    "PROPERTY_SHAPE",
    "PropertyConditions",
    "PropertyEvidence",
    "PropertyFamily",
    "PropertyQualification",
    "PropertyStatus",
    "convert_to_canonical",
    "to_ledger_record",
]
