"""Contract for nuclear results imported from external codes (F3).

Three families, never interchangeable:

* A ``pka_dpa`` - primary recoil production and displacement damage. A dpa
  value carries its convention (NRT, arc-dpa, ...) and is never converted to
  surviving defects here; that needs a cascade model.
* B ``transmutation`` - isotope-resolved inventory changes (e.g. W -> Re, Os;
  H/D/T and He-3/He-4 production).
* C ``activation`` - activity, decay heat and dose after a cooling time.
  SPECTRA-PKA computes recoils and cannot supply family C; a dose rate also
  needs a declared dose model.

This module validates and records imported results. It runs nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.execution.contracts import PROPERTY_UNITS, EvidenceLevel
from matsim_agents.execution.ledger import (
    EvidenceOutcome,
    EvidenceRecord,
    LedgerQuantity,
    MethodRecord,
)
from matsim_agents.execution.properties import (
    PropertyConditions,
    PropertyEvidence,
    PropertyQualification,
    PropertyStatus,
)
from matsim_agents.irradiation.contracts import (
    RATE_BASIS_PER_ATOM_PER_S,
    IrradiationScenario,
    PKASpectrum,
    RecoilSpecies,
)

SCHEMA = "fusion-nuclear-evidence-v1"


class NuclearEvidenceError(ValueError):
    """An imported nuclear result violates the contract."""


class NuclearFamily(StrEnum):
    PKA_DPA = "pka_dpa"
    TRANSMUTATION = "transmutation"
    ACTIVATION = "activation"


DpaConvention = Literal["NRT", "arc-dpa", "damage_energy_only"]

FAMILY_QUANTITIES: dict[NuclearFamily, dict[str, str]] = {
    NuclearFamily.PKA_DPA: {
        "dpa": "dpa",
        "dpa_rate": "dpa/s",
        "pka_rate": "1/(atom s)",
        "damage_energy_rate": "eV/(atom s)",
    },
    NuclearFamily.TRANSMUTATION: {
        "atom_fraction": "1",
        "production_rate": "appm/s",
        "concentration_appm": "appm",
    },
    NuclearFamily.ACTIVATION: {
        "specific_activity": "Bq/kg",
        "decay_heat": "W/kg",
        "shutdown_dose_rate": "Sv/h",
    },
}
# Defect survival is a cascade-model output, never a nuclear-data output.
_DEFECT_NAMES = frozenset(
    {"surviving_defects", "frenkel_pairs", "vacancy_concentration", "interstitials"}
)
# Codes that compute recoils only.
_RECOIL_ONLY_CODES = frozenset({"spectra-pka", "spectra_pka", "spectrapka"})
_HYDROGEN_ISOTOPES = {1: "H", 2: "D", 3: "T"}


class NuclearQuantity(BaseModel):
    """One reported number with its unit and one-sigma uncertainty."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float
    unit: str
    uncertainty: float | None = Field(None, ge=0, description="One sigma; None = not reported.")
    isotope: RecoilSpecies | None = None

    @model_validator(mode="after")
    def _finite(self) -> NuclearQuantity:
        if not math.isfinite(self.value):
            raise ValueError(f"{self.name} must be finite")
        if self.uncertainty is not None and not math.isfinite(self.uncertainty):
            raise ValueError(f"{self.name} uncertainty must be finite")
        return self

    @property
    def label(self) -> str:
        return self.name if self.isotope is None else f"{self.name}[{self.isotope.key}]"


class ExternalNuclearEvidence(BaseModel):
    """A validated, hash-tied result from an external nuclear code."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["fusion-nuclear-evidence-v1"] = SCHEMA
    family: NuclearFamily
    external_application: str = Field(min_length=1)
    application_version: str = Field(min_length=1)
    nuclear_data_library: str = Field(min_length=1)
    input_sha256: dict[str, str] = Field(min_length=1)
    output_sha256: dict[str, str] = Field(min_length=1)
    material_id: str = Field(min_length=1)
    isotope_inventory: dict[str, float] = Field(
        default_factory=dict, description="Isotope key (e.g. 'Re187') -> atom fraction."
    )
    irradiation_scenario_id: str = Field(min_length=1)
    exposure_time_s: float | None = Field(None, gt=0)
    cooling_time_s: float | None = Field(None, ge=0)
    quantities: tuple[NuclearQuantity, ...] = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    qualification_status: PropertyQualification = PropertyQualification.UNVALIDATED
    qualification_basis: tuple[str, ...] = ()
    dpa_convention: DpaConvention | None = None
    dose_model: str | None = Field(None, description="Dose model and geometry for dose rates.")
    synthetic: bool = Field(False, description="True for demonstration data only.")
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _contract(self) -> ExternalNuclearEvidence:
        for name, digest in {**self.input_sha256, **self.output_sha256}.items():
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{name}: sha256 must be 64 lowercase hex characters")
        allowed = FAMILY_QUANTITIES[self.family]
        for q in self.quantities:
            if q.name in _DEFECT_NAMES:
                raise ValueError(
                    f"{q.name}: surviving defects need a cascade model; "
                    "dpa is never converted to defects"
                )
            if q.name not in allowed:
                raise ValueError(f"{q.name} is not a {self.family.value} quantity")
            if q.unit != allowed[q.name]:
                raise ValueError(f"{q.name} is reported in {allowed[q.name]!r}, not {q.unit!r}")
        for key, frac in self.isotope_inventory.items():
            if not math.isfinite(frac) or frac < 0:
                raise ValueError(f"isotope fraction of {key} must be finite and >= 0")
        getattr(self, f"_check_{self.family.value}")()
        if self.qualification_status == PropertyQualification.SCIENTIFICALLY_QUALIFIED and (
            not self.qualification_basis or self.synthetic
        ):
            raise ValueError("scientific qualification needs a basis and non-synthetic data")
        return self

    def _check_pka_dpa(self) -> None:
        if any(q.name.startswith("dpa") for q in self.quantities) and not self.dpa_convention:
            raise ValueError("a dpa value must state its dpa_convention")

    def _check_transmutation(self) -> None:
        if self.exposure_time_s is None:
            raise ValueError("transmutation needs exposure_time_s")
        for q in self.quantities:
            if q.isotope is None or q.isotope.mass_number is None:
                raise ValueError(f"{q.name}: transmutation must be isotope-resolved")
        if not self.isotope_inventory:
            raise ValueError("transmutation needs an isotope_inventory")

    def _check_activation(self) -> None:
        if self.external_application.strip().lower() in _RECOIL_ONLY_CODES:
            raise ValueError("SPECTRA-PKA computes recoils, not activation")
        if self.exposure_time_s is None or self.cooling_time_s is None:
            raise ValueError("activation needs exposure_time_s and cooling_time_s")
        if any(q.name == "shutdown_dose_rate" for q in self.quantities) and not self.dose_model:
            raise ValueError("a shutdown dose rate needs a declared dose_model")

    def evidence_id(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return f"nuc-{hashlib.sha256(payload.encode()).hexdigest()[:24]}"

    def condition_label(self) -> str:
        parts = [self.irradiation_scenario_id]
        if self.exposure_time_s is not None:
            parts.append(f"exposure {self.exposure_time_s:g} s")
        if self.cooling_time_s is not None:
            parts.append(f"cooling {self.cooling_time_s:g} s")
        return "; ".join(parts)

    def hydrogen_isotopes(self) -> dict[str, float]:
        """H, D, T production kept apart; helium is never included."""
        return {
            _HYDROGEN_ISOTOPES[q.isotope.mass_number]: q.value
            for q in self.quantities
            if q.isotope
            and q.isotope.element == "H"
            and q.isotope.mass_number in _HYDROGEN_ISOTOPES
        }

    def helium_isotopes(self) -> dict[str, float]:
        return {
            f"He-{q.isotope.mass_number}": q.value
            for q in self.quantities
            if q.isotope and q.isotope.element == "He" and q.isotope.mass_number in {3, 4}
        }

    def check_scenario(self, scenario: IrradiationScenario) -> list[str]:
        problems = []
        if scenario.scenario_id != self.irradiation_scenario_id:
            problems.append(
                f"scenario {self.irradiation_scenario_id!r} != {scenario.scenario_id!r}"
            )
        if scenario.parent_alloy_id != self.material_id:
            problems.append(f"material {self.material_id!r} != {scenario.parent_alloy_id!r}")
        return problems


def to_ledger_record(
    evidence: ExternalNuclearEvidence, *, campaign_id: str | None = None
) -> EvidenceRecord:
    """Append-only ledger form (kind ``external``)."""
    assumptions = [
        f"family: {evidence.family.value}",
        f"nuclear_data_library: {evidence.nuclear_data_library}",
        f"conditions: {evidence.condition_label()}",
        f"qualification: {evidence.qualification_status.value}",
        *([f"dpa_convention: {evidence.dpa_convention}"] if evidence.dpa_convention else []),
        *([f"dose_model: {evidence.dose_model}"] if evidence.dose_model else []),
        *(["SYNTHETIC demonstration data"] if evidence.synthetic else []),
        *evidence.notes,
    ]
    return EvidenceRecord(
        kind="external",
        subject=evidence.material_id,
        campaign_id=campaign_id,
        method=MethodRecord(
            backend=evidence.external_application,
            backend_version=evidence.application_version,
            method_signature=f"{SCHEMA}:{evidence.family.value}",
            settings={
                "nuclear_data_library": evidence.nuclear_data_library,
                "irradiation_scenario_id": evidence.irradiation_scenario_id,
                "isotope_inventory": evidence.isotope_inventory,
            },
        ),
        evidence_level=EvidenceLevel.HYPOTHESIS,
        outcome=EvidenceOutcome.MEASURED,
        values=[
            LedgerQuantity(name=q.label, value=q.value, unit=q.unit, structure=evidence.material_id)
            for q in evidence.quantities
        ],
        input_sha256=evidence.input_sha256,
        output_artifact_sha256=evidence.output_sha256,
        physical_assumptions=assumptions,
        uncertainties={q.label: q.uncertainty for q in evidence.quantities if q.uncertainty},
        parent_record_ids=list(evidence.evidence_ids),
    )


def activation_properties(
    evidence: ExternalNuclearEvidence, *, composition: str, evidence_id: str
) -> list[PropertyEvidence]:
    """Activation quantities as property records for scoring.

    Inventory-code results are not DFT, so they enter as documented
    approximations and are never scientifically qualified here. A quantity
    without an uncertainty is reported, not scored.
    """
    if evidence.family != NuclearFamily.ACTIVATION:
        raise NuclearEvidenceError("only activation evidence maps to activation properties")
    out = []
    for q in evidence.quantities:
        if q.name not in PROPERTY_UNITS:
            continue
        common = dict(
            structure_id=evidence.material_id,
            composition=composition,
            property=q.name,
            unit=q.unit,
            conditions=PropertyConditions(irradiation=evidence.condition_label()),
            method=f"{evidence.external_application} ({evidence.nuclear_data_library})",
            backend=evidence.external_application,
            backend_version=evidence.application_version,
            input_sha256=evidence.input_sha256,
            output_sha256=evidence.output_sha256,
            limitations=["SYNTHETIC demonstration data"] if evidence.synthetic else [],
        )
        if q.uncertainty is None:
            out.append(
                PropertyEvidence(
                    **common,
                    status=PropertyStatus.MISSING,
                    reason=f"{q.name} reported without uncertainty; not scoreable",
                )
            )
            continue
        out.append(
            PropertyEvidence(
                **common,
                status=PropertyStatus.APPROXIMATED,
                value=q.value,
                uncertainty=q.uncertainty,
                evidence_ids=[evidence_id],
                approximation=(
                    "external inventory calculation"
                    + (f" with dose model {evidence.dose_model}" if evidence.dose_model else "")
                ),
            )
        )
    return out


def from_pka_spectrum(
    spectrum: PKASpectrum,
    *,
    material_id: str,
    application_version: str,
    evidence_ids: tuple[str, ...] = (),
) -> ExternalNuclearEvidence:
    """Family-A evidence from a parsed PKA spectrum (e.g. SPECTRA-PKA)."""
    if spectrum.rate_basis != RATE_BASIS_PER_ATOM_PER_S:
        raise NuclearEvidenceError(
            f"PKA rate basis {spectrum.rate_basis!r} is not per target atom per second"
        )
    prov = spectrum.provenance
    if not prov.input_sha256 or not prov.output_sha256:
        raise NuclearEvidenceError("PKA provenance lacks input/output hashes")
    return ExternalNuclearEvidence(
        family=NuclearFamily.PKA_DPA,
        external_application=prov.source,
        application_version=application_version,
        nuclear_data_library=prov.nuclear_data_identity,
        input_sha256=dict(prov.input_sha256),
        output_sha256=dict(prov.output_sha256),
        material_id=material_id,
        irradiation_scenario_id=spectrum.scenario.scenario_id,
        quantities=tuple(
            NuclearQuantity(
                name="pka_rate", value=s.total_rate, unit="1/(atom s)", isotope=s.species
            )
            for s in spectrum.spectra
        ),
        evidence_ids=evidence_ids,
        notes=(f"rate_basis: {spectrum.rate_basis}", "numerical uncertainty not reported"),
    )


__all__ = [
    "FAMILY_QUANTITIES",
    "SCHEMA",
    "ExternalNuclearEvidence",
    "NuclearEvidenceError",
    "NuclearFamily",
    "NuclearQuantity",
    "activation_properties",
    "from_pka_spectrum",
    "to_ledger_record",
]
