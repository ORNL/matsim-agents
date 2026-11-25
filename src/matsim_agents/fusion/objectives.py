"""Objective-to-observable map for plasma-facing-component scoring (F2).

Each of the eight policy objectives names the observables that may support
it, what they mean physically, how they are obtained, and which tempting
proxies are *not* evidence for it. :func:`check_objective_evidence` rejects
proxy evidence before any score is formed; this module never computes a
score, it only says which records may be used for which objective.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.execution.contracts import PROPERTY_UNITS
from matsim_agents.execution.properties import (
    ELASTIC_PROPERTIES,
    PropertyEvidence,
    PropertyQualification,
)
from matsim_agents.fusion.policy import FUSION_PFC_OBJECTIVES

Direction = Literal["maximize", "minimize"]


class ObservableSpec(BaseModel):
    """One observable usable for an objective, with its own direction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    direction: Direction
    meaning: str = Field(min_length=1)

    @model_validator(mode="after")
    def _known(self) -> ObservableSpec:
        if self.name not in PROPERTY_UNITS:
            raise ValueError(f"observable {self.name!r} is not a registered property")
        return self

    @property
    def unit(self) -> str:
        return PROPERTY_UNITS[self.name]


class UnsupportedProxy(BaseModel):
    """A quantity that is not evidence for an objective, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    reason: str = Field(min_length=1)


class ObjectiveDefinition(BaseModel):
    """What counts as evidence for one policy objective."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    physical_meaning: str
    required: tuple[ObservableSpec, ...] = Field(min_length=1)
    optional: tuple[ObservableSpec, ...] = ()
    compatible_conditions: tuple[str, ...] = Field(
        description="Condition fields that must match the evaluation scenario."
    )
    methods: tuple[str, ...]
    evidence_requirement: str
    minimum_qualification: PropertyQualification = PropertyQualification.NUMERICALLY_VALIDATED
    unsupported_proxies: tuple[UnsupportedProxy, ...] = ()
    qualification_note: str
    # Observables that may be used only together, never alone.
    insufficient_alone: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> ObjectiveDefinition:
        names = [o.name for o in (*self.required, *self.optional)]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: observable listed twice")
        proxies = {p.name for p in self.unsupported_proxies}
        if proxies & set(names):
            raise ValueError(f"{self.name}: an observable cannot also be an unsupported proxy")
        return self

    def observables(self) -> tuple[ObservableSpec, ...]:
        return (*self.required, *self.optional)

    def observable(self, name: str) -> ObservableSpec | None:
        return next((o for o in self.observables() if o.name == name), None)


def _o(name: str, direction: Direction, meaning: str) -> ObservableSpec:
    return ObservableSpec(name=name, direction=direction, meaning=meaning)


def _p(name: str, reason: str) -> UnsupportedProxy:
    return UnsupportedProxy(name=name, reason=reason)


_DPA = _p("dpa", "a dpa or PKA rate measures primary damage production, not damage resistance")
_PKA = _p("pka_rate", "a PKA spectrum is a damage source term, not a material response")
_FORM0K = _p(
    "formation_energy_per_atom",
    "a 0 K formation energy says nothing about stability at operating temperature",
)
_EHULL = _p("energy_above_hull", "0 K hull distance is not high-temperature phase stability")
_ELASTIC = tuple(
    _p(n, "elastic moduli are not strength, toughness, or damage tolerance")
    for n in sorted(ELASTIC_PROPERTIES)
)
_INVENTORY = _p(
    "isotope_inventory",
    "an isotope inventory gives no dose without a dose model and geometry",
)

OBJECTIVES: dict[str, ObjectiveDefinition] = {
    "radiation_damage_recovery": ObjectiveDefinition(
        name="radiation_damage_recovery",
        physical_meaning=(
            "Property retention under neutron irradiation after defect evolution and recovery."
        ),
        required=(
            _o(
                "defect_recovery_fraction",
                "maximize",
                "Fraction of initial damage annealed at the operating temperature.",
            ),
        ),
        optional=(
            _o("irradiation_swelling", "minimize", "Volumetric swelling at dose."),
            _o(
                "thermal_conductivity_degradation",
                "minimize",
                "Fractional conductivity loss at dose.",
            ),
            _o("embrittlement_dbtt_shift", "minimize", "DBTT shift at dose."),
        ),
        compatible_conditions=("temperature_K", "irradiation", "neutron_spectrum", "defect_state"),
        methods=("object kinetic Monte Carlo", "cluster dynamics", "post-irradiation test"),
        evidence_requirement="defect-evolution model or irradiation experiment at stated dose",
        unsupported_proxies=(_DPA, _PKA),
        qualification_note=(
            "Requires defect evolution beyond primary damage; primary-damage counts alone "
            "are never accepted."
        ),
    ),
    "thermal_conductivity": ObjectiveDefinition(
        name="thermal_conductivity",
        physical_meaning="Total thermal conductivity at the operating temperature and state.",
        required=(
            _o(
                "thermal_conductivity_total",
                "maximize",
                "Electronic plus lattice conductivity.",
            ),
        ),
        optional=(
            _o(
                "thermal_conductivity_degradation",
                "minimize",
                "Loss of conductivity after irradiation.",
            ),
        ),
        compatible_conditions=(
            "temperature_K",
            "irradiation",
            "transmutation_state",
            "magnetic_state",
        ),
        methods=("Boltzmann transport", "Wiedemann-Franz with lattice term", "laser flash"),
        evidence_requirement="transport calculation or measurement at the scenario temperature",
        qualification_note=(
            "Electronic-only or lattice-only values are approximations and must say so."
        ),
    ),
    "mechanical_integrity": ObjectiveDefinition(
        name="mechanical_integrity",
        physical_meaning="Strength, toughness, and ductility retained at temperature.",
        required=(
            _o("yield_strength", "maximize", "Yield stress at temperature."),
            _o("fracture_toughness", "maximize", "Resistance to crack growth."),
        ),
        optional=(
            _o("ductility_elongation", "maximize", "Uniform elongation."),
            _o("embrittlement_dbtt_shift", "minimize", "Irradiation-induced DBTT shift."),
            _o("creep_rate", "minimize", "Steady-state creep rate."),
        ),
        compatible_conditions=("temperature_K", "irradiation", "loading"),
        methods=("mechanical test", "crystal plasticity", "fracture mechanics model"),
        evidence_requirement="strength and toughness, not stiffness",
        unsupported_proxies=_ELASTIC,
        qualification_note=(
            "Elastic moduli may appear only in a separately labeled mechanical screening."
        ),
    ),
    "thermal_shock_fatigue": ObjectiveDefinition(
        name="thermal_shock_fatigue",
        physical_meaning="Resistance to cracking under transient and cyclic heat loads.",
        required=(
            _o(
                "thermal_shock_damage_threshold",
                "maximize",
                "Absorbed power density at onset of cracking.",
            ),
        ),
        optional=(
            _o(
                "thermal_fatigue_cycles_to_failure",
                "maximize",
                "Cycles to failure at stated load.",
            ),
            _o("thermal_conductivity_total", "maximize", "Heat removal."),
            _o("thermal_expansion_coefficient", "minimize", "Thermal strain per kelvin."),
            _o("yield_strength", "maximize", "Stress accommodated before yielding."),
            _o("fracture_toughness", "maximize", "Crack resistance."),
        ),
        compatible_conditions=("temperature_K", "loading"),
        methods=("electron-beam or laser heat-load test", "thermomechanical FEM with damage"),
        evidence_requirement="heat-load test or coupled thermomechanical damage model",
        unsupported_proxies=_ELASTIC,
        insufficient_alone=("thermal_conductivity_total",),
        qualification_note="Thermal conductivity alone does not predict thermal shock.",
    ),
    "plasma_erosion_resistance": ObjectiveDefinition(
        name="plasma_erosion_resistance",
        physical_meaning="Low physical and chemical sputtering under plasma ion impact.",
        required=(
            _o(
                "sputtering_yield",
                "minimize",
                "Eroded atoms per incident ion at stated ion species and energy.",
            ),
        ),
        compatible_conditions=("temperature_K", "plasma_ion", "plasma_ion_energy_eV"),
        methods=("binary-collision approximation", "MD cascade", "ion-beam or plasma exposure"),
        evidence_requirement="sputtering yield at the scenario plasma ion and energy",
        unsupported_proxies=(
            _p("pka_rate", "neutron PKA spectra are bulk recoils, not surface sputtering"),
            _DPA,
        ),
        qualification_note="Yield depends on ion species, energy, and angle; all must match.",
    ),
    "high_temperature_stability": ObjectiveDefinition(
        name="high_temperature_stability",
        physical_meaning="Phase and microstructure stability at the operating temperature.",
        required=(
            _o(
                "max_phase_stable_temperature",
                "maximize",
                "Highest temperature without phase change.",
            ),
        ),
        optional=(
            _o(
                "recrystallization_temperature",
                "maximize",
                "Onset of recrystallization.",
            ),
        ),
        compatible_conditions=("irradiation", "transmutation_state", "magnetic_state"),
        methods=("free-energy model with entropy", "CALPHAD", "annealing experiment"),
        evidence_requirement="finite-temperature free energy or annealing data",
        unsupported_proxies=(_FORM0K, _EHULL),
        qualification_note="Requires finite-temperature free energies, not 0 K energetics.",
    ),
    "low_hydrogen_retention": ObjectiveDefinition(
        name="low_hydrogen_retention",
        physical_meaning="Low retained H, D, and T inventory; isotopes reported separately.",
        required=(
            _o(
                "hydrogen_retention_fraction_T",
                "minimize",
                "Retained fraction of implanted tritium.",
            ),
        ),
        optional=(
            _o("hydrogen_retention_fraction_D", "minimize", "Retained deuterium fraction."),
            _o("hydrogen_retention_fraction_H", "minimize", "Retained protium fraction."),
        ),
        compatible_conditions=(
            "temperature_K",
            "irradiation",
            "defect_state",
            "plasma_ion_energy_eV",
        ),
        methods=("rate-equation trapping model", "thermal desorption spectroscopy"),
        evidence_requirement="isotope-specific retention at stated fluence and temperature",
        unsupported_proxies=(
            _p(
                "helium_retention_fraction",
                "helium is not a hydrogen isotope",
            ),
        ),
        qualification_note=(
            "H, D, and T differ in mass and trapping; one isotope never stands for another."
        ),
    ),
    "low_activation_shutdown_dose": ObjectiveDefinition(
        name="low_activation_shutdown_dose",
        physical_meaning="Low shutdown dose rate after a stated irradiation and cooling time.",
        required=(
            _o(
                "shutdown_dose_rate",
                "minimize",
                "Contact dose rate after the scenario cooling time.",
            ),
        ),
        optional=(
            _o("specific_activity", "minimize", "Activity per mass after cooling."),
            _o("decay_heat", "minimize", "Decay heat per mass after cooling."),
        ),
        compatible_conditions=("irradiation", "neutron_spectrum", "cooling_time_s"),
        methods=("inventory solver with dose model (e.g. FISPACT-II)",),
        evidence_requirement="activation inventory plus dose model at stated cooling time",
        unsupported_proxies=(
            _INVENTORY,
            _p("spectra_pka", "SPECTRA-PKA computes recoils, not activation"),
            _DPA,
        ),
        qualification_note=(
            "Dose needs an inventory solver and a dose model; neither alone is enough."
        ),
    ),
}
assert tuple(OBJECTIVES) == FUSION_PFC_OBJECTIVES


class ObjectiveEvidenceError(ValueError):
    """Evidence was offered for an objective it cannot support."""


def objective(name: str) -> ObjectiveDefinition:
    try:
        return OBJECTIVES[name]
    except KeyError:
        raise ObjectiveEvidenceError(f"unknown objective {name!r}") from None


def check_objective_evidence(name: str, evidence: list[PropertyEvidence]) -> list[str]:
    """Reject proxies and foreign observables; return problems with the evidence set.

    A non-empty return means the set cannot support a value for the objective.
    """
    spec = objective(name)
    proxies = {p.name: p.reason for p in spec.unsupported_proxies}
    problems: list[str] = []
    allowed = {o.name for o in spec.observables()}
    for ev in evidence:
        if ev.property in proxies:
            problems.append(f"{ev.property}: {proxies[ev.property]}")
        elif ev.property not in allowed:
            problems.append(f"{ev.property} is not an observable of {name}")
        for src in ev.derived_from:
            if src in proxies:
                problems.append(f"{ev.property} derived from {src}: {proxies[src]}")
    present = {ev.property for ev in evidence if ev.is_scoreable()}
    missing = [o.name for o in spec.required if o.name not in present]
    if missing:
        problems.append(f"missing required observable(s): {', '.join(missing)}")
    if present and present <= set(spec.insufficient_alone):
        problems.append(f"{', '.join(sorted(present))} alone is insufficient for {name}")
    return problems


def mechanical_screening(evidence: list[PropertyEvidence]) -> dict[str, float]:
    """Elastic values relabeled as a screening; never an integrity score."""
    return {
        f"mechanical_screening:{ev.property}": float(ev.value)  # type: ignore[arg-type]
        for ev in evidence
        if ev.property in ELASTIC_PROPERTIES
        and ev.is_scoreable()
        and isinstance(ev.value, int | float)
    }


__all__ = [
    "OBJECTIVES",
    "ObjectiveDefinition",
    "ObjectiveEvidenceError",
    "ObservableSpec",
    "UnsupportedProxy",
    "check_objective_evidence",
    "mechanical_screening",
    "objective",
]
