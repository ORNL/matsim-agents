import pytest
from pydantic import ValidationError

from matsim_agents.core.contracts import PROPERTY_UNITS, EvidenceLevel
from matsim_agents.fusion.objectives import (
    OBJECTIVES,
    ObjectiveEvidenceError,
    check_objective_evidence,
    mechanical_screening,
    objective,
)
from matsim_agents.fusion.policy import FUSION_PFC_OBJECTIVES
from matsim_agents.science.properties.calculators import PropertyEvidence, PropertyStatus


def _ev(prop, value=1.0, *, derived_from=(), irradiation=None, temperature=1000.0):
    return PropertyEvidence(
        structure_id="s1",
        composition="W",
        property=prop,
        status=PropertyStatus.SURROGATE_PREDICTED,
        value=value,
        uncertainty=0.1,
        unit=PROPERTY_UNITS[prop],
        conditions={"temperature_K": temperature, "irradiation": irradiation},
        evidence_ids=["e1"],
        evidence_level=EvidenceLevel.HYPOTHESIS,
        derived_from=list(derived_from),
    )


def test_every_policy_objective_is_defined_in_order():
    assert tuple(OBJECTIVES) == FUSION_PFC_OBJECTIVES
    for spec in OBJECTIVES.values():
        assert spec.required and spec.physical_meaning and spec.qualification_note
        for obs in spec.observables():
            assert obs.unit == PROPERTY_UNITS[obs.name]


def test_unknown_objective():
    with pytest.raises(ObjectiveEvidenceError):
        objective("beauty")


def test_dpa_and_pka_are_not_radiation_resistance():
    names = {p.name for p in objective("radiation_damage_recovery").unsupported_proxies}
    assert {"dpa", "pka_rate"} <= names
    problems = check_objective_evidence(
        "radiation_damage_recovery",
        [_ev("defect_recovery_fraction", 0.5, derived_from=["dpa"])],
    )
    assert any("dpa" in p for p in problems)


def test_elastic_moduli_are_not_mechanical_integrity():
    problems = check_objective_evidence("mechanical_integrity", [_ev("bulk_modulus", 300.0)])
    assert any("elastic moduli" in p for p in problems)
    assert any("missing required" in p for p in problems)
    assert mechanical_screening([_ev("bulk_modulus", 300.0)]) == {
        "mechanical_screening:bulk_modulus": 300.0
    }


def test_conductivity_alone_is_not_thermal_shock():
    problems = check_objective_evidence(
        "thermal_shock_fatigue", [_ev("thermal_conductivity_total", 150.0)]
    )
    assert any("alone is insufficient" in p for p in problems)


def test_sputtering_rejects_neutron_pka():
    names = {p.name for p in objective("plasma_erosion_resistance").unsupported_proxies}
    assert "pka_rate" in names


def test_zero_k_energy_is_not_phase_stability():
    names = {p.name for p in objective("high_temperature_stability").unsupported_proxies}
    assert {"formation_energy_per_atom", "energy_above_hull"} <= names


def test_hydrogen_isotopes_distinct_and_helium_rejected():
    spec = objective("low_hydrogen_retention")
    obs = {o.name for o in spec.observables()}
    assert {
        "hydrogen_retention_fraction_H",
        "hydrogen_retention_fraction_D",
        "hydrogen_retention_fraction_T",
    } <= obs
    problems = check_objective_evidence(
        "low_hydrogen_retention", [_ev("helium_retention_fraction", 0.1)]
    )
    assert any("not a hydrogen isotope" in p for p in problems)
    # D does not stand for T
    problems = check_objective_evidence(
        "low_hydrogen_retention", [_ev("hydrogen_retention_fraction_D", 0.1)]
    )
    assert any("hydrogen_retention_fraction_T" in p for p in problems)


def test_dose_needs_irradiation_and_rejects_inventory():
    with pytest.raises(ValidationError, match="needs conditions.irradiation"):
        _ev("shutdown_dose_rate", 1e-3)
    names = {p.name for p in objective("low_activation_shutdown_dose").unsupported_proxies}
    assert {"isotope_inventory", "spectra_pka"} <= names
    ok = check_objective_evidence(
        "low_activation_shutdown_dose",
        [_ev("shutdown_dose_rate", 1e-3, irradiation="scenario-1 cool 10y")],
    )
    assert ok == []


def test_complete_evidence_has_no_problems():
    assert (
        check_objective_evidence(
            "mechanical_integrity",
            [_ev("yield_strength", 900.0), _ev("fracture_toughness", 20.0)],
        )
        == []
    )
