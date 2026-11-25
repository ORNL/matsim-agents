import pytest
from pydantic import ValidationError

from matsim_agents.core.contracts import PROPERTY_UNITS
from matsim_agents.execution.properties import PropertyEvidence, PropertyStatus
from matsim_agents.fusion.scenario import (
    EvaluationScenario,
    EvidenceContext,
    FusionEvidence,
    ScenarioSet,
    aggregate,
)


def _item(prop, *, temperature=1000.0, irradiation=None, magnetic=None, **ctx):
    ev = PropertyEvidence(
        structure_id="s",
        composition="W",
        property=prop,
        status=PropertyStatus.SURROGATE_PREDICTED,
        value=1.0,
        uncertainty=0.1,
        unit=PROPERTY_UNITS[prop],
        conditions={
            "temperature_K": temperature,
            "irradiation": irradiation,
            "magnetic_state": magnetic,
        },
        evidence_ids=["e"],
    )
    return FusionEvidence(evidence=ev, context=EvidenceContext(**ctx))


def test_temperature_mismatch_reported():
    sc = EvaluationScenario(scenario_id="a", temperature_K=1200.0)
    problems = sc.mismatches("thermal_conductivity", _item("thermal_conductivity_total"))
    assert any("temperature" in p for p in problems)
    assert (
        sc.mismatches(
            "thermal_conductivity", _item("thermal_conductivity_total", temperature=1200.0)
        )
        == []
    )


def test_irradiation_and_magnetic_state_mismatch():
    sc = EvaluationScenario(
        scenario_id="a", temperature_K=1000.0, irradiation="fw-1dpa", magnetic_state="PM"
    )
    item = _item("thermal_conductivity_total", magnetic="FM")
    problems = sc.mismatches("thermal_conductivity", item)
    assert any("irradiation" in p for p in problems)
    assert any("magnetic_state" in p for p in problems)


def test_plasma_energy_and_ion_mismatch():
    sc = EvaluationScenario(
        scenario_id="a", temperature_K=1000.0, plasma_ion="D", plasma_ion_energy_eV=100.0
    )
    item = _item("sputtering_yield", plasma_ion="D", plasma_ion_energy_eV=50.0)
    problems = sc.mismatches("plasma_erosion_resistance", item)
    assert any("plasma_ion_energy_eV" in p for p in problems)
    ok = _item("sputtering_yield", plasma_ion="D", plasma_ion_energy_eV=100.0)
    assert sc.mismatches("plasma_erosion_resistance", ok) == []


def test_cooling_time_and_spectrum_mismatch():
    sc = EvaluationScenario(
        scenario_id="a",
        temperature_K=1000.0,
        irradiation="fw",
        neutron_spectrum="DEMO-FW",
        cooling_time_s=3.15e8,
    )
    item = _item(
        "shutdown_dose_rate", irradiation="fw", neutron_spectrum="ITER-FW", cooling_time_s=3.15e7
    )
    problems = sc.mismatches("low_activation_shutdown_dose", item)
    assert any("cooling_time_s" in p for p in problems)
    assert any("neutron_spectrum" in p for p in problems)


def test_incidence_angle_mismatch():
    sc = EvaluationScenario(
        scenario_id="a",
        temperature_K=1000.0,
        plasma_ion="D",
        plasma_ion_energy_eV=100.0,
        ion_incidence_angle_deg=0.0,
    )
    grazing = _item(
        "sputtering_yield", plasma_ion="D", plasma_ion_energy_eV=100.0, ion_incidence_angle_deg=80
    )
    problems = sc.mismatches("plasma_erosion_resistance", grazing)
    assert any("ion_incidence_angle_deg" in p for p in problems)


def test_dose_geometry_mismatch_only_for_dose_rate():
    sc = EvaluationScenario(
        scenario_id="a",
        temperature_K=1000.0,
        irradiation="fw",
        neutron_spectrum="S",
        cooling_time_s=1.0,
        dose_geometry="contact",
    )
    ctx = dict(irradiation="fw", neutron_spectrum="S", cooling_time_s=1.0)
    at_1m = _item("shutdown_dose_rate", dose_geometry="1 m", **ctx)
    problems = sc.mismatches("low_activation_shutdown_dose", at_1m)
    assert any("dose_geometry" in p for p in problems)
    assert sc.mismatches("low_activation_shutdown_dose", _item("decay_heat", **ctx)) == []


def test_unrequested_irradiation_is_a_mismatch():
    sc = EvaluationScenario(scenario_id="a", temperature_K=1000.0)
    item = _item("thermal_conductivity_total", irradiation="fw-1dpa")
    assert any("not requested" in p for p in sc.mismatches("thermal_conductivity", item))


def test_multi_scenario_no_average_without_weights():
    s = ScenarioSet(
        scenarios=(
            EvaluationScenario(scenario_id="a", temperature_K=800.0),
            EvaluationScenario(scenario_id="b", temperature_K=1200.0),
        )
    )
    agg = aggregate({"a": 0.8, "b": 0.4}, s)
    assert agg.worst_case == 0.4 and agg.worst_scenario == "b"
    assert agg.robustness == pytest.approx(0.5)
    assert agg.weighted_mean is None
    assert any("no cross-scenario average" in n for n in agg.notes)
    weighted = ScenarioSet(scenarios=s.scenarios, weights={"a": 0.25, "b": 0.75})
    assert aggregate({"a": 0.8, "b": 0.4}, weighted).weighted_mean == pytest.approx(0.5)
    gap = aggregate({"a": 0.8, "b": None}, weighted)
    assert gap.worst_case is None and gap.weighted_mean is None
    omitted = aggregate({"a": 0.8}, weighted)
    assert omitted.worst_case is None and omitted.weighted_mean is None
    assert omitted.per_scenario == {"a": 0.8, "b": None}
    with pytest.raises(ValueError, match="undeclared"):
        aggregate({"a": 0.8, "b": 0.4, "z": 0.1}, weighted)


def test_scenario_weights_validated():
    sc = (EvaluationScenario(scenario_id="a", temperature_K=800.0),)
    with pytest.raises(ValidationError):
        ScenarioSet(scenarios=sc, weights={"a": 0.5})
    with pytest.raises(ValidationError):
        ScenarioSet(scenarios=sc, weights={"z": 1.0})
