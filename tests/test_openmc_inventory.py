"""OpenMC inventory to evidence conversion, on a synthetic inventory (no OpenMC needed)."""

from __future__ import annotations

import json
import math

import pytest

from matsim_agents.backends.external.openmc_activation.inventory import (
    parse_inventory,
    read_inventory,
    select_step,
    to_activation_evidence,
    to_transmutation_evidence,
)
from matsim_agents.science.irradiation.nuclear_evidence import (
    NuclearEvidenceError,
    NuclearFamily,
)

HASH = "a" * 64


def _doc(**overrides):
    doc = {
        "openmc_version": "0.15.3",
        "nuclear_data_library": "ENDF/B-VIII.0",
        "material_name": "Fe",
        "flux_cm2_s": 1.0e14,
        "temperature_K": 293.6,
        "input_sha256": {"chain.xml": HASH},
        "output_sha256": {"results.h5": HASH},
        "steps": [
            {
                "irradiation_time_s": 0.0,
                "cooling_time_s": 0.0,
                "atoms": {"Fe56": 1.0e22},
                "mass_kg": 1.0e-3,
                "activity_bq": 0.0,
                "decay_heat_w": 0.0,
            },
            {
                "irradiation_time_s": 1000.0,
                "cooling_time_s": 0.0,
                "atoms": {"Fe56": 0.99e22, "He4": 2.0e18, "H1": 1.0e18, "Mn56": 1.0e15},
                "mass_kg": 1.0e-3,
                "activity_bq": 5.0e9,
                "decay_heat_w": 2.0e-3,
            },
            {
                "irradiation_time_s": 1000.0,
                "cooling_time_s": 3600.0,
                "atoms": {"Fe56": 0.99e22, "He4": 2.0e18, "Mn56_m1": 4.0e14, "Mn56": 1.0e14},
                "mass_kg": 1.0e-3,
                "activity_bq": 1.0e9,
                "decay_heat_w": 4.0e-4,
            },
        ],
    }
    doc.update(overrides)
    return doc


def test_activation_evidence_is_specific_and_has_no_dose_rate():
    ev = to_activation_evidence(
        parse_inventory(_doc()),
        cooling_time_s=3600.0,
        material_id="Fe",
        irradiation_scenario_id="s1",
        synthetic=True,
    )
    assert ev.family is NuclearFamily.ACTIVATION
    assert ev.external_application == "OpenMC"
    values = {q.name: q for q in ev.quantities}
    assert set(values) == {"specific_activity", "decay_heat"}
    assert values["specific_activity"].value == pytest.approx(1.0e12)
    assert values["specific_activity"].unit == "Bq/kg"
    assert values["decay_heat"].value == pytest.approx(0.4)
    assert all(q.uncertainty is None for q in ev.quantities)


def test_transmutation_gas_is_appm_per_initial_atom_and_folds_metastables():
    ev = to_transmutation_evidence(
        parse_inventory(_doc()),
        cooling_time_s=0.0,
        material_id="Fe",
        irradiation_scenario_id="s1",
        synthetic=True,
    )
    by_isotope = {(q.isotope.element, q.isotope.mass_number): q.value for q in ev.quantities}
    assert by_isotope == {("H", 1): pytest.approx(100.0), ("He", 4): pytest.approx(200.0)}
    assert ev.isotope_inventory == {"Fe56": pytest.approx(1.0)}


def test_metastable_atoms_fold_into_the_ground_state_key():
    ev = to_transmutation_evidence(
        parse_inventory(_doc()),
        cooling_time_s=3600.0,
        material_id="Fe",
        irradiation_scenario_id="s1",
        elements=("Mn",),
        synthetic=True,
    )
    (quantity,) = ev.quantities
    assert quantity.isotope.mass_number == 56
    assert quantity.value == pytest.approx((4.0e14 + 1.0e14) / 1.0e22 * 1e6)


def test_missing_cooling_time_lists_the_available_ones():
    with pytest.raises(NuclearEvidenceError, match="available: 0, 3600"):
        select_step(parse_inventory(_doc()), 60.0)


def test_evidence_requires_output_hashes():
    inventory = parse_inventory(_doc(output_sha256={}))
    with pytest.raises(NuclearEvidenceError, match="output_sha256"):
        to_activation_evidence(
            inventory, cooling_time_s=0.0, material_id="Fe", irradiation_scenario_id="s1"
        )


def test_transmutation_rejects_an_unirradiated_step():
    unirradiated_only = parse_inventory(_doc(steps=[_doc()["steps"][0]]))
    with pytest.raises(NuclearEvidenceError, match="not been irradiated"):
        to_transmutation_evidence(
            unirradiated_only, cooling_time_s=0.0, material_id="Fe", irradiation_scenario_id="s1"
        )


def test_transmutation_without_the_requested_element_fails():
    with pytest.raises(NuclearEvidenceError, match="no Li isotopes"):
        to_transmutation_evidence(
            parse_inventory(_doc()),
            cooling_time_s=0.0,
            material_id="Fe",
            irradiation_scenario_id="s1",
            elements=("Li",),
        )


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan])
def test_inventory_rejects_bad_atom_counts(bad):
    doc = _doc()
    doc["steps"][1]["atoms"]["Fe56"] = bad
    with pytest.raises(NuclearEvidenceError):
        parse_inventory(doc)


def test_inventory_rejects_unknown_fields_and_bad_nuclide_names():
    with pytest.raises(NuclearEvidenceError):
        parse_inventory(_doc(uncertainty=0.1))
    doc = _doc()
    doc["steps"][1]["atoms"]["iron-56"] = 1.0
    with pytest.raises(NuclearEvidenceError, match="not a nuclide name"):
        parse_inventory(doc)


def test_read_inventory_round_trips_and_reports_a_missing_file(tmp_path):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(_doc()), encoding="utf-8")
    assert read_inventory(path).material_name == "Fe"
    with pytest.raises(NuclearEvidenceError, match="missing.json"):
        read_inventory(tmp_path / "missing.json")
