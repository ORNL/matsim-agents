import json
import os
from pathlib import Path

import pytest

from matsim_agents.backends.external.fispact.inventory import (
    parse_inventory,
    read_inventory,
    select_step,
    to_activation_evidence,
    to_transmutation_evidence,
)
from matsim_agents.science.irradiation.nuclear_evidence import (
    NuclearEvidenceError,
    activation_properties,
)


def _nuclide(grams, activity, el="Fe", iso=56):
    return {
        "element": el,
        "isotope": iso,
        "state": "",
        "grams": grams,
        "activity": activity,
        "heat": 0.0,
    }


def _document():
    dose = {"type": "Plane source", "distance": 1.0, "mass": 0.0, "dose": 2.0}
    return {
        "run_data": {"run_name": "demo", "flux_name": "flux"},
        "inventory_data": [
            {
                "irradiation_time": 0.0,
                "cooling_time": 0.0,
                "flux": 0.0,
                "total_heat": 0.0,
                "dose_rate": {**dose, "dose": 0.0},
                "nuclides": [_nuclide(1000.0, 0.0)],
            },
            {
                "irradiation_time": 100.0,
                "cooling_time": 0.0,
                "flux": 1e14,
                "total_heat": 0.5,
                "dose_rate": dose,
                "nuclides": [_nuclide(400.0, 1e6), _nuclide(100.0, 3e6, "Mn", 54)],
            },
        ],
    }


@pytest.fixture
def files(tmp_path):
    out = tmp_path / "out.json"
    out.write_text(json.dumps(_document()))
    inp = tmp_path / "run.i"
    inp.write_text("<< demo >>")
    return out, inp


def _evidence(out, inp, **kw):
    args = dict(
        cooling_time_s=0.0,
        material_id="m",
        irradiation_scenario_id="s",
        application_version="5.0",
        nuclear_data_library="TENDL-2019",
        input_files={"run.i": inp},
        synthetic=True,
    )
    args.update(kw)
    return to_activation_evidence(read_inventory(out), **args)


def test_units_are_normalised_per_kg(files):
    ev = _evidence(*files)
    values = {q.name: (q.value, q.unit) for q in ev.quantities}
    assert values["specific_activity"] == (pytest.approx(4e6 / 0.5), "Bq/kg")
    assert values["decay_heat"] == (pytest.approx(500.0 / 0.5), "W/kg")
    assert values["shutdown_dose_rate"] == (2.0, "Sv/h")
    assert ev.exposure_time_s == 100.0
    assert ev.dose_geometry == "Plane source, distance 1 m"
    assert set(ev.input_sha256) == {"run.i"}
    assert len(ev.output_sha256["inventory.json"]) == 64


def test_uncertainty_is_not_invented(files):
    ev = _evidence(*files)
    assert all(q.uncertainty is None for q in ev.quantities)
    assert activation_properties(ev, composition="Fe", evidence_id=ev.evidence_id())


def test_unknown_cooling_time_lists_the_choices(files):
    with pytest.raises(NuclearEvidenceError, match="available: 0"):
        select_step(read_inventory(files[0]), 3600.0)


def test_initial_step_has_no_exposure_time(files):
    inv = read_inventory(files[0])
    first = inv.steps[0]
    assert first.irradiation_time == 0.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("inventory_data"),
        lambda d: d.update(inventory_data=[]),
        lambda d: d["inventory_data"][1].update(total_heat=float("nan")),
        lambda d: d["inventory_data"][1]["nuclides"][0].update(grams=-1.0),
    ],
)
def test_malformed_results_are_rejected(mutate):
    doc = _document()
    mutate(doc)
    with pytest.raises(NuclearEvidenceError):
        parse_inventory(doc)


def test_requires_inputs_and_a_hashed_source(files):
    out, inp = files
    with pytest.raises(NuclearEvidenceError, match="input_files"):
        _evidence(out, inp, input_files={})
    with pytest.raises(NuclearEvidenceError, match="hash"):
        to_activation_evidence(
            parse_inventory(_document()),
            cooling_time_s=0.0,
            material_id="m",
            irradiation_scenario_id="s",
            application_version="5.0",
            nuclear_data_library="x",
            input_files={"run.i": inp},
        )


def test_non_object_json_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1]")
    with pytest.raises(NuclearEvidenceError, match="JSON object"):
        read_inventory(bad)


@pytest.mark.skipif(
    not os.environ.get("FISPACT_REFERENCE_JSON"), reason="set FISPACT_REFERENCE_JSON to a file"
)
def test_real_fispact_reference_file(tmp_path):
    inp = tmp_path / "run.i"
    inp.write_text("reference")
    inv = read_inventory(Path(os.environ["FISPACT_REFERENCE_JSON"]))
    ev = to_activation_evidence(
        inv,
        cooling_time_s=inv.steps[-1].cooling_time,
        material_id="reference",
        irradiation_scenario_id="reference",
        application_version="unknown",
        nuclear_data_library="unknown",
        input_files={"run.i": inp},
        synthetic=True,
    )
    assert ev.quantities


def _transmutation_document():
    doc = _document()
    doc["inventory_data"][0]["nuclides"] = [_nuclide(1000.0, 0.0, "Al", 27)]
    doc["inventory_data"][1]["nuclides"] = [
        _nuclide(999.0, 0.0, "Al", 27),
        _nuclide(0.9, 0.0, "H", 1),
        _nuclide(0.1, 0.0, "He", 4),
    ]
    return doc


def test_transmutation_appm_from_grams_per_mass_number(tmp_path):
    out = tmp_path / "t.json"
    out.write_text(json.dumps(_transmutation_document()))
    inp = tmp_path / "run.i"
    inp.write_text("x")
    ev = to_transmutation_evidence(
        read_inventory(out),
        cooling_time_s=0.0,
        material_id="Al",
        irradiation_scenario_id="s",
        application_version="5.0",
        nuclear_data_library="TENDL-2019",
        input_files={"run.i": inp},
        synthetic=True,
    )
    total = 1000.0 / 27
    assert ev.hydrogen_isotopes("concentration_appm") == {"H": pytest.approx(0.9 / total * 1e6)}
    assert ev.helium_isotopes("concentration_appm") == {
        "He-4": pytest.approx(0.1 / 4 / total * 1e6)
    }
    assert ev.isotope_inventory == {"Al27": 1.0}
    assert all(q.uncertainty is None for q in ev.quantities)


def test_transmutation_needs_an_irradiated_step_and_gas(tmp_path):
    inp = tmp_path / "run.i"
    inp.write_text("x")
    doc = _transmutation_document()
    doc["inventory_data"][1]["nuclides"] = [_nuclide(1000.0, 0.0, "Al", 27)]
    out = tmp_path / "t.json"
    out.write_text(json.dumps(doc))
    args = dict(
        material_id="Al",
        irradiation_scenario_id="s",
        application_version="5.0",
        nuclear_data_library="x",
        input_files={"run.i": inp},
    )
    with pytest.raises(NuclearEvidenceError, match="no H, He isotopes"):
        to_transmutation_evidence(read_inventory(out), cooling_time_s=0.0, **args)
    doc["inventory_data"][1]["irradiation_time"] = 0.0
    out.write_text(json.dumps(doc))
    with pytest.raises(NuclearEvidenceError, match="not been irradiated"):
        to_transmutation_evidence(read_inventory(out), cooling_time_s=0.0, **args)


def _missing(mutate):
    doc = _document()
    mutate(doc["inventory_data"][1])
    return parse_inventory(doc, source_sha256="0" * 64)


def _quantities(inventory, tmp_path):
    inp = tmp_path / "run.i"
    inp.write_text("x")
    ev = to_activation_evidence(
        inventory,
        cooling_time_s=0.0,
        material_id="m",
        irradiation_scenario_id="s",
        application_version="5",
        nuclear_data_library="L",
        input_files={"run.i": inp},
        synthetic=True,
    )
    return {q.name for q in ev.quantities}, ev.notes


def test_missing_heat_and_dose_are_omitted_not_zero(tmp_path):
    def mutate(step):
        del step["total_heat"]
        step["dose_rate"] = {"type": "Plane source", "distance": 1.0}

    names, notes = _quantities(_missing(mutate), tmp_path)
    assert names == {"specific_activity"}
    assert any("total_heat not reported" in n for n in notes)


def test_missing_activity_is_omitted_but_reported_zero_is_kept(tmp_path):
    def mutate(step):
        del step["nuclides"][0]["activity"]

    assert "specific_activity" not in _quantities(_missing(mutate), tmp_path)[0]
    zero = _missing(lambda step: step.update(total_heat=0.0))
    assert "decay_heat" in _quantities(zero, tmp_path)[0]


def test_missing_mass_is_an_error(tmp_path):
    def mutate(step):
        del step["nuclides"][1]["grams"]

    with pytest.raises(NuclearEvidenceError, match="mass"):
        _quantities(_missing(mutate), tmp_path)
