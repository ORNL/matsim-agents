from __future__ import annotations

import pytest
from pydantic import ValidationError

from matsim_agents.workflows.candidate_registry import (
    NOT_YET_ESTABLISHED,
    CandidateRegistry,
    DatasetEntry,
    calculation_id,
    chemical_system,
    hierarchy_subspaces,
    ingest_dataset,
)

ELEMENTS = ("Cr", "Hf", "Mo", "Ta", "V", "W")


def _entry(composition, h, **kw):
    return DatasetEntry(structure_hash=h, composition=composition, **kw)


def test_hierarchy_component_counts_for_six_elements():
    sub = hierarchy_subspaces(ELEMENTS)
    assert {k: len(v) for k, v in sub.items()} == {2: 15, 3: 20, 4: 15, 5: 6, 6: 1}
    assert sub[6] == [tuple(sorted(ELEMENTS))]
    assert len(set(sub[2])) == 15


def test_nb_data_never_counts_as_target_coverage():
    reg = CandidateRegistry()
    rec = ingest_dataset(
        reg,
        "nb-ta-v",
        [
            _entry({"Nb": 8, "Ta": 8}, "a"),
            _entry({"Nb": 8, "V": 8}, "b"),
            _entry({"Ta": 8, "V": 8}, "c"),
            _entry({"Nb": 6, "Ta": 5, "V": 5}, "d"),
        ],
        target_elements=ELEMENTS,
    )
    assert rec.off_target_elements == ["Nb"]
    assert rec.target_elements_covered == ["Ta", "V"]
    assert rec.covered_subspaces == ["Ta-V"]
    coverage = reg.target_coverage(hierarchy_subspaces(ELEMENTS))
    assert coverage["Ta-V"] == 1
    assert all(coverage[k] == 0 for k in coverage if k != "Ta-V")
    assert not any(
        r.transferability("bulk") != NOT_YET_ESTABLISHED for r in reg.candidates.values()
    )


def test_record_fields_and_serialization_round_trip():
    reg = CandidateRegistry()
    ingest_dataset(
        reg,
        "w-ta",
        [
            _entry(
                {"W": 8, "Ta": 8},
                "h1",
                supercell=(2, 2, 2),
                ordering="random",
                dft_backend="qe",
                dft_settings={"ecutwfc": 80},
                magnetic_initial={"mode": "nm"},
                cost_node_hours=1.5,
                provenance={"doi": "x"},
            )
        ],
        target_elements=ELEMENTS,
    )
    record = next(iter(reg.candidates.values()))
    assert record.chemical_system == "Ta-W" and record.n_components == 2
    assert record.atom_count == 16 and record.origin == "existing"
    assert record.source_dataset == "w-ta" and record.provenance == {"doi": "x"}
    restored = CandidateRegistry.model_validate_json(reg.model_dump_json())
    assert restored == reg


def test_duplicate_calculation_detected_but_changed_conditions_kept():
    reg = CandidateRegistry()
    base = {"dft_backend": "qe", "dft_settings": {"ecutwfc": 80}, "magnetic_initial": {"m": 0}}
    first = ingest_dataset(
        reg, "d1", [_entry({"W": 4, "Ta": 4}, "h", **base)], target_elements=ELEMENTS
    )
    again = ingest_dataset(
        reg, "d2", [_entry({"W": 4, "Ta": 4}, "h", **base)], target_elements=ELEMENTS
    )
    assert (first.n_new_calculations, again.n_duplicate_calculations) == (1, 1)
    assert len(reg.candidates) == 1 and len(reg.calculations) == 1
    ingest_dataset(
        reg,
        "d3",
        [_entry({"W": 4, "Ta": 4}, "h", **{**base, "dft_settings": {"ecutwfc": 60}})],
        target_elements=ELEMENTS,
    )
    ingest_dataset(
        reg,
        "d4",
        [_entry({"W": 4, "Ta": 4}, "h", **{**base, "magnetic_initial": {"m": 1}})],
        target_elements=ELEMENTS,
    )
    assert len(reg.candidates) == 1 and len(reg.calculations) == 3


def test_calculation_id_is_stable_and_order_independent():
    a = calculation_id("h", "qe", {"a": 1, "b": 2}, None)
    assert a == calculation_id("h", "qe", {"b": 2, "a": 1}, {})
    assert a != calculation_id("h", "vasp", {"a": 1, "b": 2}, None)


def test_chemical_system_is_order_independent_and_entry_validated():
    assert chemical_system(["W", "Ta", "W"]) == "Ta-W"
    with pytest.raises(ValidationError):
        _entry({}, "x")
    with pytest.raises(ValidationError):
        _entry({"W": 0}, "x")
