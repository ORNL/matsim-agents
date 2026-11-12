from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("ase")

from benchmarks.e3.structure_generation import (
    StructureGenerationError,
    generate_structures,
)


def _action(material, **overrides):
    defaults = {
        "action_id": f"test-{material}",
        "family": "prototype_strain",
        "formula": "NbO2",
        "size_tier": 2,
        "generator": "prototype",
        "random_seed": 13,
        "count": 1,
    }
    defaults.update(overrides)
    return defaults


def _read_record(record):
    from ase.io import read

    atoms = read(record["structure_path"], index=0, format="extxyz")
    assert atoms.calc is None
    assert set(atoms.arrays) == {"numbers", "positions"}
    assert all(atoms.pbc)
    assert atoms.info.keys() == {"generation_json"}
    generation = json.loads(atoms.info["generation_json"])
    assert generation == record["generation"]
    assert (
        hashlib.sha256(Path(record["structure_path"]).read_bytes()).hexdigest()
        == (record["structure_sha256"])
    )
    return atoms


@pytest.mark.parametrize(
    ("material", "formula", "tier", "generator", "expected_counts"),
    [
        (
            "bcc_refractory",
            "NbTaVZr",
            4,
            "native_lattice_occupations",
            {"Nb": 4, "Ta": 4, "V": 4, "Zr": 4},
        ),
        (
            "fcc_cantor",
            "CrMnFeCoNi",
            4,
            "native_lattice_occupations",
            {"Cr": 4, "Mn": 4, "Fe": 4, "Co": 4, "Ni": 4},
        ),
    ],
)
def test_native_alloy_occupations_are_periodic_exact_and_unique(
    tmp_path, material, formula, tier, generator, expected_counts
):
    records = generate_structures(
        material,
        _action(
            material,
            family="equiatomic_ordering",
            formula=formula,
            size_tier=tier,
            generator=generator,
            random_seed=51,
            count=3,
        ),
        tmp_path,
    )

    assert len(records) == 3
    assert len({record["generation"]["canonical_geometry_sha256"] for record in records}) == 3
    for record in records:
        atoms = _read_record(record)
        assert Counter(atoms.get_chemical_symbols()) == Counter(expected_counts)
        assert record["num_atoms"] == len(atoms)
        assert record["formula_units"] == tier
        assert record["generation"]["family"] == "equiatomic_ordering"
        assert record["generation"]["prototype"] in {"bcc", "fcc"}


def test_lifepo4_antisite_batch_keeps_source_composition_and_provenance(tmp_path):
    records = generate_structures(
        "lifepo4",
        _action(
            "lifepo4",
            family="antisite_pair",
            formula="LiFePO4",
            size_tier=4,
            generator="class_specific",
            random_seed=8,
            count=2,
            defect="antisite_pair",
        ),
        tmp_path,
    )

    assert len(records) == 2
    for record in records:
        atoms = _read_record(record)
        assert Counter(atoms.get_chemical_symbols()) == Counter({"Li": 4, "Fe": 4, "P": 4, "O": 16})
        assert record["generation"]["source_file"].endswith("lifepo4_olivine.extxyz")
        assert record["generation"]["operation"]["type"] == "Li-Fe antisite pair"


def test_candidate_geometry_rejection_retries_within_the_bounded_batch(monkeypatch, tmp_path):
    import benchmarks.e3.structure_generation as generation

    validate = generation._validate_geometry
    calls = 0

    def reject_first_candidate(atoms):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StructureGenerationError("test overlap")
        validate(atoms)

    monkeypatch.setattr(generation, "_validate_geometry", reject_first_candidate)
    records = generate_structures(
        "lifepo4",
        _action(
            "lifepo4-retry",
            family="olivine",
            formula="LiFePO4",
            size_tier=4,
            generator="class_specific",
            count=1,
        ),
        tmp_path,
    )

    assert len(records) == 1
    assert calls == 2
    assert Counter(_read_record(records[0]).get_chemical_symbols()) == Counter(
        {"Li": 4, "Fe": 4, "P": 4, "O": 16}
    )


def test_symmetry_matching_recognizes_translated_lifepo4_antisite_cell(tmp_path):
    pytest.importorskip("pymatgen")
    from benchmarks.e3.structure_generation import (
        make_structure_matcher,
        structures_symmetry_equivalent,
    )

    record = generate_structures(
        "lifepo4",
        _action(
            "lifepo4-symmetry",
            family="antisite_pair",
            formula="LiFePO4",
            size_tier=8,
            generator="class_specific",
            random_seed=83,
            count=1,
            defect="antisite_pair",
        ),
        tmp_path,
    )[0]
    atoms = _read_record(record)
    translated = atoms.copy()
    translated.translate(atoms.cell.array[0] * 0.37 + atoms.cell.array[1] * 0.19)

    assert structures_symmetry_equivalent(atoms, translated, matcher=make_structure_matcher())


def test_pyxtal_proposals_scale_with_requested_count(monkeypatch):
    pytest.importorskip("pyxtal")
    from benchmarks.e3 import structure_generation
    from matsim_agents.discovery import seeds

    captured = {}

    def fake_random_search(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(seeds, "random_search", fake_random_search)
    with pytest.raises(StructureGenerationError, match="PyXtal produced no oxide candidate"):
        structure_generation._pyxtal_oxide_candidates(
            "Nb2O5", tier=2, seed=3, count=80, prototype_ids=(), space_groups=()
        )

    assert captured["n"] == 160


def test_phosphorene_is_generated_from_2d_template_with_fixed_vacuum(tmp_path):
    from ase.io import read

    records = generate_structures(
        "phosphorene",
        _action(
            "phosphorene",
            family="in_plane_strain",
            formula="P",
            size_tier=16,
            generator="2d_template",
            random_seed=91,
            strain=0.01,
        ),
        tmp_path,
    )
    atoms = _read_record(records[0])

    assert len(atoms) == 16
    assert set(atoms.get_chemical_symbols()) == {"P"}
    assert records[0]["formula_units"] == 16
    assert records[0]["generation"]["source_file"].endswith("phosphorene_black_P.extxyz")
    assert records[0]["generation"]["operation"]["type"] == "bounded in-plane strain"
    source_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "paper_cases"
        / "seeds"
        / "phosphorene_black_P.extxyz"
    )
    source_atoms = read(source_path, index=0, format="extxyz")
    assert atoms.cell[2].tolist() == pytest.approx(source_atoms.cell[2].tolist())


def test_blue_phosphorene_uses_the_blue_template(tmp_path):
    records = generate_structures(
        "phosphorene",
        _action(
            "blue-phosphorene",
            family="blue_phosphorene",
            formula="P",
            size_tier=16,
            generator="2d_template",
        ),
        tmp_path,
    )

    assert records[0]["generation"]["source_file"].endswith("phosphorene_blue_P.extxyz")
    assert len(_read_record(records[0])) == 16


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("Nb2O5", {"Nb": 4, "O": 10}),
        ("Ta2O5", {"Ta": 4, "O": 10}),
        ("NbTaO5", {"Nb": 2, "Ta": 2, "O": 10}),
        ("NbO2", {"Nb": 2, "O": 4}),
        ("TaO2", {"Ta": 2, "O": 4}),
        ("NbTaO4", {"Nb": 2, "Ta": 2, "O": 8}),
    ],
)
def test_declared_nb_ta_oxide_prototypes_preserve_composition(tmp_path, formula, expected):
    records = generate_structures(
        "nb_ta_o",
        _action(
            f"oxide-{formula}",
            family="oxide_prototype",
            formula=formula,
            size_tier=2,
            generator="prototype",
            random_seed=3,
        ),
        tmp_path,
    )
    atoms = _read_record(records[0])

    assert Counter(atoms.get_chemical_symbols()) == Counter(expected)
    if formula == "NbTaO5":
        assert records[0]["generation"]["prototype_id"] == "A2B5_oP7_local_prototype"
    assert records[0]["generation"]["source_file"] is None


def test_oxide_prototype_and_space_group_filters_affect_generation(tmp_path):
    record = generate_structures(
        "nb_ta_o",
        _action(
            "filtered-oxide",
            family="oxide_prototype",
            formula="NbO2",
            size_tier=2,
            generator="prototype",
            prototype_ids=["AB2_cF12_225_a_c"],
            space_groups=[225],
        ),
        tmp_path,
    )[0]
    assert record["generation"]["prototype_ids"] == ["AB2_cF12_225_a_c"]
    assert record["generation"]["space_groups"] == [225]
    assert record["generation"]["space_group"].endswith("(225)")

    with pytest.raises(StructureGenerationError, match="does not match prototype_ids"):
        generate_structures(
            "nb_ta_o",
            _action(
                "mismatched-prototype-filter",
                family="oxide_prototype",
                formula="NbO2",
                size_tier=2,
                generator="prototype",
                prototype_ids=["not-a-prototype"],
            ),
            tmp_path,
        )


def test_charge_compensated_oxygen_vacancies_derive_from_pentoxide(tmp_path, monkeypatch):
    from ase import Atoms

    import benchmarks.e3.structure_generation as generation

    parent = Atoms(
        symbols=["Nb"] * 4 + ["O"] * 10,
        positions=[[1 + 2 * index, 1, 1] for index in range(14)],
        cell=[32, 32, 32],
        pbc=True,
    )
    monkeypatch.setattr(
        generation,
        "_pyxtal_oxide_candidates",
        lambda formula, tier, *args: [
            (
                parent,
                {
                    "formula_units": tier,
                    "prototype": "stoichiometric parent fixture",
                    "source_file": None,
                    "source_sha256": None,
                    "source_description": "test fixture",
                },
            )
        ],
    )

    record = generate_structures(
        "nb_ta_o",
        _action(
            "oxygen-vacancy",
            family="oxygen_defect",
            formula="Nb2O4",
            size_tier=2,
            generator="pyxtal_vacancy",
            defect="oxygen_defect",
        ),
        tmp_path,
    )[0]
    atoms = _read_record(record)
    operation = record["generation"]["operation"]

    assert Counter(atoms.get_chemical_symbols()) == Counter({"Nb": 4, "O": 8})
    assert operation["parent_formula"] == "Nb2O5"
    assert len(operation["removed_oxygen_sites"]) == 2
    assert operation["charge_compensation"] == (
        "charge-neutral Nb2O4; formal cation(IV) with O(II-)"
    )

    with pytest.raises(StructureGenerationError, match="outside space_groups"):
        generate_structures(
            "nb_ta_o",
            _action(
                "mismatched-space-group-filter",
                family="oxide_prototype",
                formula="NbO2",
                size_tier=2,
                generator="prototype",
                space_groups=[221],
            ),
            tmp_path,
        )


def test_aflow_filters_are_passed_to_the_candidate_generator(tmp_path, monkeypatch):
    from ase import Atoms

    import benchmarks.e3.structure_generation as generation

    atoms = Atoms(
        symbols="Nb2O4",
        scaled_positions=[
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ],
        cell=[8, 8, 8],
        pbc=True,
    )
    observed = {}

    def candidates(formula, tier, prototype_ids, space_groups):
        observed.update(
            formula=formula,
            tier=tier,
            prototype_ids=prototype_ids,
            space_groups=space_groups,
        )
        return [
            (
                atoms,
                {
                    "formula_units": tier,
                    "prototype": "AFLOW test prototype",
                    "prototype_id": prototype_ids[0],
                    "source_file": None,
                    "source_sha256": None,
                    "operation": {"type": "AFLOW prototype decoration", "space_group": 225},
                },
            )
        ]

    monkeypatch.setattr(generation, "_aflow_oxide_candidates", candidates)
    record = generate_structures(
        "nb_ta_o",
        _action(
            "filtered-aflow",
            family="aflow_prototype",
            formula="NbO2",
            size_tier=2,
            generator="aflow_prototype",
            prototype_ids=["AB2_cF12_225_a_c"],
            space_groups=[225],
        ),
        tmp_path,
    )[0]

    assert observed == {
        "formula": "NbO2",
        "tier": 2,
        "prototype_ids": ("AB2_cF12_225_a_c",),
        "space_groups": (225,),
    }
    assert record["generation"]["prototype_ids"] == ["AB2_cF12_225_a_c"]
    assert record["generation"]["space_groups"] == [225]


def test_mof_is_cod_only_and_perturbation_preserves_connectivity(tmp_path):
    records = generate_structures(
        "zn_formate_mof",
        _action(
            "zn_formate_mof",
            family="bounded_perturbation",
            formula="Zn(HCOO)2",
            size_tier=24,
            generator="class_specific",
            random_seed=7,
            count=2,
        ),
        tmp_path,
    )

    assert len(records) == 2
    assert len({record["generation"]["canonical_geometry_sha256"] for record in records}) == 2
    for record in records:
        atoms = _read_record(record)
        assert Counter(atoms.get_chemical_symbols()) == Counter(
            {"Zn": 24, "C": 48, "H": 48, "O": 96}
        )
        assert record["formula_units"] == 24
        assert record["generation"]["source_file"].endswith("zn_formate_cod_7205949.cif")
        assert record["generation"]["source_sha256"] == (
            "df49d464c2cffdf50abe5c5442f0e2e2719fff77e950a43463ab8476a1887d10"
        )
        assert record["generation"]["operation"]["type"].startswith(
            "bounded connectivity-preserving"
        )


def test_incomplete_batches_and_unsupported_tiers_raise_explicit_errors(tmp_path):
    with pytest.raises(StructureGenerationError, match="Requested 2 unique"):
        generate_structures(
            "lifepo4",
            _action(
                "lifepo4-duplicate",
                family="olivine",
                formula="LiFePO4",
                size_tier=4,
                generator="class_specific",
                count=2,
            ),
            tmp_path,
        )

    with pytest.raises(StructureGenerationError, match="Unsupported phosphorene size_tier"):
        generate_structures(
            "phosphorene",
            _action(
                "blue-small",
                family="blue_phosphorene",
                formula="P",
                size_tier=8,
                generator="2d_template",
            ),
            tmp_path,
        )
