from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from ase.neighborlist import neighbor_list
from pydantic import ValidationError

from matsim_agents.discovery.alloys import (
    BCC_FIRST_SHELL,
    BCC_SECOND_SHELL,
    AlloySpec,
    assign_species,
    bcc_site_count,
    build_bcc_supercell,
    first_neighbor_list,
    generate_alloy,
    second_neighbor_list,
)

REFERENCE_COUNTS = {"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}


def test_reference_spec_is_valid_and_has_1024_sites():
    spec = AlloySpec(counts=REFERENCE_COUNTS)
    assert spec.n_sites == 1024
    assert spec.repeats == (8, 8, 8)
    assert spec.composition["W"] == pytest.approx(0.625)


@pytest.mark.parametrize(
    ("counts", "repeats"),
    [
        ({"W": 2}, (1, 1, 1)),
        ({"W": 1, "Mo": 1}, (1, 1, 1)),
        ({"W": 128}, (4, 4, 4)),
    ],
)
def test_valid_small_specs_are_accepted(counts, repeats):
    assert AlloySpec(counts=counts, repeats=repeats).n_sites == bcc_site_count(repeats)


def test_zero_counts_are_dropped_not_rejected():
    spec = AlloySpec(counts={"W": 2, "Mo": 0}, repeats=(1, 1, 1))
    assert spec.counts == {"W": 2}


@pytest.mark.parametrize(
    ("counts", "repeats", "message"),
    [
        ({}, (1, 1, 1), "at least"),
        ({"W": 0}, (1, 1, 1), "at least"),
        ({"W": 1}, (1, 1, 1), "sum to 1"),
        ({"W": 3}, (1, 1, 1), "sum to 3"),
        ({"W": 1, "Mo": 2}, (1, 1, 1), "sum to 3"),
        ({"W": -1, "Mo": 3}, (1, 1, 1), "nonnegative"),
        ({"W": True, "Mo": True}, (1, 1, 1), "valid integer"),
        ({"W": 2.0}, (1, 1, 1), "valid integer"),
        ({"Xx": 2}, (1, 1, 1), "unknown element"),
        ({"W": 2}, (0, 1, 1), "positive"),
    ],
)
def test_invalid_specs_are_rejected(counts, repeats, message):
    with pytest.raises(ValidationError, match=message):
        AlloySpec(counts=counts, repeats=repeats)


def test_spec_is_immutable():
    spec = AlloySpec(counts={"W": 2}, repeats=(1, 1, 1))
    with pytest.raises(ValidationError):
        spec.seed = 5


def test_counts_mapping_cannot_be_mutated_after_validation():
    spec = AlloySpec(counts={"W": 1, "Mo": 1}, repeats=(1, 1, 1))
    with pytest.raises(TypeError):
        spec.counts["W"] = 3
    with pytest.raises(TypeError):
        del spec.counts["Mo"]
    assert dict(spec.counts) == {"Mo": 1, "W": 1}


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_lattice_parameter_is_rejected(value):
    with pytest.raises(ValidationError):
        AlloySpec(counts={"W": 2}, repeats=(1, 1, 1), lattice_parameter=value)


def test_alloy_spec_json_round_trips_counts_as_plain_mapping():
    spec = AlloySpec(counts={"W": 1, "Mo": 1}, repeats=(1, 1, 1), seed=4)
    assert AlloySpec.model_validate_json(spec.model_dump_json()) == spec


def test_bcc_supercell_geometry_and_periodicity():
    spec = AlloySpec(counts=REFERENCE_COUNTS)
    atoms = build_bcc_supercell(spec)
    assert len(atoms) == 1024
    assert atoms.pbc.all()
    assert np.allclose(atoms.cell.lengths(), 8 * spec.lattice_parameter)
    assert np.allclose(atoms.cell.angles(), 90.0)


def test_bcc_first_shell_has_eight_neighbours_per_site():
    spec = AlloySpec(counts=REFERENCE_COUNTS)
    atoms = generate_alloy(spec)
    first_shell = spec.lattice_parameter * np.sqrt(3) / 2
    i, _ = neighbor_list("ij", atoms, 1.1 * first_shell)
    coordination = np.bincount(i, minlength=len(atoms))
    assert set(coordination) == {BCC_FIRST_SHELL}


@pytest.mark.parametrize("repeats", [(1, 1, 1), (2, 2, 2), (3, 2, 4)])
def test_first_neighbor_list_gives_eight_periodic_body_diagonals_per_site(repeats):
    spec = AlloySpec(counts={"W": bcc_site_count(repeats)}, repeats=repeats)
    atoms = build_bcc_supercell(spec)
    i, _, vectors = first_neighbor_list(atoms, spec.lattice_parameter)
    half_diagonal = spec.lattice_parameter / 2

    assert len(i) == BCC_FIRST_SHELL * len(atoms)
    assert np.allclose(np.linalg.norm(vectors, axis=1), np.sqrt(3) * half_diagonal)
    directions = np.rint(vectors / half_diagonal).astype(int)
    expected = {(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)}
    for site in range(len(atoms)):
        assert {tuple(d) for d in directions[i == site]} == expected


def test_first_neighbor_list_on_reference_structure_excludes_second_shell():
    spec = AlloySpec(counts=REFERENCE_COUNTS, seed=5)
    atoms = generate_alloy(spec)
    i, _, vectors = first_neighbor_list(atoms, spec.lattice_parameter)
    coordination = np.bincount(i, minlength=len(atoms))
    assert set(coordination) == {BCC_FIRST_SHELL}
    assert np.allclose(np.linalg.norm(vectors, axis=1), np.sqrt(3) * spec.lattice_parameter / 2)


@pytest.mark.parametrize("repeats", [(1, 1, 1), (2, 2, 2), (3, 2, 4)])
def test_second_neighbor_list_gives_six_axis_aligned_neighbours_per_site(repeats):
    spec = AlloySpec(counts={"W": bcc_site_count(repeats)}, repeats=repeats)
    atoms = build_bcc_supercell(spec)
    i, _, vectors = second_neighbor_list(atoms, spec.lattice_parameter)
    a = spec.lattice_parameter

    assert len(i) == BCC_SECOND_SHELL * len(atoms)
    assert np.allclose(np.linalg.norm(vectors, axis=1), a)
    directions = np.rint(vectors / a).astype(int)
    expected = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    for site in range(len(atoms)):
        assert {tuple(d) for d in directions[i == site]} == expected


def test_second_shell_is_disjoint_from_first_and_reciprocal_on_reference_structure():
    spec = AlloySpec(counts=REFERENCE_COUNTS, seed=5)
    atoms = generate_alloy(spec)
    first_i, _, _ = first_neighbor_list(atoms, spec.lattice_parameter)
    i, j, vectors = second_neighbor_list(atoms, spec.lattice_parameter)

    assert set(np.bincount(i, minlength=len(atoms))) == {BCC_SECOND_SHELL}
    assert len(first_i) + len(i) == (BCC_FIRST_SHELL + BCC_SECOND_SHELL) * len(atoms)

    units = np.rint(vectors / spec.lattice_parameter).astype(int)
    forward = set(zip(i.tolist(), j.tolist(), map(tuple, units.tolist()), strict=True))
    reverse = {(b, a, tuple(-x for x in d)) for a, b, d in forward}
    assert forward == reverse


def test_generated_alloy_has_exact_species_counts():
    atoms = generate_alloy(AlloySpec(counts=REFERENCE_COUNTS, seed=7))
    assert Counter(atoms.get_chemical_symbols()) == Counter(REFERENCE_COUNTS)


def test_same_seed_gives_identical_occupations_and_structure_data():
    first = generate_alloy(AlloySpec(counts=REFERENCE_COUNTS, seed=11))
    second = generate_alloy(AlloySpec(counts=REFERENCE_COUNTS, seed=11))
    assert first.get_chemical_symbols() == second.get_chemical_symbols()
    assert np.array_equal(first.positions, second.positions)
    assert first.info == second.info


def test_different_seeds_give_distinct_orderings_with_same_composition():
    first = generate_alloy(AlloySpec(counts=REFERENCE_COUNTS, seed=1))
    second = generate_alloy(AlloySpec(counts=REFERENCE_COUNTS, seed=2))
    assert first.get_chemical_symbols() != second.get_chemical_symbols()
    assert Counter(second.get_chemical_symbols()) == Counter(REFERENCE_COUNTS)


def test_assign_species_is_a_permutation_of_the_counts():
    spec = AlloySpec(counts={"W": 1, "Mo": 1}, repeats=(1, 1, 1), seed=3)
    assert Counter(assign_species(spec)) == Counter({"W": 1, "Mo": 1})
