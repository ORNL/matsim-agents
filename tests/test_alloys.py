from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from ase.neighborlist import neighbor_list
from pydantic import ValidationError

from matsim_agents.science.structures.alloys import (
    AlloySpec,
    assign_species,
    build_supercell,
    first_neighbor_list,
    generate_alloy,
    second_neighbor_list,
    warren_cowley_sro,
)

REFERENCE_COUNTS = {"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}


def test_reference_spec_is_valid_and_has_1024_sites():
    spec = AlloySpec(counts=REFERENCE_COUNTS, seed=0)
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
    assert AlloySpec(counts=counts, repeats=repeats, seed=0).n_sites == (2 * int(np.prod(repeats)))


def test_zero_counts_are_dropped_not_rejected():
    spec = AlloySpec(counts={"W": 2, "Mo": 0}, repeats=(1, 1, 1), seed=0)
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
        AlloySpec(counts=counts, repeats=repeats, seed=0)


@pytest.mark.parametrize("seed", [-1, True, 1.0, "1", None])
def test_seed_must_be_a_nonnegative_integer(seed):
    with pytest.raises(ValidationError):
        AlloySpec(counts={"W": 2}, repeats=(1, 1, 1), seed=seed)


def test_seed_is_required():
    with pytest.raises(ValidationError, match="seed"):
        AlloySpec(counts={"W": 2}, repeats=(1, 1, 1))


def test_spec_is_immutable():
    spec = AlloySpec(counts={"W": 2}, repeats=(1, 1, 1), seed=0)
    with pytest.raises(ValidationError):
        spec.seed = 5


def test_counts_mapping_cannot_be_mutated_after_validation():
    spec = AlloySpec(counts={"W": 1, "Mo": 1}, repeats=(1, 1, 1), seed=0)
    with pytest.raises(TypeError):
        spec.counts["W"] = 3
    with pytest.raises(TypeError):
        del spec.counts["Mo"]
    assert dict(spec.counts) == {"Mo": 1, "W": 1}


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_lattice_parameter_is_rejected(value):
    with pytest.raises(ValidationError):
        AlloySpec(counts={"W": 2}, repeats=(1, 1, 1), lattice_parameter=value, seed=0)


def test_alloy_spec_json_round_trips_counts_as_plain_mapping():
    spec = AlloySpec(counts={"W": 1, "Mo": 1}, repeats=(1, 1, 1), seed=4)
    assert AlloySpec.model_validate_json(spec.model_dump_json()) == spec


def test_bcc_supercell_geometry_and_periodicity():
    spec = AlloySpec(counts=REFERENCE_COUNTS, seed=0)
    atoms = build_supercell(spec)
    assert len(atoms) == 1024
    assert atoms.pbc.all()
    assert np.allclose(atoms.cell.lengths(), 8 * spec.lattice_parameter)
    assert np.allclose(atoms.cell.angles(), 90.0)


def test_bcc_first_shell_has_eight_neighbours_per_site():
    spec = AlloySpec(counts=REFERENCE_COUNTS, seed=0)
    atoms = generate_alloy(spec)
    first_shell = spec.lattice_parameter * np.sqrt(3) / 2
    i, _ = neighbor_list("ij", atoms, 1.1 * first_shell)
    coordination = np.bincount(i, minlength=len(atoms))
    assert set(coordination) == {8}


@pytest.mark.parametrize("repeats", [(1, 1, 1), (2, 2, 2), (3, 2, 4)])
def test_first_neighbor_list_gives_eight_periodic_body_diagonals_per_site(repeats):
    spec = AlloySpec(counts={"W": (2 * int(np.prod(repeats)))}, repeats=repeats, seed=0)
    atoms = build_supercell(spec)
    i, _, vectors = first_neighbor_list(atoms, spec.lattice_parameter)
    half_diagonal = spec.lattice_parameter / 2

    assert len(i) == 8 * len(atoms)
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
    assert set(coordination) == {8}
    assert np.allclose(np.linalg.norm(vectors, axis=1), np.sqrt(3) * spec.lattice_parameter / 2)


@pytest.mark.parametrize("repeats", [(1, 1, 1), (2, 2, 2), (3, 2, 4)])
def test_second_neighbor_list_gives_six_axis_aligned_neighbours_per_site(repeats):
    spec = AlloySpec(counts={"W": (2 * int(np.prod(repeats)))}, repeats=repeats, seed=0)
    atoms = build_supercell(spec)
    i, _, vectors = second_neighbor_list(atoms, spec.lattice_parameter)
    a = spec.lattice_parameter

    assert len(i) == 6 * len(atoms)
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

    assert set(np.bincount(i, minlength=len(atoms))) == {6}
    assert len(first_i) + len(i) == (8 + 6) * len(atoms)

    units = np.rint(vectors / spec.lattice_parameter).astype(int)
    forward = set(zip(i.tolist(), j.tolist(), map(tuple, units.tolist()), strict=True))
    reverse = {(b, a, tuple(-x for x in d)) for a, b, d in forward}
    assert forward == reverse


def _b2_binary(repeats):
    """W on cube corners, Mo on body centres: perfectly ordered CsCl-type BCC."""
    spec = AlloySpec(
        counts={"Mo": (2 * int(np.prod(repeats))) // 2, "W": (2 * int(np.prod(repeats))) // 2},
        repeats=repeats,
        seed=0,
    )
    atoms = build_supercell(spec)
    parity = np.rint(2 * atoms.positions / spec.lattice_parameter).astype(int).sum(axis=1) % 2
    atoms.set_chemical_symbols(["W" if p == 0 else "Mo" for p in parity])
    return atoms, spec.lattice_parameter


@pytest.mark.parametrize("repeats", [(1, 1, 1), (2, 2, 2), (3, 2, 4)])
def test_ordered_b2_binary_has_analytic_sro_in_both_shells(repeats):
    atoms, a = _b2_binary(repeats)
    result = warren_cowley_sro(atoms, a)
    n_each = len(atoms) // 2

    assert result.species == ("Mo", "W")
    assert np.allclose(result.composition, [0.5, 0.5])
    # First shell: every neighbour is on the other sublattice.
    assert np.allclose(result.first_shell.alpha, [[1, -1], [-1, 1]])
    assert np.array_equal(result.first_shell.pair_counts, [[0, 8 * n_each], [8 * n_each, 0]])
    # Second shell: every neighbour is on the same sublattice.
    assert np.allclose(result.second_shell.alpha, [[-1, 1], [1, -1]])
    assert np.array_equal(result.second_shell.pair_counts, [[6 * n_each, 0], [0, 6 * n_each]])


def test_pair_counts_reuse_neighbour_lists_without_double_counting():
    atoms, a = _b2_binary((2, 2, 2))
    result = warren_cowley_sro(atoms, a)
    assert result.first_shell.pair_counts.sum() == len(first_neighbor_list(atoms, a)[0])
    assert result.second_shell.pair_counts.sum() == len(second_neighbor_list(atoms, a)[0])


def test_reference_alloy_gives_two_6x6_matrices_satisfying_identities():
    spec = AlloySpec(counts=REFERENCE_COUNTS, seed=5)
    result = warren_cowley_sro(generate_alloy(spec), spec.lattice_parameter)
    species_counts = np.array([REFERENCE_COUNTS[s] for s in result.species])

    assert result.species == ("Cr", "Hf", "Mo", "Ta", "V", "W")
    assert np.allclose(result.composition, species_counts / 1024)
    for shell, z in (
        (result.first_shell, 8),
        (result.second_shell, 6),
    ):
        assert shell.alpha.shape == (6, 6)
        assert shell.pair_counts.sum() == z * 1024
        assert np.array_equal(shell.pair_counts, shell.pair_counts.T)
        assert np.array_equal(shell.pair_counts.sum(axis=1), z * species_counts)
        assert np.allclose(shell.conditional_probability.sum(axis=1), 1.0)
        assert np.allclose(shell.alpha @ result.composition, 0.0)
        assert np.allclose(shell.alpha, shell.alpha.T)


def test_random_alloy_reports_measured_nonzero_sro_that_depends_on_seed():
    first_spec = AlloySpec(counts=REFERENCE_COUNTS, seed=1)
    second_spec = AlloySpec(counts=REFERENCE_COUNTS, seed=2)
    first = warren_cowley_sro(generate_alloy(first_spec), first_spec.lattice_parameter)
    second = warren_cowley_sro(generate_alloy(second_spec), second_spec.lattice_parameter)
    w = first.species.index("W")

    assert np.any(first.first_shell.alpha != 0.0)
    assert not np.allclose(first.first_shell.alpha, second.first_shell.alpha)
    assert abs(first.first_shell.alpha[w, w]) < 0.05


def test_sro_is_deterministic_for_identical_inputs():
    spec = AlloySpec(counts=REFERENCE_COUNTS, seed=9)
    one = warren_cowley_sro(generate_alloy(spec), spec.lattice_parameter)
    two = warren_cowley_sro(generate_alloy(spec), spec.lattice_parameter)
    assert np.array_equal(one.first_shell.alpha, two.first_shell.alpha)
    assert np.array_equal(one.second_shell.alpha, two.second_shell.alpha)


def test_single_species_structure_has_zero_sro():
    atoms = build_supercell(AlloySpec(counts={"W": 16}, repeats=(2, 2, 2), seed=0))
    atoms.set_chemical_symbols(["W"] * 16)
    result = warren_cowley_sro(atoms, 3.16)
    assert np.allclose(result.first_shell.alpha, [[0.0]])
    assert np.allclose(result.second_shell.alpha, [[0.0]])


def test_explicit_species_order_is_respected():
    atoms, a = _b2_binary((2, 2, 2))
    result = warren_cowley_sro(atoms, a, species=("W", "Mo"))
    assert result.species == ("W", "Mo")
    assert np.allclose(result.first_shell.alpha, [[1, -1], [-1, 1]])


def test_sro_rejects_invalid_inputs():
    atoms, a = _b2_binary((2, 2, 2))
    with pytest.raises(ValueError, match="not listed"):
        warren_cowley_sro(atoms, a, species=("W",))
    with pytest.raises(ValueError, match="absent"):
        warren_cowley_sro(atoms, a, species=("Mo", "W", "Ta"))
    with pytest.raises(ValueError, match="duplicate"):
        warren_cowley_sro(atoms, a, species=("Mo", "W", "W"))
    with pytest.raises(ValueError, match="lattice parameter"):
        warren_cowley_sro(atoms, 0.0)
    with pytest.raises(ValueError, match="lattice parameter"):
        warren_cowley_sro(atoms, float("inf"))
    with pytest.raises(ValueError, match="no atoms"):
        warren_cowley_sro(atoms[:0], a)
    open_cell = atoms.copy()
    open_cell.pbc = False
    with pytest.raises(ValueError, match="periodic"):
        warren_cowley_sro(open_cell, a)


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
