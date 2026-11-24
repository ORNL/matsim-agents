"""Independent validation of the Warren-Cowley SRO descriptors (A2.4).

Nothing here uses ``first_neighbor_list`` / ``second_neighbor_list`` or ASE cell lists: the
reference counts pairs from integer half-lattice offsets with modular wrap-around, and the
analytic cases come from closed-form BCC geometry and combinatorics.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest

from matsim_agents.discovery.alloys import (
    AlloySpec,
    bcc_site_count,
    build_bcc_supercell,
    generate_alloy,
    warren_cowley_sro,
)

REFERENCE_COUNTS = {"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}
FIRST_OFFSETS = list(product((-1, 1), repeat=3))
SECOND_OFFSETS = [
    tuple(s * 2 if axis == k else 0 for k in range(3)) for axis in range(3) for s in (-1, 1)
]


def _half_lattice_coordinates(atoms, repeats, a):
    box = np.array([2 * n for n in repeats])
    coords = np.rint(2 * atoms.positions / a).astype(int) % box
    return coords, box


def _reference_sro(atoms, repeats, a):
    """Warren-Cowley (counts, alpha) per shell from integer offsets on the BCC lattice."""
    coords, box = _half_lattice_coordinates(atoms, repeats, a)
    n_atoms = len(atoms)
    grid = np.full(tuple(box), -1)
    grid[tuple(coords.T)] = np.arange(n_atoms)
    assert (grid >= 0).sum() == n_atoms

    species = sorted(set(atoms.get_chemical_symbols()))
    index = np.array([species.index(s) for s in atoms.get_chemical_symbols()])
    composition = np.bincount(index, minlength=len(species)) / n_atoms

    shells = []
    for offsets in (FIRST_OFFSETS, SECOND_OFFSETS):
        counts = np.zeros((len(species), len(species)))
        for offset in offsets:
            neighbour = grid[tuple(((coords + offset) % box).T)]
            np.add.at(counts, (index, index[neighbour]), 1)
        conditional = counts / counts.sum(axis=1, keepdims=True)
        shells.append((counts, 1.0 - conditional / composition[None, :]))
    return species, shells


@pytest.mark.parametrize(
    ("counts", "repeats"),
    [
        (REFERENCE_COUNTS, (8, 8, 8)),
        ({"W": 60, "Mo": 40, "Ta": 28}, (4, 4, 4)),
        ({"W": 30, "Mo": 20, "Ta": 10, "V": 12}, (3, 3, 4)),
        ({"W": 20, "Mo": 12}, (2, 2, 4)),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_independent_integer_offset_reference(counts, repeats, seed):
    spec = AlloySpec(counts=counts, repeats=repeats, seed=seed)
    atoms = generate_alloy(spec)
    result = warren_cowley_sro(atoms, spec.lattice_parameter)
    species, shells = _reference_sro(atoms, repeats, spec.lattice_parameter)

    assert list(result.species) == species
    for measured, (ref_counts, ref_alpha) in zip(
        (result.first_shell, result.second_shell), shells, strict=True
    ):
        assert np.array_equal(measured.pair_counts, ref_counts)
        assert np.allclose(measured.alpha, ref_alpha, rtol=0, atol=1e-12)


def _layered(repeats):
    """Species by block of two (001) half-planes: floor(z_half / 2) mod 2."""
    spec = AlloySpec(
        counts={"Mo": bcc_site_count(repeats) // 2, "W": bcc_site_count(repeats) // 2},
        repeats=repeats,
    )
    atoms = build_bcc_supercell(spec)
    coords, _ = _half_lattice_coordinates(atoms, repeats, spec.lattice_parameter)
    atoms.set_chemical_symbols(["W" if (z // 2) % 2 == 0 else "Mo" for z in coords[:, 2]])
    return atoms, spec.lattice_parameter


@pytest.mark.parametrize("repeats", [(2, 2, 2), (3, 2, 4), (2, 3, 6)])
def test_layered_ordering_has_zero_first_shell_and_nonzero_second_shell_sro(repeats):
    atoms, a = _layered(repeats)
    result = warren_cowley_sro(atoms, a)

    # First shell: 4 of 8 neighbours share the block, so P(j|i) = c_j and alpha = 0.
    assert np.allclose(result.first_shell.alpha, 0.0, atol=1e-12)
    # Second shell: 4 in-plane neighbours share the block, the two (0, 0, +-2) ones do not.
    assert np.allclose(
        result.second_shell.alpha, [[-1 / 3, 1 / 3], [1 / 3, -1 / 3]], rtol=0, atol=1e-12
    )


def _unique_arrangements(counts, n_sites):
    """Every distinct assignment of the multiset ``counts`` to ``n_sites`` sites."""
    symbols = list(counts)

    def place(free, k):
        if k == len(symbols) - 1:
            yield {site: symbols[k] for site in free}
            return
        for chosen in combinations(free, counts[symbols[k]]):
            chosen_set = set(chosen)
            rest = [site for site in free if site not in chosen_set]
            head = {site: symbols[k] for site in chosen}
            for tail in place(rest, k + 1):
                yield {**head, **tail}

    for assignment in place(list(range(n_sites)), 0):
        yield [assignment[site] for site in range(n_sites)]


def _random_alloy_expectation(counts):
    """E[alpha_ij] for a uniformly random occupation with fixed counts (sorted species)."""
    species = sorted(counts)
    n_i = np.array([counts[s] for s in species], dtype=float)
    n = n_i.sum()
    expected = np.full((len(species), len(species)), -1.0 / (n - 1))
    np.fill_diagonal(expected, (n - n_i) / ((n - 1) * n_i))
    return expected


@pytest.mark.parametrize("counts", [{"Mo": 13, "W": 3}, {"Mo": 2, "Ta": 1, "W": 13}])
def test_random_alloy_sro_averages_exactly_to_finite_composition_expectation(counts):
    repeats = (2, 2, 2)
    spec = AlloySpec(counts=counts, repeats=repeats)
    atoms = build_bcc_supercell(spec)

    arrangements = list(_unique_arrangements(counts, len(atoms)))
    first_total = np.zeros((len(counts), len(counts)))
    second_total = np.zeros_like(first_total)
    for symbols in arrangements:
        atoms.set_chemical_symbols(symbols)
        result = warren_cowley_sro(atoms, spec.lattice_parameter)
        first_total += result.first_shell.alpha
        second_total += result.second_shell.alpha

    expected = _random_alloy_expectation(counts)
    assert np.allclose(first_total / len(arrangements), expected, rtol=0, atol=1e-12)
    assert np.allclose(second_total / len(arrangements), expected, rtol=0, atol=1e-12)


def test_random_expectation_is_nonzero_and_vanishes_only_in_the_large_cell_limit():
    small = _random_alloy_expectation({"Mo": 8, "W": 8})
    large = _random_alloy_expectation({"Mo": 512, "W": 512})
    assert np.allclose(small, [[1 / 15, -1 / 15], [-1 / 15, 1 / 15]])
    assert np.abs(large).max() < np.abs(small).max() / 50
