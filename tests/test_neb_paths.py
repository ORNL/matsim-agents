"""Identity-based correspondence and periodic displacements (small synthetic cells)."""

from itertools import product

import numpy as np
import pytest
from ase import Atoms

from matsim_agents.discovery.neb_contracts import AtomCorrespondence
from matsim_agents.discovery.neb_paths import (
    correspond_atoms,
    periodic_displacements,
)

CUBIC = np.diag([10.0, 10.0, 10.0])
SKEWED = np.array([[10.0, 0.0, 0.0], [9.9, 1.0, 0.0], [3.0, 2.0, 8.0]])


def _atoms(cell, positions, symbols=("W", "Ta", "Mo"), ids=None):
    atoms = Atoms(list(symbols), positions=positions, cell=cell, pbc=True)
    atoms.set_array("atom_id", np.array(ids if ids is not None else range(len(symbols))))
    return atoms


def _brute_force(vector, cell, n=4):
    shifts = np.array(list(product(range(-n, n + 1), repeat=3))) @ cell
    images = vector + shifts
    return images[np.argmin(np.linalg.norm(images, axis=1))]


def test_correspondence_follows_identity_not_index():
    initial = _atoms(CUBIC, [[1, 1, 1], [3, 3, 3], [5, 5, 5]])
    final = _atoms(CUBIC, [[5, 5, 5.5], [1, 1, 1.5], [3, 3, 3.5]], ("Mo", "W", "Ta"), ids=[2, 0, 1])
    corr = correspond_atoms(initial, final)
    assert corr.atom_ids == (0, 1, 2)
    assert corr.final_indices == (1, 2, 0)
    disp = periodic_displacements(initial, final)
    assert np.allclose(disp.vectors, [[0, 0, 0.5]] * 3)
    assert disp.species == ("W", "Ta", "Mo")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("species", "species of atom 1"),
        ("ids", "same atomic identities"),
        ("duplicate", "duplicate atom_id"),
        ("count", "atom counts differ"),
        ("cell", "cells differ"),
        ("pbc", "fully periodic"),
        ("nan", "non-finite"),
        ("noid", "no atom_id"),
    ],
)
def test_invalid_pairs_are_rejected(change, match):
    initial = _atoms(CUBIC, [[1, 1, 1], [3, 3, 3], [5, 5, 5]])
    final = initial.copy()
    if change == "species":
        final.symbols[1] = "Hf"
    elif change == "ids":
        final.set_array("atom_id", np.array([0, 1, 7]))
    elif change == "duplicate":
        final.set_array("atom_id", np.array([0, 1, 1]))
    elif change == "count":
        final = final[:2]
    elif change == "cell":
        final.set_cell(CUBIC * 1.001)
    elif change == "pbc":
        final.pbc = (True, True, False)
    elif change == "nan":
        final.positions[0, 0] = np.nan
    else:
        del final.arrays["atom_id"]
    with pytest.raises(ValueError, match=match):
        correspond_atoms(initial, final)


def test_wrapped_final_gives_the_short_displacement():
    initial = _atoms(CUBIC, [[0.2, 5, 5], [3, 3, 3], [5, 5, 5]])
    final = initial.copy()
    final.positions[0] = [9.9, 5, 5]  # wrapped across the boundary: true move is -0.3
    final.positions[1] += CUBIC[0] * 3  # a whole-cell shift is no displacement
    disp = periodic_displacements(initial, final)
    assert np.allclose(disp.vectors[0], [-0.3, 0, 0])
    assert np.allclose(disp.vectors[1], 0.0)
    assert np.allclose(disp.unwrapped_final[0], [-0.1, 5, 5])  # unwrapped, outside the cell
    assert disp.max_length == pytest.approx(0.3)


def test_skewed_cell_matches_brute_force_where_rounding_fails():
    inverse = np.linalg.inv(SKEWED)
    naive_failures = 0
    for fx, fy, fz in product(np.linspace(-0.45, 0.45, 7), repeat=3):
        raw = np.array([fx, fy, fz]) @ SKEWED * 1.0
        initial = _atoms(SKEWED, [[0.0, 0.0, 0.0], [4, 4, 4]], ("W", "Ta"))
        final = initial.copy()
        final.positions[0] = raw
        try:
            disp = periodic_displacements(initial, final)
        except ValueError:  # an exact tie; covered by the ambiguity test
            continue
        best = _brute_force(raw, SKEWED)
        assert np.linalg.norm(disp.vectors[0]) == pytest.approx(np.linalg.norm(best), abs=1e-9)
        fractional = raw @ inverse
        naive = (fractional - np.round(fractional)) @ SKEWED
        if np.linalg.norm(naive) > np.linalg.norm(best) + 1e-9:
            naive_failures += 1
    assert naive_failures > 0  # the fixture really distinguishes true MIC from rounding


def test_minimum_image_is_exact_on_a_skewed_cell_where_ase_find_mic_is_not():
    from ase.geometry import find_mic

    from matsim_agents.discovery.neb_paths import minimum_image

    vectors = np.array(
        [np.array(f) @ SKEWED for f in product(np.linspace(-0.45, 0.45, 7), repeat=3)]
    )
    ours, lengths, _ = minimum_image(vectors, SKEWED)
    exact = np.array([np.linalg.norm(_brute_force(v, SKEWED, n=6)) for v in vectors])
    assert np.allclose(lengths, exact, atol=1e-9)
    assert np.allclose(np.linalg.norm(ours, axis=1), lengths)
    single = np.array([find_mic(v[None, :], SKEWED, True)[1][0] for v in vectors])
    # Documents why ASE's routine is not relied on: single-vector calls miss the shortest image.
    assert (single > exact + 1e-9).any()


def test_ambiguous_half_cell_displacement_is_detected():
    initial = _atoms(CUBIC, [[1, 1, 1], [3, 3, 3], [5, 5, 5]])
    final = initial.copy()
    final.positions[2, 0] += 5.0  # exactly half a cell vector: two equally short images
    with pytest.raises(ValueError, match="ambiguous periodic displacement for atom_id \\[2\\]"):
        periodic_displacements(initial, final)
    disp = periodic_displacements(initial, final, allow_ambiguous=True)
    assert disp.ambiguous_atom_ids == (2,)
    assert disp.lengths[2] == pytest.approx(5.0)
    near = initial.copy()
    near.positions[2, 0] += 4.9  # clearly shorter the other way: not ambiguous
    assert periodic_displacements(initial, near).ambiguous_atom_ids == ()


def test_explicit_correspondence_must_match_the_structures():
    initial = _atoms(CUBIC, [[1, 1, 1], [3, 3, 3], [5, 5, 5]])
    final = initial.copy()
    good = correspond_atoms(initial, final)
    assert periodic_displacements(initial, final, good).max_length == 0.0
    bad = AtomCorrespondence(
        atom_ids=good.atom_ids,
        species=good.species,
        initial_indices=good.initial_indices,
        final_indices=(1, 0, 2),
    )
    with pytest.raises(ValueError, match="does not match"):
        periodic_displacements(initial, final, bad)
    with pytest.raises(ValueError, match="ambiguity_tolerance"):
        periodic_displacements(initial, final, ambiguity_tolerance=-1.0)
