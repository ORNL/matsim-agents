"""B1.4: linear periodic image generation (small cells)."""

import numpy as np
import pytest
from ase import Atoms

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.neb_endpoints import generate_vacancy_hop, vacancy_neighbors
from matsim_agents.discovery.neb_images import generate_linear_images
from matsim_agents.discovery.radiation_defects import generate_vacancy

A = 3.16
PARENT = generate_alloy(AlloySpec(counts={"Mo": 4, "Ta": 4, "W": 8}, repeats=(2, 2, 2), seed=1))


def _hop():
    initial, config = generate_vacancy(PARENT, "p", 0)
    mover = vacancy_neighbors(initial, config, A)[0]
    return generate_vacancy_hop(initial, config, PARENT, mover, A)


def test_images_are_deterministic_ordered_and_endpoints_are_fixed():
    pair = _hop()
    first = generate_linear_images(pair.initial, pair.final, 7)
    again = generate_linear_images(pair.initial, pair.final, 7)
    assert len(first.images) == 7
    assert first.fingerprints == again.fingerprints
    assert np.array_equal(first.images[0].positions, pair.initial.positions)
    assert np.allclose(first.images[-1].positions, pair.final.positions)
    for image in first.images:
        assert image.get_chemical_symbols() == pair.initial.get_chemical_symbols()
        assert np.array_equal(image.get_array("atom_id"), pair.initial.get_array("atom_id"))
        assert np.array_equal(image.cell.array, pair.initial.cell.array)
    shift = np.linalg.norm(pair.final.positions - pair.initial.positions, axis=1)
    mover = int(np.flatnonzero(shift > 0)[0])
    track = [im.positions[mover] for im in first.images]
    spacing = np.linalg.norm(np.diff(track, axis=0), axis=1)
    assert np.allclose(spacing, spacing[0])
    assert first.max_step == pytest.approx(spacing[0])
    assert first.parameters["method"] == "linear_periodic"


def test_inputs_are_not_modified():
    pair = _hop()
    i0, f0 = pair.initial.positions.copy(), pair.final.positions.copy()
    generate_linear_images(pair.initial, pair.final, 5)
    assert np.array_equal(pair.initial.positions, i0) and np.array_equal(pair.final.positions, f0)


def test_atom_crossing_the_cell_boundary_takes_the_short_route():
    initial = Atoms("W2", positions=[[0.2, 5, 5], [5, 5, 5]], cell=np.eye(3) * 10.0, pbc=True)
    initial.set_array("atom_id", np.array([0, 1]))
    final = initial.copy()
    final.positions[0] = [9.8, 5, 5]  # wrapped: the true move is -0.4
    images = generate_linear_images(initial, final, 5)
    xs = [im.positions[0, 0] for im in images.images]
    assert np.allclose(xs, [0.2, 0.1, 0.0, -0.1, -0.2])  # unwrapped, continuous
    assert images.displacements.max_length == pytest.approx(0.4)


def test_reordered_final_gives_identity_matched_images():
    pair = _hop()
    reordered = pair.final[list(reversed(range(len(pair.final))))]
    images = generate_linear_images(pair.initial, reordered, 5)
    straight = generate_linear_images(pair.initial, pair.final, 5)
    assert images.fingerprints == straight.fingerprints


@pytest.mark.parametrize("n", [0, 1, 3, -2])
def test_too_few_images_are_rejected(n):
    pair = _hop()
    with pytest.raises(ValueError, match="at least 4"):
        generate_linear_images(pair.initial, pair.final, n)


@pytest.mark.parametrize("n", [5.0, True, "5", None])
def test_non_integer_image_counts_are_rejected(n):
    pair = _hop()
    with pytest.raises(ValueError, match="integer"):
        generate_linear_images(pair.initial, pair.final, n)


def test_large_jump_and_overlap_are_rejected():
    pair = _hop()
    with pytest.raises(ValueError, match="exceeds"):
        generate_linear_images(pair.initial, pair.final, 4, max_step=0.2)
    initial = Atoms("W2", positions=[[1, 1, 1], [3.5, 1, 1]], cell=np.eye(3) * 10.0, pbc=True)
    initial.set_array("atom_id", np.array([0, 1]))
    final = initial.copy()
    final.positions[1] = [1.4, 1, 1]  # passes through the other atom's neighbourhood
    with pytest.raises(ValueError, match="apart"):
        generate_linear_images(initial, final, 6, max_step=5.0)
    with pytest.raises(ValueError, match="min_distance"):
        generate_linear_images(pair.initial, pair.final, 5, min_distance=0.0)
    with pytest.raises(ValueError, match="max_step"):
        generate_linear_images(pair.initial, pair.final, 5, max_step=float("nan"))


def test_broken_correspondence_and_invalid_inputs_are_rejected():
    pair = _hop()
    bad = pair.final.copy()
    bad.symbols[0] = "Hf"
    with pytest.raises(ValueError, match="species of atom"):
        generate_linear_images(pair.initial, bad, 5)
    nan = pair.final.copy()
    nan.positions[2, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        generate_linear_images(pair.initial, nan, 5)
    scaled = pair.final.copy()
    scaled.set_cell(pair.final.cell.array * 1.01)
    with pytest.raises(ValueError, match="cells differ"):
        generate_linear_images(pair.initial, scaled, 5)
    flat = pair.initial.copy()
    flat.set_cell([[10, 0, 0], [0, 10, 0], [0, 0, 0]])
    with pytest.raises(ValueError):
        generate_linear_images(flat, flat.copy(), 5)


def test_ambiguous_displacement_is_rejected_unless_allowed():
    initial = Atoms("W2", positions=[[1, 1, 1], [5, 5, 5]], cell=np.eye(3) * 10.0, pbc=True)
    initial.set_array("atom_id", np.array([0, 1]))
    final = initial.copy()
    final.positions[0, 0] += 5.0
    with pytest.raises(ValueError, match="ambiguous"):
        generate_linear_images(initial, final, 20, max_step=1.0)
    allowed = generate_linear_images(initial, final, 20, max_step=1.0, allow_ambiguous=True)
    assert allowed.displacements.ambiguous_atom_ids == (0,)


def test_identical_endpoints_produce_identical_images():
    pair = _hop()
    images = generate_linear_images(pair.initial, pair.initial.copy(), 4)
    assert len(set(images.fingerprints)) == 1 and images.max_step == 0.0
