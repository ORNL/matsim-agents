import numpy as np
import pytest
from ase.build import bulk
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor

from matsim_agents.discovery.selection import (
    CandidateSelection,
    _neighbor_fingerprint,
    select_diverse_candidates,
)


@pytest.fixture
def structures(tmp_path):
    def save(kind, name=None):
        path = tmp_path / f"{name or kind}.extxyz"
        write(path, bulk("Cu", kind, a=3.6))
        return str(path)

    return save


def test_farthest_first_is_not_generator_order(structures):
    paths = [structures(kind) for kind in ("fcc", "bcc", "sc")]
    plan = select_diverse_candidates(paths, [], formula="Cu", budget=2)
    assert plan.selected == [0, 2]
    assert [decision.disposition for decision in plan.decisions] == [
        "selected",
        "deferred",
        "selected",
    ]
    assert plan.decisions[0].reason == "first_unique_candidate"
    assert plan.decisions[2].novelty_distance_A > 1
    assert select_diverse_candidates(paths, [], formula="Cu", budget=2) == plan
    assert CandidateSelection.model_validate_json(plan.model_dump_json()) == plan


def test_prior_structures_steer_first_choice(structures):
    prior = structures("fcc")
    paths = [structures("bcc"), structures("sc")]
    plan = select_diverse_candidates(paths, [prior], formula="Cu", budget=1)
    assert plan.selected == [1]
    assert plan.references == [prior]
    assert plan.decisions[1].reason == "farthest_from_prior_and_selected"


def test_periodic_equivalence_across_cells_and_order(tmp_path, structures):
    prior = structures("fcc")
    atoms = bulk("Cu", "fcc", a=3.6).repeat((2, 1, 1))
    atoms.translate([0.75, -1.2, 0.3])
    atoms.wrap()
    atoms = atoms[::-1]
    path = tmp_path / "equivalent.extxyz"
    write(path, atoms)
    plan = select_diverse_candidates([str(path)], [prior], formula="Cu", budget=2)
    assert plan.selected == []
    assert plan.decisions[0].disposition == "duplicate"
    assert plan.decisions[0].duplicate_of == prior


def test_within_round_duplicates_have_explicit_representative(structures):
    paths = [structures("fcc"), structures("fcc", "other-name"), structures("bcc")]
    plan = select_diverse_candidates(paths, [], formula="Cu", budget=2)
    assert plan.selected == [0, 2]
    assert plan.decisions[1].duplicate_of == paths[0]


def test_matching_ignores_uniform_volume_scaling(tmp_path, structures):
    prior = structures("fcc")
    path = tmp_path / "scaled.extxyz"
    write(path, bulk("Cu", "fcc", a=4.5))
    plan = select_diverse_candidates([str(path)], [prior], formula="Cu", budget=1)
    assert plan.decisions[0].disposition == "duplicate"


def test_neighbor_descriptor_is_periodic_and_permutation_invariant():
    atoms = bulk("NaCl", "rocksalt", a=5.6)
    species = ["Cl", "Na"]
    expected = _neighbor_fingerprint(AseAtomsAdaptor.get_structure(atoms), species)
    repeated = atoms.repeat((2, 1, 1))[::-1]
    repeated.translate([0.2, 1.5, -0.5])
    actual = _neighbor_fingerprint(AseAtomsAdaptor.get_structure(repeated), species)
    assert expected.shape == (16,)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_fingerprint_ties_keep_input_order(structures, monkeypatch):
    monkeypatch.setattr(
        "matsim_agents.discovery.selection._neighbor_fingerprint",
        lambda structure, species: np.zeros(8),
    )
    paths = [structures(kind) for kind in ("fcc", "bcc", "sc")]
    plan = select_diverse_candidates(paths, [], formula="Cu", budget=2)
    assert plan.selected == [0, 1]
    assert plan.decisions[1].novelty_distance_A == 0


@pytest.mark.parametrize("invalid", ["nonperiodic", "composition", "cell"])
def test_invalid_structure_fails_explicitly(tmp_path, invalid):
    atoms = bulk("Cu", "fcc", a=3.6)
    if invalid == "nonperiodic":
        atoms.pbc = False
    elif invalid == "composition":
        atoms.set_chemical_symbols(["Al"])
    else:
        atoms.set_cell(np.zeros((3, 3)))
    path = tmp_path / "invalid.extxyz"
    write(path, atoms)
    with pytest.raises(ValueError, match="Selection"):
        select_diverse_candidates([str(path)], [], formula="Cu", budget=1)


def test_empty_batch_and_invalid_budget():
    assert select_diverse_candidates([], [], formula="Cu", budget=1).selected == []
    with pytest.raises(ValueError, match="budget"):
        select_diverse_candidates([], [], formula="Cu", budget=0)
