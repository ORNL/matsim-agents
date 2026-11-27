import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.science.defects.local_structure import (
    centrosymmetry,
    cna_labels,
    coordination_numbers,
    label_change_fraction,
    voronoi_volumes,
)

BCC = bulk("W", "bcc", a=3.165, cubic=True).repeat((3, 3, 3))
FCC = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((3, 3, 3))


def test_coordination_of_perfect_lattices():
    assert (coordination_numbers(BCC, cutoff=3.0) == 8).all()
    assert (coordination_numbers(FCC, cutoff=3.0) == 12).all()


def test_cna_identifies_bcc_and_fcc():
    assert set(cna_labels(BCC, cutoff=3.8)) == {"bcc"}
    assert set(cna_labels(FCC, cutoff=3.0)) == {"fcc"}


def test_cna_bcc_in_cell_smaller_than_twice_the_cutoff():
    small = bulk("W", "bcc", a=3.165, cubic=True).repeat((2, 2, 2))
    assert set(cna_labels(small, cutoff=3.8)) == {"bcc"}


def test_cna_marks_vacancy_neighbours_other():
    atoms = BCC.copy()
    del atoms[0]
    labels = cna_labels(atoms, cutoff=3.8)
    assert 0 < labels.count("other") < len(atoms)
    assert label_change_fraction(labels, ["bcc"] * len(atoms)) == labels.count("other") / len(atoms)


def test_centrosymmetry_zero_in_perfect_and_positive_near_vacancy():
    assert np.allclose(centrosymmetry(BCC, n_neighbors=8), 0.0, atol=1e-8)
    assert np.allclose(centrosymmetry(FCC, n_neighbors=12), 0.0, atol=1e-8)
    atoms = BCC.copy()
    del atoms[0]
    csp = centrosymmetry(atoms, n_neighbors=8)
    assert csp.max() > 1.0
    assert (csp > 1e-6).sum() < len(atoms)


def test_voronoi_volumes_fill_the_cell():
    vols = voronoi_volumes(BCC)
    assert np.isfinite(vols).all()
    assert vols.sum() == pytest.approx(BCC.cell.volume, rel=1e-6)
    assert vols == pytest.approx(BCC.cell.volume / len(BCC), rel=1e-6)


def test_voronoi_volume_grows_around_a_vacancy():
    atoms = BCC.copy()
    del atoms[0]
    vols = voronoi_volumes(atoms)
    assert vols.sum() == pytest.approx(atoms.cell.volume, rel=1e-6)
    assert vols.max() > BCC.cell.volume / len(BCC) * 1.05


def test_voronoi_volumes_are_translation_invariant():
    atoms = BCC.copy()
    del atoms[0]
    moved = atoms.copy()
    moved.positions += np.array([7.3, -11.1, 25.0])
    assert voronoi_volumes(moved) == pytest.approx(voronoi_volumes(atoms), rel=1e-6)


def test_centrosymmetry_of_a_two_atom_bcc_cell():
    small = bulk("W", "bcc", a=3.165, cubic=True)
    assert len(small) == 2
    assert np.allclose(centrosymmetry(small, n_neighbors=8), 0.0, atol=1e-8)
    with pytest.raises(ValueError):
        centrosymmetry(small[:0], n_neighbors=8)


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        coordination_numbers(BCC, cutoff=0.0)
    with pytest.raises(ValueError):
        cna_labels(BCC, cutoff=float("nan"))
    with pytest.raises(ValueError):
        centrosymmetry(BCC, n_neighbors=7)
    with pytest.raises(ValueError):
        label_change_fraction(["bcc"], ["bcc", "fcc"])
