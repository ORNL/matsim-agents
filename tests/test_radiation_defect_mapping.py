import numpy as np
import pytest

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.radiation_defect_mapping import assign_lattice_sites
from matsim_agents.discovery.radiation_defects import (
    generate_bcc_interstitial,
    generate_frenkel_pair,
    generate_vacancy,
)

SPEC = AlloySpec(counts={"Mo": 18, "Ta": 18, "W": 18}, repeats=(3, 3, 3), seed=2)
A = SPEC.lattice_parameter
PARENT = generate_alloy(SPEC)
N = len(PARENT)


def test_perfect_lattice_maps_every_atom_to_its_own_site():
    result = assign_lattice_sites(PARENT, PARENT.copy(), A)
    assert result.resolved
    assert result.atom_site == tuple(range(N))
    assert set(result.status) == {"on_site"}
    assert result.vacant_sites == result.uncertain_sites == result.conflict_sites == ()
    assert max(result.nearest_distance) == pytest.approx(0.0, abs=1e-9)
    assert all(len(o) == 1 for o in result.site_occupants)


def test_small_thermal_displacements_and_wrapped_atoms_still_map_to_their_sites():
    moved = PARENT.copy()
    signs = np.where(np.arange(N)[:, None] % 2 == 0, 1.0, -1.0)
    moved.positions += 0.08 * signs * np.array([1.0, -1.0, 0.5])
    moved.positions[0] -= moved.cell[0]  # image of site 0 outside the cell
    result = assign_lattice_sites(PARENT, moved, A)
    assert result.resolved and result.atom_site == tuple(range(N))


def test_vacancy_leaves_one_vacant_site_and_no_off_lattice_atoms():
    defective, _ = generate_vacancy(PARENT, "p", 10)
    result = assign_lattice_sites(PARENT, defective, A)
    assert result.resolved and result.vacant_sites == (10,)
    assert result.off_lattice == () and result.n_atoms == N - 1
    assert result.site_occupants[10] == ()


def test_interstitial_is_off_lattice_and_frenkel_pair_has_vacancy_and_interstitial():
    inter, _ = generate_bcc_interstitial(PARENT, "p", 0, "Ta", "tetrahedral", A)
    result = assign_lattice_sites(PARENT, inter, A)
    assert result.off_lattice == (N,) and result.vacant_sites == ()
    assert result.atom_site[N] is None
    frenkel, _ = generate_frenkel_pair(PARENT, "p", 5, 0, A)
    result = assign_lattice_sites(PARENT, frenkel, A)
    assert result.resolved and result.off_lattice == (5,) and result.vacant_sites == (5,)


def test_two_atoms_claiming_one_site_are_a_conflict_not_merged():
    moved = PARENT.copy()
    moved.positions[1] = PARENT.positions[0] + 0.05
    result = assign_lattice_sites(PARENT, moved, A)
    assert not result.resolved
    assert result.conflict_sites == (0,) and result.conflicts == (0, 1)
    assert result.atom_site[0] == result.atom_site[1] == 0
    assert result.site_occupants[0] == (0, 1)
    assert 1 in result.vacant_sites


def test_atom_midway_between_two_sites_is_ambiguous_and_marks_sites_uncertain():
    moved = PARENT.copy()
    moved.positions[0] = 0.5 * (PARENT.positions[0] + PARENT.positions[1])  # midpoint, 0.43 a
    result = assign_lattice_sites(PARENT, moved, A, site_tolerance=0.5)
    assert result.ambiguous == (0,) and not result.resolved
    assert result.atom_site[0] is None
    assert result.uncertain_sites == (0,)
    assert 0 not in result.vacant_sites


def test_atom_far_from_every_site_is_off_lattice_not_forced():
    moved = PARENT.copy()
    moved.positions[0] = PARENT.positions[0] + np.array([0.5, 0.25, 0.0]) * A
    result = assign_lattice_sites(PARENT, moved, A, site_tolerance=0.3)
    assert result.off_lattice == (0,) and result.atom_site[0] is None
    assert result.vacant_sites == (0,)


def test_mapping_is_deterministic_and_validates_inputs():
    defective, _ = generate_vacancy(PARENT, "p", 10)
    assert assign_lattice_sites(PARENT, defective, A) == assign_lattice_sites(
        PARENT, defective.copy(), A
    )
    bad_cell = PARENT.copy()
    bad_cell.set_cell(PARENT.cell.array * 1.01, scale_atoms=True)
    with pytest.raises(ValueError, match="cells differ"):
        assign_lattice_sites(PARENT, bad_cell, A)
    nan = PARENT.copy()
    nan.positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        assign_lattice_sites(PARENT, nan, A)
    open_cell = PARENT.copy()
    open_cell.pbc = False
    with pytest.raises(ValueError, match="periodic"):
        assign_lattice_sites(PARENT, open_cell, A)
    for kwargs in (
        {"lattice_parameter": 0.0},
        {"site_tolerance": 0.0},
        {"site_tolerance": float("nan")},
        {"ambiguity_margin": -0.1},
    ):
        arguments = {"lattice_parameter": A} | kwargs
        with pytest.raises(ValueError):
            assign_lattice_sites(PARENT, PARENT, arguments.pop("lattice_parameter"), **arguments)
