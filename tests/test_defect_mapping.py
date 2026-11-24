import numpy as np
import pytest

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.bcc_defects import generate_bcc_interstitial, generate_frenkel_pair
from matsim_agents.discovery.defect_mapping import (
    assign_lattice_sites,
    identify_defect_sites,
)
from matsim_agents.discovery.point_defects import generate_vacancy

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


def _identify(relaxed, **kwargs):
    assignment = assign_lattice_sites(PARENT, relaxed, A, **kwargs)
    return identify_defect_sites(PARENT, relaxed, assignment)


def test_perfect_lattice_has_no_defect_sites():
    found = _identify(PARENT.copy())
    assert found.resolved and found.final_sites == () and found.antisites == ()
    assert found.unresolved_atoms == () and found.uncertain_sites == ()


def test_vacancy_and_frenkel_pair_sites_match_the_generated_configuration():
    defective, config = generate_vacancy(PARENT, "p", 10)
    found = _identify(defective)
    assert found.vacancies == config.sites and found.interstitials == ()
    frenkel, config = generate_frenkel_pair(PARENT, "p", 5, 0, A)
    found = _identify(frenkel)
    (vacancy,), (interstitial,) = found.vacancies, found.interstitials
    expected_vacancy, expected_interstitial = config.sites
    assert vacancy == expected_vacancy
    assert (interstitial.atom_id, interstitial.species) == (5, expected_interstitial.species)
    assert interstitial.position == expected_interstitial.position
    assert found.final_sites == (vacancy, interstitial)


def test_swapped_species_are_antisites_with_no_vacancy_or_interstitial():
    swapped = PARENT.copy()
    i, j = 0, next(k for k, s in enumerate(PARENT.symbols) if s != PARENT.symbols[0])
    symbols = list(PARENT.symbols)
    symbols[i], symbols[j] = symbols[j], symbols[i]
    swapped.set_chemical_symbols(symbols)
    found = _identify(swapped)
    assert found.final_sites == () and found.resolved
    assert {(a.lattice_site, a.parent_species, a.occupant_species) for a in found.antisites} == {
        (i, PARENT.symbols[i], PARENT.symbols[j]),
        (j, PARENT.symbols[j], PARENT.symbols[i]),
    }


def test_ambiguous_and_conflicting_atoms_are_left_unresolved_not_classified():
    moved = PARENT.copy()
    moved.positions[0] = 0.5 * (PARENT.positions[0] + PARENT.positions[1])
    found = _identify(moved, site_tolerance=0.5)
    assert not found.resolved and found.unresolved_atoms == (0,)
    assert found.uncertain_sites == (0,) and found.interstitials == ()
    assert all(site.lattice_site != 0 for site in found.vacancies)
    clash = PARENT.copy()
    clash.positions[1] = PARENT.positions[0] + 0.05
    found = _identify(clash)
    assert not found.resolved and found.unresolved_atoms == (0, 1)
    assert found.antisites == ()


def test_identification_rejects_a_mismatched_assignment():
    defective, _ = generate_vacancy(PARENT, "p", 10)
    assignment = assign_lattice_sites(PARENT, defective, A)
    with pytest.raises(ValueError, match="does not match"):
        identify_defect_sites(PARENT, PARENT, assignment)
