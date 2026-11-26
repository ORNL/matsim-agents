import numpy as np
import pytest
from ase import Atoms
from ase.geometry import find_mic

from matsim_agents.science.defects.bcc import generate_frenkel_pair
from matsim_agents.science.defects.mapping import assign_lattice_sites
from matsim_agents.science.defects.survival import (
    cluster_defects,
    cluster_sensitivity,
    size_distribution,
    summarize_surviving_defects,
)
from matsim_agents.science.defects.wigner_seitz import wigner_seitz_analysis
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy

SPEC = AlloySpec(counts={"Mo": 18, "Ta": 18, "W": 18}, repeats=(3, 3, 3), seed=2)
A = SPEC.lattice_parameter
REF = generate_alloy(SPEC)
N = len(REF)
CUTOFF = 0.9 * A  # between the bcc first (0.866 a) and second (1.0 a) shells


def _delete(atoms, indices):
    out = atoms.copy()
    del out[list(indices)]
    return out


def _first_neighbors(site):
    d = REF.get_distances(site, range(N), mic=True)
    return [int(i) for i in np.flatnonzero((d > 0) & (d < CUTOFF))]


def _summary(current, result, **kwargs):
    arguments = {"cluster_cutoff": CUTOFF, "cluster_justification": "between bcc shells 1 and 2"}
    return summarize_surviving_defects(REF, current, result, **(arguments | kwargs))


def test_perfect_crystal_has_no_defects():
    result = wigner_seitz_analysis(REF, REF.copy())
    assert result.occupancy == (1,) * N
    assert result.n_vacancies == result.n_interstitials == result.n_frenkel_pairs == 0
    assert result.antisites == () and result.multiply_occupied_sites == ()
    assert result.translation == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_single_vacancy_and_single_interstitial():
    vacancy = wigner_seitz_analysis(REF, _delete(REF, [10]))
    assert vacancy.vacant_sites == (10,) and vacancy.n_interstitials == 0
    interstitial = REF.copy()
    interstitial.append("Ta")
    interstitial.positions[-1] = REF.positions[0] + np.array([0.5, 0.25, 0.0]) * A
    result = wigner_seitz_analysis(REF, interstitial)
    assert result.n_vacancies == 0 and result.interstitial_atoms == (N,)
    assert result.n_frenkel_pairs == 0 and len(result.multiply_occupied_sites) == 1


def test_known_frenkel_pair_is_found_and_paired():
    frenkel, _ = generate_frenkel_pair(REF, "p", 5, 0, A)
    result = wigner_seitz_analysis(REF, frenkel)
    assert result.vacant_sites == (5,) and result.interstitial_atoms == (5,)
    (pair,) = result.frenkel_pairs
    assert (pair.vacancy_site, pair.interstitial_atom) == (5, 5)
    _, expected = find_mic(frenkel.positions[5] - REF.positions[5], REF.cell, pbc=True)
    assert pair.separation == pytest.approx(float(expected), abs=1e-9) and pair.separation > 0.0


def test_two_vacancies_two_interstitials_are_paired_by_minimum_total_distance():
    hosts = {0: _first_neighbors(0)[0], 40: _first_neighbors(40)[0]}
    current = _delete(REF, [0, 40])
    for host in hosts.values():
        current.append("Mo")
        current.positions[-1] = REF.positions[host] + 0.1 * A
    result = wigner_seitz_analysis(REF, current)
    assert result.vacant_sites == (0, 40) and result.n_interstitials == 2
    first, second = N - 2, N - 1  # appended atoms sit on the hosts of sites 0 and 40
    assert {(p.vacancy_site, p.interstitial_atom) for p in result.frenkel_pairs} == {
        (0, first),
        (40, second),
    }
    positions = {first: current.positions[first], second: current.positions[second]}

    def dist(site, atom):
        return float(find_mic(positions[atom] - REF.positions[site], REF.cell, pbc=True)[1])

    chosen = dist(0, first) + dist(40, second)
    assert chosen <= dist(0, second) + dist(40, first) + 1e-9
    assert sum(p.separation for p in result.frenkel_pairs) == pytest.approx(chosen, abs=1e-9)


def test_known_vacancy_cluster_and_species_aware_split():
    symbols = REF.get_chemical_symbols()
    neighbors = _first_neighbors(0)
    same = [i for i in neighbors if symbols[i] == symbols[0]]
    other = [i for i in neighbors if symbols[i] != symbols[0]]
    assert same and other, "fixture needs like and unlike first neighbors of site 0"
    for partner, expected in ((same[0], {2: 1}), (other[0], {1: 2})):
        current = _delete(REF, [0, partner])
        result = wigner_seitz_analysis(REF, current)
        assert result.vacant_sites == tuple(sorted((0, partner)))
        assert _summary(current, result).vacancy_size_distribution == {2: 1}
        aware = _summary(current, result, species_aware=True)
        assert aware.vacancy_size_distribution == expected


def test_vacancies_across_the_periodic_boundary_form_one_cluster():
    pair = next(
        (a, b)
        for a in range(N)
        for b in _first_neighbors(a)
        if np.linalg.norm(REF.positions[a] - REF.positions[b]) > CUTOFF
    )
    current = _delete(REF, pair)
    result = wigner_seitz_analysis(REF, current)
    assert result.vacant_sites == tuple(sorted(pair))
    positions = REF.positions[list(result.vacant_sites)]
    assert size_distribution(cluster_defects(positions, REF.cell.array, CUTOFF)) == {2: 1}
    assert _summary(current, result).vacancy_size_distribution == {2: 1}


def test_rigid_translation_is_removed_and_recorded():
    reference = _delete(REF, [10])
    moved = reference.copy()
    shift = np.array([0.4, -0.3, 0.2])
    moved.positions += shift
    plain = wigner_seitz_analysis(REF, reference)
    result = wigner_seitz_analysis(REF, moved)
    assert result.vacant_sites == plain.vacant_sites == (10,)
    assert np.array(result.translation) == pytest.approx(shift, abs=1e-6)
    assert result.n_interstitials == 0


def test_homogeneous_deformation_needs_affine_mode():
    defective = _delete(REF, [10])
    strained = defective.copy()
    strained.set_cell(REF.cell.array * np.array([1.03, 0.98, 1.01]), scale_atoms=True)
    with pytest.raises(ValueError, match="cells differ"):
        wigner_seitz_analysis(REF, strained)
    result = wigner_seitz_analysis(REF, strained, cell_mode="affine")
    assert result.vacant_sites == (10,) and result.n_interstitials == 0
    assert result.max_nearest_distance < 1e-8


def test_chemical_exchange_is_an_antisite_pair_not_a_vacancy_or_interstitial():
    species = REF.get_chemical_symbols()
    i = species.index("Mo")
    j = species.index("W")
    swapped = REF.copy()
    swapped.symbols[i], swapped.symbols[j] = "W", "Mo"
    result = wigner_seitz_analysis(REF, swapped)
    assert result.n_vacancies == result.n_interstitials == 0
    assert {(s.site, s.reference_species, s.occupant_species) for s in result.antisites} == {
        (i, "Mo", "W"),
        (j, "W", "Mo"),
    }


def test_thermal_noise_below_half_spacing_changes_no_count_and_is_measured():
    rng = np.random.default_rng(0)
    noisy = _delete(REF, [10])
    noisy.positions += rng.normal(0.0, 0.05 * A, size=noisy.positions.shape)
    result = wigner_seitz_analysis(REF, noisy)
    assert result.vacant_sites == (10,) and result.n_interstitials == 0
    assert result.n_ambiguous == 0
    wild = _delete(REF, [10])
    wild.positions += rng.normal(0.0, 0.3 * A, size=wild.positions.shape)
    assert wigner_seitz_analysis(REF, wild, ambiguity_margin=0.3 * A).n_ambiguous > 0


def test_agrees_with_the_independent_site_mapping_on_counts():
    frenkel, _ = generate_frenkel_pair(REF, "p", 5, 0, A)
    cases = [REF.copy(), _delete(REF, [10]), frenkel, _delete(REF, [0, _first_neighbors(0)[0]])]
    for current in cases:
        ws = wigner_seitz_analysis(REF, current)
        mapped = assign_lattice_sites(REF, current, A)
        assert mapped.resolved
        assert set(ws.vacant_sites) == set(mapped.vacant_sites)
        assert ws.n_interstitials == len(mapped.off_lattice)


def test_brute_force_nearest_site_agrees():
    frenkel, _ = generate_frenkel_pair(REF, "p", 5, 0, A)
    result = wigner_seitz_analysis(REF, frenkel, align="none")
    for atom in range(len(frenkel)):
        vectors = frenkel.positions[atom] - REF.positions
        scaled = vectors @ np.linalg.inv(REF.cell.array)
        scaled -= np.round(scaled)
        nearest = int(np.argmin(np.linalg.norm(scaled @ REF.cell.array, axis=1)))
        assert result.atom_site[atom] == nearest


def test_substituted_gas_atom_at_a_site_is_geometrically_occupied():
    current = REF.copy()
    current.symbols[10] = "He"
    result = wigner_seitz_analysis(REF, current)
    assert result.n_vacancies == 0 and len(result.antisites) == 1
    assert result.antisites[0].occupant_species == "He"


def test_input_validation():
    with pytest.raises(ValueError, match="cell_mode"):
        wigner_seitz_analysis(REF, REF, cell_mode="bogus")
    with pytest.raises(ValueError, match="align"):
        wigner_seitz_analysis(REF, REF, align="bogus")
    open_cell = REF.copy()
    open_cell.pbc = False
    with pytest.raises(ValueError, match="periodic"):
        wigner_seitz_analysis(REF, open_cell)
    nan = REF.copy()
    nan.positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        wigner_seitz_analysis(REF, nan)
    bad_ref = REF.copy()
    bad_ref.positions[0, 0] = np.inf
    with pytest.raises(ValueError, match="reference structure has non-finite"):
        wigner_seitz_analysis(bad_ref, REF)
    with pytest.raises(ValueError, match="no atoms"):
        wigner_seitz_analysis(REF, Atoms(cell=REF.cell, pbc=True))


def test_cluster_cutoff_sensitivity_is_reported_and_not_unique():
    current = _delete(REF, [0, *_first_neighbors(0)[:1], 40])
    result = wigner_seitz_analysis(REF, current)
    positions = REF.positions[list(result.vacant_sites)]
    tight, loose = cluster_sensitivity(positions, REF.cell.array, [0.5 * A, 2.5 * A])
    assert tight.largest == 1 and tight.n_clusters == 3
    assert loose.n_clusters < tight.n_clusters
    with pytest.raises(ValueError, match="cutoff"):
        cluster_defects(positions, REF.cell.array, 0.0)
    with pytest.raises(ValueError, match="at least one"):
        cluster_sensitivity(positions, REF.cell.array, [])


def test_survival_fraction_requires_an_explicit_reference_population():
    frenkel, _ = generate_frenkel_pair(REF, "p", 5, 0, A)
    result = wigner_seitz_analysis(REF, frenkel)
    assert _summary(frenkel, result).survival_fraction is None
    summary = _summary(
        frenkel,
        result,
        reference_population=4.0,
        reference_population_definition="Frenkel pairs at the thermal-spike peak",
    )
    assert summary.survival_fraction == pytest.approx(0.25)
    assert summary.n_frenkel_pairs == 1 and "not a DPA estimate" in summary.scope
    with pytest.raises(ValueError, match="together"):
        _summary(frenkel, result, reference_population=4.0)
    with pytest.raises(ValueError, match="positive"):
        _summary(frenkel, result, reference_population=0.0, reference_population_definition="peak")
    with pytest.raises(ValueError, match="justification"):
        _summary(frenkel, result, cluster_justification=" ")


def test_summary_is_deterministic_and_records_the_cutoff():
    current = _delete(REF, [10])
    result = wigner_seitz_analysis(REF, current)
    first = _summary(current, result)
    assert first == _summary(current.copy(), wigner_seitz_analysis(REF, current.copy()))
    assert first.cluster_cutoff == CUTOFF and first.cluster_justification
    with pytest.raises(ValueError, match="does not match"):
        _summary(REF, result)
