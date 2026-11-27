import numpy as np
import pytest

from matsim_agents.science.defects.gas import (
    GasComplexSpec,
    GasPlacement,
    generate_gas_complex,
)
from matsim_agents.science.defects.survival import cluster_defects
from matsim_agents.science.defects.wigner_seitz import wigner_seitz_analysis
from matsim_agents.science.structures.alloys import AlloySpec, first_neighbor_list, generate_alloy

A = 3.165
GAS = ("H", "He")


@pytest.fixture(scope="module")
def parent():
    spec = AlloySpec(counts={"W": 100, "Ta": 28}, repeats=(4, 4, 4), seed=1, lattice_parameter=A)
    return generate_alloy(spec)


def _neighbor(parent, site):
    i, j, _ = first_neighbor_list(parent, A)
    return sorted(int(b) for a, b in zip(i, j, strict=True) if a == site)[0]


def _complex(parent, vacancies, gas):
    spec = GasComplexSpec(vacancy_sites=tuple(vacancies), gas=tuple(gas), lattice_parameter=A)
    atoms, _ = generate_gas_complex(parent, "p", spec)
    return atoms


def _host_only(atoms):
    return atoms[[a.index for a in atoms if a.symbol not in GAS]]


def _gp(species, placement, anchor=None):
    return GasPlacement(species=species, placement=placement, anchor_site=anchor)


def test_host_sublattice_recovers_the_vacancies_of_a_known_complex(parent):
    n = _neighbor(parent, 5)
    atoms = _complex(
        parent,
        [5, n],
        [_gp("He", "vacancy", 5), _gp("He", "vacancy", n), _gp("H", "cluster_centroid")],
    )
    assert sum(a.symbol in GAS for a in atoms) == 3
    result = wigner_seitz_analysis(parent, _host_only(atoms))
    assert result.vacant_sites == tuple(sorted((5, n)))
    assert result.n_interstitials == 0 and result.antisites == ()
    positions = parent.positions[list(result.vacant_sites)]
    (cluster,) = cluster_defects(positions, parent.cell.array, 0.9 * A)
    assert sorted(cluster) == [0, 1]  # the two vacancies are first neighbors


def test_raw_analysis_hides_the_vacancy_when_gas_sits_in_it(parent):
    atoms = _complex(parent, [5], [_gp("He", "vacancy", 5)])
    result = wigner_seitz_analysis(parent, atoms)
    assert result.n_vacancies == 0 and result.n_interstitials == 0
    assert [(s.site, s.occupant_species) for s in result.antisites] == [(5, "He")]
    assert wigner_seitz_analysis(parent, _host_only(atoms)).vacant_sites == (5,)


def test_gas_interstitial_leaves_the_host_sublattice_intact(parent):
    atoms = _complex(parent, [], [_gp("He", "tetrahedral", 7)])
    raw = wigner_seitz_analysis(parent, atoms)
    assert raw.n_vacancies == 0 and raw.n_interstitials == 1
    assert atoms[raw.interstitial_atoms[0]].symbol == "He"
    host = wigner_seitz_analysis(parent, _host_only(atoms))
    assert host.n_vacancies == host.n_interstitials == 0 and host.antisites == ()


def test_each_vacancy_is_found_for_several_cavity_gas_counts(parent):
    for count in (1, 2, 4):
        gas = [_gp("He", "vacancy", 12) for _ in range(count)]
        atoms = _complex(parent, [12], gas)
        assert np.count_nonzero(atoms.symbols == "He") == count
        assert wigner_seitz_analysis(parent, _host_only(atoms)).vacant_sites == (12,)
