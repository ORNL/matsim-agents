from __future__ import annotations

import numpy as np
import pytest
from ase.geometry import get_distances

from matsim_agents.science.defects.gas import (
    GasComplexConfiguration,
    GasComplexSpec,
    GasDefectError,
    GasPlacement,
    complex_notation,
    generate_gas_complex,
)
from matsim_agents.science.structures.alloys import AlloySpec, first_neighbor_list, generate_alloy

A = 3.165


@pytest.fixture(scope="module")
def parent():
    spec = AlloySpec(counts={"W": 100, "Ta": 28}, repeats=(4, 4, 4), seed=1, lattice_parameter=A)
    return generate_alloy(spec)


def neighbors(parent, site):
    i, j, _ = first_neighbor_list(parent, A)
    return sorted(int(b) for a, b in zip(i, j, strict=True) if a == site)


def spec(vac=(), gas=()):
    return GasComplexSpec(vacancy_sites=tuple(vac), gas=tuple(gas), lattice_parameter=A)


def gp(species, placement, anchor=None, variant=0):
    return GasPlacement(species=species, placement=placement, anchor_site=anchor, variant=variant)


def check_invariants(parent, atoms, cfg):
    n, m, p = cfg.n_vacancies, cfg.n_he, cfg.n_h
    assert len(atoms) == len(parent) - n + m + p == cfg.n_atoms
    ids = atoms.get_array("atom_id")
    assert len(set(ids.tolist())) == len(atoms)
    removed = {v.atom_id for v in cfg.vacancies}
    survivors = [int(i) for i in ids if i < len(parent)]
    assert set(survivors) == set(range(len(parent))) - removed
    for k, i in enumerate(ids):
        if i < len(parent):
            assert atoms[k].symbol == parent[int(i)].symbol
    gas_ids = [g.atom_id for g in cfg.gas_atoms]
    assert all(g >= len(parent) for g in gas_ids) and not removed & set(gas_ids)
    assert len(parent) == 128  # parent untouched


@pytest.mark.parametrize("site_type", ["tetrahedral", "octahedral"])
@pytest.mark.parametrize("species", ["H", "He"])
def test_isolated_interstitial(parent, site_type, species):
    atoms, cfg = generate_gas_complex(parent, "p", spec(gas=[gp(species, site_type, 5)]))
    check_invariants(parent, atoms, cfg)
    assert cfg.notation == f"{species}1" and cfg.n_vacancies == 0
    assert len(atoms) == 129


@pytest.mark.parametrize(("species", "n_he", "n_h"), [("H", 0, 1), ("He", 1, 0)])
def test_single_gas_in_vacancy_keeps_site_count(parent, species, n_he, n_h):
    atoms, cfg = generate_gas_complex(parent, "p", spec([10], [gp(species, "vacancy", 10)]))
    check_invariants(parent, atoms, cfg)
    assert len(atoms) == 128 and (cfg.n_he, cfg.n_h) == (n_he, n_h)
    assert cfg.gas_atoms[0].trapped_vacancy_site == 10
    np.testing.assert_allclose(cfg.gas_atoms[0].position, parent.positions[10], atol=1e-9)
    assert cfg.gas_atoms[0].atom_id == 128 and cfg.vacancies[0].atom_id == 10


def test_multiple_gas_in_one_vacancy_and_mixed(parent):
    gas = [gp("He", "vacancy", 10), gp("He", "vacancy", 10), gp("H", "vacancy", 10)]
    atoms, cfg = generate_gas_complex(parent, "p", spec([10], gas))
    check_invariants(parent, atoms, cfg)
    assert cfg.notation == "V1He2H1" and len(atoms) == 130
    assert [g.atom_id for g in cfg.gas_atoms] == [128, 129, 130]
    pos = np.array([g.position for g in cfg.gas_atoms])
    _, d = get_distances(pos, cell=parent.cell, pbc=True)
    assert d[np.triu_indices(3, 1)].min() >= 0.6


def test_vacancy_cluster_with_gas(parent):
    nb = neighbors(parent, 20)
    sites = [20, nb[0], nb[1]]
    gas = [gp("He", "cluster_centroid"), gp("H", "cluster_centroid"), gp("He", "vacancy", 20)]
    atoms, cfg = generate_gas_complex(parent, "p", spec(sites, gas))
    check_invariants(parent, atoms, cfg)
    assert cfg.notation == "V3He2H1" and len(atoms) == 128 - 3 + 3


def test_divacancy_with_interstitial_gas_elsewhere(parent):
    nb = neighbors(parent, 20)
    atoms, cfg = generate_gas_complex(
        parent, "p", spec([20, nb[0]], [gp("He", "tetrahedral", 90), gp("H", "octahedral", 91)])
    )
    check_invariants(parent, atoms, cfg)
    assert cfg.notation == "V2He1H1" and len(atoms) == 128


def test_n_atoms_for_1024_parent_examples():
    big = generate_alloy(
        AlloySpec(
            counts={"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64},
            repeats=(8, 8, 8),
            seed=1,
            lattice_parameter=A,
        )
    )
    cases = {
        ((), (gp("H", "tetrahedral", 3),)): 1025,
        ((7,), ()): 1023,
        ((7,), (gp("He", "vacancy", 7),)): 1024,
        ((7,), (gp("He", "vacancy", 7), gp("He", "vacancy", 7))): 1025,
        ((7,), (gp("He", "vacancy", 7), gp("H", "vacancy", 7))): 1025,
    }
    for (vac, gas), expected in cases.items():
        atoms, cfg = generate_gas_complex(big, "p", spec(vac, gas))
        assert len(atoms) == cfg.n_atoms == expected
        assert cfg.n_parent_atoms == 1024
    nb = [b for b in range(1024) if b != 7][0]
    atoms, _ = generate_gas_complex(
        big, "p", spec([7, nb], [gp("He", "cluster_centroid"), gp("He", "cluster_centroid")])
    )
    assert len(atoms) == 1024


def test_deterministic_and_serializable(parent):
    s = spec([10], [gp("He", "vacancy", 10), gp("H", "tetrahedral", 50, variant=3)])
    a1, c1 = generate_gas_complex(parent, "p", s)
    a2, c2 = generate_gas_complex(parent, "p", s)
    assert c1 == c2 and c1.geometry_fingerprint == c2.geometry_fingerprint
    assert np.array_equal(a1.positions, a2.positions)
    assert GasComplexConfiguration.model_validate_json(c1.model_dump_json()) == c1
    _, c3 = generate_gas_complex(parent, "p", spec([10], [gp("He", "vacancy", 10)]))
    assert c3.config_id != c1.config_id
    assert "not asserted" in c1.validation_notes[0]


def test_gas_ids_never_reuse_removed_ids(parent):
    _, cfg = generate_gas_complex(parent, "p", spec([0, 1], [gp("He", "vacancy", 0)]))
    assert {v.atom_id for v in cfg.vacancies} == {0, 1}
    assert cfg.gas_atoms[0].atom_id == 128


def test_input_parent_with_prior_ids_is_not_mutated(parent):
    before = parent.positions.copy()
    generate_gas_complex(parent, "p", spec([10], [gp("He", "vacancy", 10)]))
    assert np.array_equal(parent.positions, before) and "atom_id" not in parent.arrays


def test_rejections(parent):
    with pytest.raises(GasDefectError, match="outside"):
        generate_gas_complex(parent, "p", spec([500]))
    with pytest.raises(GasDefectError, match="not a vacancy"):
        generate_gas_complex(parent, "p", spec([10], [gp("He", "vacancy", 11)]))
    with pytest.raises(GasDefectError, match="vacated"):
        generate_gas_complex(parent, "p", spec([10], [gp("He", "tetrahedral", 10)]))
    with pytest.raises(GasDefectError, match="needs vacancies"):
        generate_gas_complex(parent, "p", spec(gas=[gp("He", "cluster_centroid")]))
    with pytest.raises(GasDefectError, match="outside"):
        generate_gas_complex(parent, "p", spec(gas=[gp("He", "tetrahedral", 999)]))
    with pytest.raises(GasDefectError, match="variant"):
        generate_gas_complex(parent, "p", spec(gas=[gp("He", "tetrahedral", 3, variant=99)]))
    with pytest.raises(ValueError, match="gas species"):
        gp("Ar", "vacancy", 1)
    with pytest.raises(ValueError, match="anchor_site"):
        gp("He", "tetrahedral")
    with pytest.raises(ValueError, match="at least one"):
        spec()
    with pytest.raises(ValueError, match="distinct"):
        spec([1, 1])


def test_overlap_rejections(parent):
    # two He into the same tetrahedral site -> gas-gas overlap
    with pytest.raises(GasDefectError, match="apart"):
        generate_gas_complex(
            parent, "p", spec(gas=[gp("He", "tetrahedral", 5), gp("H", "tetrahedral", 5)])
        )
    # two gas atoms at a vacancy centre closer than the lattice allows (default layout is OK, so
    # force a lattice atom overlap with a huge cavity request by using a tiny lattice atom gap)
    with pytest.raises(GasDefectError, match="overlap"):
        generate_gas_complex(
            parent,
            "p",
            spec([10], [gp("He", "vacancy", 10)]),
            min_distance=5.0,
        )


def test_non_bcc_parent_rejected_for_interstitial_sites(parent):
    bad = parent.copy()
    bad.rattle(stdev=0.5, seed=1)
    with pytest.raises(GasDefectError, match="not BCC"):
        generate_gas_complex(bad, "p", spec(gas=[gp("He", "tetrahedral", 5)]))


def test_notation():
    assert complex_notation(0, 0, 1) == "H1"
    assert complex_notation(2, 3, 0) == "V2He3"
    assert complex_notation(1, 2, 1) == "V1He2H1"
