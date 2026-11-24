"""Migration endpoint generation on small BCC cells (no calculations)."""

import numpy as np
import pytest
from ase.geometry import get_distances

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.bcc_defects import (
    generate_dumbbell_interstitial,
    generate_frenkel_pair,
)
from matsim_agents.discovery.neb_endpoints import (
    generate_frenkel_recombination,
    generate_interstitial_migration,
    generate_vacancy_hop,
    vacancy_neighbors,
)
from matsim_agents.discovery.point_defects import generate_vacancy

A = 3.16
PARENT = generate_alloy(AlloySpec(counts={"Mo": 4, "Ta": 4, "W": 8}, repeats=(2, 2, 2), seed=1))
NN = np.sqrt(3.0) / 2.0 * A


def _vacancy(site=0):
    return generate_vacancy(PARENT, "p", site)


def test_vacancy_neighbors_are_the_eight_bcc_neighbors():
    initial, config = _vacancy()
    assert len(vacancy_neighbors(initial, config, A)) == 8


def test_vacancy_hop_moves_the_neighbor_into_the_vacancy_and_keeps_identities():
    initial, config = _vacancy()
    before = initial.positions.copy()
    mover = vacancy_neighbors(initial, config, A)[0]
    pair = generate_vacancy_hop(initial, config, PARENT, mover, A)
    assert np.array_equal(initial.positions, before)  # initial is not modified
    assert np.array_equal(pair.final.get_array("atom_id"), initial.get_array("atom_id"))
    assert pair.final.get_chemical_symbols() == initial.get_chemical_symbols()
    index = int(np.flatnonzero(initial.get_array("atom_id") == mover)[0])
    moved = np.flatnonzero(np.linalg.norm(pair.final.positions - initial.positions, axis=1) > 1e-9)
    assert moved.tolist() == [index]
    _, d = get_distances(
        pair.final.positions[index], PARENT.positions[0], cell=PARENT.cell, pbc=True
    )
    assert d[0, 0] < 1e-9
    params = pair.transition.mechanism.parameters
    assert params["to_site"] == 0
    assert params["displacement_length"] == pytest.approx(NN, abs=1e-9)
    assert pair.transition.mechanism.moving_atom_ids == (mover,)
    assert pair.transition.mechanism.requires_validation
    assert pair.transition.initial.status == pair.transition.final.status == "generated"
    # the new vacancy is at the mover's former parent site, and the atom id there is that site
    new_vacancy = pair.final_config.sites[0]
    assert new_vacancy.lattice_site == params["from_site"] == new_vacancy.atom_id
    assert new_vacancy.species == PARENT[new_vacancy.lattice_site].symbol


def test_occupant_exchange_flag_follows_species():
    initial, config = _vacancy()
    vacancy_species = PARENT[0].symbol
    flags = {}
    for mover in vacancy_neighbors(initial, config, A):
        pair = generate_vacancy_hop(initial, config, PARENT, mover, A)
        species = pair.transition.mechanism.parameters["mover_species"]
        flags[mover] = (species, pair.transition.mechanism.parameters["occupant_exchange"])
    assert all(flag == (species != vacancy_species) for species, flag in flags.values())
    assert len({species for species, _ in flags.values()}) > 1  # both outcomes are exercised


def test_vacancy_hops_chain_and_are_deterministic():
    initial, config = _vacancy()
    mover = vacancy_neighbors(initial, config, A)[0]
    first = generate_vacancy_hop(initial, config, PARENT, mover, A)
    again = generate_vacancy_hop(initial, config, PARENT, mover, A)
    assert first.transition == again.transition
    nxt = vacancy_neighbors(first.final, first.final_config, A)
    assert len(nxt) == 8
    second = generate_vacancy_hop(first.final, first.final_config, PARENT, nxt[-1], A)
    assert second.transition.initial.structure_id == first.final_config.config_id


def test_vacancy_hop_rejects_bad_requests():
    initial, config = _vacancy()
    neighbors = set(vacancy_neighbors(initial, config, A))
    far = next(i for i in initial.get_array("atom_id") if int(i) not in neighbors)
    with pytest.raises(ValueError, match="not a nearest neighbour"):
        generate_vacancy_hop(initial, config, PARENT, int(far), A)
    with pytest.raises(ValueError, match="exactly once"):
        generate_vacancy_hop(initial, config, PARENT, 0, A)  # atom 0 was removed
    with pytest.raises(ValueError, match="lattice_parameter"):
        generate_vacancy_hop(initial, config, PARENT, next(iter(neighbors)), float("nan"))
    no_ids = initial.copy()
    del no_ids.arrays["atom_id"]
    with pytest.raises(ValueError, match="atom_id"):
        generate_vacancy_hop(no_ids, config, PARENT, next(iter(neighbors)), A)
    interstitial, i_config = generate_dumbbell_interstitial(PARENT, "p", 0, "W", "110", A)
    with pytest.raises(ValueError, match="needs a vacancy"):
        generate_vacancy_hop(interstitial, i_config, PARENT, 1, A)
    with pytest.raises(ValueError, match="parent has"):
        generate_vacancy_hop(initial, config, PARENT[:-2], next(iter(neighbors)), A)


def test_interstitial_migration_moves_only_the_named_atom():
    initial, config = generate_dumbbell_interstitial(PARENT, "p", 0, "W", "110", A)
    inserted = len(PARENT)
    target = initial.positions[-1] + np.array([0.0, 0.0, 0.3])
    pair = generate_interstitial_migration(
        initial, config, {inserted: tuple(target)}, min_distance=1.0
    )
    moved = np.flatnonzero(np.linalg.norm(pair.final.positions - initial.positions, axis=1) > 1e-9)
    assert moved.tolist() == [len(initial) - 1]
    mechanism = pair.transition.mechanism
    assert mechanism.kind == "interstitial_migration" and mechanism.requires_validation
    assert mechanism.parameters["max_displacement"] == pytest.approx(0.3)
    assert pair.final_config.n_atoms == config.n_atoms
    assert pair.final_config.config_id != config.config_id


def test_interstitial_migration_rejects_overlap_unknown_atoms_and_no_move():
    initial, config = generate_dumbbell_interstitial(PARENT, "p", 0, "W", "110", A)
    inserted = len(PARENT)
    with pytest.raises(ValueError, match="from another atom"):
        generate_interstitial_migration(initial, config, {inserted: tuple(initial.positions[3])})
    with pytest.raises(ValueError, match="not an interstitial or displaced"):
        generate_interstitial_migration(initial, config, {5: (0.0, 0.0, 0.0)})
    with pytest.raises(ValueError, match="finite"):
        generate_interstitial_migration(initial, config, {inserted: (np.nan, 0.0, 0.0)})
    with pytest.raises(ValueError, match="at least one"):
        generate_interstitial_migration(initial, config, {})
    with pytest.raises(ValueError, match="equals the initial"):
        generate_interstitial_migration(
            initial, config, {inserted: tuple(initial.positions[-1])}, min_distance=0.1
        )
    vacancy, v_config = _vacancy()
    with pytest.raises(ValueError, match="needs an interstitial"):
        generate_interstitial_migration(vacancy, v_config, {1: (0.0, 0.0, 0.0)})


def _frenkel():
    return generate_frenkel_pair(PARENT, "p", 0, 9, A)


def test_direct_frenkel_recombination_restores_the_lattice_ordering():
    initial, config = _frenkel()
    pair = generate_frenkel_recombination(initial, config, PARENT, A)
    assert pair.final_config is None
    params = pair.transition.mechanism.parameters
    assert params["recombination"] == "direct" and params["ordering"] == "restored"
    assert params["vacancy_atom_id"] == params["interstitial_atom_id"] == 0
    assert np.allclose(pair.final.positions, PARENT.positions)
    assert pair.final.get_chemical_symbols() == PARENT.get_chemical_symbols()
    assert pair.transition.mechanism.moving_atom_ids == (0,)


def test_exchange_recombination_records_the_ordering_it_computes():
    initial, config = _frenkel()
    mover_species = PARENT[0].symbol
    outcomes = set()
    for partner in vacancy_neighbors(
        generate_vacancy(PARENT, "p", 0)[0], generate_vacancy(PARENT, "p", 0)[1], A
    ):
        pair = generate_frenkel_recombination(initial, config, PARENT, A, exchange_with=partner)
        params = pair.transition.mechanism.parameters
        different = PARENT[partner].symbol != mover_species
        assert params["recombination"] == "exchange"
        assert params["ordering"] == ("exchanged" if different else "restored")
        assert pair.transition.mechanism.moving_atom_ids == (0, partner)
        assert len(pair.final) == len(PARENT)
        outcomes.add(params["ordering"])
    assert outcomes == {"restored", "exchanged"}  # both branches are exercised


def test_frenkel_recombination_rejects_bad_requests():
    initial, config = _frenkel()
    vac, vac_config = _vacancy()
    near = set(vacancy_neighbors(vac, vac_config, A)) | {0}
    far = next(i for i in range(len(PARENT)) if i not in near)
    with pytest.raises(ValueError, match="not a nearest neighbour"):
        generate_frenkel_recombination(initial, config, PARENT, A, exchange_with=far)
    with pytest.raises(ValueError, match="not the interstitial"):
        generate_frenkel_recombination(initial, config, PARENT, A, exchange_with=0)
    with pytest.raises(ValueError, match="exactly once"):
        generate_frenkel_recombination(initial, config, PARENT, A, exchange_with=99)
    vacancy, v_config = _vacancy()
    with pytest.raises(ValueError, match="needs a Frenkel pair"):
        generate_frenkel_recombination(vacancy, v_config, PARENT, A)
