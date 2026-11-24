"""C1.3: candidate transitions from B pathways (no barrier is invented)."""

import numpy as np
import pytest

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
from matsim_agents.kinetics.contracts import NetworkProvenance
from matsim_agents.kinetics.graph import NetworkError, TransitionNetwork
from matsim_agents.kinetics.transitions import add_candidate_transition

A = 3.16
PARENT = generate_alloy(AlloySpec(counts={"Mo": 4, "Ta": 4, "W": 8}, repeats=(2, 2, 2), seed=1))


def _network():
    ordering = _hop().transition.initial.parent_ordering_id
    prov = NetworkProvenance(
        network_id="n", parent_id="p", parent_ordering_id=ordering, synthetic=True
    )
    return TransitionNetwork(prov)


def _hop():
    initial, config = generate_vacancy(PARENT, "p", 0)
    mover = vacancy_neighbors(initial, config, A)[0]
    return generate_vacancy_hop(initial, config, PARENT, mover, A)


def test_vacancy_hop_gives_candidate_forward_and_reverse_edges():
    net = _network()
    forward, back = add_candidate_transition(net, _hop().transition)
    assert forward.mechanism == back.mechanism == "vacancy_hop"
    assert (forward.source_id, forward.target_id) == (back.target_id, back.source_id)
    assert forward.qualification == "candidate" and forward.barrier_eV is None
    assert net.reverse_of(forward) == back
    assert len(net.states) == 2 and not any(s.absorbing for s in net.states)


def test_interstitial_migration_is_a_reversible_candidate():
    initial, config = generate_dumbbell_interstitial(PARENT, "p", 0, "W", "110", A)
    target = initial.positions[-1] + np.array([0.0, 0.0, 0.3])
    pair = generate_interstitial_migration(
        initial, config, {len(PARENT): tuple(target)}, min_distance=1.0
    )
    events = add_candidate_transition(_network(), pair.transition)
    assert [e.mechanism for e in events] == ["interstitial_migration"] * 2


def test_frenkel_recombination_ends_in_an_absorbing_state_without_reverse():
    initial, config = generate_frenkel_pair(PARENT, "p", 0, 9, A)
    pair = generate_frenkel_recombination(initial, config, PARENT, A)
    net = _network()
    (event,) = add_candidate_transition(net, pair.transition)
    target = net.state(event.target_id)
    assert target.absorbing and target.absorbing_kind == "recombined"
    assert event.provenance["recombination"] == "direct"
    assert net.outgoing(event.target_id) == ()


def test_adding_the_same_pathway_twice_is_a_duplicate_error():
    net = _network()
    transition = _hop().transition
    add_candidate_transition(net, transition)
    with pytest.raises(NetworkError, match="duplicate"):
        add_candidate_transition(net, transition)


def test_pathway_from_another_parent_is_rejected():
    net = TransitionNetwork(NetworkProvenance(network_id="n", parent_id="other"))
    with pytest.raises(NetworkError, match="another parent"):
        add_candidate_transition(net, _hop().transition)
