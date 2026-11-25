"""Candidate transitions from B pathways (no barrier is invented)."""

import numpy as np
import pytest

from matsim_agents.science.defects.bcc import (
    generate_dumbbell_interstitial,
    generate_frenkel_pair,
)
from matsim_agents.science.defects.point_defects import generate_vacancy
from matsim_agents.science.kinetics.contracts import NetworkProvenance
from matsim_agents.science.kinetics.graph import NetworkError, TransitionNetwork
from matsim_agents.science.kinetics.transitions import add_candidate_transition
from matsim_agents.science.neb.endpoints import (
    generate_frenkel_recombination,
    generate_interstitial_migration,
    generate_vacancy_hop,
    vacancy_neighbors,
)
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy

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


@pytest.fixture
def custom_mechanisms():
    from matsim_agents.science.kinetics import transitions

    names = ("dimer_glide", "one_way_jump")
    transitions.register_mechanism("dimer_glide", reversible=True)
    transitions.register_mechanism("one_way_jump")
    yield names
    for name in names:
        transitions._MECHANISMS.pop(name, None)


def _as(kind):
    transition = _hop().transition
    mechanism = transition.mechanism.model_copy(update={"kind": kind})
    return transition.model_copy(update={"mechanism": mechanism})


def test_builtin_mechanisms_are_registered_with_their_policy():
    from matsim_agents.science.kinetics.transitions import get_mechanism, list_mechanisms

    assert {"vacancy_hop", "interstitial_migration", "frenkel_recombination"} <= set(
        list_mechanisms()
    )
    assert get_mechanism("vacancy_hop").reversible
    assert get_mechanism("interstitial_migration").reversible
    assert not get_mechanism("frenkel_recombination").reversible


def test_custom_reversible_mechanism_generates_a_reverse_event(custom_mechanisms):
    net = _network()
    events = add_candidate_transition(net, _as("dimer_glide"))
    assert [e.mechanism for e in events] == ["dimer_glide"] * 2
    assert net.reverse_of(events[0]) == events[1]


def test_custom_irreversible_mechanism_has_no_reverse(custom_mechanisms):
    net = _network()
    (event,) = add_candidate_transition(net, _as("one_way_jump"))
    assert event.mechanism == "one_way_jump" and net.reverse_of(event) is None


def test_custom_mechanism_round_trips_through_serialization(custom_mechanisms):
    from matsim_agents.science.kinetics.contracts import TransitionEvent

    (event, *_) = add_candidate_transition(_network(), _as("dimer_glide"))
    again = TransitionEvent.model_validate_json(event.model_dump_json())
    assert again == event and again.mechanism == "dimer_glide"


def test_duplicate_and_invalid_registrations_are_rejected(custom_mechanisms):
    from matsim_agents.science.kinetics.transitions import register_mechanism

    with pytest.raises(ValueError, match="already registered"):
        register_mechanism("dimer_glide")
    with pytest.raises(ValueError, match="already registered"):
        register_mechanism("vacancy_hop", reversible=True)
    for bad in ("", "  ", " padded "):
        with pytest.raises(ValueError, match="invalid mechanism"):
            register_mechanism(bad)


def test_unregistered_mechanism_is_rejected_when_building_the_graph():
    net = _network()
    with pytest.raises(ValueError, match="unregistered mechanism"):
        add_candidate_transition(net, _as("teleport"))
    assert not net.states


def test_consecutive_vacancy_hops_share_the_intermediate_state():
    first = _hop()
    initial, config = generate_vacancy(PARENT, "p", 0)
    second_initial = first.final
    second_config = first.final_config
    mover = vacancy_neighbors(second_initial, second_config, A)[0]
    second = generate_vacancy_hop(second_initial, second_config, PARENT, mover, A)
    net = _network()
    (f1, _) = add_candidate_transition(net, first.transition)
    (f2, _) = add_candidate_transition(net, second.transition)
    assert f1.target_id == f2.source_id
    assert net.outgoing(f1.target_id)
