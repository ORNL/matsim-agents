"""Transition graph (synthetic states)."""

import pytest

from matsim_agents.kinetics.contracts import DefectState, NetworkProvenance, TransitionEvent
from matsim_agents.kinetics.graph import NetworkError, TransitionNetwork, state_id, transition_id

PROV = NetworkProvenance(network_id="n", parent_id="p", parent_ordering_id="o1", synthetic=True)
SPECIES = {"W": 14, "Mo": 2}


def _state(label, **over):
    base = {
        "state_id": state_id("p", "o1", None, label),
        "parent_id": "p",
        "parent_ordering_id": "o1",
        "n_atoms": 16,
        "atom_ids": tuple(range(16)),
        "species_counts": dict(SPECIES),
        "provenance": {"origin": "fixture"},
    }
    base.update(over)
    return DefectState(**base)


def _event(src, dst, atoms=(3,), mechanism="vacancy_hop", **over):
    base = {
        "transition_id": transition_id(src.state_id, dst.state_id, mechanism, atoms),
        "source_id": src.state_id,
        "target_id": dst.state_id,
        "mechanism": mechanism,
        "moving_atom_ids": atoms,
        "provenance": {"origin": "fixture"},
    }
    base.update(over)
    return TransitionEvent(**base)


def _three():
    net = TransitionNetwork(PROV)
    a, b = _state("a"), _state("b")
    done = _state("done", absorbing=True, absorbing_kind="recombined")
    for s in (a, b, done):
        net.add_state(s)
    return net, a, b, done


def test_ids_are_deterministic_and_context_dependent():
    assert state_id("p", "o1", None, "a") == state_id("p", "o1", None, "a")
    assert state_id("p", "o1", None, "a") != state_id("p", "o2", None, "a")
    assert transition_id("a", "b", "vacancy_hop", (2, 1)) == transition_id(
        "a", "b", "vacancy_hop", (1, 2)
    )


def test_three_state_graph_with_explicit_reverse_and_absorbing_state():
    net, a, b, done = _three()
    forward = _event(a, b)
    net.add_transition(forward)
    assert net.reverse_of(forward) is None
    back = _event(b, a)
    net.add_transition(back)
    net.add_transition(_event(b, done, atoms=(5,), mechanism="frenkel_recombination"))
    assert net.reverse_of(forward) == back
    assert [e.target_id for e in net.outgoing(b.state_id)] == sorted(
        [a.state_id, done.state_id]
    ) or len(net.outgoing(b.state_id)) == 2
    with pytest.raises(NetworkError, match="absorbing"):
        net.add_transition(_event(done, a, atoms=(7,)))


def test_duplicate_events_are_rejected():
    net, a, b, _ = _three()
    net.add_transition(_event(a, b))
    with pytest.raises(NetworkError, match="duplicate transition id"):
        net.add_transition(_event(a, b))
    with pytest.raises(NetworkError, match="repeats"):
        net.add_transition(_event(a, b, transition_id="other"))
    net.add_state(a)
    with pytest.raises(NetworkError, match="other content"):
        net.add_state(_state("a", n_atoms=17))


def test_disconnected_states_form_separate_components():
    net, a, b, done = _three()
    net.add_transition(_event(a, b))
    assert net.components() == tuple(
        sorted([tuple(sorted((a.state_id, b.state_id))), (done.state_id,)])
    )


def test_cross_system_and_non_conserving_transitions_are_rejected():
    net, a, b, _ = _three()
    with pytest.raises(NetworkError, match="another parent"):
        net.add_state(_state("x", parent_id="q"))
    with pytest.raises(NetworkError, match="chemical ordering"):
        net.add_state(_state("y", parent_ordering_id="o2"))
    with pytest.raises(NetworkError, match="unknown state"):
        net.add_transition(_event(a, _state("ghost")))
    other = _state("c", atom_ids=tuple(range(1, 17)))
    net.add_state(other)
    with pytest.raises(NetworkError, match="atom identities"):
        net.add_transition(_event(a, other))
    swapped = _state("d", species_counts={"W": 13, "Mo": 3})
    net.add_state(swapped)
    with pytest.raises(NetworkError, match="species counts"):
        net.add_transition(_event(a, swapped))
    with pytest.raises(NetworkError, match="provenance"):
        net.add_transition(_event(a, b, provenance={}))
    with pytest.raises(NetworkError, match="no provenance"):
        net.add_state(_state("e", provenance={}))
