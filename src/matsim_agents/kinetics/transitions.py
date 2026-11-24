"""Candidate elementary transitions from B migration pathways (C1.3).

Barriers are never guessed: every event created here is a ``candidate`` without evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from matsim_agents.discovery.neb_contracts import NebEndpoint, NebTransition
from matsim_agents.kinetics.contracts import DefectState, TransitionEvent
from matsim_agents.kinetics.graph import TransitionNetwork, state_id, transition_id


@dataclass(frozen=True)
class MechanismPolicy:
    """Graph-building policy for a mechanism; the KMC solver never reads it."""

    name: str
    reversible: bool = False
    absorbing_kind: str | None = None


_MECHANISMS: dict[str, MechanismPolicy] = {}


def register_mechanism(
    name: str, *, reversible: bool = False, absorbing_kind: str | None = None
) -> MechanismPolicy:
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ValueError(f"invalid mechanism identifier {name!r}")
    if name in _MECHANISMS:
        raise ValueError(f"mechanism {name!r} is already registered")
    if reversible and absorbing_kind is not None:
        raise ValueError("a reversible mechanism cannot end in an absorbing state")
    policy = MechanismPolicy(name, reversible, absorbing_kind)
    _MECHANISMS[name] = policy
    return policy


def get_mechanism(name: str) -> MechanismPolicy:
    try:
        return _MECHANISMS[name]
    except KeyError:
        raise ValueError(
            f"unregistered mechanism {name!r}; registered: {', '.join(list_mechanisms())}"
        ) from None


def list_mechanisms() -> tuple[str, ...]:
    return tuple(sorted(_MECHANISMS))


def is_reversible(name: str) -> bool:
    """False for unregistered names, so checks on loaded networks never raise."""
    policy = _MECHANISMS.get(name)
    return policy is not None and policy.reversible


register_mechanism("vacancy_hop", reversible=True)
register_mechanism("interstitial_migration", reversible=True)
register_mechanism("frenkel_recombination", absorbing_kind="recombined")


def state_from_endpoint(endpoint: NebEndpoint, *, absorbing_kind: str | None = None) -> DefectState:
    """Identity comes from the parent context and structure id, never from a fingerprint match."""
    return DefectState(
        state_id=state_id(
            endpoint.parent_id,
            endpoint.parent_ordering_id,
            endpoint.defect_config_id,
            endpoint.structure_id,
        ),
        config_id=endpoint.defect_config_id,
        parent_id=endpoint.parent_id,
        parent_ordering_id=endpoint.parent_ordering_id,
        defect_type=endpoint.defect_type,
        n_atoms=len(endpoint.atom_ids),
        atom_ids=endpoint.atom_ids,
        species_counts=dict(sorted(Counter(endpoint.species).items())),
        absorbing=absorbing_kind is not None,
        absorbing_kind=absorbing_kind,
        provenance={
            "structure_id": endpoint.structure_id,
            "geometry_fingerprint": endpoint.geometry_fingerprint,
            "endpoint_status": endpoint.status,
        },
    )


def _event(
    transition: NebTransition, source: DefectState, target: DefectState, *, reverse: bool
) -> TransitionEvent:
    kind = transition.mechanism.kind
    atoms = tuple(transition.mechanism.moving_atom_ids)
    return TransitionEvent(
        transition_id=transition_id(source.state_id, target.state_id, kind, atoms),
        source_id=source.state_id,
        target_id=target.state_id,
        mechanism=kind,
        moving_atom_ids=atoms,
        provenance={
            "neb_transition_id": transition.transition_id,
            "direction": "reverse" if reverse else "forward",
            **{str(k): str(v) for k, v in transition.mechanism.parameters.items()},
        },
    )


def add_candidate_transition(
    network: TransitionNetwork, transition: NebTransition, *, add_reverse: bool = True
) -> tuple[TransitionEvent, ...]:
    """Add the endpoint states and a candidate event (plus its reverse where physical).

    The registered mechanism decides reversibility and the absorbing kind of the target.
    """
    mechanism = get_mechanism(transition.mechanism.kind)
    source = state_from_endpoint(transition.initial)
    target = state_from_endpoint(transition.final, absorbing_kind=mechanism.absorbing_kind)
    network.add_state(source)
    network.add_state(target)
    events = [_event(transition, source, target, reverse=False)]
    if add_reverse and mechanism.reversible:
        events.append(_event(transition, target, source, reverse=True))
    for event in events:
        network.add_transition(event)
    return tuple(events)


__all__ = [
    "MechanismPolicy",
    "add_candidate_transition",
    "get_mechanism",
    "is_reversible",
    "list_mechanisms",
    "register_mechanism",
    "state_from_endpoint",
]
