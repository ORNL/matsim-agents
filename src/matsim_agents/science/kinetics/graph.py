"""Directed graph of defect states and elementary transitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from matsim_agents.science.kinetics.contracts import DefectState, NetworkProvenance, TransitionEvent


class NetworkError(ValueError):
    """The requested node or edge would make the network inconsistent."""


def state_id(
    parent_id: str,
    parent_ordering_id: str | None,
    config_id: str | None,
    label: str = "",
) -> str:
    """Deterministic ID; the chemical context is part of the identity."""
    return _digest("state", parent_id, parent_ordering_id, config_id, label)


def transition_id(
    source_id: str, target_id: str, mechanism: str, moving_atom_ids: Iterable[int]
) -> str:
    return _digest("transition", source_id, target_id, mechanism, sorted(moving_atom_ids))


def _digest(*parts: object) -> str:
    text = json.dumps(parts, sort_keys=True, default=str)
    return f"{parts[0]}-{hashlib.sha256(text.encode()).hexdigest()[:16]}"


class TransitionNetwork:
    """Nodes are states in one parent context; edges are directed events."""

    def __init__(self, provenance: NetworkProvenance) -> None:
        self.provenance = provenance
        self._states: dict[str, DefectState] = {}
        self._events: dict[str, TransitionEvent] = {}

    @property
    def states(self) -> tuple[DefectState, ...]:
        return tuple(self._states[k] for k in sorted(self._states))

    @property
    def events(self) -> tuple[TransitionEvent, ...]:
        return tuple(self._events[k] for k in sorted(self._events))

    def state(self, sid: str) -> DefectState:
        try:
            return self._states[sid]
        except KeyError:
            raise NetworkError(f"unknown state {sid!r}") from None

    def add_state(self, state: DefectState) -> None:
        if state.parent_id != self.provenance.parent_id:
            raise NetworkError(f"state {state.state_id!r} belongs to another parent system")
        if state.parent_ordering_id != self.provenance.parent_ordering_id:
            raise NetworkError(f"state {state.state_id!r} has a different chemical ordering")
        known = self._states.get(state.state_id)
        if known is not None:
            if known != state:
                raise NetworkError(f"state {state.state_id!r} already exists with other content")
            return
        if not state.provenance:
            raise NetworkError(f"state {state.state_id!r} has no provenance")
        self._states[state.state_id] = state

    def add_transition(self, event: TransitionEvent) -> None:
        source = self.state(event.source_id)
        target = self.state(event.target_id)
        if source.absorbing:
            raise NetworkError(f"absorbing state {source.state_id!r} cannot have outgoing events")
        if not event.provenance:
            raise NetworkError(f"transition {event.transition_id!r} has no provenance")
        self._check_conservation(source, target, event)
        if event.transition_id in self._events:
            raise NetworkError(f"duplicate transition id {event.transition_id!r}")
        key = (
            event.source_id,
            event.target_id,
            event.mechanism,
            tuple(sorted(event.moving_atom_ids)),
        )
        for other in self._events.values():
            other_key = (
                other.source_id,
                other.target_id,
                other.mechanism,
                tuple(sorted(other.moving_atom_ids)),
            )
            if key == other_key:
                raise NetworkError(
                    f"duplicate event {event.transition_id!r} repeats {other.transition_id!r}"
                )
        self._events[event.transition_id] = event

    @staticmethod
    def _check_conservation(
        source: DefectState, target: DefectState, event: TransitionEvent
    ) -> None:
        if target.absorbing and target.absorbing_kind == "sink":
            return
        if source.atom_ids is not None and target.atom_ids is not None:
            if set(source.atom_ids) != set(target.atom_ids):
                raise NetworkError("transition does not conserve atom identities")
            if not set(event.moving_atom_ids) <= set(source.atom_ids):
                raise NetworkError("moving atoms are not part of the states")
        counts = (source.species_counts, target.species_counts)
        if None not in counts and counts[0] != counts[1]:
            raise NetworkError("transition does not conserve species counts")

    def replace_transition(self, event: TransitionEvent) -> None:
        """Swap in updated evidence for an existing event; its endpoints must not change."""
        old = self._events.get(event.transition_id)
        if old is None:
            raise NetworkError(f"unknown transition {event.transition_id!r}")
        same = (old.source_id, old.target_id, old.mechanism, old.moving_atom_ids) == (
            event.source_id,
            event.target_id,
            event.mechanism,
            event.moving_atom_ids,
        )
        if not same:
            raise NetworkError("replacing an event cannot change its endpoints or mechanism")
        self._events[event.transition_id] = event

    def outgoing(self, sid: str) -> tuple[TransitionEvent, ...]:
        self.state(sid)
        return tuple(e for e in self.events if e.source_id == sid)

    def reverse_of(self, event: TransitionEvent) -> TransitionEvent | None:
        """The explicitly stored reverse event, or None; it is never synthesised."""
        for other in self.events:
            if (
                other.source_id == event.target_id
                and other.target_id == event.source_id
                and other.mechanism == event.mechanism
            ):
                return other
        return None

    def components(self) -> tuple[tuple[str, ...], ...]:
        """Weakly connected components, sorted deterministically."""
        parent = {sid: sid for sid in self._states}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in self._events.values():
            parent[find(e.source_id)] = find(e.target_id)
        groups: dict[str, list[str]] = {}
        for sid in self._states:
            groups.setdefault(find(sid), []).append(sid)
        return tuple(sorted(tuple(sorted(g)) for g in groups.values()))


__all__ = ["NetworkError", "TransitionNetwork", "state_id", "transition_id"]
