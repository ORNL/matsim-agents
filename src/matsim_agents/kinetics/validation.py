"""Network completeness report.

Structural integrity is already enforced by ``TransitionNetwork``; this module reports what
is still missing for kinetics: barrier evidence, attempt frequencies, reachability, dead ends.
"""

from __future__ import annotations

from collections import deque

from pydantic import BaseModel, ConfigDict

from matsim_agents.kinetics.graph import TransitionNetwork
from matsim_agents.kinetics.transitions import is_reversible


class NetworkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_states: int
    n_events: int
    n_candidate: int
    n_unqualified: int
    n_qualified: int
    n_synthetic_evidence: int
    qualified_fraction: float
    evidence_fraction: float
    missing_barrier: tuple[str, ...]
    missing_attempt_frequency: tuple[str, ...]
    missing_reverse: tuple[str, ...]
    unreachable_states: tuple[str, ...]
    dead_end_states: tuple[str, ...]
    absorbing_states: tuple[str, ...]
    ready_for_qualified_kmc: bool
    ready_for_exploratory_kmc: bool
    issues: tuple[str, ...]


def validate_network(network: TransitionNetwork, initial_state_id: str) -> NetworkReport:
    """Summarise evidence coverage and reachability from ``initial_state_id``."""
    network.state(initial_state_id)
    events = network.events
    states = network.states

    seen = {initial_state_id}
    queue = deque([initial_state_id])
    while queue:
        for event in network.outgoing(queue.popleft()):
            if event.target_id not in seen:
                seen.add(event.target_id)
                queue.append(event.target_id)

    counts = {"candidate": 0, "unqualified": 0, "qualified": 0}
    for event in events:
        counts[event.qualification] += 1
    n = len(events)
    with_barrier = sum(e.barrier_eV is not None for e in events)

    missing_barrier = tuple(e.transition_id for e in events if e.barrier_eV is None)
    missing_freq = tuple(e.transition_id for e in events if e.attempt_frequency_hz is None)
    missing_reverse = tuple(
        e.transition_id
        for e in events
        if is_reversible(e.mechanism)
        and not network.state(e.target_id).absorbing
        and network.reverse_of(e) is None
    )
    unreachable = tuple(s.state_id for s in states if s.state_id not in seen)
    dead_ends = tuple(
        s.state_id for s in states if not s.absorbing and not network.outgoing(s.state_id)
    )
    absorbing = tuple(s.state_id for s in states if s.absorbing)

    reachable_events = [e for e in events if e.source_id in seen]
    reach_qualified = all(e.qualification == "qualified" for e in reachable_events)
    reach_barriers = all(e.barrier_eV is not None for e in reachable_events)
    reach_freq = all(e.attempt_frequency_hz is not None for e in reachable_events)
    runnable = bool(reachable_events)

    issues: list[str] = []
    if not runnable:
        issues.append("no transitions leave the initial state")
    if missing_barrier:
        issues.append(f"{len(missing_barrier)} of {n} transitions have no barrier evidence")
    if counts["unqualified"]:
        issues.append(f"{counts['unqualified']} transitions rest on unqualified evidence")
    if missing_reverse:
        issues.append(f"{len(missing_reverse)} reversible transitions have no reverse event")
    if dead_ends:
        issues.append(f"{len(dead_ends)} non-absorbing states have no outgoing transition")
    if unreachable:
        issues.append(f"{len(unreachable)} states are unreachable from the initial state")
    if network.provenance.synthetic:
        issues.append("network is synthetic; it is not scientific evidence")
    n_synthetic = sum(1 for e in events if e.evidence and e.evidence.synthetic)
    if n_synthetic:
        issues.append(f"{n_synthetic} transitions rest on synthetic barrier evidence")

    return NetworkReport(
        n_states=len(states),
        n_events=n,
        n_candidate=counts["candidate"],
        n_unqualified=counts["unqualified"],
        n_qualified=counts["qualified"],
        n_synthetic_evidence=n_synthetic,
        qualified_fraction=counts["qualified"] / n if n else 0.0,
        evidence_fraction=with_barrier / n if n else 0.0,
        missing_barrier=missing_barrier,
        missing_attempt_frequency=missing_freq,
        missing_reverse=missing_reverse,
        unreachable_states=unreachable,
        dead_end_states=dead_ends,
        absorbing_states=absorbing,
        ready_for_qualified_kmc=(
            runnable
            and reach_qualified
            and not missing_reverse
            and not network.provenance.synthetic
            and not n_synthetic
        ),
        ready_for_exploratory_kmc=runnable and reach_barriers and reach_freq,
        issues=tuple(issues),
    )


__all__ = ["NetworkReport", "validate_network"]
