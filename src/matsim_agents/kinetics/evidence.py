"""Attach B2 NEB results to transition events (C1.4)."""

from __future__ import annotations

import math
from collections.abc import Mapping

from matsim_agents.discovery.neb_contracts import NebResult, NebTransition
from matsim_agents.kinetics.contracts import BarrierEvidence, TransitionEvent
from matsim_agents.kinetics.graph import NetworkError, TransitionNetwork
from matsim_agents.kinetics.transitions import state_from_endpoint

ENERGY_TOLERANCE_EV = 1e-3


class EvidenceError(ValueError):
    """The NEB result cannot be attached to this transition."""


def _events_for(network: TransitionNetwork, neb_transition_id: str) -> dict[str, TransitionEvent]:
    found = {
        e.provenance["direction"]: e
        for e in network.events
        if e.provenance.get("neb_transition_id") == neb_transition_id
        and "direction" in e.provenance
    }
    if "forward" not in found:
        raise EvidenceError(f"no forward event for NEB transition {neb_transition_id!r}")
    return found


def attach_neb_result(
    network: TransitionNetwork,
    transition: NebTransition,
    result: NebResult,
    *,
    attempt_frequency_hz: float | None = None,
    prefactor_note: str | None = None,
    synthetic: bool = False,
    artifacts: Mapping[str, str] | None = None,
    energy_tolerance_eV: float = ENERGY_TOLERANCE_EV,
) -> tuple[TransitionEvent, ...]:
    """Record NEB barriers on the forward event and, if present, its reverse event.

    ``qualified`` needs a qualified result, qualified endpoints, a climbing image (a discrete
    image maximum is not a saddle) and an explicit attempt frequency; everything else with
    usable barriers becomes ``unqualified``. Failed or incomplete runs are rejected.
    """
    if result.state in ("failed", "incomplete"):
        raise EvidenceError(f"a {result.state} NEB run provides no barrier evidence")
    if result.forward_barrier_eV is None or result.reverse_barrier_eV is None:
        raise EvidenceError("the NEB result lacks a forward or reverse barrier")
    forward, reverse = result.forward_barrier_eV, result.reverse_barrier_eV
    if forward < 0 or reverse < 0:
        raise EvidenceError("barriers must be nonnegative")
    energies = result.relative_energies_eV
    if energies:
        if len(energies) != result.n_images:
            raise EvidenceError("energy profile does not match the image count")
        delta = energies[-1] - energies[0]
        if not math.isclose(forward - reverse, delta, abs_tol=energy_tolerance_eV):
            raise EvidenceError(
                f"E_f - E_r = {forward - reverse:.4f} eV disagrees with E_N - E_0 = {delta:.4f} eV"
            )

    events = _events_for(network, transition.transition_id)
    source = state_from_endpoint(transition.initial)
    if events["forward"].source_id != source.state_id:
        raise EvidenceError("NEB initial endpoint does not match the event source state")
    if network.state(events["forward"].target_id).parent_id != transition.final.parent_id:
        raise EvidenceError("NEB final endpoint does not match the event target state")

    warnings = list(result.warnings)
    can_qualify = (
        result.state == "qualified"
        and result.endpoints_qualified
        and result.climbing_image_used
        and attempt_frequency_hz is not None
    )
    if result.state == "qualified" and not result.climbing_image_used:
        warnings.append("no climbing image: the image maximum is not a refined saddle")
    if result.state == "qualified" and attempt_frequency_hz is None:
        warnings.append("no attempt frequency supplied")
    level = "qualified" if can_qualify else "unqualified"

    def evidence(fwd: float, rev: float) -> BarrierEvidence:
        return BarrierEvidence(
            method_id=result.method_id,
            neb_state=result.state,
            forward_barrier_eV=fwd,
            reverse_barrier_eV=rev,
            endpoints_qualified=result.endpoints_qualified,
            climbing_image_used=result.climbing_image_used,
            n_images=result.n_images,
            synthetic=synthetic,
            source_hash=result.provenance.get("pathway_id"),
            artifacts=dict(artifacts or {}),
            warnings=tuple(warnings),
        )

    updated: list[TransitionEvent] = []
    for direction, event in events.items():
        pair = (forward, reverse) if direction == "forward" else (reverse, forward)
        new = event.model_copy(
            update={
                "evidence": evidence(*pair),
                "qualification": level,
                "attempt_frequency_hz": attempt_frequency_hz,
                "prefactor_note": prefactor_note,
            }
        )
        new = TransitionEvent.model_validate(new.model_dump())
        try:
            network.replace_transition(new)
        except NetworkError as exc:
            raise EvidenceError(str(exc)) from None
        updated.append(new)
    return tuple(updated)


__all__ = ["ENERGY_TOLERANCE_EV", "EvidenceError", "attach_neb_result"]
