"""Transition-network records (synthetic evidence only)."""

import pytest
from pydantic import ValidationError

from matsim_agents.kinetics.contracts import (
    BarrierEvidence,
    DefectState,
    TransitionEvent,
    TransitionRate,
)


def _evidence(**over):
    base = {
        "method_id": "synthetic-fixture",
        "neb_state": "qualified",
        "forward_barrier_eV": 0.8,
        "reverse_barrier_eV": 0.9,
        "endpoints_qualified": True,
        "n_images": 6,
        "synthetic": True,
    }
    base.update(over)
    return BarrierEvidence(**base)


def _event(**over):
    base = {
        "transition_id": "t1",
        "source_id": "s1",
        "target_id": "s2",
        "mechanism": "vacancy_hop",
        "moving_atom_ids": (3,),
        "attempt_frequency_hz": 1e13,
        "evidence": _evidence(),
        "qualification": "qualified",
    }
    base.update(over)
    return TransitionEvent(**base)


def test_valid_qualified_event_round_trips_through_json():
    event = _event()
    assert event.barrier_eV == pytest.approx(0.8)
    assert TransitionEvent.model_validate_json(event.model_dump_json()) == event


def test_missing_evidence_is_none_not_zero():
    event = _event(evidence=None, qualification="candidate", attempt_frequency_hz=None)
    assert event.barrier_eV is None
    assert _event(evidence=_evidence(forward_barrier_eV=0.0)).barrier_eV == 0.0


def test_invalid_transitions_are_rejected():
    with pytest.raises(ValidationError, match="two different states"):
        _event(target_id="s1")
    with pytest.raises(ValidationError, match="candidate has no evidence"):
        _event(qualification="candidate")
    with pytest.raises(ValidationError, match="needs barrier evidence"):
        _event(evidence=None)
    with pytest.raises(ValidationError, match="attempt frequency"):
        _event(attempt_frequency_hz=None)
    with pytest.raises(ValidationError, match="qualified NEB evidence"):
        _event(evidence=_evidence(neb_state="unconverged"))
    with pytest.raises(ValidationError, match="qualified NEB evidence"):
        _event(evidence=_evidence(endpoints_qualified=False))
    with pytest.raises(ValidationError):
        _event(multiplicity=0)
    with pytest.raises(ValidationError):
        _evidence(forward_barrier_eV=-0.1)
    with pytest.raises(ValidationError):
        _event(attempt_frequency_hz=-1.0)


def test_unqualified_evidence_is_allowed_but_not_qualified():
    event = _event(qualification="unqualified", evidence=_evidence(neb_state="unconverged"))
    assert event.qualification == "unqualified"


def test_absorbing_state_requires_a_kind():
    assert DefectState(state_id="a", parent_id="p", n_atoms=10).absorbing is False
    state = DefectState(
        state_id="a", parent_id="p", n_atoms=10, absorbing=True, absorbing_kind="recombined"
    )
    assert state.absorbing_kind == "recombined"
    with pytest.raises(ValidationError, match="absorbing_kind"):
        DefectState(state_id="a", parent_id="p", n_atoms=10, absorbing=True)
    with pytest.raises(ValidationError, match="absorbing_kind"):
        DefectState(state_id="a", parent_id="p", n_atoms=10, absorbing_kind="sink")


def test_rate_record_validates_temperature_and_rate():
    assert TransitionRate(
        transition_id="t1", temperature_K=300.0, rate_per_s=0.0, qualification="candidate"
    )
    with pytest.raises(ValidationError):
        TransitionRate(
            transition_id="t1", temperature_K=0.0, rate_per_s=1.0, qualification="candidate"
        )
    with pytest.raises(ValidationError):
        TransitionRate(
            transition_id="t1",
            temperature_K=300.0,
            rate_per_s=float("inf"),
            qualification="candidate",
        )
