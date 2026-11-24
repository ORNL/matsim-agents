"""C1.4: NEB evidence ingestion (synthetic results; no calculation)."""

import pytest

from matsim_agents.discovery.neb_contracts import NebResult
from matsim_agents.kinetics.evidence import EvidenceError, attach_neb_result
from matsim_agents.kinetics.transitions import add_candidate_transition
from tests.test_kinetics_transitions import _hop, _network

ENERGIES = (0.0, 0.4, 0.8, 0.5, 0.1, -0.2)


def _result(**over):
    base = {
        "state": "qualified",
        "method_id": "synthetic-fixture",
        "n_images": 6,
        "relative_energies_eV": ENERGIES,
        "forward_barrier_eV": 0.8,
        "reverse_barrier_eV": 1.0,
        "endpoints_qualified": True,
        "climbing_image_used": True,
    }
    base.update(over)
    return NebResult(**base)


def _setup():
    net, transition = _network(), _hop().transition
    add_candidate_transition(net, transition)
    return net, transition


def test_qualified_evidence_sets_both_directions_with_swapped_barriers():
    net, transition = _setup()
    forward, back = attach_neb_result(
        net, transition, _result(), attempt_frequency_hz=1e13, synthetic=True
    )
    assert forward.qualification == back.qualification == "qualified"
    assert forward.barrier_eV == pytest.approx(0.8) and back.barrier_eV == pytest.approx(1.0)
    assert forward.evidence.synthetic and forward.evidence.method_id == "synthetic-fixture"
    assert net.reverse_of(forward).barrier_eV == pytest.approx(1.0)


def test_unconverged_or_unrefined_evidence_is_never_qualified():
    for result, kwargs in (
        (_result(state="unconverged", endpoints_qualified=False), {}),
        (_result(state="converged_unqualified", endpoints_qualified=False), {}),
        (_result(climbing_image_used=False), {}),
        (_result(), {"attempt_frequency_hz": None}),
    ):
        net, transition = _setup()
        kwargs.setdefault("attempt_frequency_hz", 1e13)
        event, _ = attach_neb_result(net, transition, result, **kwargs)
        assert event.qualification == "unqualified"


def test_failed_runs_and_missing_barriers_are_rejected():
    net, transition = _setup()
    with pytest.raises(EvidenceError, match="no barrier evidence"):
        attach_neb_result(net, transition, NebResult(state="failed", method_id="m", n_images=6))
    with pytest.raises(EvidenceError, match="lacks"):
        attach_neb_result(
            net,
            transition,
            _result(state="unconverged", reverse_barrier_eV=None, endpoints_qualified=False),
        )


def test_inconsistent_forward_reverse_barriers_are_rejected():
    net, transition = _setup()
    with pytest.raises(EvidenceError, match="disagrees"):
        attach_neb_result(net, transition, _result(reverse_barrier_eV=0.5))
    with pytest.raises(EvidenceError, match="nonnegative"):
        attach_neb_result(
            net,
            transition,
            _result(
                state="unconverged",
                endpoints_qualified=False,
                forward_barrier_eV=-0.1,
                relative_energies_eV=(),
            ),
        )


def test_evidence_for_another_pathway_is_rejected():
    net, _ = _setup()
    other = _hop().transition.model_copy(update={"transition_id": "something-else"})
    with pytest.raises(EvidenceError, match="no forward event"):
        attach_neb_result(net, other, _result())
