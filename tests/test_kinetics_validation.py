"""Network completeness report (synthetic fixtures)."""

from matsim_agents.kinetics.evidence import attach_neb_result
from matsim_agents.kinetics.transitions import add_candidate_transition
from matsim_agents.kinetics.validation import validate_network
from tests.test_kinetics_evidence import _result
from tests.test_kinetics_transitions import _hop, _network


def _net():
    net, transition = _network(), _hop().transition
    add_candidate_transition(net, transition)
    source = next(e.source_id for e in net.events if e.provenance["direction"] == "forward")
    return net, transition, source


def test_candidate_only_network_reports_missing_evidence_and_is_not_ready():
    net, _, source = _net()
    report = validate_network(net, source)
    assert report.n_events == 2 and report.n_candidate == 2
    assert report.evidence_fraction == 0.0 and report.qualified_fraction == 0.0
    assert len(report.missing_barrier) == 2
    assert not report.ready_for_qualified_kmc and not report.ready_for_exploratory_kmc
    assert any("no barrier evidence" in i for i in report.issues)


def test_qualified_evidence_makes_exploratory_ready_but_synthetic_never_qualified_ready():
    net, transition, source = _net()
    attach_neb_result(net, transition, _result(), attempt_frequency_hz=1e13, synthetic=True)
    report = validate_network(net, source)
    assert report.qualified_fraction == 1.0 and report.n_synthetic_evidence == 2
    assert report.ready_for_exploratory_kmc
    assert not report.ready_for_qualified_kmc
    assert any("synthetic" in i for i in report.issues)


def test_unqualified_evidence_is_flagged_and_not_qualified_ready():
    net, transition, source = _net()
    attach_neb_result(
        net,
        transition,
        _result(state="unconverged", endpoints_qualified=False),
        attempt_frequency_hz=1e13,
    )
    report = validate_network(net, source)
    assert report.n_unqualified == 2 and report.evidence_fraction == 1.0
    assert report.ready_for_exploratory_kmc and not report.ready_for_qualified_kmc
    assert any("unqualified" in i for i in report.issues)


def test_missing_attempt_frequency_blocks_exploratory_use():
    net, transition, source = _net()
    attach_neb_result(net, transition, _result())
    report = validate_network(net, source)
    assert len(report.missing_attempt_frequency) == 2
    assert not report.ready_for_exploratory_kmc


def test_unknown_initial_state_is_rejected():
    net, _, _ = _net()
    try:
        validate_network(net, "nope")
    except Exception as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected an error")
