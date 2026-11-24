"""C2.1: HTST rates (synthetic fixtures)."""

import math

import pytest

from matsim_agents.kinetics.evidence import attach_neb_result
from matsim_agents.kinetics.rates import (
    KB_EV_PER_K,
    RateError,
    detailed_balance_violations,
    event_rate,
    htst_rate,
    state_rates,
)
from matsim_agents.kinetics.transitions import add_candidate_transition
from tests.test_kinetics_evidence import _result
from tests.test_kinetics_transitions import _hop, _network


def test_rate_matches_analytic_value():
    got = htst_rate(0.5, 300.0, 1e13, 4)
    assert got == pytest.approx(4e13 * math.exp(-0.5 / (KB_EV_PER_K * 300.0)), rel=1e-12)


def test_zero_barrier_gives_attempt_frequency_times_multiplicity():
    assert htst_rate(0.0, 500.0, 1e13, 3) == pytest.approx(3e13)


def test_huge_barrier_underflows_to_zero_not_nan():
    assert htst_rate(500.0, 100.0, 1e13) == 0.0


def test_rate_increases_with_temperature_and_decreases_with_barrier():
    assert htst_rate(0.5, 600, 1e13) > htst_rate(0.5, 300, 1e13)
    assert htst_rate(0.8, 300, 1e13) < htst_rate(0.5, 300, 1e13)


@pytest.mark.parametrize(
    "args",
    [(-0.1, 300, 1e13), (0.5, 0, 1e13), (0.5, 300, 0), (float("nan"), 300, 1e13), (0.5, -5, 1e13)],
)
def test_invalid_inputs_are_rejected(args):
    with pytest.raises(RateError):
        htst_rate(*args)


def _net(**over):
    net, transition = _network(), _hop().transition
    add_candidate_transition(net, transition)
    return net, transition, over


def test_candidate_without_barrier_has_no_rate():
    net, _, _ = _net()
    with pytest.raises(RateError, match="no barrier"):
        state_rates(net, net.events[0].source_id, 300.0, require="exploratory")


def test_qualified_policy_rejects_unqualified_and_missing_frequency():
    net, transition, _ = _net()
    attach_neb_result(
        net,
        transition,
        _result(state="unconverged", endpoints_qualified=False),
        attempt_frequency_hz=1e13,
    )
    forward = next(e for e in net.events if e.provenance["direction"] == "forward")
    with pytest.raises(RateError, match="not qualified"):
        event_rate(forward, 300.0)
    assert event_rate(forward, 300.0, require="exploratory").qualification == "unqualified"


def test_fallback_frequency_is_recorded_and_not_allowed_for_qualified_policy():
    net, transition, _ = _net()
    attach_neb_result(net, transition, _result(state="unconverged", endpoints_qualified=False))
    forward = next(e for e in net.events if e.provenance["direction"] == "forward")
    with pytest.raises(RateError, match="attempt frequency"):
        event_rate(forward, 300.0, require="exploratory")
    rate = event_rate(forward, 300.0, require="exploratory", fallback_attempt_frequency_hz=1e12)
    assert any("fallback" in a for a in rate.assumptions)


def test_detailed_balance_holds_for_consistent_energies_and_flags_inconsistent():
    net, transition, _ = _net()
    attach_neb_result(net, transition, _result(), attempt_frequency_hz=1e13, synthetic=True)
    forward = next(e for e in net.events if e.provenance["direction"] == "forward")
    energies = {forward.source_id: 0.0, forward.target_id: -0.2}
    assert detailed_balance_violations(net, energies, 300.0, require="qualified") == ()
    wrong = {forward.source_id: 0.0, forward.target_id: 0.3}
    assert len(detailed_balance_violations(net, wrong, 300.0, require="qualified")) == 1
    with pytest.raises(RateError, match="missing state energy"):
        detailed_balance_violations(net, {}, 300.0)
