"""Multifidelity promotion, magnetic-claim gating and cost accounting (M15)."""

from __future__ import annotations

import pytest

from matsim_agents.workflows.exploration.multifidelity import (
    F1_LIMITATION,
    CostLedger,
    FidelityEvidence,
    PromotionRule,
    budget_remaining,
    decide_promotions,
    magnetic_claims_allowed,
    savings_versus_all_f2,
)


def _e(cid, fid="F1", value=0.0, conv=True, cost=1.0, unc=None, resolved=False):
    return FidelityEvidence(cid, fid, value, conv, cost, unc, resolved)


def test_cutoff_promotion_and_unconverged_handling():
    rule = PromotionRule("F1", "F2", max_value=0.05)
    out = {
        d.candidate_id: d
        for d in decide_promotions(
            [_e("a", value=0.01), _e("b", value=0.2), _e("c", value=None, conv=False)], rule
        )
    }
    assert out["a"].promoted and out["a"].target == "F2"
    assert not out["b"].promoted and out["b"].reason == "above cutoff"
    assert not out["c"].promoted and out["c"].reason == "no converged result"


def test_top_fraction_and_combined_cutoff():
    ev = [_e(f"c{i}", value=float(i)) for i in range(10)]
    promoted = {
        d.candidate_id
        for d in decide_promotions(ev, PromotionRule("F1", "F2", top_fraction=0.3))
        if d.promoted
    }
    assert promoted == {"c0", "c1", "c2"}
    tighter = {
        d.candidate_id
        for d in decide_promotions(ev, PromotionRule("F1", "F2", max_value=1.0, top_fraction=0.3))
        if d.promoted
    }
    assert tighter == {"c0", "c1"}


def test_high_uncertainty_is_escalated_not_discarded():
    rule = PromotionRule("F1", "F2", max_value=0.05, max_uncertainty=0.1)
    (decision,) = decide_promotions([_e("a", value=0.9, unc=0.5)], rule)
    assert decision.promoted and "escalated" in decision.reason


def test_only_matching_source_is_considered_and_downward_rule_rejected():
    rule = PromotionRule("F1", "F2", max_value=1.0)
    ids = {d.candidate_id for d in decide_promotions([_e("a"), _e("b", fid="F0")], rule)}
    assert ids == {"a"}
    with pytest.raises(ValueError, match="higher fidelity"):
        decide_promotions([], PromotionRule("F2", "F1"))


def test_mlip_evidence_never_supports_magnetic_claims():
    ok, why = magnetic_claims_allowed([_e("a", "F1", resolved=True)])
    assert not ok and why == F1_LIMITATION
    assert magnetic_claims_allowed([]) == (False, "no spin-polarized evidence")
    assert not magnetic_claims_allowed([_e("a", "F2", resolved=False)])[0]
    assert not magnetic_claims_allowed([_e("a", "F2", conv=False, resolved=True)])[0]
    ok, _ = magnetic_claims_allowed([_e("a", "F1"), _e("a", "F2", resolved=True)])
    assert ok


def test_cost_ledger_accumulates_measured_cost_only():
    ledger = CostLedger()
    for e in [
        _e("a", "F1", cost=0.01),
        _e("b", "F1", cost=0.01),
        _e("a", "F2", cost=2.0),
        _e("b", "F2", cost=3.0, conv=False),
    ]:
        ledger.record(e)
    summary = ledger.summary()
    assert set(summary) == {"F1", "F2"}
    assert summary["F2"] == {
        "evaluations": 2.0,
        "cost": 5.0,
        "wasted_on_unconverged": 3.0,
        "mean_cost": 2.5,
    }
    assert ledger.total() == pytest.approx(5.02)
    assert budget_remaining(ledger, 10.0) == pytest.approx(4.98)
    with pytest.raises(ValueError):
        ledger.record(_e("x", cost=-1.0))


def test_savings_are_none_without_measured_f2_cost():
    ledger = CostLedger()
    ledger.record(_e("a", "F1", cost=0.1))
    assert savings_versus_all_f2(ledger, n_candidates=100, measured_f2_mean_cost=2.0) is None
    ledger.record(_e("a", "F2", cost=2.0))
    saved = savings_versus_all_f2(ledger, n_candidates=100, measured_f2_mean_cost=2.0)
    assert saved == pytest.approx(200.0 - 2.1)
