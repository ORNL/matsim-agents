import pytest
from fusion_pfc_app import POLICY, synthetic

from matsim_agents.execution.budget import BudgetLedger
from matsim_agents.science.decision.acquisition import (
    ObjectiveCalculation,
    decision_confidence,
    plan_objective_acquisition,
    render_objective_plan,
    reserve_selected,
)
from matsim_agents.science.decision.scoring import score_candidates


def _result(cands=None):
    return score_candidates(
        POLICY,
        synthetic.BENCHMARKS,
        synthetic.SCENARIO,
        cands if cands is not None else synthetic.candidates(),
    )


def _ledger(allocation=100.0):
    return BudgetLedger(total_node_hours=1000.0, allocation={"property": allocation})


def _calc(calc_id, candidate, objective, observable, cost=10.0, sigma=1.0, **kw):
    return ObjectiveCalculation(
        calc_id=calc_id,
        candidate_id=candidate,
        objective=objective,
        observable=observable,
        backend="qe",
        cost_node_hours=cost,
        expected_uncertainty=sigma,
        **kw,
    )


def _plan(proposals, ledger=None, result=None, **kw):
    return plan_objective_acquisition(
        result or _result(),
        POLICY,
        synthetic.BENCHMARKS,
        proposals,
        ledger or _ledger(),
        draws=4,
        samples=100,
        seed=kw.pop("seed", 0),
        **kw,
    )


def test_missing_objective_without_prior_is_unquantified():
    plan = _plan([_calc("c1", "C", "low_activation_shutdown_dose", "shutdown_dose_rate")])
    (item,) = plan.items
    assert item.decision == "unquantified" and item.expected_improvement is None
    assert plan.selected == []


def test_declared_prior_completes_candidate_and_is_labelled():
    calc = _calc(
        "c1",
        "C",
        "low_activation_shutdown_dose",
        "shutdown_dose_rate",
        sigma=1e-4,
        prior_value=1e-3,
        prior_uncertainty=5e-4,
        prior_source="analogue alloy, labelled estimate",
    )
    plan = _plan([calc])
    (item,) = plan.items
    assert item.used_prior and item.expected_improvement is not None
    assert item.expected_improvement > 0
    assert plan.selected == ["c1"]
    assert any("declared priors" in x for x in plan.limitations)


def test_prior_needs_source():
    with pytest.raises(ValueError, match="value, uncertainty and source"):
        _calc("c1", "C", "low_activation_shutdown_dose", "shutdown_dose_rate", prior_value=1.0)


def test_proxy_observable_refused():
    plan = _plan([_calc("c1", "A", "plasma_erosion_resistance", "pka_rate")])
    assert plan.items[0].decision == "refused"
    assert "proxy" in plan.items[0].reason


def test_budget_limits_selection_and_reports_real_ledger():
    calc = _calc(
        "c1",
        "C",
        "low_activation_shutdown_dose",
        "shutdown_dose_rate",
        cost=50.0,
        sigma=1e-4,
        prior_value=1e-3,
        prior_uncertainty=5e-4,
        prior_source="estimate",
    )
    plan = _plan([calc], ledger=_ledger(allocation=20.0))
    assert plan.selected == []
    assert "exceeds spendable budget" in plan.items[0].reason
    assert plan.budget.spendable == pytest.approx(20.0)
    assert not plan.dispatched and plan.requires_approval


def test_prerequisite_and_duplicate_and_blocked_backend():
    a = _calc("a", "A", "thermal_conductivity", "thermal_conductivity_total")
    dup = _calc("b", "A", "thermal_conductivity", "thermal_conductivity_total")
    need = _calc("c", "B", "thermal_conductivity", "thermal_conductivity_total", requires=("z",))
    plan = _plan([a, dup, need], blockers={"qe": "GPU build broken"})
    by = {s.calc_id: s for s in plan.items}
    assert by["a"].decision == "blocked" and "GPU build broken" in by["a"].reason
    assert by["b"].decision == "duplicate"
    assert by["c"].decision == "blocked"


def test_policy_mismatch_rejected():
    other = POLICY.with_weights(
        {"thermal_conductivity": 0.21, "radiation_damage_recovery": 0.24}, policy_version="x"
    )
    with pytest.raises(ValueError, match="differ"):
        plan_objective_acquisition(_result(), other, synthetic.BENCHMARKS, [], _ledger(), seed=0)


def test_confidence_counts_incomplete_and_is_deterministic():
    from matsim_agents.science.decision.acquisition import _state

    state, ranked = _state(_result())
    c1 = decision_confidence(state, ranked, POLICY, synthetic.BENCHMARKS, samples=200, seed=0)
    c2 = decision_confidence(state, ranked, POLICY, synthetic.BENCHMARKS, samples=200, seed=0)
    assert c1 == c2
    assert c1 <= 2 / 3 + 1e-12  # C is incomplete


def test_plan_digest_reserve_and_render():
    calc = _calc(
        "c1",
        "C",
        "low_activation_shutdown_dose",
        "shutdown_dose_rate",
        sigma=1e-4,
        prior_value=1e-3,
        prior_uncertainty=5e-4,
        prior_source="estimate",
    )
    p1 = _plan([calc], observed_cost={"node_hours": 3.5, "bad": "x"})
    p2 = _plan([calc], observed_cost={"node_hours": 3.5, "bad": "x"})
    assert p1.plan_digest == p2.plan_digest and p1.plan_digest.startswith("facq-")
    assert p1.observed_cost == {"node_hours": 3.5}
    text = render_objective_plan(p1)
    assert "SYNTHETIC" in text and "[prior]" in text and "nothing dispatched" in text
    ledger = _ledger()
    with pytest.raises(ValueError, match="approved"):
        reserve_selected(p1, ledger, ["c1"], authorized_by=" ")
    with pytest.raises(ValueError, match="not a selected item"):
        reserve_selected(p1, ledger, ["zz"], authorized_by="PI")
    assert reserve_selected(p1, ledger, ["c1"], authorized_by="PI") == ["c1"]
