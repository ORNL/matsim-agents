from __future__ import annotations

import pytest

from matsim_agents.execution.budget import (
    BudgetError,
    BudgetExceeded,
    BudgetLedger,
    DuplicateCalculation,
    RetryNotAuthorized,
)


def ledger(total=1000.0, **kw):
    return BudgetLedger(total_node_hours=total, **kw)


def test_default_allocation_keeps_a_nonzero_validation_reserve():
    led = ledger()
    s = led.summary()
    assert s.validation_reserve == pytest.approx(150.0) and s.validation_reserve > 0
    assert sum(led.allocation.values()) == pytest.approx(850.0)
    assert s.remaining == 1000.0 and s.consumed == s.reserved == s.committed == 0


def test_invalid_allocations_fail():
    with pytest.raises(ValueError):
        ledger(allocation={"bridges": 990.0})
    with pytest.raises(ValueError):
        ledger(allocation={"nonsense": 1.0})
    with pytest.raises(ValueError):
        ledger(validation_reserve_fraction=0.0)
    with pytest.raises(ValueError):
        BudgetLedger(total_node_hours=10, bogus=1)


def test_no_dispatch_when_cost_exceeds_the_uncommitted_budget():
    led = ledger(allocation={"lower_foundations": 400.0, "bridges": 450.0})
    led.reserve("a", "lower_foundations", 300.0)
    assert not led.can_dispatch("lower_foundations", 101.0)
    with pytest.raises(BudgetExceeded):
        led.reserve("b", "lower_foundations", 101.0)
    led.reserve("b", "lower_foundations", 100.0)
    with pytest.raises(BudgetExceeded):
        led.reserve("c", "lower_foundations", 0.5)
    assert len(led.entries) == 2


def test_total_uncommitted_budget_is_a_hard_cap_across_categories():
    led = ledger(total=100.0, allocation={"bridges": 85.0})
    led.reserve("a", "bridges", 85.0)
    with pytest.raises(BudgetExceeded):
        led.reserve("b", "defect", 1.0)
    led.reserve("v", "six_component_validation", 15.0)
    with pytest.raises(BudgetExceeded):
        led.reserve("w", "convergence_validation", 0.1)


def test_reserve_is_only_available_to_validation_categories():
    led = ledger(allocation={"bridges": 850.0})
    assert led.available("bridges") == 850.0
    assert led.available("six_component_validation") == pytest.approx(150.0)
    led.reserve("v1", "six_component_validation", 100.0)
    assert led.available("six_component_validation") == pytest.approx(50.0)
    led.reserve("b", "bridges", 850.0)
    assert led.summary().validation_reserve_remaining == pytest.approx(50.0)
    with pytest.raises(BudgetExceeded):
        led.reserve("b2", "bridges", 1.0)


def test_failed_and_unconverged_jobs_count_as_consumed():
    led = ledger(allocation={"defect": 500.0})
    led.reserve("a", "defect", 100.0)
    led.commit("a", "job-1")
    assert led.summary().committed == 100.0
    led.record_result("a", 80.0, "failed")
    s = led.summary()
    assert s.consumed == 80.0 and s.committed == 0 and s.actual == 80.0 and s.predicted == 100.0
    led.reserve("b", "defect", 50.0)
    led.record_result("b", 50.0, "unconverged")
    assert led.summary().consumed == 130.0 and led.uncommitted() == 870.0


def test_retry_needs_reason_authorization_and_budget():
    led = ledger(allocation={"defect": 500.0})
    led.reserve("a", "defect", 100.0)
    led.record_result("a", 100.0, "failed")
    with pytest.raises(RetryNotAuthorized):
        led.retry("a", 100.0, reason="", authorized_by="pi")
    with pytest.raises(RetryNotAuthorized):
        led.retry("a", 100.0, reason="walltime hit", authorized_by=" ")
    with pytest.raises(DuplicateCalculation):
        led.reserve("a", "defect", 100.0)
    entry = led.retry("a", 120.0, reason="raised walltime", authorized_by="pi")
    assert entry.attempt == 2 and entry.retry_reason == "raised walltime"
    with pytest.raises(DuplicateCalculation):
        led.retry("a", 10.0, reason="again", authorized_by="pi")
    led.record_result("a", 120.0, "converged")
    with pytest.raises(RetryNotAuthorized):
        led.retry("a", 1.0, reason="why", authorized_by="pi")
    assert led.summary().consumed == 220.0


def test_release_returns_node_hours_only_when_unsubmitted():
    led = ledger()
    led.reserve("a", "bridges", 200.0)
    led.release("a")
    assert led.uncommitted() == 1000.0
    led.reserve("b", "bridges", 10.0)
    led.commit("b")
    with pytest.raises(BudgetError):
        led.release("b")
    with pytest.raises(BudgetError):
        led.commit("missing")


def test_reallocation_preserves_spend_and_the_reserve():
    led = ledger()
    led.reserve("a", "lower_foundations", 100.0)
    led.record_result("a", 60.0)
    new = led.reallocate({"bridges": 3, "defect": 1})
    assert sum(new.values()) == pytest.approx(850.0)
    assert new["lower_foundations"] == pytest.approx(60.0)
    assert new["bridges"] > led.default_allocation()["bridges"]
    assert led.summary().validation_reserve == pytest.approx(150.0)
    assert led.reallocation_log
    with pytest.raises(BudgetError):
        led.reallocate({"bridges": 0})


def test_ledger_round_trips_through_disk(tmp_path):
    led = ledger()
    led.reserve("a", "magnetic", 25.0)
    led.commit("a", "42")
    led.save(tmp_path / "sub" / "ledger.json")
    again = BudgetLedger.load(tmp_path / "sub" / "ledger.json")
    assert again.model_dump() == led.model_dump()
    with pytest.raises(DuplicateCalculation):
        again.reserve("a", "magnetic", 25.0)
