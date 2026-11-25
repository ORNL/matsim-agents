"""MP8: acquisition by expected gain in selection confidence per node-hour (synthetic values)."""

from __future__ import annotations

import pytest

from matsim_agents.campaign.budget import BudgetExceeded, BudgetLedger
from matsim_agents.campaign.property_acquisition import (
    ProposedCalculation,
    candidate_confidence,
    plan_property_acquisition,
    render_plan,
    reserve_approved,
)
from matsim_agents.science.acquisition.selection import (
    CandidateEvidence,
    Constraint,
    Objective,
    SelectionSpec,
    select,
)
from matsim_agents.science.properties.calculators import PropertyStatus
from tests.test_selection import BULK, KAPPA, T, ev


def spec():
    return SelectionSpec(
        objectives=[
            Objective(property=KAPPA, direction="maximize", temperature_K=T),
            Objective(property=BULK, direction="maximize"),
        ],
        constraints=[Constraint(property=KAPPA, minimum=20.0, temperature_K=T)],
        seed=3,
        n_samples=300,
    )


def candidates():
    def c(cid, kappa, ku, bulk, bu):
        return CandidateEvidence(
            candidate_id=cid,
            evidence=[ev(KAPPA, kappa, ku, cid=cid), ev(BULK, bulk, bu, cid=cid)],
        )

    return [
        c("A", 21.0, 6.0, 250.0, 1.0),  # kappa sits on the limit and is very uncertain
        c("B", 60.0, 1.0, 180.0, 1.0),  # well known
        CandidateEvidence(
            candidate_id="C",
            evidence=[
                ev(KAPPA, 40.0, 1.0, cid="C"),
                ev(BULK, None, cid="C", status=PropertyStatus.MISSING),
            ],
        ),
    ]


def ledger(allocation=200.0, total=1000.0):
    return BudgetLedger(total_node_hours=total, allocation={"property": allocation})


def proposals():
    return [
        ProposedCalculation(
            calc_id="A-kappa",
            candidate_id="A",
            property=KAPPA,
            temperature_K=T,
            backend="shengbte",
            cost_node_hours=20.0,
            expected_uncertainty=0.5,
        ),
        ProposedCalculation(
            calc_id="B-bulk",
            candidate_id="B",
            property=BULK,
            backend="qe-elastic",
            cost_node_hours=20.0,
            expected_uncertainty=0.2,
        ),
        ProposedCalculation(
            calc_id="C-bulk",
            candidate_id="C",
            property=BULK,
            backend="qe-elastic",
            cost_node_hours=20.0,
            expected_uncertainty=0.5,
        ),
    ]


def by_id(plan):
    return {i.calc_id: i for i in plan.items}


def test_uncertain_decision_boundary_property_has_the_best_gain_per_node_hour():
    plan = plan_property_acquisition(spec(), candidates(), proposals(), ledger())
    items = by_id(plan)
    assert items["A-kappa"].expected_gain > items["B-bulk"].expected_gain
    assert items["A-kappa"].decision == "selected" and items["A-kappa"].rank == 1
    assert items["A-kappa"].gain_per_node_hour == pytest.approx(
        items["A-kappa"].expected_gain / 20.0
    )
    assert plan.requires_approval and not plan.dispatched


def test_property_with_no_value_is_unquantified_not_given_an_invented_gain():
    plan = plan_property_acquisition(spec(), candidates(), proposals(), ledger())
    c = by_id(plan)["C-bulk"]
    assert c.decision == "unquantified" and c.expected_gain is None
    assert c.gain_per_node_hour is None and "not invented" in c.reason
    assert candidate_confidence(select(spec(), candidates()))["C"] == 0.0


def test_blocked_backends_are_never_selected_and_say_why():
    plan = plan_property_acquisition(
        spec(),
        candidates(),
        proposals(),
        ledger(),
        blockers={"shengbte": "not installed"},
    )
    a = by_id(plan)["A-kappa"]
    assert a.decision == "blocked" and "not installed" in a.reason and a.expected_gain is None
    assert "A-kappa" not in plan.selected


def test_duplicates_are_dropped():
    dup = [*proposals(), proposals()[0].model_copy(update={"calc_id": "A-kappa-again"})]
    plan = plan_property_acquisition(spec(), candidates(), dup, ledger())
    assert by_id(plan)["A-kappa-again"].decision == "duplicate"
    done = plan_property_acquisition(
        spec(), candidates(), proposals(), ledger(), existing_calc_ids={"A-kappa"}
    )
    assert by_id(done)["A-kappa"].decision == "duplicate"


def test_budget_report_uses_the_real_ledger_and_protects_the_validation_reserve():
    led = ledger(allocation=100.0)
    led.reserve("prior-1", "property", 30.0)
    led.commit("prior-1")
    led.reserve("prior-2", "property", 10.0)
    led.record_result("prior-2", 8.0)
    plan = plan_property_acquisition(spec(), candidates(), proposals(), led)
    b = plan.budget
    assert (b.committed, b.consumed, b.reserved) == (30.0, 8.0, 0.0)
    assert b.total == 1000.0 and b.validation_reserve == pytest.approx(150.0)
    assert b.category_allocation == 100.0 and b.category_available == pytest.approx(62.0)
    assert b.spendable == pytest.approx(62.0)
    assert b.remaining == pytest.approx(962.0)
    assert b.expected == pytest.approx(sum(by_id(plan)[c].cost_node_hours for c in plan.selected))
    assert b.remaining_after_plan == pytest.approx(b.remaining - b.expected)
    text = render_plan(plan)
    assert "validation reserve" in text and "nothing dispatched" in text


def test_nothing_is_selected_beyond_the_property_allocation():
    plan = plan_property_acquisition(spec(), candidates(), proposals(), ledger(allocation=0.0))
    assert plan.selected == [] and plan.budget.spendable == 0.0
    assert any("exceeds spendable budget" in i.reason for i in plan.items if i.expected_gain)


def test_default_allocation_funds_no_property_work():
    led = BudgetLedger(total_node_hours=1000.0)
    assert led.allocation["property"] == 0.0
    plan = plan_property_acquisition(spec(), candidates(), proposals(), led)
    assert plan.selected == []


def test_plan_is_deterministic_and_does_not_mutate_evidence():
    cands = candidates()
    before = [c.model_dump() for c in cands]
    a = plan_property_acquisition(spec(), cands, proposals(), ledger())
    b = plan_property_acquisition(spec(), cands, proposals(), ledger())
    assert a.plan_digest == b.plan_digest and a.plan_digest.startswith("acq-")
    assert [c.model_dump() for c in cands] == before


def test_reserving_needs_a_named_person_and_a_selected_item_and_never_submits():
    led = ledger()
    plan = plan_property_acquisition(spec(), candidates(), proposals(), led)
    with pytest.raises(ValueError):
        reserve_approved(plan, led, ["A-kappa"], authorized_by=" ")
    with pytest.raises(ValueError):
        reserve_approved(plan, led, ["C-bulk"], authorized_by="alice")
    assert led.summary().reserved == 0.0
    reserve_approved(plan, led, ["A-kappa"], authorized_by="alice")
    s = led.summary()
    assert s.reserved == 20.0 and s.committed == 0.0
    with pytest.raises(BudgetExceeded):
        led.reserve("too-big", "property", 1e6)
