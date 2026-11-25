from __future__ import annotations

import pytest
from pydantic import ValidationError

from matsim_agents.campaign.stages import (
    CampaignEvidence,
    StageThresholds,
    decide,
    evaluate_stages,
)


def ev(**kw) -> CampaignEvidence:
    base = dict(budget_remaining=500.0, budget_total=1000.0)
    base.update(kw)
    return CampaignEvidence(**base)


STAGE0 = dict(inventory_complete=True, compatibility_checked=True)
STAGE1 = dict(**STAGE0, coverage_by_order={2: 0.95}, converged_fraction=0.97)
STAGE2 = dict(
    **{**STAGE1, "coverage_by_order": {2: 0.95, 3: 0.7, 4: 0.6}},
    transfer_status={"2->3": "validated"},
)
STAGE3 = dict(**STAGE2, heldout_mae=0.05)


def test_empty_campaign_starts_with_reuse_and_inventory_not_new_dft():
    plan = decide(ev())
    assert plan.decision == "reuse_inventory" and plan.current_stage == 0
    assert plan.completed_stages == []
    assert all(g.status == "failed" for g in plan.stages[0].gates)


def test_counts_alone_never_advance_a_stage():
    plan = decide(ev(n_reused_records=10**6, n_new_calculations=10**6))
    assert plan.current_stage == 0 and plan.decision == "reuse_inventory"
    flooded = decide(ev(**STAGE0, n_new_calculations=10**6))
    assert flooded.current_stage == 1 and flooded.decision == "continue_lower"
    assert {g.status for g in flooded.stages[1].gates} == {"NOT_YET_ESTABLISHED"}


def test_stages_progress_strictly_in_order_on_evidence():
    assert decide(ev(**STAGE0)).decision == "continue_lower"
    assert decide(ev(**STAGE1)).decision == "add_bridges"
    assert decide(ev(**STAGE1)).current_stage == 2
    three = decide(ev(**STAGE2))
    assert three.current_stage == 3 and three.decision == "add_bridges"
    four = decide(ev(**STAGE3))
    assert four.current_stage == 4 and four.decision == "improve_magnetic"


def test_a_later_gate_cannot_be_passed_by_skipping_an_earlier_one():
    skipped = decide(ev(heldout_mae=0.001, defect_status="validated", magnetic_status="validated"))
    assert skipped.completed_stages == [] and skipped.decision == "reuse_inventory"


def test_low_coverage_or_poor_convergence_keeps_stage_one_open():
    low = decide(ev(**{**STAGE1, "coverage_by_order": {2: 0.3}}))
    assert low.current_stage == 1 and low.decision == "continue_lower"
    bad = decide(ev(**{**STAGE1, "converged_fraction": 0.4}))
    assert bad.current_stage == 1
    assert {g.name: g.status for g in bad.stages[1].gates}["converged_fraction"] == "failed"


def test_unvalidated_or_negative_transfer_does_not_open_stage_three():
    neg = decide(ev(**{**STAGE2, "transfer_status": {"2->3": "unsupported"}}))
    assert neg.current_stage == 2
    none = decide(ev(**{**STAGE2, "transfer_status": {}}))
    assert none.current_stage == 2
    assert {g.name: g.status for g in none.stages[2].gates}["validated_transfer"] == (
        "NOT_YET_ESTABLISHED"
    )


def test_six_component_validation_requires_accuracy_and_an_intact_reserve():
    drawn = decide(ev(**{**STAGE3, "validation_reserve_intact": False}))
    assert drawn.current_stage == 3
    strict = decide(ev(**STAGE3), StageThresholds(max_heldout_mae=0.01))
    assert strict.current_stage == 3
    assert decide(ev(**STAGE2)).stages[3].gates[0].status == "NOT_YET_ESTABLISHED"


def test_defects_then_neb_decisions():
    base = {**STAGE3, "magnetic_status": "validated"}
    assert decide(ev(**base)).decision == "add_defects"
    ready = {**base, "defect_status": "validated", "neb_endpoints_ready": True}
    plan = decide(ev(**ready))
    assert plan.current_stage == 5 and plan.decision == "promote_neb"
    not_ready = decide(ev(**{**ready, "neb_endpoints_ready": False}))
    assert not_ready.decision == "add_defects"
    unsupported = decide(ev(**{**base, "magnetic_status": "unsupported"}))
    assert unsupported.decision == "improve_magnetic"


def test_stop_when_everything_passes_or_the_budget_is_gone():
    done = decide(
        ev(
            **STAGE3,
            magnetic_status="validated",
            defect_status="validated",
            neb_endpoints_ready=True,
            neb_status="validated",
        )
    )
    assert done.decision == "stop" and done.current_stage is None
    assert done.completed_stages == [0, 1, 2, 3, 4, 5]
    broke = decide(ev(**STAGE1, budget_remaining=0.0))
    assert broke.decision == "stop" and "budget" in broke.reason
    inventory_first = decide(ev(budget_remaining=0.0))
    assert inventory_first.decision == "reuse_inventory"


def test_thresholds_and_evidence_reject_unknown_or_invalid_fields():
    with pytest.raises(ValidationError):
        StageThresholds(mystery=1)
    with pytest.raises(ValidationError):
        CampaignEvidence(converged_fraction=1.5)
    assert [r.stage for r in evaluate_stages(ev())] == [0, 1, 2, 3, 4, 5]
