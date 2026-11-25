"""MP10: one integrated, explicitly SYNTHETIC campaign over candidates A, B and C.

Every value comes from examples/characterization/fusion_alloy_selection.yaml and is invented. The
test checks that the pieces connect (evidence -> selection -> report -> LLM-checked revision ->
acquisition -> budget reservation) and that nothing is invented, dispatched or qualified on the
way. It makes no statement about any real alloy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from matsim_agents.campaign import radiation_properties as rp
from matsim_agents.campaign.budget import BudgetLedger
from matsim_agents.campaign.property_acquisition import (
    ProposedCalculation,
    plan_property_acquisition,
    render_plan,
    reserve_approved,
)
from matsim_agents.campaign.selection import (
    CandidateEvidence,
    SelectionSpec,
    SelectionStatus,
    render_report,
    select,
)
from matsim_agents.campaign.selection_revision import revise_selection_hypothesis
from matsim_agents.core.contracts import PROPERTY_UNITS, EvidenceLevel
from matsim_agents.execution.ledger import EvidenceLedger, HypothesisStatus
from matsim_agents.execution.properties import (
    PropertyConditions,
    PropertyEvidence,
    PropertyQualification,
    PropertyStatus,
)
from matsim_agents.integrations.external.spectra_pka.config import SpectraPKAConfig
from matsim_agents.workflows.radiation_hypothesis import RevisionBudget

CONFIG = Path(__file__).parents[1] / "examples/characterization/fusion_alloy_selection.yaml"
NEEDS_T = {"thermal_conductivity_total", "irradiation_swelling"}


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG.read_text())


def _evidence(cid: str, prop: str, row: dict, temperature: float, irradiation: str):
    cond = PropertyConditions(
        temperature_K=temperature if prop in NEEDS_T else None,
        irradiation=irradiation if prop == "irradiation_swelling" else None,
        defect_state="synthetic" if prop == "irradiation_swelling" else None,
    )
    base = dict(
        structure_id=f"{cid}-s",
        composition=cid,
        property=prop,
        unit=PROPERTY_UNITS[prop],
        conditions=cond,
    )
    if row["value"] is None:
        return PropertyEvidence(
            **base,
            status=PropertyStatus.MISSING,
            reason="synthetic example: no record for this property",
        )
    status = PropertyStatus(row["status"])
    common = dict(
        **base,
        status=status,
        value=row["value"],
        uncertainty=row["uncertainty"],
        evidence_ids=[f"synthetic-{cid}-{prop}"],
    )
    if status is PropertyStatus.CALCULATED:
        return PropertyEvidence(
            **common,
            evidence_level=EvidenceLevel.CONVERGED_DFT,
            converged=True,
            qualification=PropertyQualification.NUMERICALLY_VALIDATED,
        )
    if status is PropertyStatus.MEASURED:
        return PropertyEvidence(
            **common,
            evidence_level=EvidenceLevel.EXPERIMENTAL,
            qualification=PropertyQualification.NUMERICALLY_VALIDATED,
        )
    return PropertyEvidence(**common, evidence_level=EvidenceLevel.MLIP_PREDICTION)


def _selection_spec(cfg) -> SelectionSpec:
    return SelectionSpec.model_validate(cfg["selection"])


def _candidates(cfg) -> list[CandidateEvidence]:
    t, irr = cfg["temperature_K"], cfg["irradiation"]
    return [
        CandidateEvidence(
            candidate_id=cid,
            evidence=[_evidence(cid, p, row, t, irr) for p, row in props.items()],
        )
        for cid, props in cfg["candidates"].items()
    ]


def _proposals(cfg) -> list[ProposedCalculation]:
    t = cfg["temperature_K"]
    return [
        ProposedCalculation(
            calc_id=p["calc_id"],
            candidate_id=p["candidate"],
            property=p["property"],
            temperature_K=t if p["property"] in NEEDS_T else None,
            backend=p["backend"],
            cost_node_hours=p["cost_node_hours"],
            expected_uncertainty=p["expected_uncertainty"],
        )
        for p in cfg["acquisition"]["proposals"]
    ]


def test_config_declares_itself_synthetic_and_names_the_reference_alloy(cfg):
    assert cfg["synthetic"] is True
    assert cfg["reference_alloy"]["composition"] == rp.REFERENCE_ALLOY_COUNTS
    assert tuple(cfg["reference_alloy"]["repeats"]) == rp.REFERENCE_REPEATS


def test_selection_separates_infeasible_qualified_and_unscored_candidates(cfg):
    result = select(_selection_spec(cfg), _candidates(cfg))
    a, b, c = (result.by_id(x) for x in "ABC")
    assert a.status is SelectionStatus.INFEASIBLE  # conductivity 12 < 20 limit
    assert c.status is not SelectionStatus.QUALIFIED  # surrogate values, swelling missing
    assert b.status is not SelectionStatus.INFEASIBLE
    assert "A" not in result.pareto_front or a.status is not SelectionStatus.QUALIFIED


def test_report_shows_each_objective_and_how_they_were_combined(cfg):
    result = select(_selection_spec(cfg), _candidates(cfg))
    text = render_report(result)
    for prop in ("thermal_conductivity_total", "bulk_modulus", "irradiation_swelling"):
        assert prop in text
    assert result.combination and result.score_digest in text


def test_selection_is_reproducible(cfg):
    one = select(_selection_spec(cfg), _candidates(cfg))
    two = select(_selection_spec(cfg), _candidates(cfg))
    assert one.score_digest == two.score_digest


def test_revision_cites_evidence_and_the_llm_cannot_change_numbers(cfg, tmp_path):
    spec, candidates = _selection_spec(cfg), _candidates(cfg)
    result = select(spec, candidates)
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    out = revise_selection_hypothesis(
        ledger,
        "h-A",
        "A",
        result,
        candidates,
        budget=RevisionBudget(max_llm_calls=1),
        drafter=lambda _packet: "Candidate A has thermal conductivity 99.9 W/m/K.",
        agent_model_id="mock",
    )
    assert out.record.status is HypothesisStatus.CONTRADICTED
    assert not out.used_llm and "rejected" in out.notes[0]
    assert out.record.traceability["score_digest"] == result.score_digest
    after = select(spec, candidates)
    assert after.score_digest == result.score_digest  # the draft changed no score


def test_acquisition_ranks_the_uncertain_boundary_and_blocks_the_missing_backend(cfg):
    spec, candidates = _selection_spec(cfg), _candidates(cfg)
    ledger = BudgetLedger(total_node_hours=1000.0, allocation={"property": 200.0})
    plan = plan_property_acquisition(
        spec,
        candidates,
        _proposals(cfg),
        ledger,
        blockers=cfg["acquisition"]["backend_blockers"],
    )
    items = {i.calc_id: i for i in plan.items}
    assert items["C-swelling"].decision == "blocked"
    assert "NOT_INSTALLED" in items["C-swelling"].reason
    assert items["C-kappa"].decision in {"selected", "not_selected", "unquantified"}
    assert plan.requires_approval and not plan.dispatched
    assert "approval" in render_plan(plan).lower()


def test_reservation_needs_a_named_person_and_an_explicit_selected_id(cfg):
    spec, candidates = _selection_spec(cfg), _candidates(cfg)
    ledger = BudgetLedger(total_node_hours=1000.0, allocation={"property": 200.0})
    plan = plan_property_acquisition(spec, candidates, _proposals(cfg), ledger)
    selected = list(plan.selected)
    assert selected, "the synthetic boundary candidate should yield a selected calculation"
    with pytest.raises(ValueError):
        reserve_approved(plan, ledger, selected, authorized_by=" ")
    with pytest.raises(ValueError):
        reserve_approved(plan, ledger, ["C-swelling"], authorized_by="tester")
    reserved = reserve_approved(plan, ledger, selected[:1], authorized_by="tester")
    assert reserved and ledger.summary().reserved > 0 and ledger.summary().committed == 0


def test_radiation_readiness_is_reported_and_no_degradation_is_claimed(cfg, tmp_path):
    ready = rp.assess_radiation_readiness(SpectraPKAConfig(), tmp_path, {})
    state = rp.RadiationState(
        irradiation=cfg["irradiation"], temperature_K=cfg["temperature_K"], defect_state="V1He1"
    )
    for e in rp.radiation_evidence("C-s", "C", state, readiness=ready):
        assert e.status is PropertyStatus.MISSING and e.value is None
    assert set(ready.blockers) == {"spectra_pka", "qe_gas_defects"}


def test_nothing_in_the_synthetic_campaign_is_scientifically_qualified(cfg):
    for cand in _candidates(cfg):
        for e in cand.evidence:
            assert e.qualification is not PropertyQualification.SCIENTIFICALLY_QUALIFIED
