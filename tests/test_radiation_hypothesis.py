"""Deterministic hypothesis revision. Energies are synthetic fixtures."""

import pytest

from matsim_agents.core.revision import RevisionBudget, RevisionBudgetError
from matsim_agents.execution.ledger import EvidenceLedger, HypothesisStatus
from matsim_agents.workflows.radiation.damage import run_radiation_damage
from matsim_agents.workflows.radiation.hypothesis import RevisionGateError, revise_hypothesis
from tests.test_neb_workflow import _fake
from tests.test_radiation_damage_workflow import _config, _forbidden

HID = "h-vacancy-hop"


def _dry(tmp_path):
    config = _config(tmp_path)
    return config, run_radiation_damage(config, tmp_path / "dry", launcher=_forbidden)


def _dispatched(tmp_path, ledger):
    config = _config(tmp_path)
    run = run_radiation_damage(
        config,
        tmp_path / "run",
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
        ledger=ledger,
    )
    return config, run


def test_dry_run_gives_a_proposed_claim_with_no_evidence(tmp_path):
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    config, run = _dry(tmp_path)
    out = revise_hypothesis(ledger, HID, config, run)
    assert out.record.status is HypothesisStatus.PROPOSED
    assert out.record.evidence_ids == [] and out.record.falsifiable_prediction is None
    assert out.record.prior_revision_id is None and not out.used_llm


def test_exploratory_result_is_unresolved_and_cites_measured_evidence(tmp_path):
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    config, run = _dispatched(tmp_path, ledger)
    out = revise_hypothesis(ledger, HID, config, run)
    assert out.record.status is HypothesisStatus.UNRESOLVED
    assert out.record.evidence_ids == [e.evidence_id for e in ledger.evidence()]
    assert out.record.support_level.value == "low_fidelity_dft"
    assert "no qualification" in out.record.scientific_rationale
    assert out.record.proposed_tests[0].startswith("qualified NEB barrier")


def test_revisions_chain_and_budget_stops_them(tmp_path):
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    config, dry = _dry(tmp_path)
    budget = RevisionBudget(max_revisions=2)
    first = revise_hypothesis(ledger, HID, config, dry, budget=budget)
    config2, run = _dispatched(tmp_path, ledger)
    second = revise_hypothesis(ledger, HID, config2, run, budget=budget)
    assert second.record.prior_revision_id == first.record.revision_id
    with pytest.raises(RevisionBudgetError, match="2 of 2"):
        revise_hypothesis(ledger, HID, config2, run, budget=budget)
    assert len(ledger.history(HID)) == 2


def test_llm_text_is_labelled_and_cannot_change_status(tmp_path):
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    config, run = _dispatched(tmp_path, ledger)
    budget = RevisionBudget(max_llm_calls=1)
    out = revise_hypothesis(
        ledger,
        HID,
        config,
        run,
        budget=budget,
        rationale_writer=lambda s: "This is SUPPORTED and qualified.",
    )
    assert out.used_llm and budget.llm_calls_used == 1
    assert out.record.status is HypothesisStatus.UNRESOLVED
    assert "[LLM-drafted, unverified]" in out.record.scientific_rationale


def test_llm_budget_zero_never_calls_the_writer(tmp_path):
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    config, run = _dry(tmp_path)

    def writer(summary):
        raise AssertionError("budget is zero")

    out = revise_hypothesis(ledger, HID, config, run, rationale_writer=writer)
    assert not out.used_llm
    assert any("budget exhausted" in n for n in out.notes)


def test_failing_writer_falls_back_without_losing_the_record(tmp_path):
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    config, run = _dry(tmp_path)

    def writer(summary):
        raise TimeoutError("down")

    budget = RevisionBudget(max_llm_calls=1)
    out = revise_hypothesis(ledger, HID, config, run, budget=budget, rationale_writer=writer)
    assert not out.used_llm and budget.llm_calls_used == 1
    assert any("TimeoutError" in n for n in out.notes)
    assert len(ledger.history(HID)) == 1


def test_run_without_a_transition_is_refused(tmp_path):
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    config, run = _dry(tmp_path)
    broken = run.model_copy(update={"state": run.state.model_copy(update={"transition_id": None})})
    with pytest.raises(RevisionGateError, match="no transition"):
        revise_hypothesis(ledger, HID, config, broken)


def test_invalid_budget_is_rejected():
    with pytest.raises(ValueError):
        RevisionBudget(max_revisions=0)
    with pytest.raises(ValueError):
        RevisionBudget(max_llm_calls=-1)


def test_ledger_chain_verifies_after_revisions(tmp_path):
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    config, run = _dispatched(tmp_path, ledger)
    revise_hypothesis(ledger, HID, config, run)
    reopened = EvidenceLedger(tmp_path / "l.jsonl")
    assert reopened.current_hypotheses()[HID].status is HypothesisStatus.UNRESOLVED
