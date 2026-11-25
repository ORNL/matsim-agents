"""MP7: selection scores drive hypothesis revision. All values are synthetic."""

from __future__ import annotations

import pytest

from matsim_agents.campaign.selection_revision import (
    build_evidence_packet,
    revise_selection_hypothesis,
    validate_draft,
)
from matsim_agents.core.revision import RevisionBudget, RevisionBudgetError
from matsim_agents.execution.ledger import EvidenceLedger, HypothesisRecord, HypothesisStatus
from matsim_agents.science.acquisition.selection import (
    CandidateEvidence,
    Constraint,
    Objective,
    SelectionSpec,
    select,
)
from matsim_agents.science.properties.calculators import PropertyConditions, PropertyStatus
from tests.test_selection import BULK, KAPPA, T, ev

SWELL = "irradiation_swelling"  # lower is better
HID = "h-select"


def cand(cid, kappa, swelling, status=PropertyStatus.CALCULATED):
    return CandidateEvidence(
        candidate_id=cid,
        evidence=[
            ev(KAPPA, kappa, 1.0, cid=cid, status=status),
            ev(BULK, 200.0, 2.0, cid=cid, status=status),
            ev(
                SWELL,
                swelling,
                0.001,
                cid=cid,
                status=status,
                conditions=PropertyConditions(temperature_K=T),
            ),
        ],
    )


def scenario():
    """A: excellent radiation recovery but poor thermal transport. B: balanced. C: surrogate."""
    spec = SelectionSpec(
        objectives=[
            Objective(property=KAPPA, direction="maximize", temperature_K=T),
            Objective(property=SWELL, direction="minimize"),
        ],
        constraints=[Constraint(property=KAPPA, minimum=20.0, temperature_K=T)],
        seed=1,
        n_samples=200,
    )
    candidates = [
        cand("A", 8.0, 0.001),
        cand("B", 45.0, 0.010),
        cand("C", 60.0, 0.020, status=PropertyStatus.SURROGATE_PREDICTED),
    ]
    return spec, candidates, select(spec, candidates)


@pytest.fixture
def ledger(tmp_path):
    return EvidenceLedger(tmp_path / "l.jsonl")


def test_thermal_shortfall_contradicts_despite_good_radiation(ledger):
    _, candidates, result = scenario()
    out = revise_selection_hypothesis(ledger, HID, "A", result, candidates)
    assert out.record.status == HypothesisStatus.CONTRADICTED
    row = next(c for c in out.packet.candidates if c.candidate_id == "A")
    assert row.status == "INFEASIBLE" and any(KAPPA in v for v in row.violated)
    assert {v.property: v.value for v in row.values}[SWELL] == 0.001
    assert out.record.evidence_ids and all(i in ledger for i in out.record.evidence_ids)
    assert not out.used_llm and out.record.scientific_rationale.startswith("Deterministic")


def test_balanced_candidate_is_supported_on_calculated_evidence_only(ledger):
    _, candidates, result = scenario()
    out = revise_selection_hypothesis(ledger, HID, "B", result, candidates)
    assert out.record.status == HypothesisStatus.SUPPORTED
    assert "scientifically" not in out.record.scientific_rationale.lower()
    assert out.record.traceability["selection_status"] == "QUALIFIED"
    assert any("experimental" in t or "qualify" in t for t in out.record.proposed_tests)


def test_surrogate_only_candidate_stays_unresolved_and_cites_nothing(ledger):
    _, candidates, result = scenario()
    out = revise_selection_hypothesis(ledger, HID, "C", result, candidates)
    assert out.record.status == HypothesisStatus.UNRESOLVED
    assert out.record.evidence_ids == []
    assert any("replace the surrogate_predicted" in t for t in out.record.proposed_tests)


def test_traceability_and_ledger_chain(ledger):
    _, candidates, result = scenario()
    first = revise_selection_hypothesis(ledger, HID, "C", result, candidates).record
    second = revise_selection_hypothesis(ledger, HID, "B", result, candidates).record
    assert second.prior_revision_id == first.revision_id
    trace = second.traceability
    assert trace["score_digest"] == result.score_digest
    assert trace["scoring_version"] == result.scoring_version
    assert trace["packet_digest"].startswith("pk-") and trace["candidate_id"] == "B"
    reopened = EvidenceLedger(ledger.path)
    assert reopened.current_hypotheses()[HID].revision_id == second.revision_id
    assert reopened.current_hypotheses()[HID].traceability == trace


def test_records_without_traceability_keep_their_identifiers():
    old = HypothesisRecord(hypothesis_id="h", scientific_claim="c", status="proposed")
    assert old.revision_id == HypothesisRecord.model_validate(old.model_dump()).revision_id
    assert "traceability" in old.model_dump() and old.traceability is None


def test_mock_llm_cannot_change_numbers(ledger):
    _, candidates, result = scenario()
    seen = {}

    def drafter(packet_text: str) -> str:
        seen["packet"] = packet_text
        return (
            "Candidate A recovers well under irradiation (swelling 0.001) but its thermal "
            "conductivity of 8 W/(m K) is below the 20 limit."
        )

    budget = RevisionBudget(max_llm_calls=1)
    out = revise_selection_hypothesis(
        ledger, HID, "A", result, candidates, budget=budget, drafter=drafter, agent_model_id="m"
    )
    assert out.used_llm and budget.llm_calls_used == 1
    assert out.record.status == HypothesisStatus.CONTRADICTED  # the LLM does not set status
    assert out.record.agent_model_id == "m"
    assert "numbers verified" in out.record.scientific_rationale
    assert result.score_digest in seen["packet"]
    assert result.by_id("A").values[KAPPA].value == 8.0  # scores are untouched


def test_invented_number_or_evidence_is_rejected(ledger):
    _, candidates, result = scenario()
    lies = [
        "Thermal conductivity is actually 35.7 W/(m K).",
        "Supported by ev-0123456789abcdef0123.",
        "This candidate is scientifically qualified.",
    ]
    for lie in lies:
        out = revise_selection_hypothesis(
            ledger,
            f"h-{abs(hash(lie))}",
            "A",
            result,
            candidates,
            budget=RevisionBudget(max_llm_calls=1),
            drafter=lambda _text, lie=lie: lie,
        )
        assert not out.used_llm and out.notes and "rejected" in out.notes[0]
        assert out.record.traceability["llm"]["rejected"]
        assert lie not in out.record.scientific_rationale
        assert out.record.scientific_rationale.startswith("Deterministic")


def test_validate_draft_accepts_packet_numbers_and_small_integers():
    _, candidates, result = scenario()
    packet = build_evidence_packet(result, {})
    assert validate_draft("Of the 3 candidates, A has swelling 0.001.", packet) == []
    assert validate_draft("A has swelling 0.002.", packet)


def test_llm_budget_and_revision_budget_are_enforced(ledger):
    _, candidates, result = scenario()
    calls = []
    out = revise_selection_hypothesis(
        ledger, HID, "B", result, candidates, drafter=lambda t: calls.append(t) or "x"
    )
    assert calls == [] and "budget" in out.notes[0] and not out.used_llm
    tight = RevisionBudget(max_revisions=1)
    with pytest.raises(RevisionBudgetError):
        revise_selection_hypothesis(ledger, HID, "A", result, candidates, budget=tight)


def test_drafter_failure_does_not_block_the_record(ledger):
    _, candidates, result = scenario()

    def boom(_):
        raise TimeoutError

    out = revise_selection_hypothesis(
        ledger, HID, "B", result, candidates, budget=RevisionBudget(max_llm_calls=1), drafter=boom
    )
    assert out.record.status == HypothesisStatus.SUPPORTED and not out.used_llm
