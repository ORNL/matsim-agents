import json

import pytest

from matsim_agents.execution.ledger import EvidenceLedger, HypothesisStatus
from matsim_agents.fusion import synthetic
from matsim_agents.fusion.policy import builtin_policy
from matsim_agents.fusion.revision import build_packet, parse_draft, revise_fusion_hypothesis
from matsim_agents.fusion.scoring import score_candidates
from matsim_agents.workflows.radiation_hypothesis import RevisionBudget, RevisionBudgetError

POLICY = builtin_policy("fusion_pfc_initial_v1")


@pytest.fixture
def scored():
    cands = synthetic.candidates()
    return cands, score_candidates(POLICY, synthetic.BENCHMARKS, synthetic.SCENARIO, cands)


def _draft(**kw):
    base = {"rationale": "C lacks a shutdown dose rate.", "missing_evidence": ["dose"]}
    base.update(kw)
    return json.dumps(base)


def test_packet_carries_policy_and_digests(scored):
    _, result = scored
    p = build_packet(result, POLICY)
    assert p.policy_version == "fusion_pfc_initial_v1"
    assert p.score_digest == result.digest() and p.digest.startswith("fpk-")
    assert p.synthetic_benchmarks
    assert 0.25 in p.numbers
    with pytest.raises(ValueError, match="policy"):
        build_packet(result, POLICY.with_weights({**POLICY.weights}, policy_version="other"))


def test_draft_checks(scored):
    _, result = scored
    p = build_packet(result, POLICY)
    assert parse_draft(_draft(), p)[1] == []
    assert "schema" in parse_draft(_draft(score=0.9), p)[1][0]
    assert "schema" in parse_draft("not json", p)[1][0]
    assert any("0.123456" in x for x in parse_draft(_draft(rationale="S is 0.123456"), p)[1])
    assert any(
        "not in the packet" in x
        for x in parse_draft(_draft(supporting_evidence_ids=["ev-0123456789abcdef"]), p)[1]
    )
    for text in (
        "We should increase the weight of conductivity.",
        "Relax the yield constraint for C.",
        "Reweighting would favour C.",
    ):
        assert any("alter" in x for x in parse_draft(_draft(rationale=text), p)[1]), text
    q = parse_draft(_draft(rationale="C is scientifically qualified."), p)[1]
    assert any("qualification" in x for x in q)


def test_revision_records_provenance(tmp_path, scored):
    cands, result = scored
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    out = revise_fusion_hypothesis(ledger, "h-pfc", "A", result, POLICY, cands)
    r = out.record
    # surrogate-only synthetic evidence can never support the claim
    assert r.status == HypothesisStatus.UNRESOLVED
    t = r.traceability
    assert t["policy_version"] == "fusion_pfc_initial_v1"
    assert t["parent_revision_id"] is None
    assert t["score_digest"] == result.digest()
    assert t["supporting_evidence"] and set(t["supporting_evidence"]) <= set(r.evidence_ids)
    assert t["recommended_calculation"]
    c = revise_fusion_hypothesis(ledger, "h-pfc", "C", result, POLICY, cands).record
    assert c.prior_revision_id == r.revision_id
    assert c.traceability["parent_revision_id"] == r.revision_id
    assert any("low_activation_shutdown_dose" in m for m in c.traceability["missing_evidence"])
    assert c.proposed_tests[0].startswith("low_activation_shutdown_dose")


def test_llm_cannot_change_score_or_status(tmp_path, scored):
    cands, result = scored
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    budget = RevisionBudget(max_llm_calls=2)
    bad = revise_fusion_hypothesis(
        ledger,
        "h",
        "A",
        result,
        POLICY,
        cands,
        budget=budget,
        drafter=lambda _: _draft(rationale="Lower the weight of dose; A scores 0.99."),
    )
    assert not bad.used_llm and bad.record.traceability["llm"]["rejected"]
    good = revise_fusion_hypothesis(
        ledger,
        "h",
        "A",
        result,
        POLICY,
        cands,
        budget=budget,
        drafter=lambda _: _draft(),
        agent_model_id="m",
    )
    assert good.used_llm and good.record.status == bad.record.status
    assert good.record.agent_model_id == "m"
    assert (
        result.by_id("A").score
        == score_candidates(POLICY, synthetic.BENCHMARKS, synthetic.SCENARIO, cands)
        .by_id("A")
        .score
    )
    third = revise_fusion_hypothesis(
        ledger, "h", "A", result, POLICY, cands, budget=budget, drafter=lambda _: _draft()
    )
    assert not third.used_llm and "exhausted" in third.notes[0]
    with pytest.raises(RevisionBudgetError):
        revise_fusion_hypothesis(
            ledger, "h", "A", result, POLICY, cands, budget=RevisionBudget(max_revisions=3)
        )
