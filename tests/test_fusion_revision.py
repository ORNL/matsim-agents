import json

import pytest

from matsim_agents.core.revision import RevisionBudget, RevisionBudgetError
from matsim_agents.execution.ledger import EvidenceLedger, HypothesisStatus
from matsim_agents.fusion import synthetic
from matsim_agents.fusion.policy import builtin_policy
from matsim_agents.fusion.revision import build_packet, parse_draft, revise_fusion_hypothesis
from matsim_agents.fusion.scoring import score_candidates

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


def test_overflowing_number_is_rejected_not_raised(scored):
    _, result = scored
    p = build_packet(result, POLICY)
    assert any("1e999" in x for x in parse_draft(_draft(rationale="A has 1e999."), p)[1])


def test_synthetic_benchmarks_never_support_or_contradict(tmp_path):
    from matsim_agents.core.contracts import EvidenceLevel
    from matsim_agents.execution.properties import PropertyQualification, PropertyStatus
    from matsim_agents.fusion.revision import _status
    from matsim_agents.fusion.scenario import FusionEvidence

    upd = {
        "status": PropertyStatus.CALCULATED,
        "evidence_level": EvidenceLevel.CONVERGED_DFT,
        "converged": True,
        "qualification": PropertyQualification.NUMERICALLY_VALIDATED,
        "uncertainty": 0.0,
    }
    cands = {
        k: [
            FusionEvidence(evidence=i.evidence.model_copy(update=upd), context=i.context) for i in v
        ]
        for k, v in synthetic.candidates().items()
        if k != "C"
    }
    real = synthetic.BENCHMARKS.model_copy(update={"synthetic": False})
    r_real = score_candidates(POLICY, real, synthetic.SCENARIO, cands)
    assert _status(r_real.by_id("A"), r_real)[0] == HypothesisStatus.SUPPORTED
    r_syn = score_candidates(POLICY, synthetic.BENCHMARKS, synthetic.SCENARIO, cands)
    assert _status(r_syn.by_id("A"), r_syn) == (
        HypothesisStatus.UNRESOLVED,
        "benchmarks are synthetic",
    )


def test_context_changes_ledger_identity_and_is_recorded(tmp_path, scored):
    cands, result = scored
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    out = revise_fusion_hypothesis(ledger, "h", "A", result, POLICY, cands)
    t = out.record.traceability
    assert t["scenario"]["scenario_id"] == result.scenario_id
    assert t["scenario"]["plasma_ion_energy_eV"] == 100.0
    assert out.packet.scenario == t["scenario"]
    obs = out.packet.candidates[0].objectives[0].observables[0]
    assert obs.context and obs.context["plasma_ion"] == "D"
    rec = ledger.get(obs.ledger_evidence_id)
    assert any("fusion context" in a for a in rec.physical_assumptions)

    # Same values, different context: new ledger ids.
    sc = synthetic.SCENARIO.model_copy(update={"plasma_ion_energy_eV": 200.0})
    moved = {
        k: [
            i.model_copy(
                update={"context": i.context.model_copy(update={"plasma_ion_energy_eV": 200.0})}
            )
            for i in v
        ]
        for k, v in cands.items()
    }
    r2 = score_candidates(POLICY, synthetic.BENCHMARKS, sc, moved)
    out2 = revise_fusion_hypothesis(ledger, "h2", "A", r2, POLICY, moved)
    obs2 = out2.packet.candidates[0].objectives[0].observables[0]
    assert obs2.property_id == obs.property_id
    assert obs2.ledger_evidence_id != obs.ledger_evidence_id


def test_packet_carries_constraint_spec(scored):
    from matsim_agents.campaign.selection import Constraint

    cands, _ = scored
    c = Constraint(property="yield_strength", minimum=123.0, temperature_K=1200.0)
    r = score_candidates(POLICY, synthetic.BENCHMARKS, synthetic.SCENARIO, cands, constraints=[c])
    p = build_packet(r, POLICY)
    pc = p.candidates[0].constraints[0]
    assert (pc.minimum, pc.temperature_K) == (123.0, 1200.0)
    assert pc.property_id in p.allowed_evidence_ids and 123.0 in p.numbers
    r2 = score_candidates(
        POLICY,
        synthetic.BENCHMARKS,
        synthetic.SCENARIO,
        cands,
        constraints=[c.model_copy(update={"minimum": 124.0})],
    )
    assert build_packet(r2, POLICY).digest != p.digest
