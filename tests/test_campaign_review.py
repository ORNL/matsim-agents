from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from matsim_agents.backends.llm.transcript import Contribution
from matsim_agents.campaign.evidence import open_campaign_ledger, sync_stability_evidence
from matsim_agents.campaign.review import (
    CampaignDebateReviewConfig,
    _deactivation_block_reason,
    _parse_verdict,
    _review_evidence,
    run_campaign_debate_review,
)
from matsim_agents.campaign.state import CampaignState, FormulaRunRecord
from matsim_agents.discovery.formula import FormulaCandidate, FormulaGenerationPolicy
from matsim_agents.discovery.stability import PhaseStability, StabilityReport
from matsim_agents.execution.contracts import EvidenceLevel, WorkflowStatus
from matsim_agents.workflows.debate import DebateParticipant, DebateVerdict, ScientificDebateResult


def _proposal():
    return {
        "claim": "Sampled oxide structures may differ in MLIP energy",
        "status": "proposed",
        "falsifiable_prediction": "Relaxed oxide prototypes will have different MLIP energies",
        "rationale": "This is a testable proposal, not thermodynamic validation.",
    }


def _validate_verdicts(verdicts, callbacks):
    for verdict in verdicts:
        try:
            payload = json.loads(verdict.response)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid mandatory hypothesis review") from exc
        hypotheses = []
        for index, revision in enumerate(payload.get("revisions", [])):
            hypotheses.append(
                {
                    "id": revision.get("hypothesis_id") or f"hypothesis-{index}",
                    "claim": revision["claim"],
                    "prediction": revision["falsifiable_prediction"],
                    "reason": revision["rationale"],
                    "status": (
                        "supported_by_mlip"
                        if revision["status"] == "supported"
                        else revision["status"]
                    ),
                    "evidence_ids": revision.get("evidence_ids") or ["campaign-design"],
                }
            )
        payload["scientific_hypotheses"] = hypotheses
        verdict.response = json.dumps(payload)
        callbacks["final_response_validator"](
            Contribution(
                id=verdict.contribution_id,
                kind="verdict",
                speaker=verdict.participant,
                model=verdict.model,
                text=verdict.response,
            ),
            hypotheses,
        )


@pytest.mark.parametrize("invalid_first", [False, True])
def test_real_debate_validates_every_round_before_continuing(tmp_path, invalid_first):
    campaign = CampaignState(
        campaign_id="round-review",
        element_set=["Nb", "O"],
        formula_policy=FormulaGenerationPolicy(elements=["Nb", "O"], require_charge_balance=False),
    )
    ledger = open_campaign_ledger(campaign, tmp_path)
    calls = []

    class Model:
        def invoke(self, messages, **kwargs):
            calls.append(messages)
            revision = _proposal()
            revision["hypothesis_id"] = "oxide-energy" if len(calls) > 1 else None
            hypothesis = {
                "id": "oxide-energy",
                "claim": revision["claim"],
                "prediction": revision["falsifiable_prediction"],
                "status": "proposed",
                "reason": revision["rationale"],
                "evidence_ids": ["campaign-design"],
            }
            if invalid_first and len(calls) == 1:
                revision["rationale"] = "Contradictory persisted ledger interpretation"
            return AIMessage(
                content=json.dumps(
                    {
                        "scientific_hypotheses": [hypothesis],
                        "decisions": [],
                        "revisions": [revision],
                        "new_formula_proposals": [],
                    }
                )
            )

    config = CampaignDebateReviewConfig(
        participants=[
            DebateParticipant(name=name, provider="vllm", model=name) for name in ("a", "b")
        ],
        output_root=str(tmp_path / "debates"),
        rounds=2,
    )
    if invalid_first:
        with pytest.raises(ValueError, match="hypothesis and ledger revision disagree"):
            run_campaign_debate_review(
                campaign,
                [],
                config=config,
                model_factory=lambda **kwargs: Model(),
            )
        assert len(calls) == 1
        assert ledger.history("oxide-energy") == []
        contributions = list((tmp_path / "debates").glob("*/contributions.jsonl"))
        assert contributions and "Contradictory" in contributions[0].read_text()
    else:
        decision = run_campaign_debate_review(
            campaign,
            [],
            config=config,
            model_factory=lambda **kwargs: Model(),
        )
        assert len(calls) == 6
        assert len(ledger.history("oxide-energy")) == 4
        assert len(decision.hypothesis_revisions) == 2
        assert all(row.hypothesis_id == "oxide-energy" for row in decision.hypothesis_revisions)


def test_review_evidence_defines_authoritative_numerical_semantics(tmp_path) -> None:
    policy = FormulaGenerationPolicy(
        elements=["Nb", "Ta", "O"],
        require_charge_balance=True,
    )
    campaign = CampaignState(
        campaign_id="evidence-semantics-test",
        element_set=["Nb", "Ta", "O"],
        formula_policy=policy,
        formulas={
            "Nb2O5Ta": FormulaCandidate(
                formula_id="det-Nb2O5Ta",
                reduced_formula="Nb2O5Ta",
                elements={"Nb": 2, "Ta": 1, "O": 5},
                charge_balanced=True,
            )
        },
    )
    record = FormulaRunRecord(
        formula="Nb2O5Ta",
        status=WorkflowStatus.COMPLETE,
        n_mlip_relaxations=0,
        evidence={
            "mlip_labels": [
                {"candidate_id": "one", "converged": True},
                {"candidate_id": "two", "converged": False},
            ]
        },
    )

    open_campaign_ledger(campaign, tmp_path)
    evidence = json.loads(_review_evidence(campaign, [record]))
    facts = evidence["formulas"][0]["authoritative_facts"]
    assert evidence["formulas"][0]["evidence_records"] == []
    assert evidence["hypotheses"] == []

    assert facts["canonical_formula"] == "Nb2O5Ta"
    assert facts["element_counts"] == {"Nb": 2, "Ta": 1, "O": 5}
    assert facts["charge_balanced_by_policy"] is True
    assert facts["observed_mlip_labels"] == 2
    assert facts["converged_mlip_labels"] == 1
    assert facts["has_formation_energy"] is False
    assert facts["has_energy_above_hull"] is False
    assert (
        "never infer composition from prototype_id" in evidence["evidence_semantics"]["composition"]
    )
    assert "does not establish formation energy" in evidence["evidence_semantics"]["energy_scope"]


def test_campaign_review_requires_cross_model_agreement(tmp_path):
    policy = FormulaGenerationPolicy(
        elements=["Nb", "O"],
        require_charge_balance=False,
    )
    campaign = CampaignState(
        campaign_id="review-test",
        element_set=["Nb", "O"],
        formula_policy=policy,
        formulas={
            formula: FormulaCandidate(
                formula_id=f"det-{formula}",
                reduced_formula=formula,
                elements=elements,
                charge_balanced=True,
            )
            for formula, elements in {
                "NbO": {"Nb": 1, "O": 1},
                "NbO2": {"Nb": 1, "O": 2},
            }.items()
        },
    )
    records = [
        FormulaRunRecord(formula="NbO", status=WorkflowStatus.COMPLETE),
        FormulaRunRecord(formula="NbO2", status=WorkflowStatus.COMPLETE),
    ]
    ranked_phase = PhaseStability(
        structure_path="NbO-seed.vasp",
        optimized_structure_path="NbO-relaxed.vasp",
        final_energy_eV=-10.0,
        energy_per_atom_eV=-5.0,
        energy_above_hull_eV_per_atom=0.2,
        delta_e_above_min_eV_per_atom=0.0,
        final_max_force_eV_per_A=0.01,
        converged=True,
        dynamically_stable_proxy=True,
    )
    campaign.stability_reports["NbO"] = StabilityReport(
        formula="NbO",
        ground_state=ranked_phase,
        ranking=[ranked_phase],
        chemically_stable_proxy=False,
        summary="Ranked above the hull.",
        evidence_level=EvidenceLevel.MLIP_RELAXATION,
    )
    ledger = open_campaign_ledger(campaign, tmp_path)
    [nbo_evidence] = sync_stability_evidence(ledger, campaign)
    participants = [
        DebateParticipant(name="model-a", provider="vllm", model="a"),
        DebateParticipant(name="model-b", provider="vllm", model="b"),
    ]

    def debate_runner(config, *, model_factory, **callbacks):
        assert "NbO2" in config.hypothesis
        assert nbo_evidence in config.hypothesis
        deactivate_nbo = {
            "formula": "NbO",
            "action": "deactivate",
            "rationale": "poor evidence",
            "evidence_ids": [nbo_evidence],
        }
        responses = [
            {
                "revisions": [_proposal()],
                "decisions": [
                    deactivate_nbo,
                    {"formula": "NbO2", "action": "deactivate", "rationale": "uncertain"},
                ],
            },
            {
                "revisions": [_proposal()],
                "decisions": [
                    deactivate_nbo,
                    {"formula": "NbO2", "action": "keep", "rationale": "needs more evidence"},
                ],
            },
        ]
        verdicts = [
            DebateVerdict(
                contribution_id=f"verdict-{index}",
                participant=participant.name,
                provider=participant.provider,
                model=participant.model,
                response=json.dumps(response),
            )
            for index, (participant, response) in enumerate(
                zip(participants, responses, strict=True)
            )
        ]
        _validate_verdicts(verdicts, callbacks)
        return ScientificDebateResult(
            run_id="review-001",
            run_directory=str(tmp_path),
            status=WorkflowStatus.COMPLETE,
            hypothesis=config.hypothesis,
            rounds_completed=config.rounds,
            turns=[],
            verdicts=verdicts,
            synthesis="",
            transcript_path=str(tmp_path / "transcript.md"),
            contributions_path=str(tmp_path / "contributions.jsonl"),
        )

    decision = run_campaign_debate_review(
        campaign,
        records,
        config=CampaignDebateReviewConfig(
            participants=participants,
            output_root=str(tmp_path),
            minimum_agreement_fraction=1.0,
        ),
        model_factory=lambda **_: None,
        debate_runner=debate_runner,
    )

    assert decision.debate_run_id == "review-001"
    assert decision.deactivate_formulas == ["NbO"]
    assert decision.evidence_ids == {"NbO": [nbo_evidence]}
    assert decision.reactivate_formulas == []


def test_campaign_review_cannot_deactivate_converged_result_without_hull_evidence(
    tmp_path,
) -> None:
    policy = FormulaGenerationPolicy(elements=["Nb", "O"], require_charge_balance=False)
    campaign = CampaignState(
        campaign_id="review-numerical-guard-test",
        element_set=["Nb", "O"],
        formula_policy=policy,
        formulas={
            "Nb2O3": FormulaCandidate(
                formula_id="det-Nb2O3",
                reduced_formula="Nb2O3",
                elements={"Nb": 2, "O": 3},
                charge_balanced=True,
            )
        },
    )
    phase = PhaseStability(
        structure_path="Nb2O3-seed.vasp",
        optimized_structure_path="Nb2O3-relaxed.vasp",
        final_energy_eV=-43.5,
        energy_per_atom_eV=-8.7,
        delta_e_above_min_eV_per_atom=0.0,
        final_max_force_eV_per_A=0.015,
        converged=True,
        dynamically_stable_proxy=True,
        prototype_id="A3B2_hR5_155_e_c",
    )
    campaign.stability_reports["Nb2O3"] = StabilityReport(
        formula="Nb2O3",
        ground_state=phase,
        ranking=[phase],
        chemically_stable_proxy=True,
        summary="Relative MLIP ranking only.",
        evidence_level=EvidenceLevel.MLIP_RELAXATION,
    )
    records = [FormulaRunRecord(formula="Nb2O3", status=WorkflowStatus.COMPLETE)]
    open_campaign_ledger(campaign, tmp_path)
    participants = [
        DebateParticipant(name="model-a", provider="vllm", model="a"),
        DebateParticipant(name="model-b", provider="vllm", model="b"),
    ]

    def debate_runner(config, *, model_factory, **callbacks):
        response = json.dumps(
            {
                "revisions": [_proposal()],
                "decisions": [
                    {
                        "formula": "Nb2O3",
                        "action": "deactivate",
                        "rationale": "The prototype identifier has the wrong stoichiometry.",
                    }
                ],
            }
        )
        verdicts = [
            DebateVerdict(
                contribution_id=f"verdict-{participant.name}",
                participant=participant.name,
                provider=participant.provider,
                model=participant.model,
                response=response,
            )
            for participant in participants
        ]
        _validate_verdicts(verdicts, callbacks)
        return ScientificDebateResult(
            run_id="review-numerical-guard",
            run_directory=str(tmp_path),
            status=WorkflowStatus.COMPLETE,
            hypothesis=config.hypothesis,
            rounds_completed=config.rounds,
            turns=[],
            verdicts=verdicts,
            synthesis="",
            transcript_path=str(tmp_path / "transcript.md"),
            contributions_path=str(tmp_path / "contributions.jsonl"),
        )

    decision = run_campaign_debate_review(
        campaign,
        records,
        config=CampaignDebateReviewConfig(
            participants=participants,
            output_root=str(tmp_path),
            minimum_agreement_fraction=1.0,
        ),
        model_factory=lambda **_: None,
        debate_runner=debate_runner,
    )

    assert decision.deactivate_formulas == []
    assert sum("rejected Nb2O3 deactivation" in note for note in decision.notes) == 2


def test_missing_stability_report_blocks_deactivation() -> None:
    campaign = CampaignState(
        campaign_id="missing-evidence-test",
        element_set=["Nb", "O"],
        formula_policy=FormulaGenerationPolicy(elements=["Nb", "O"], require_charge_balance=False),
    )

    reason = _deactivation_block_reason(
        campaign,
        FormulaRunRecord(formula="NbO", status=WorkflowStatus.FAILED),
    )

    assert reason == "authoritative numerical evidence has no stability report"


def test_parse_verdict_accepts_json_after_reasoning() -> None:
    verdict = _parse_verdict(
        'The schema is {"decisions": []}, followed by the final assessment.\n'
        + json.dumps(
            {
                "decisions": [{"formula": "NbO2", "action": "keep", "rationale": "stable"}],
                "revisions": [_proposal()],
            }
        )
    )

    assert verdict.decisions[0].formula == "NbO2"
    assert verdict.decisions[0].action == "keep"


def test_campaign_review_preserves_revisions_and_validates_new_formulas(tmp_path) -> None:
    policy = FormulaGenerationPolicy(
        elements=["Nb", "O"],
        require_charge_balance=True,
        oxidation_states={"Nb": [4], "O": [-2]},
    )
    campaign = CampaignState(
        campaign_id="revision-test",
        element_set=["Nb", "O"],
        formula_policy=policy,
        formulas={
            "NbO2": FormulaCandidate(
                formula_id="det-NbO2",
                reduced_formula="NbO2",
                elements={"Nb": 1, "O": 2},
                charge_balanced=True,
            )
        },
    )
    records = [FormulaRunRecord(formula="NbO2", status=WorkflowStatus.COMPLETE)]
    phase = PhaseStability(
        structure_path="NbO2-seed.vasp",
        optimized_structure_path="NbO2-relaxed.vasp",
        final_energy_eV=-30.0,
        energy_per_atom_eV=-10.0,
        delta_e_above_min_eV_per_atom=0.0,
        final_max_force_eV_per_A=0.01,
        converged=True,
        dynamically_stable_proxy=True,
    )
    campaign.stability_reports["NbO2"] = StabilityReport(
        formula="NbO2",
        ground_state=phase,
        ranking=[phase],
        chemically_stable_proxy=True,
        summary="MLIP ranking.",
        evidence_level=EvidenceLevel.MLIP_RELAXATION,
    )
    ledger = open_campaign_ledger(campaign, tmp_path)
    [mlip_ranking] = sync_stability_evidence(ledger, campaign)
    participants = [
        DebateParticipant(name="model-a", provider="vllm", model="a"),
        DebateParticipant(name="model-b", provider="vllm", model="b"),
    ]

    def debate_runner(config, *, model_factory, **callbacks):
        response = {
            "decisions": [],
            "revisions": [
                {
                    "claim": "The relaxed NbO2 seed is its lowest-energy structure",
                    "claim_type": "energy_ranking",
                    "status": "supported",
                    "evidence_ids": [mlip_ranking],
                    "falsifiable_prediction": "No other NbO2 prototype relaxes lower",
                    "rationale": "The cited MLIP ranking supports only the sampled ordering.",
                    "proposed_test": "phonons",
                },
                {
                    "claim": "NbO2 may be metallic",
                    "status": "proposed",
                    "falsifiable_prediction": "DFT may show no band gap",
                    "rationale": "Not tested by the MLIP calculation.",
                },
            ],
            "new_formula_proposals": [
                {
                    "formula": "Nb2O4",
                    "rationale": "equivalent notation",
                    "falsification_tests": ["DFT"],
                },
                {
                    "formula": "TaO2",
                    "rationale": "out of scope",
                    "falsification_tests": ["DFT"],
                },
            ],
        }
        verdicts = [
            DebateVerdict(
                contribution_id=f"verdict-{participant.name}",
                participant=participant.name,
                provider=participant.provider,
                model=participant.model,
                response=json.dumps(response),
            )
            for participant in participants
        ]
        _validate_verdicts(verdicts, callbacks)
        return ScientificDebateResult(
            run_id="review-revision",
            run_directory=str(tmp_path),
            status=WorkflowStatus.COMPLETE,
            hypothesis=config.hypothesis,
            rounds_completed=config.rounds,
            turns=[],
            verdicts=verdicts,
            synthesis="",
            transcript_path=str(tmp_path / "transcript.md"),
            contributions_path=str(tmp_path / "contributions.jsonl"),
        )

    decision = run_campaign_debate_review(
        campaign,
        records,
        config=CampaignDebateReviewConfig(participants=participants, output_root=str(tmp_path)),
        model_factory=lambda **_: None,
        debate_runner=debate_runner,
    )

    # Each participant: the cited ranking claim and the uncited proposal are kept.
    assert len(decision.hypothesis_revisions) == 4
    kept = {(r.scientific_claim, r.agent_model_id) for r in decision.hypothesis_revisions}
    assert ("NbO2 may be metallic", "model-a:a") in kept
    assert {proposal.formula for proposal in decision.formula_proposals} == {"NbO2"}
    assert sum("out-of-scope" in note for note in decision.notes) == 2


def test_campaign_review_rejects_invalid_verdict_instead_of_abstaining(tmp_path) -> None:
    policy = FormulaGenerationPolicy(elements=["Nb", "O"], require_charge_balance=False)
    campaign = CampaignState(
        campaign_id="invalid-review-test",
        element_set=["Nb", "O"],
        formula_policy=policy,
        formulas={
            "NbO2": FormulaCandidate(
                formula_id="det-NbO2",
                reduced_formula="NbO2",
                elements={"Nb": 1, "O": 2},
                charge_balanced=True,
            )
        },
    )
    records = [FormulaRunRecord(formula="NbO2", status=WorkflowStatus.COMPLETE)]
    open_campaign_ledger(campaign, tmp_path)
    participants = [
        DebateParticipant(name="model-a", provider="vllm", model="a"),
        DebateParticipant(name="model-b", provider="vllm", model="b"),
    ]

    def debate_runner(config, *, model_factory, **callbacks):
        verdicts = [
            DebateVerdict(
                contribution_id="verdict-a",
                participant="model-a",
                provider="vllm",
                model="a",
                response=(
                    '{"decisions": [{"formula": "NbO2", "action": "deactivate", '
                    '"rationale": "poor evidence"}]}'
                ),
            ),
            DebateVerdict(
                contribution_id="verdict-b",
                participant="model-b",
                provider="vllm",
                model="b",
                response="analysis without a JSON verdict",
            ),
        ]
        _validate_verdicts(verdicts, callbacks)
        return ScientificDebateResult(
            run_id="review-invalid",
            run_directory=str(tmp_path),
            status=WorkflowStatus.COMPLETE,
            hypothesis=config.hypothesis,
            rounds_completed=config.rounds,
            turns=[],
            verdicts=verdicts,
            synthesis="",
            transcript_path=str(tmp_path / "transcript.md"),
            contributions_path=str(tmp_path / "contributions.jsonl"),
        )

    with pytest.raises(ValueError, match="invalid mandatory hypothesis review"):
        run_campaign_debate_review(
            campaign,
            records,
            config=CampaignDebateReviewConfig(
                participants=participants,
                output_root=str(tmp_path),
                minimum_agreement_fraction=1.0,
            ),
            model_factory=lambda **_: None,
            debate_runner=debate_runner,
        )
