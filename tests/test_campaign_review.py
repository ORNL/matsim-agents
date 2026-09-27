from __future__ import annotations

import json

from matsim_agents.campaign.review import (
    CampaignDebateReviewConfig,
    _parse_verdict,
    run_campaign_debate_review,
)
from matsim_agents.campaign.state import CampaignState, FormulaRunRecord
from matsim_agents.discovery.formula import FormulaCandidate, FormulaGenerationPolicy
from matsim_agents.execution.contracts import WorkflowStatus
from matsim_agents.workflows.debate import DebateParticipant, DebateVerdict, ScientificDebateResult


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
    participants = [
        DebateParticipant(name="model-a", provider="vllm", model="a"),
        DebateParticipant(name="model-b", provider="vllm", model="b"),
    ]

    def debate_runner(config, *, model_factory):
        assert "NbO2" in config.hypothesis
        responses = [
            {
                "decisions": [
                    {"formula": "NbO", "action": "deactivate", "rationale": "poor evidence"},
                    {"formula": "NbO2", "action": "deactivate", "rationale": "uncertain"},
                ]
            },
            {
                "decisions": [
                    {"formula": "NbO", "action": "deactivate", "rationale": "poor evidence"},
                    {"formula": "NbO2", "action": "keep", "rationale": "needs more evidence"},
                ]
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
        return ScientificDebateResult(
            run_id="review-001",
            run_directory=str(tmp_path),
            status=WorkflowStatus.COMPLETE,
            hypothesis=config.hypothesis,
            rounds_completed=config.rounds,
            turns=[],
            verdicts=verdicts,
            synthesis="",
            transcript_path=str(tmp_path / "transcript.json"),
            dialogue_path=str(tmp_path / "dialogue.json"),
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
    assert decision.reactivate_formulas == []


def test_parse_verdict_accepts_json_after_reasoning() -> None:
    verdict = _parse_verdict(
        'The schema is {"decisions": []}, followed by the final assessment.\n'
        '{"decisions": [{"formula": "NbO2", "action": "keep", "rationale": "stable"}]}'
    )

    assert verdict.decisions[0].formula == "NbO2"
    assert verdict.decisions[0].action == "keep"


def test_campaign_review_treats_invalid_verdict_as_abstention(tmp_path) -> None:
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
    participants = [
        DebateParticipant(name="model-a", provider="vllm", model="a"),
        DebateParticipant(name="model-b", provider="vllm", model="b"),
    ]

    def debate_runner(config, *, model_factory):
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
        return ScientificDebateResult(
            run_id="review-invalid",
            run_directory=str(tmp_path),
            status=WorkflowStatus.COMPLETE,
            hypothesis=config.hypothesis,
            rounds_completed=config.rounds,
            turns=[],
            verdicts=verdicts,
            synthesis="",
            transcript_path=str(tmp_path / "transcript.json"),
            dialogue_path=str(tmp_path / "dialogue.json"),
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
    assert decision.notes[-1] == (
        "[model-b] invalid review verdict: response contains no JSON object"
    )
