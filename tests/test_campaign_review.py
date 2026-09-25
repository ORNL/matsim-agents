from __future__ import annotations

import json

from matsim_agents.campaign.review import CampaignDebateReviewConfig, run_campaign_debate_review
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
