from __future__ import annotations

import json

from matsim_agents.campaign.review import (
    CampaignDebateReviewConfig,
    _parse_verdict,
    _review_evidence,
    run_campaign_debate_review,
)
from matsim_agents.campaign.state import CampaignState, FormulaRunRecord
from matsim_agents.discovery.formula import FormulaCandidate, FormulaGenerationPolicy
from matsim_agents.discovery.stability import PhaseStability, StabilityReport
from matsim_agents.execution.contracts import WorkflowStatus
from matsim_agents.workflows.debate import DebateParticipant, DebateVerdict, ScientificDebateResult


def test_review_evidence_defines_authoritative_numerical_semantics() -> None:
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

    evidence = json.loads(_review_evidence(campaign, [record]))
    facts = evidence["formulas"][0]["authoritative_facts"]

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
    )
    records = [FormulaRunRecord(formula="Nb2O3", status=WorkflowStatus.COMPLETE)]
    participants = [
        DebateParticipant(name="model-a", provider="vllm", model="a"),
        DebateParticipant(name="model-b", provider="vllm", model="b"),
    ]

    def debate_runner(config, *, model_factory):
        response = json.dumps(
            {
                "decisions": [
                    {
                        "formula": "Nb2O3",
                        "action": "deactivate",
                        "rationale": "The prototype identifier has the wrong stoichiometry.",
                    }
                ]
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
        return ScientificDebateResult(
            run_id="review-numerical-guard",
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
    assert sum("rejected Nb2O3 deactivation" in note for note in decision.notes) == 2


def test_parse_verdict_accepts_json_after_reasoning() -> None:
    verdict = _parse_verdict(
        'The schema is {"decisions": []}, followed by the final assessment.\n'
        '{"decisions": [{"formula": "NbO2", "action": "keep", "rationale": "stable"}]}'
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
    participants = [
        DebateParticipant(name="model-a", provider="vllm", model="a"),
        DebateParticipant(name="model-b", provider="vllm", model="b"),
    ]

    def debate_runner(config, *, model_factory):
        response = {
            "decisions": [],
            "revisions": [
                {
                    "claim": "NbO2 is competitive",
                    "status": "supported",
                    "evidence": ["DFT hull distance"],
                    "proposed_test": "phonons",
                }
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
        return ScientificDebateResult(
            run_id="review-revision",
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
        config=CampaignDebateReviewConfig(participants=participants, output_root=str(tmp_path)),
        model_factory=lambda **_: None,
        debate_runner=debate_runner,
    )

    assert len(decision.hypothesis_revisions) == 2
    assert {proposal.formula for proposal in decision.formula_proposals} == {"NbO2"}
    assert sum("out-of-scope" in note for note in decision.notes) == 2


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
