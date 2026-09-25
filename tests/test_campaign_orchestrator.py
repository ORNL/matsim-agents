from __future__ import annotations

from matsim_agents.campaign.orchestrator import (
    CampaignReviewDecision,
    CampaignRunPolicy,
    run_campaign,
)
from matsim_agents.campaign.state import CampaignState
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.formula import FormulaCandidate, FormulaGenerationPolicy
from matsim_agents.discovery.wrapper import CompositionExplorationResult
from matsim_agents.execution.contracts import ComputeBudget, WorkflowStatus
from matsim_agents.workflows.phase_exploration import PhaseExplorationWorkflowResult


def _campaign(*, max_candidates: int | None = None) -> CampaignState:
    policy = FormulaGenerationPolicy(
        elements=["Nb", "O"],
        maximum_coefficient=4,
        require_charge_balance=False,
    )
    formulas = {
        "NbO": FormulaCandidate(
            formula_id="det-NbO",
            reduced_formula="NbO",
            elements={"Nb": 1, "O": 1},
            charge_balanced=True,
        ),
        "NbO2": FormulaCandidate(
            formula_id="llm-NbO2",
            reduced_formula="NbO2",
            elements={"Nb": 1, "O": 2},
            charge_balanced=True,
            generation_source="llm",
            llm_contributors=["qwen"],
        ),
        "Nb2O3": FormulaCandidate(
            formula_id="det-Nb2O3",
            reduced_formula="Nb2O3",
            elements={"Nb": 2, "O": 3},
            charge_balanced=True,
        ),
    }
    return CampaignState(
        campaign_id="nb-o-campaign",
        element_set=["Nb", "O"],
        formula_policy=policy,
        formulas=formulas,
        budget=ComputeBudget(max_candidates=max_candidates),
    )


def _result(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
    composition = parse_composition(formula)
    assert composition is not None
    return PhaseExplorationWorkflowResult(
        composition=formula,
        initial=CompositionExplorationResult(
            composition=composition,
            phase_candidates=[],
            relaxations=[],
        ),
        active_learning_result={
            "n_dft_converged": 2,
            "iteration_states": [{"status": "complete", "score_mean": 0.5}],
        },
    )


def test_campaign_enforces_budget_and_prioritizes_llm_formula(tmp_path):
    calls: list[str] = []

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        calls.append(formula)
        return _result(formula, output_dir)

    result = run_campaign(
        _campaign(max_candidates=1),
        output_dir=tmp_path,
        formula_runner=runner,
        policy=CampaignRunPolicy(formulas_per_iteration=2),
    )

    assert calls == ["NbO2"]
    assert result.stop_reason == "max_candidates budget reached"
    assert result.campaign.status == WorkflowStatus.PARTIAL
    assert result.campaign.formula_runs["NbO2"].status == WorkflowStatus.COMPLETE
    assert result.campaign.formula_runs["NbO2"].evidence["iteration_states"][0]["score_mean"] == 0.5
    assert CampaignState.load(result.state_path) == result.campaign


def test_campaign_resume_does_not_repeat_completed_formula(tmp_path):
    calls: list[str] = []

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        calls.append(formula)
        return _result(formula, output_dir)

    first = run_campaign(
        _campaign(),
        output_dir=tmp_path,
        formula_runner=runner,
        policy=CampaignRunPolicy(max_iterations=1),
    )
    assert first.formulas_completed == ["NbO2"]

    second = run_campaign(
        _campaign(),
        output_dir=tmp_path,
        formula_runner=runner,
        policy=CampaignRunPolicy(max_iterations=1),
    )
    assert calls == ["NbO2", "NbO"]
    assert second.formulas_completed == ["NbO"]
    assert second.campaign.formula_runs["NbO2"].attempts == 1


def test_campaign_applies_review_decision_before_next_iteration(tmp_path):
    calls: list[str] = []

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        calls.append(formula)
        return _result(formula, output_dir)

    def review(campaign, records):
        if records[0].formula == "NbO2":
            return CampaignReviewDecision(
                debate_run_id="review-001",
                deactivate_formulas=["NbO"],
            )
        return CampaignReviewDecision()

    result = run_campaign(
        _campaign(),
        output_dir=tmp_path,
        formula_runner=runner,
        review_runner=review,
    )

    assert calls == ["NbO2", "Nb2O3"]
    assert not result.campaign.formulas["NbO"].active
    assert result.campaign.debate_run_ids == ["review-001"]
    assert result.campaign.status == WorkflowStatus.COMPLETE
