from __future__ import annotations

from matsim_agents.campaign.acquisition import (
    CampaignAcquisitionPolicy,
    FormulaAcquisitionMetrics,
    select_formula_batch,
)
from matsim_agents.campaign.execution import latest_promoted_model
from matsim_agents.campaign.orchestrator import (
    CampaignReviewDecision,
    CampaignRunPolicy,
    run_campaign,
)
from matsim_agents.campaign.state import CampaignState, FormulaRunRecord
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.formula import FormulaCandidate, FormulaGenerationPolicy
from matsim_agents.discovery.stability import ReferenceEnergySet, ReferencePhaseEntry
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
            outcome_class="generation_failure",
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
    assert result.campaign.formula_runs["NbO2"].candidate_counts == {
        "generated": 0,
        "attempted": 0,
        "completed": 0,
        "converged": 0,
        "failed": 0,
    }
    assert result.campaign.formula_runs["NbO2"].outcome_class == "generation_failure"
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


def test_campaign_resume_retries_interrupted_running_formula(tmp_path):
    campaign = _campaign()
    campaign.formula_runs["NbO2"] = FormulaRunRecord(
        formula="NbO2",
        status=WorkflowStatus.RUNNING,
        attempts=1,
    )
    calls: list[str] = []

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        calls.append(formula)
        return _result(formula, output_dir)

    result = run_campaign(
        campaign,
        output_dir=tmp_path,
        formula_runner=runner,
        policy=CampaignRunPolicy(max_iterations=1),
        resume=False,
    )

    assert calls == ["NbO2"]
    assert result.campaign.formula_runs["NbO2"].attempts == 2


def test_latest_promotion_uses_persisted_execution_order(tmp_path):
    campaign = _campaign()
    campaign.formula_runs = {
        "NbO": FormulaRunRecord(formula="NbO", status=WorkflowStatus.PLANNED),
        "NbO2": FormulaRunRecord(formula="NbO2", status=WorkflowStatus.PLANNED),
    }
    checkpoints = {"NbO2": "promoted-first.pt", "NbO": "promoted-last.pt"}

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        result = _result(formula, output_dir)
        result.model_promoted = True
        result.active_learning_result["iteration_states"] = [
            {"model_promoted": True, "new_logdir": checkpoints[formula]}
        ]
        return result

    result = run_campaign(
        campaign,
        output_dir=tmp_path,
        formula_runner=runner,
        policy=CampaignRunPolicy(formulas_per_iteration=2, max_iterations=1),
        resume=False,
    )

    assert result.campaign.formula_runs["NbO2"].promotion_sequence == 0
    assert result.campaign.formula_runs["NbO"].promotion_sequence == 1
    assert latest_promoted_model(result.campaign) == "promoted-last.pt"
    persisted = CampaignState.load(result.state_path)
    assert latest_promoted_model(persisted) == "promoted-last.pt"


def test_campaign_retry_inconclusive_skips_usable_completed_formula(tmp_path):
    campaign = _campaign()
    campaign.formula_runs["NbO2"] = FormulaRunRecord(
        formula="NbO2",
        status=WorkflowStatus.COMPLETE,
        outcome_class="generation_failure",
    )
    campaign.formula_runs["NbO"] = FormulaRunRecord(
        formula="NbO",
        status=WorkflowStatus.COMPLETE,
        outcome_class="usable_minimum",
    )
    calls: list[str] = []

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        calls.append(formula)
        return _result(formula, output_dir)

    result = run_campaign(
        campaign,
        output_dir=tmp_path,
        formula_runner=runner,
        policy=CampaignRunPolicy(max_iterations=1, retry_inconclusive=True),
        resume=False,
    )

    assert calls == ["NbO2"]
    assert result.campaign.formula_runs["NbO2"].attempts == 1
    assert result.campaign.formula_runs["NbO"].attempts == 0


def test_campaign_persists_reference_registry_from_formula_evidence(tmp_path):
    references = ReferenceEnergySet(
        identifier="nb-o-pbe-v1",
        method_signature="pbe-v1",
        backend="qe",
        elemental_energies_eV_per_atom={"Nb": -10.0, "O": -5.0},
        phase_entries=[
            ReferencePhaseEntry(
                phase_id="NbO2-rutile",
                formula="NbO2",
                formation_energy_eV_per_atom=-2.0,
                method_signature="pbe-v1",
                backend="qe",
            )
        ],
    )

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        result = _result(formula, output_dir)
        result.active_learning_result["dft_refinement"] = {
            "reference_energy_set": references.model_dump(mode="json")
        }
        return result

    result = run_campaign(
        _campaign(max_candidates=1),
        output_dir=tmp_path,
        formula_runner=runner,
    )

    persisted = CampaignState.load(result.state_path)
    assert persisted.reference_energies is not None
    assert persisted.reference_energies.phase_entries[0].phase_id == "NbO2-rutile"


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


def test_adaptive_acquisition_reserves_branches_updates_lambda_and_persists(tmp_path):
    campaign = _campaign()
    campaign.acquisition.formula_metrics = {
        "NbO": FormulaAcquisitionMetrics(
            predicted_hull_proximity=1.0,
            mlip_uncertainty=0.0,
        ),
        "NbO2": FormulaAcquisitionMetrics(
            mlip_uncertainty=1.0,
            structural_novelty=1.0,
            structural_diversity=1.0,
        ),
        "Nb2O3": FormulaAcquisitionMetrics(
            predicted_hull_proximity=0.5,
            mlip_uncertainty=0.5,
        ),
    }
    calls: list[str] = []

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        calls.append(formula)
        result = _result(formula, output_dir)
        result.active_learning_result["useful_outcome"] = formula == "NbO2"
        return result

    result = run_campaign(
        campaign,
        output_dir=tmp_path,
        formula_runner=runner,
        policy=CampaignRunPolicy(
            formulas_per_iteration=2,
            max_iterations=1,
            acquisition=CampaignAcquisitionPolicy(
                enabled=True,
                lambda_initial=0.5,
                update_rate=0.2,
                minimum_exploitation_fraction=0.5,
                minimum_exploration_fraction=0.5,
            ),
        ),
    )

    assert calls == ["NbO", "NbO2"]
    assert result.campaign.formula_runs["NbO"].acquisition_branch == "exploitation"
    assert result.campaign.formula_runs["NbO2"].acquisition_branch == "exploration"
    assert result.campaign.acquisition.lambda_value == 0.3
    selection = result.campaign.acquisition.selection_history[0]
    assert selection.selected_formulas == calls
    assert selection.scores["NbO"].exploitation_score > 0
    assert selection.scores["NbO2"].exploration_score > 0
    assert CampaignState.load(result.state_path).acquisition == result.campaign.acquisition


def test_adaptive_single_item_batches_eventually_reserve_exploration():
    campaign = _campaign()
    policy = CampaignAcquisitionPolicy(
        enabled=True,
        minimum_exploitation_fraction=0.2,
        minimum_exploration_fraction=0.2,
    )

    for iteration in range(5):
        select_formula_batch(
            eligible=list(campaign.formulas),
            batch_size=1,
            iteration=iteration,
            policy=policy,
            state=campaign.acquisition,
        )

    branches = [
        record.scores[record.selected_formulas[0]].assigned_branch
        for record in campaign.acquisition.selection_history
    ]
    assert "exploitation" in branches
    assert "exploration" in branches


def test_registry_ingestion_uses_mlip_provenance(tmp_path, monkeypatch):
    observed = {}

    def capture_ingestion(*args, **kwargs):
        observed.update(kwargs)
        return []

    monkeypatch.setattr(
        "matsim_agents.campaign.orchestrator.ingest_exploration_result",
        capture_ingestion,
    )

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        result = _result(formula, output_dir)
        assert result.active_learning_result is not None
        result.active_learning_result.update(
            {
                "mlip_backend": "mace",
                "model_identifier": "mace:checkpoint:/models/promoted.model",
                "dft_refinement": {"backend": "qe"},
            }
        )
        return result

    run_campaign(
        _campaign(max_candidates=1),
        output_dir=tmp_path,
        formula_runner=runner,
    )

    assert observed["backend"] == "mace"
    assert observed["model_identifier"] == "mace:checkpoint:/models/promoted.model"


def test_campaign_does_not_admit_formula_beyond_reserved_budget(tmp_path):
    campaign = _campaign()
    campaign.budget.max_dft_calculations = 3
    campaign.formula_runs["NbO2"] = FormulaRunRecord(
        formula="NbO2",
        status=WorkflowStatus.COMPLETE,
        n_dft_calculations=2,
    )
    calls: list[str] = []

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        calls.append(formula)
        return _result(formula, output_dir)

    result = run_campaign(
        campaign,
        output_dir=tmp_path,
        formula_runner=runner,
        policy=CampaignRunPolicy(
            formulas_per_iteration=2,
            reserved_dft_calculations_per_formula=2,
        ),
    )

    assert calls == []
    assert result.stop_reason == "remaining budget is below per-formula reservation"
    assert result.campaign.status == WorkflowStatus.PARTIAL


def test_campaign_passes_remaining_dft_allowance_to_formula_runner(tmp_path):
    campaign = _campaign(max_candidates=1)
    campaign.budget.max_dft_calculations = 3
    allowances: list[int] = []

    def runner(formula: str, output_dir: str, dft_allowance: int) -> PhaseExplorationWorkflowResult:
        allowances.append(dft_allowance)
        result = _result(formula, output_dir)
        result.active_learning_result = {"n_dft_calculations": dft_allowance}
        return result

    result = run_campaign(campaign, output_dir=tmp_path, formula_runner=runner)

    assert allowances == [3]
    assert result.campaign.formula_runs["NbO2"].n_dft_calculations == 3


def test_pure_acquisition_mode_records_forced_branch():
    campaign = _campaign()
    campaign.acquisition.formula_metrics["NbO"] = FormulaAcquisitionMetrics(
        structural_novelty=1.0,
    )
    selection = select_formula_batch(
        eligible=["NbO"],
        batch_size=1,
        iteration=0,
        policy=CampaignAcquisitionPolicy(enabled=True, mode="exploitation"),
        state=campaign.acquisition,
    )

    assert selection.scores["NbO"].exploration_score > 0
    assert selection.scores["NbO"].assigned_branch == "exploitation"


def test_campaign_stops_at_formula_coverage_threshold(tmp_path):
    calls: list[str] = []

    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        calls.append(formula)
        return _result(formula, output_dir)

    result = run_campaign(
        _campaign(),
        output_dir=tmp_path,
        formula_runner=runner,
        policy=CampaignRunPolicy(
            formulas_per_iteration=1,
            minimum_formula_coverage=2 / 3,
        ),
    )

    assert calls == ["NbO2", "NbO"]
    assert result.stop_reason == "minimum formula coverage reached"
    assert result.campaign.status == WorkflowStatus.PARTIAL


def test_campaign_runs_final_review_after_stopping(tmp_path):
    review_calls: list[list[str]] = []

    def final_review(campaign, records):
        review_calls.append([record.formula for record in records])
        return CampaignReviewDecision(debate_run_id="final-review")

    result = run_campaign(
        _campaign(max_candidates=1),
        output_dir=tmp_path,
        formula_runner=_result,
        final_review_runner=final_review,
    )

    assert review_calls == [["NbO2"]]
    assert result.campaign.review_history[-1].debate_run_id == "final-review"
