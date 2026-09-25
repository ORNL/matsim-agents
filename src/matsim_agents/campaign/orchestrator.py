"""Resumable orchestration for multi-formula discovery campaigns."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from matsim_agents.campaign.state import CampaignReviewRecord, CampaignState, FormulaRunRecord
from matsim_agents.discovery.formula import enumerate_formulas
from matsim_agents.discovery.formula_merge import extract_llm_formula_proposals, merge_formulas
from matsim_agents.execution.contracts import WorkflowStatus
from matsim_agents.workflows.debate import ScientificDebateResult
from matsim_agents.workflows.phase_exploration import PhaseExplorationWorkflowResult


class CampaignRunPolicy(BaseModel):
    """Control limits and failure behavior for a campaign run."""

    formulas_per_iteration: int = Field(1, ge=1)
    max_iterations: int | None = Field(None, ge=1)
    continue_on_failure: bool = True
    retry_failed: bool = False


class CampaignReviewDecision(BaseModel):
    """Validated campaign changes returned by an evidence-review stage."""

    debate_run_id: str | None = None
    deactivate_formulas: list[str] = Field(default_factory=list)
    reactivate_formulas: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CampaignRunResult(BaseModel):
    campaign: CampaignState
    state_path: str
    formulas_completed: list[str] = Field(default_factory=list)
    formulas_failed: list[str] = Field(default_factory=list)
    stop_reason: str


FormulaRunner = Callable[[str, str], PhaseExplorationWorkflowResult]
ReviewRunner = Callable[[CampaignState, list[FormulaRunRecord]], CampaignReviewDecision]


def run_formula_discovery_stage(
    campaign: CampaignState,
    debate_result: ScientificDebateResult,
) -> CampaignState:
    """Merge deterministic and LLM-proposed formulas into ``campaign`` in place."""
    deterministic = enumerate_formulas(campaign.formula_policy)
    llm_proposals = extract_llm_formula_proposals(debate_result.verdicts, campaign.formula_policy)
    merged = merge_formulas(
        deterministic, llm_proposals, campaign.formula_policy, iteration=campaign.iteration
    )
    campaign.upsert_formulas(merged)
    campaign.debate_run_ids.append(debate_result.run_id)
    campaign.iteration += 1
    return campaign


def _formula_priority(campaign: CampaignState, formula: str) -> tuple[int, int, str]:
    candidate = campaign.formulas[formula]
    return (
        0 if candidate.llm_contributors else 1,
        sum(candidate.elements.values()),
        formula,
    )


def _eligible_formulas(campaign: CampaignState, *, retry_failed: bool) -> list[str]:
    eligible: list[str] = []
    for candidate in campaign.active_formulas():
        record = campaign.formula_runs.get(candidate.reduced_formula)
        if (
            record is None
            or record.status == WorkflowStatus.PLANNED
            or (retry_failed and record.status == WorkflowStatus.FAILED)
        ):
            eligible.append(candidate.reduced_formula)
    return sorted(eligible, key=lambda formula: _formula_priority(campaign, formula))


def _budget_stop_reason(campaign: CampaignState) -> str | None:
    completed_or_failed = sum(
        record.status in {WorkflowStatus.COMPLETE, WorkflowStatus.FAILED}
        for record in campaign.formula_runs.values()
    )
    if (
        campaign.budget.max_candidates is not None
        and completed_or_failed >= campaign.budget.max_candidates
    ):
        return "max_candidates budget reached"

    relaxations = sum(record.n_mlip_relaxations for record in campaign.formula_runs.values())
    if (
        campaign.budget.max_mlip_relaxations is not None
        and relaxations >= campaign.budget.max_mlip_relaxations
    ):
        return "max_mlip_relaxations budget reached"

    dft_calculations = sum(record.n_dft_calculations for record in campaign.formula_runs.values())
    if (
        campaign.budget.max_dft_calculations is not None
        and dft_calculations >= campaign.budget.max_dft_calculations
    ):
        return "max_dft_calculations budget reached"

    al_iterations = sum(
        record.n_active_learning_iterations for record in campaign.formula_runs.values()
    )
    if (
        campaign.budget.max_active_learning_iterations is not None
        and al_iterations >= campaign.budget.max_active_learning_iterations
    ):
        return "max_active_learning_iterations budget reached"

    node_hours = sum(record.node_hours for record in campaign.formula_runs.values())
    if campaign.budget.max_node_hours is not None and node_hours >= campaign.budget.max_node_hours:
        return "max_node_hours budget reached"
    return None


def _apply_review(campaign: CampaignState, decision: CampaignReviewDecision) -> None:
    unknown = (set(decision.deactivate_formulas) | set(decision.reactivate_formulas)) - set(
        campaign.formulas
    )
    if unknown:
        raise ValueError(f"review references unknown formulas: {sorted(unknown)}")
    for formula in decision.deactivate_formulas:
        campaign.formulas[formula].active = False
        campaign.formulas[formula].rejection_reason = "deactivated by campaign evidence review"
    for formula in decision.reactivate_formulas:
        campaign.formulas[formula].active = True
        campaign.formulas[formula].rejection_reason = None
    if decision.debate_run_id and decision.debate_run_id not in campaign.debate_run_ids:
        campaign.debate_run_ids.append(decision.debate_run_id)
    campaign.review_history.append(
        CampaignReviewRecord(
            iteration=campaign.iteration,
            debate_run_id=decision.debate_run_id,
            deactivate_formulas=decision.deactivate_formulas,
            reactivate_formulas=decision.reactivate_formulas,
            notes=decision.notes,
        )
    )


def run_campaign(
    campaign: CampaignState,
    *,
    output_dir: str | Path,
    formula_runner: FormulaRunner,
    policy: CampaignRunPolicy | None = None,
    review_runner: ReviewRunner | None = None,
    resume: bool = True,
) -> CampaignRunResult:
    """Run active formulas through phase exploration with durable checkpoints."""

    policy = policy or CampaignRunPolicy()
    root = Path(output_dir)
    state_path = root / "campaign_state.json"
    if resume and state_path.exists():
        saved = CampaignState.load(state_path)
        if saved.campaign_id != campaign.campaign_id:
            raise ValueError(
                f"saved campaign_id {saved.campaign_id!r} does not match {campaign.campaign_id!r}"
            )
        campaign = saved

    root.mkdir(parents=True, exist_ok=True)
    campaign.status = WorkflowStatus.RUNNING
    campaign.save(state_path)
    completed: list[str] = []
    failed: list[str] = []
    iterations_run = 0
    stop_reason = "no active unprocessed formulas remain"

    while True:
        budget_reason = _budget_stop_reason(campaign)
        if budget_reason:
            stop_reason = budget_reason
            break
        if policy.max_iterations is not None and iterations_run >= policy.max_iterations:
            stop_reason = "iteration limit reached"
            break

        eligible = _eligible_formulas(campaign, retry_failed=policy.retry_failed)
        if not eligible:
            break
        batch = eligible[: policy.formulas_per_iteration]
        iteration_records: list[FormulaRunRecord] = []
        for formula in batch:
            budget_reason = _budget_stop_reason(campaign)
            if budget_reason:
                stop_reason = budget_reason
                break
            formula_dir = root / "formulas" / formula
            record = campaign.formula_runs.get(formula) or FormulaRunRecord(formula=formula)
            record.status = WorkflowStatus.RUNNING
            record.iteration = campaign.iteration
            record.attempts += 1
            record.output_dir = str(formula_dir)
            record.failure_reason = None
            campaign.formula_runs[formula] = record
            campaign.save(state_path)
            try:
                result = formula_runner(formula, str(formula_dir))
                exploration = result.after_retraining or result.initial
                record.n_mlip_relaxations += len(exploration.relaxations)
                al_result = result.active_learning_result or {}
                record.n_dft_calculations += int(
                    al_result.get("n_dft_calculations", al_result.get("n_dft_converged", 0))
                )
                record.n_active_learning_iterations += int(
                    al_result.get("n_active_learning_iterations", al_result.get("n_iterations", 0))
                )
                record.node_hours += float(al_result.get("node_hours", 0.0))
                record.model_promoted = result.model_promoted
                record.evidence = {
                    key: value for key, value in al_result.items() if key != "exploration_kwargs"
                }
                if exploration.stability is not None:
                    campaign.record_stability(exploration.stability)
                record.status = WorkflowStatus.COMPLETE
                completed.append(formula)
            except Exception as exc:  # noqa: BLE001
                record.status = WorkflowStatus.FAILED
                record.failure_reason = repr(exc)
                failed.append(formula)
                if not policy.continue_on_failure:
                    campaign.status = WorkflowStatus.FAILED
                    campaign.save(state_path)
                    raise
            finally:
                campaign.formula_runs[formula] = record
                campaign.save(state_path)
            iteration_records.append(record)

        if review_runner is not None and iteration_records:
            try:
                _apply_review(campaign, review_runner(campaign, iteration_records))
            except Exception:
                campaign.status = WorkflowStatus.FAILED
                campaign.save(state_path)
                raise
        campaign.iteration += 1
        iterations_run += 1
        campaign.save(state_path)
        if budget_reason:
            break

    remaining = _eligible_formulas(campaign, retry_failed=policy.retry_failed)
    has_failures = any(
        record.status == WorkflowStatus.FAILED for record in campaign.formula_runs.values()
    )
    campaign.status = (
        WorkflowStatus.PARTIAL if remaining or has_failures else WorkflowStatus.COMPLETE
    )
    campaign.save(state_path)
    return CampaignRunResult(
        campaign=campaign,
        state_path=str(state_path),
        formulas_completed=completed,
        formulas_failed=failed,
        stop_reason=stop_reason,
    )
