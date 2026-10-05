"""Resumable orchestration for multi-formula discovery campaigns."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from matsim_agents.campaign.acquisition import (
    CampaignAcquisitionPolicy,
    select_formula_batch,
    update_adaptive_lambda,
)
from matsim_agents.campaign.registry import formula_acquisition_metrics, ingest_exploration_result
from matsim_agents.campaign.state import CampaignReviewRecord, CampaignState, FormulaRunRecord
from matsim_agents.discovery.formula import enumerate_formulas
from matsim_agents.discovery.formula_merge import (
    LLMFormulaProposal,
    extract_llm_formula_proposals,
    merge_formulas,
)
from matsim_agents.discovery.stability import (
    ReferenceEnergySet,
    recalibrate_hull_reports,
)
from matsim_agents.execution.contracts import WorkflowStatus
from matsim_agents.workflows.debate import ScientificDebateResult
from matsim_agents.workflows.phase_exploration import PhaseExplorationWorkflowResult


class CampaignRunPolicy(BaseModel):
    """Control limits and failure behavior for a campaign run."""

    formulas_per_iteration: int = Field(1, ge=1)
    max_iterations: int | None = Field(None, ge=1)
    continue_on_failure: bool = True
    retry_failed: bool = False
    retry_inconclusive: bool = False
    reserved_dft_calculations_per_formula: int = Field(0, ge=0)
    reserved_node_hours_per_formula: float = Field(0.0, ge=0.0)
    no_new_hull_vertex_iterations: int | None = Field(None, ge=1)
    hull_energy_change_eV_per_atom: float | None = Field(None, ge=0.0)
    minimum_formula_coverage: float | None = Field(None, ge=0.0, le=1.0)
    require_low_uncertainty_near_hull: bool = False
    low_uncertainty_threshold: float = Field(0.1, ge=0.0, le=1.0)
    acquisition: CampaignAcquisitionPolicy = Field(default_factory=CampaignAcquisitionPolicy)


class CampaignReviewDecision(BaseModel):
    """Validated campaign changes returned by an evidence-review stage."""

    debate_run_id: str | None = None
    deactivate_formulas: list[str] = Field(default_factory=list)
    reactivate_formulas: list[str] = Field(default_factory=list)
    formula_proposals: list[LLMFormulaProposal] = Field(default_factory=list)
    hypothesis_revisions: list[dict[str, str | list[str]]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CampaignRunResult(BaseModel):
    campaign: CampaignState
    state_path: str
    formulas_completed: list[str] = Field(default_factory=list)
    formulas_failed: list[str] = Field(default_factory=list)
    stop_reason: str


FormulaRunner = Callable[..., PhaseExplorationWorkflowResult]
ReviewRunner = Callable[[CampaignState, list[FormulaRunRecord]], CampaignReviewDecision]


def _merge_reference_energy_sets(
    existing: ReferenceEnergySet | None,
    incoming: ReferenceEnergySet,
) -> ReferenceEnergySet:
    """Merge formula-local DFT references with the campaign's accumulated phases."""
    if existing is None:
        return incoming
    if (
        existing.identifier != incoming.identifier
        or existing.method_signature != incoming.method_signature
        or existing.backend != incoming.backend
    ):
        raise ValueError("campaign formula runs produced incompatible DFT reference sets")
    merged = existing.model_copy(deep=True)

    for element, energy in incoming.elemental_energies_eV_per_atom.items():
        previous = merged.elemental_energies_eV_per_atom.get(element)
        if previous is not None and previous != energy:
            raise ValueError(f"incompatible elemental reference energy for {element}")
        merged.elemental_energies_eV_per_atom[element] = energy
    for element, entry in incoming.elemental_entries.items():
        previous = merged.elemental_entries.get(element)
        if previous is not None and previous != entry:
            raise ValueError(f"incompatible elemental reference entry for {element}")
        merged.elemental_entries[element] = entry

    candidates = {entry.phase_id: entry for entry in merged.elemental_reference_candidates}
    for entry in incoming.elemental_reference_candidates:
        previous = candidates.get(entry.phase_id)
        if previous is not None and previous != entry:
            raise ValueError(f"incompatible unary reference candidate {entry.phase_id!r}")
        candidates[entry.phase_id] = entry
    merged.elemental_reference_candidates = list(candidates.values())

    phases = {entry.phase_id: entry for entry in merged.phase_entries}
    for entry in incoming.phase_entries:
        previous = phases.get(entry.phase_id)
        if previous is not None and previous != entry:
            raise ValueError(f"incompatible competing-phase reference {entry.phase_id!r}")
        phases[entry.phase_id] = entry
    merged.phase_entries = list(phases.values())

    for formula, energy in incoming.competing_phases.items():
        previous = merged.competing_phases.get(formula)
        if previous is None:
            merged.competing_phases[formula] = energy
        else:
            merged.competing_phases[formula] = min(previous, energy)
    merged.completeness_policy.required_formulas = sorted(
        set(merged.completeness_policy.required_formulas)
        | set(incoming.completeness_policy.required_formulas)
    )
    merged.completeness_policy.require_binary_subsystems = (
        merged.completeness_policy.require_binary_subsystems
        or incoming.completeness_policy.require_binary_subsystems
    )
    merged.completeness_policy.require_ternary_competitor = (
        merged.completeness_policy.require_ternary_competitor
        or incoming.completeness_policy.require_ternary_competitor
    )
    return merged


def _recover_partial_formula_evidence(formula_dir: Path) -> dict[str, Any]:
    """Recover resource usage and persisted stage evidence after runner failure."""
    stage_path = formula_dir / "campaign_stages.json"
    stages: dict[str, dict[str, Any]] = {}
    if stage_path.is_file():
        payload = json.loads(stage_path.read_text(encoding="utf-8"))
        stages = {
            str(stage["name"]): stage
            for stage in payload.get("stages", [])
            if isinstance(stage, dict) and stage.get("name")
        }

    al_root = formula_dir / "active_learning"
    iteration_states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(al_root.glob("iteration_*/state.json"))
    ]
    al_stage = stages.get("active_learning_labels_and_training", {})
    dft_stage = stages.get("independent_dft_ranking", {})
    al_dft = int(
        al_stage.get(
            "n_dft_calculations",
            sum(
                int(state.get("n_dft_converged", 0)) + int(state.get("n_dft_failed", 0))
                for state in iteration_states
            ),
        )
    )
    al_node_hours = float(
        al_stage.get(
            "node_hours",
            sum(float(state.get("timings_sec", {}).get("total", 0.0)) for state in iteration_states)
            / 3600.0,
        )
    )
    refinement_dft = int(dft_stage.get("dft_attempts", 0))
    refinement_node_hours = float(dft_stage.get("dft_node_hours", 0.0))
    return {
        "stages": stages,
        "iteration_states": iteration_states,
        "active_learning_dft_calculations": al_dft,
        "active_learning_node_hours": al_node_hours,
        "refinement_dft_attempts": refinement_dft,
        "refinement_node_hours": refinement_node_hours,
        "n_dft_calculations": al_dft + refinement_dft,
        "node_hours": al_node_hours + refinement_node_hours,
    }


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


def _record_has_usable_minimum(campaign: CampaignState, record: FormulaRunRecord) -> bool:
    if record.outcome_class == "usable_minimum" or record.formula in campaign.stability_reports:
        return True
    labels = record.evidence.get("mlip_labels", [])
    return isinstance(labels, list) and any(
        isinstance(label, dict) and bool(label.get("converged")) for label in labels
    )


def _eligible_formulas(
    campaign: CampaignState,
    *,
    retry_failed: bool,
    retry_inconclusive: bool = False,
) -> list[str]:
    eligible: list[str] = []
    for candidate in campaign.active_formulas():
        record = campaign.formula_runs.get(candidate.reduced_formula)
        if (
            record is None
            or record.status in {WorkflowStatus.PLANNED, WorkflowStatus.RUNNING}
            or (retry_failed and record.status == WorkflowStatus.FAILED)
            or (
                retry_inconclusive
                and record.status == WorkflowStatus.COMPLETE
                and not _record_has_usable_minimum(campaign, record)
            )
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


def _admissible_batch_size(
    campaign: CampaignState,
    policy: CampaignRunPolicy,
    requested: int,
) -> int:
    """Limit a batch to work that fits conservative per-formula reservations."""
    admitted = requested
    if campaign.budget.max_dft_calculations is not None:
        used = sum(record.n_dft_calculations for record in campaign.formula_runs.values())
        remaining = max(0, campaign.budget.max_dft_calculations - used)
        if policy.reserved_dft_calculations_per_formula:
            admitted = min(admitted, remaining // policy.reserved_dft_calculations_per_formula)
    if campaign.budget.max_node_hours is not None:
        used = sum(record.node_hours for record in campaign.formula_runs.values())
        remaining = max(0.0, campaign.budget.max_node_hours - used)
        if policy.reserved_node_hours_per_formula:
            admitted = min(admitted, int(remaining // policy.reserved_node_hours_per_formula))
    return admitted


def _scientific_stop_reason(
    campaign: CampaignState,
    policy: CampaignRunPolicy,
) -> str | None:
    if policy.minimum_formula_coverage is not None:
        active = {candidate.reduced_formula for candidate in campaign.active_formulas()}
        completed = {
            formula
            for formula, record in campaign.formula_runs.items()
            if record.status == WorkflowStatus.COMPLETE and formula in active
        }
        coverage = len(completed) / len(active) if active else 1.0
        if coverage >= policy.minimum_formula_coverage:
            return "minimum formula coverage reached"

    stale = policy.no_new_hull_vertex_iterations
    if (
        stale is not None
        and len(campaign.hull_history) >= stale
        and all(not state.new_hull_vertices for state in campaign.hull_history[-stale:])
    ):
        return "no new hull vertex threshold reached"

    tolerance = policy.hull_energy_change_eV_per_atom
    if tolerance is not None and len(campaign.hull_history) >= 2:
        previous, current = campaign.hull_history[-2:]
        common = set(previous.energy_above_hull_eV_per_atom) & set(
            current.energy_above_hull_eV_per_atom
        )
        if (
            common
            and max(
                abs(
                    current.energy_above_hull_eV_per_atom[formula]
                    - previous.energy_above_hull_eV_per_atom[formula]
                )
                for formula in common
            )
            <= tolerance
        ):
            if not policy.require_low_uncertainty_near_hull:
                return "hull energy change tolerance reached"
            near_hull = set(current.hull_vertices) | set(current.near_hull_phases)
            if near_hull and all(
                campaign.acquisition.formula_metrics.get(formula) is not None
                and campaign.acquisition.formula_metrics[formula].mlip_uncertainty
                <= policy.low_uncertainty_threshold
                for formula in near_hull
            ):
                return "hull and near-hull uncertainty converged"
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
    if decision.formula_proposals:
        merged = merge_formulas(
            list(campaign.formulas.values()),
            decision.formula_proposals,
            campaign.formula_policy,
            iteration=campaign.iteration,
        )
        campaign.upsert_formulas(merged)
    campaign.review_history.append(
        CampaignReviewRecord(
            iteration=campaign.iteration,
            debate_run_id=decision.debate_run_id,
            deactivate_formulas=decision.deactivate_formulas,
            reactivate_formulas=decision.reactivate_formulas,
            notes=decision.notes,
            metadata={"hypothesis_revisions": decision.hypothesis_revisions},
        )
    )


def run_campaign(
    campaign: CampaignState,
    *,
    output_dir: str | Path,
    formula_runner: FormulaRunner,
    policy: CampaignRunPolicy | None = None,
    review_runner: ReviewRunner | None = None,
    final_review_runner: ReviewRunner | None = None,
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
        scientific_reason = _scientific_stop_reason(campaign, policy)
        if scientific_reason:
            stop_reason = scientific_reason
            break
        if policy.max_iterations is not None and iterations_run >= policy.max_iterations:
            stop_reason = "iteration limit reached"
            break

        eligible = _eligible_formulas(
            campaign,
            retry_failed=policy.retry_failed,
            retry_inconclusive=policy.retry_inconclusive,
        )
        if not eligible:
            break
        batch_size = _admissible_batch_size(
            campaign,
            policy,
            min(policy.formulas_per_iteration, len(eligible)),
        )
        if batch_size == 0:
            stop_reason = "remaining budget is below per-formula reservation"
            break
        selection = None
        if policy.acquisition.enabled:
            selection = select_formula_batch(
                eligible=eligible,
                batch_size=batch_size,
                iteration=campaign.iteration,
                policy=policy.acquisition,
                state=campaign.acquisition,
            )
            batch = selection.selected_formulas
        else:
            batch = eligible[:batch_size]
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
            record.evidence["accounted_refinement_stage_attempts"] = 0
            record.evidence["accounted_refinement_stage_node_hours"] = 0.0
            if selection is not None:
                record.acquisition_branch = selection.scores[formula].assigned_branch
            campaign.formula_runs[formula] = record
            campaign.save(state_path)
            try:
                dft_allowance = None
                if campaign.budget.max_dft_calculations is not None:
                    used_dft = sum(
                        item.n_dft_calculations for item in campaign.formula_runs.values()
                    )
                    dft_allowance = max(0, campaign.budget.max_dft_calculations - used_dft)
                mlip_allowance = None
                mlip_attempts = 0
                if campaign.budget.max_mlip_relaxations is not None:
                    used_mlip = sum(
                        item.n_mlip_relaxations for item in campaign.formula_runs.values()
                    )
                    mlip_allowance = max(0, campaign.budget.max_mlip_relaxations - used_mlip)

                def record_mlip_attempt(
                    allowance: int | None = mlip_allowance,
                    attempt_record: FormulaRunRecord = record,
                ) -> None:
                    nonlocal mlip_attempts
                    if allowance is not None and mlip_attempts >= allowance:
                        raise RuntimeError("candidate MLIP relaxation allowance exhausted")
                    mlip_attempts += 1
                    attempt_record.n_mlip_relaxations += 1
                    campaign.save(state_path)

                result = (
                    formula_runner(
                        formula,
                        str(formula_dir),
                        dft_allowance,
                        mlip_allowance=mlip_allowance,
                        on_mlip_relaxation_attempt=record_mlip_attempt,
                    )
                    if mlip_allowance is not None
                    else (
                        formula_runner(formula, str(formula_dir), dft_allowance)
                        if dft_allowance is not None
                        else formula_runner(formula, str(formula_dir))
                    )
                )
                exploration = result.after_retraining or result.initial
                attempt_count = result.initial.candidate_counts["attempted"] + (
                    result.after_retraining.candidate_counts["attempted"]
                    if result.after_retraining is not None
                    else 0
                )
                if mlip_allowance is None:
                    record.n_mlip_relaxations += attempt_count
                elif attempt_count != mlip_attempts:
                    raise RuntimeError(
                        "budgeted formula runner must record every candidate relaxation attempt"
                    )
                record.outcome_class = exploration.outcome_class
                record.candidate_counts = exploration.candidate_counts
                al_result = result.active_learning_result or {}
                formula_dft = int(
                    al_result.get("n_dft_calculations", al_result.get("n_dft_converged", 0))
                )
                prior_evidence = record.evidence
                previous_al_dft = int(
                    prior_evidence.get("accounted_active_learning_dft_calculations", 0)
                )
                previous_al_node_hours = float(
                    prior_evidence.get("accounted_active_learning_node_hours", 0.0)
                )
                previous_al_iterations = int(
                    prior_evidence.get("accounted_active_learning_iterations", 0)
                )
                previous_refinement_dft = int(
                    prior_evidence.get("accounted_refinement_dft_attempts", 0)
                )
                previous_dft_total = max(
                    record.n_dft_calculations,
                    int(prior_evidence.get("accounted_total_dft_calculations", 0)),
                )
                previous_node_hours = max(
                    record.node_hours,
                    float(prior_evidence.get("accounted_total_node_hours", 0.0)),
                )
                durable_progress = _recover_partial_formula_evidence(formula_dir)
                refinement_result = al_result.get("dft_refinement", {})
                if not isinstance(refinement_result, dict):
                    refinement_result = {}
                reported_refinement_dft = int(refinement_result.get("reference_calculations", 0))
                reported_refinement_dft += int(refinement_result.get("candidate_calculations", 0))
                current_refinement_dft_total = max(
                    durable_progress["refinement_dft_attempts"],
                    reported_refinement_dft,
                )
                previous_refinement_stage_dft = int(
                    prior_evidence.get("accounted_refinement_stage_attempts", 0)
                )
                current_refinement_dft = max(
                    0,
                    current_refinement_dft_total - previous_refinement_stage_dft,
                )
                reported_al_dft = max(0, formula_dft - reported_refinement_dft)
                current_al_dft = max(
                    durable_progress["active_learning_dft_calculations"],
                    reported_al_dft,
                )
                if (
                    not durable_progress["active_learning_dft_calculations"]
                    and not reported_refinement_dft
                ):
                    current_al_dft = formula_dft

                reported_total_node_hours = float(al_result.get("node_hours", 0.0))
                durable_al_node_hours = durable_progress["active_learning_node_hours"]
                current_refinement_node_hours_total = max(
                    durable_progress["refinement_node_hours"],
                    max(0.0, reported_total_node_hours - durable_al_node_hours),
                )
                previous_refinement_stage_node_hours = float(
                    prior_evidence.get("accounted_refinement_stage_node_hours", 0.0)
                )
                current_refinement_node_hours = max(
                    0.0,
                    current_refinement_node_hours_total - previous_refinement_stage_node_hours,
                )
                current_al_node_hours = max(
                    durable_al_node_hours,
                    reported_total_node_hours - current_refinement_node_hours,
                )
                reported_iterations = int(
                    al_result.get(
                        "n_active_learning_iterations",
                        al_result.get("n_iterations", 0),
                    )
                )
                current_al_iterations = max(
                    reported_iterations,
                    len(durable_progress["iteration_states"]),
                )

                additional_al_dft = max(0, current_al_dft - previous_al_dft)
                additional_al_node_hours = max(0.0, current_al_node_hours - previous_al_node_hours)
                additional_al_iterations = max(0, current_al_iterations - previous_al_iterations)
                attempt_dft = additional_al_dft + current_refinement_dft
                if dft_allowance is not None and attempt_dft > dft_allowance:
                    raise RuntimeError(
                        f"formula used {attempt_dft} new DFT calculations with allowance "
                        f"{dft_allowance}"
                    )
                record.n_dft_calculations = (
                    previous_dft_total + additional_al_dft + current_refinement_dft
                )
                record.n_active_learning_iterations += additional_al_iterations
                record.node_hours = (
                    previous_node_hours + additional_al_node_hours + current_refinement_node_hours
                )
                record.model_promoted = result.model_promoted
                if result.model_promoted:
                    record.promotion_sequence = (
                        max(
                            (
                                item.promotion_sequence
                                for item in campaign.formula_runs.values()
                                if item.promotion_sequence is not None
                            ),
                            default=-1,
                        )
                        + 1
                    )
                record.evidence = {
                    key: value for key, value in al_result.items() if key != "exploration_kwargs"
                }
                record.evidence.update(
                    {
                        "accounted_active_learning_dft_calculations": max(
                            previous_al_dft, current_al_dft
                        ),
                        "accounted_active_learning_node_hours": max(
                            previous_al_node_hours, current_al_node_hours
                        ),
                        "accounted_active_learning_iterations": max(
                            previous_al_iterations, current_al_iterations
                        ),
                        "accounted_refinement_dft_attempts": (
                            previous_refinement_dft + current_refinement_dft
                        ),
                        "accounted_refinement_stage_attempts": (current_refinement_dft_total),
                        "accounted_refinement_stage_node_hours": (
                            current_refinement_node_hours_total
                        ),
                        "accounted_total_dft_calculations": record.n_dft_calculations,
                        "accounted_total_node_hours": record.node_hours,
                    }
                )
                record.evidence["candidate_counts"] = record.candidate_counts
                record.evidence["candidate_relaxation_budget"] = {
                    "scope": "candidate_structure_attempts",
                    "allowance": mlip_allowance,
                    "attempted": attempt_count,
                    "initial_exhausted": result.initial.relaxation_budget_exhausted,
                    "initial_unrelaxed": result.initial.unrelaxed_candidate_paths,
                    "reevaluation_exhausted": (
                        result.after_retraining.relaxation_budget_exhausted
                        if result.after_retraining is not None
                        else False
                    ),
                    "reevaluation_unrelaxed": (
                        result.after_retraining.unrelaxed_candidate_paths
                        if result.after_retraining is not None
                        else []
                    ),
                }
                record.evidence["outcome_class"] = record.outcome_class
                record.evidence["relaxation_failures"] = list(exploration.failures)
                record.evidence["ranking_failure"] = exploration.ranking_failure
                prior_families = set(campaign.candidate_registry.relaxed_families)
                novelty_references = [
                    candidate.optimized_structure_path
                    for candidate in campaign.candidate_registry.candidates.values()
                    if candidate.optimized_structure_path is not None
                ]
                uncertainty_by_candidate = {
                    str(key): float(value)
                    for key, value in record.evidence.get("candidate_uncertainty", {}).items()
                }
                ingest_exploration_result(
                    campaign.candidate_registry,
                    exploration,
                    iteration=campaign.iteration,
                    backend=record.evidence.get("mlip_backend"),
                    model_identifier=record.evidence.get("model_identifier"),
                    model_checkpoint_hash=record.evidence.get("model_checkpoint_hash"),
                    uncertainty_by_candidate=uncertainty_by_candidate,
                    novelty_reference_paths=novelty_references,
                )
                reference_payload = record.evidence.get("dft_refinement", {}).get(
                    "reference_energy_set"
                )
                previous_references = campaign.reference_energies
                incoming_references = None
                if reference_payload is not None:
                    incoming_references = ReferenceEnergySet.model_validate(reference_payload)
                    campaign.reference_energies = _merge_reference_energy_sets(
                        previous_references,
                        incoming_references,
                    )
                if exploration.stability is not None:
                    report = exploration.stability
                    references = campaign.reference_energies
                    if (
                        references is not None
                        and report.ranking_mode.value == "convex_hull_ranking"
                        and report.reference_set_id == references.identifier
                    ):
                        if (
                            previous_references is not None
                            and incoming_references is not None
                            and report.formula in previous_references.competing_phases
                            and report.formula not in incoming_references.competing_phases
                        ):
                            references.competing_phases.pop(report.formula, None)
                        campaign.stability_reports[report.formula] = report
                        recalibrate_hull_reports(
                            campaign.stability_reports.values(),
                            references,
                        )
                    campaign.record_stability(exploration.stability)
                candidate = campaign.formulas[formula]
                disagreement_count = len(candidate.model_disagreements)
                panel_evidence_count = len(candidate.llm_contributors) + disagreement_count
                disagreement = (
                    disagreement_count / panel_evidence_count if panel_evidence_count else 0.0
                )
                hull_energy = None
                if exploration.stability is not None:
                    hull_energy = exploration.stability.ground_state.energy_above_hull_eV_per_atom
                campaign.acquisition.formula_metrics[formula] = formula_acquisition_metrics(
                    campaign.candidate_registry,
                    formula,
                    energy_above_hull_eV_per_atom=hull_energy,
                    llm_disagreement=disagreement,
                )
                new_families = set(campaign.candidate_registry.relaxed_families) - prior_families
                near_hull = hull_energy is not None and hull_energy <= 0.05
                dft_verified_family = bool(new_families and record.n_dft_calculations)
                record.evidence.setdefault(
                    "useful_outcome",
                    bool(near_hull or dft_verified_family or record.model_promoted),
                )
                record.status = WorkflowStatus.COMPLETE
                completed.append(formula)
            except Exception as exc:  # noqa: BLE001
                partial = _recover_partial_formula_evidence(formula_dir)
                previous_al_dft = int(
                    record.evidence.get("accounted_active_learning_dft_calculations", 0)
                )
                previous_al_node_hours = float(
                    record.evidence.get("accounted_active_learning_node_hours", 0.0)
                )
                previous_al_iterations = int(
                    record.evidence.get("accounted_active_learning_iterations", 0)
                )
                previous_refinement_stage_dft = int(
                    record.evidence.get("accounted_refinement_stage_attempts", 0)
                )
                previous_refinement_stage_node_hours = float(
                    record.evidence.get("accounted_refinement_stage_node_hours", 0.0)
                )
                additional_al_dft = max(
                    0,
                    partial["active_learning_dft_calculations"] - previous_al_dft,
                )
                additional_al_node_hours = max(
                    0.0,
                    partial["active_learning_node_hours"] - previous_al_node_hours,
                )
                additional_refinement_dft = max(
                    0,
                    partial["refinement_dft_attempts"] - previous_refinement_stage_dft,
                )
                additional_refinement_node_hours = max(
                    0.0,
                    partial["refinement_node_hours"] - previous_refinement_stage_node_hours,
                )
                record.n_dft_calculations += additional_al_dft + additional_refinement_dft
                record.node_hours += additional_al_node_hours + additional_refinement_node_hours
                record.n_active_learning_iterations = max(
                    record.n_active_learning_iterations,
                    len(partial["iteration_states"]),
                )
                record.evidence.update(
                    {
                        "partial_campaign_stages": partial["stages"],
                        "iteration_states": partial["iteration_states"],
                        "accounted_active_learning_dft_calculations": partial[
                            "active_learning_dft_calculations"
                        ],
                        "accounted_active_learning_node_hours": partial[
                            "active_learning_node_hours"
                        ],
                        "accounted_active_learning_iterations": max(
                            previous_al_iterations,
                            len(partial["iteration_states"]),
                        ),
                        "accounted_refinement_dft_attempts": (
                            int(record.evidence.get("accounted_refinement_dft_attempts", 0))
                            + additional_refinement_dft
                        ),
                        "accounted_refinement_stage_attempts": partial["refinement_dft_attempts"],
                        "accounted_refinement_stage_node_hours": partial["refinement_node_hours"],
                        "accounted_total_dft_calculations": record.n_dft_calculations,
                        "accounted_total_node_hours": record.node_hours,
                        "partial_progress": partial,
                    }
                )
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

        if policy.acquisition.enabled and policy.acquisition.mode == "adaptive":
            for record in iteration_records:
                useful = bool(record.evidence.get("useful_outcome", False))
                if record.acquisition_branch == "exploitation":
                    campaign.acquisition.exploitation_attempts += 1
                    campaign.acquisition.exploitation_useful += int(useful)
                elif record.acquisition_branch == "exploration":
                    campaign.acquisition.exploration_attempts += 1
                    campaign.acquisition.exploration_useful += int(useful)
            update_adaptive_lambda(campaign.acquisition, policy.acquisition)

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

    remaining = _eligible_formulas(
        campaign,
        retry_failed=policy.retry_failed,
        retry_inconclusive=False,
    )
    has_failures = any(
        record.status == WorkflowStatus.FAILED for record in campaign.formula_runs.values()
    )
    campaign.status = (
        WorkflowStatus.PARTIAL if remaining or has_failures else WorkflowStatus.COMPLETE
    )
    if final_review_runner is not None and campaign.formula_runs:
        final_records = sorted(
            campaign.formula_runs.values(),
            key=lambda record: (record.iteration, record.formula),
        )
        _apply_review(campaign, final_review_runner(campaign, final_records))
    campaign.save(state_path)
    return CampaignRunResult(
        campaign=campaign,
        state_path=str(state_path),
        formulas_completed=completed,
        formulas_failed=failed,
        stop_reason=stop_reason,
    )
