"""Composable phase exploration built from relaxation and active learning."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from matsim_agents.discovery.wrapper import CompositionExplorationResult, explore_composition
from matsim_agents.execution.contracts import (
    ApprovalPolicy,
    ComputeBudget,
    EvidenceLevel,
    ProvenanceRecord,
)
from matsim_agents.execution.ledger import file_sha256
from matsim_agents.execution.run_directory import ScientificRunDirectory

_EXPLORATION_DEFAULTS = {
    name: parameter.default
    for name, parameter in inspect.signature(explore_composition).parameters.items()
    if name not in {"composition", "output_dir"}
    and parameter.default is not inspect.Parameter.empty
}


class PhaseExplorationContext(BaseModel):
    """Scientific context that may inform a later stability-study decision."""

    model_config = ConfigDict(extra="forbid")

    objective: str | None = None
    thermodynamic_stability_required: bool | None = None
    temperature_K: float | None = None
    pressure_GPa: float | None = None
    chemical_potentials_eV_per_atom: dict[str, float] = Field(default_factory=dict)
    known_competing_phases: list[str] = Field(default_factory=list)
    experimental_decomposition_evidence: str | None = None
    phase_separation_suspected: bool | None = None
    existing_reference_set_id: str | None = None
    known_energy_above_hull_eV_per_atom: float | None = None


class HullFollowUpAssessment(BaseModel):
    """Recorded decision boundary between exploration and hull analysis."""

    status: Literal["not_indicated", "review_required"]
    campaign_started: Literal[False] = False
    reasons: list[str] = Field(default_factory=list)
    observations: dict[str, Any] = Field(default_factory=dict)
    required_before_campaign: list[str] = Field(
        default_factory=lambda: [
            "explicit thermodynamic-stability objective and scope",
            "method-compatible elemental and competing-phase references",
            "DFT settings, compute budget, and approval",
        ]
    )


class PhaseExplorationPolicy(BaseModel):
    relax_structures: bool = True
    active_learning: bool = False
    retrain_mlip: bool = False
    promote_model: bool = False
    reevaluate_after_retraining: bool = False
    continue_on_promotion_rejection: bool = False
    ranking_mode: Literal["relative_phase_ranking"] = "relative_phase_ranking"
    study_context: PhaseExplorationContext = Field(default_factory=PhaseExplorationContext)
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    approvals: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    dft_approved: bool = False
    retraining_approved: bool = False
    promotion_approved: bool = False

    @field_validator("ranking_mode", mode="before")
    @classmethod
    def _relative_ranking_only(cls, value: object) -> object:
        if value != "relative_phase_ranking":
            raise ValueError(
                "phase exploration supports only relative_phase_ranking; "
                "use thermodynamic_stability characterization for convex-hull analysis"
            )
        return value

    @model_validator(mode="after")
    def _consistent_options(self) -> PhaseExplorationPolicy:
        if self.retrain_mlip and not self.active_learning:
            raise ValueError("retrain_mlip requires active_learning")
        if self.promote_model and not self.retrain_mlip:
            raise ValueError("promote_model requires retrain_mlip")
        if self.reevaluate_after_retraining and not self.retrain_mlip:
            raise ValueError("reevaluate_after_retraining requires retrain_mlip")
        if self.reevaluate_after_retraining and not self.promote_model:
            raise ValueError("reevaluate_after_retraining requires promote_model")
        if self.continue_on_promotion_rejection and not self.promote_model:
            raise ValueError("continue_on_promotion_rejection requires promote_model")
        return self


class PhaseExplorationWorkflowResult(BaseModel):
    composition: str
    initial: CompositionExplorationResult
    after_retraining: CompositionExplorationResult | None = None
    active_learning_result: dict[str, Any] | None = None
    model_promoted: bool = False
    run_id: str | None = None
    run_directory: str | None = None
    record_path: str | None = None
    hull_follow_up: HullFollowUpAssessment | None = None


class PhaseExplorationEvidenceRecord(BaseModel):
    """Portable evidence dossier for review and possible follow-up studies."""

    schema_version: Literal[1] = 1
    run_id: str
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context: PhaseExplorationContext
    policy: PhaseExplorationPolicy
    exploration_options: dict[str, Any]
    omitted_runtime_options: list[str] = Field(default_factory=list)
    structure_sha256: dict[str, str] = Field(default_factory=dict)
    status: Literal["complete", "failed"] = "complete"
    failure_reason: str | None = None
    failure_type: str | None = None
    result: PhaseExplorationWorkflowResult | None = None


def _recordable_exploration_options(
    options: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    effective = {**_EXPLORATION_DEFAULTS, **options}
    recorded: dict[str, Any] = {}
    omitted: list[str] = []
    runtime_names = {"mlip_pool", "relax_fn"}
    for name, value in effective.items():
        if callable(value) or (name in runtime_names and value is not None):
            omitted.append(name)
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            omitted.append(name)
        else:
            recorded[name] = value
    return recorded, sorted(omitted)


def run_phase_exploration(
    composition: str,
    *,
    policy: PhaseExplorationPolicy,
    output_dir: str,
    exploration_kwargs: dict[str, Any] | None = None,
    active_learning_runner: Callable[[str, str, bool, bool, bool], dict[str, Any]] | None = None,
) -> PhaseExplorationWorkflowResult:
    """Run exploration, optional AL, and optional post-promotion reevaluation.

    The AL callback receives ``(composition, output_dir, retrain, promote_model,
    promotion_approved)`` and must apply those controls before training or
    promotion. It returns a mapping containing ``model_promoted`` plus any
    provenance, keeping this workflow independent of facility-specific launch mechanics.
    """

    kwargs = dict(exploration_kwargs or {})
    limits = [
        limit
        for limit in (policy.budget.max_mlip_relaxations, kwargs.get("max_relaxations"))
        if limit is not None
    ]
    relaxation_allowance = min(limits) if limits else None
    if relaxation_allowance is not None:
        kwargs["max_relaxations"] = relaxation_allowance
    recorded_options, omitted_options = _recordable_exploration_options(kwargs)
    run = ScientificRunDirectory.create(
        output_dir,
        workflow="phase_exploration",
        request={
            "composition": composition,
            "study_context": policy.study_context.model_dump(mode="json"),
        },
        resolved_config={
            "policy": policy.model_dump(mode="json"),
            "exploration_options": recorded_options,
            "omitted_runtime_options": omitted_options,
        },
        provenance=ProvenanceRecord(
            workflow="phase_exploration",
            evidence_level=(
                EvidenceLevel.MLIP_RELAXATION
                if policy.relax_structures
                else EvidenceLevel.MLIP_PREDICTION
            ),
            numerical_settings=recorded_options,
            random_seed=recorded_options.get("random_seed"),
        ),
    )
    calculations_dir = run.path / "calculations"
    initial: CompositionExplorationResult | None = None
    al_result: dict[str, Any] | None = None
    promoted = False
    after: CompositionExplorationResult | None = None
    try:
        if policy.active_learning and policy.approvals.before_dft and not policy.dft_approved:
            raise PermissionError("active learning requires explicit DFT approval")
        if (
            policy.retrain_mlip
            and policy.approvals.before_retraining
            and not policy.retraining_approved
        ):
            raise PermissionError("MLIP retraining requires explicit approval")
        if (
            policy.promote_model
            and policy.approvals.before_model_promotion
            and not policy.promotion_approved
        ):
            raise PermissionError("model promotion requires explicit approval")

        if not policy.relax_structures:
            from matsim_agents.discovery.composition import parse_composition
            from matsim_agents.discovery.seeds import generate_seeds

            parsed = parse_composition(composition)
            if parsed is None:
                raise ValueError(f"Could not parse composition {composition!r}")
            candidates = generate_seeds(
                parsed,
                str(calculations_dir / parsed.formula / "seeds"),
                n_random=kwargs.get("n_random", 50),
                random_seed=kwargs.get("random_seed", 0),
            )
            initial = CompositionExplorationResult(
                composition=parsed,
                phase_candidates=candidates,
                outcome_class="seed_only" if candidates else "generation_failure",
            )
        else:
            initial = explore_composition(composition, output_dir=str(calculations_dir), **kwargs)

        if policy.active_learning:
            if active_learning_runner is None:
                raise ValueError("active_learning=True requires active_learning_runner")
            al_result = active_learning_runner(
                composition,
                str(calculations_dir),
                policy.retrain_mlip,
                policy.promote_model,
                policy.promotion_approved,
            )
            promoted = bool(al_result.get("model_promoted", False))
            if promoted and not policy.promote_model:
                raise RuntimeError(
                    "active learning promoted a model without promotion being requested"
                )
            if (
                promoted
                and policy.approvals.before_model_promotion
                and not policy.promotion_approved
            ):
                raise PermissionError("model promotion requires explicit approval")
            if policy.reevaluate_after_retraining and not promoted:
                deferred = bool(al_result.get("training_deferred", False))
                if not deferred and not policy.continue_on_promotion_rejection:
                    raise RuntimeError("cannot reevaluate: active learning did not promote a model")
                logging.getLogger(__name__).warning(
                    "No model promoted for %s%s; retaining incumbent MLIP exploration",
                    composition,
                    " (training deferred for insufficient cumulative data)" if deferred else "",
                )
            if policy.reevaluate_after_retraining and promoted:
                updated = dict(kwargs)
                updated.update(dict(al_result.get("exploration_kwargs", {})))
                if relaxation_allowance is not None:
                    updated["max_relaxations"] = max(
                        0, relaxation_allowance - initial.candidate_counts["attempted"]
                    )
                if "on_relaxation_attempt" in kwargs:
                    updated["on_relaxation_attempt"] = kwargs["on_relaxation_attempt"]
                after = explore_composition(
                    composition,
                    output_dir=str(calculations_dir / "after_retraining"),
                    **updated,
                )

        result = PhaseExplorationWorkflowResult(
            composition=composition,
            initial=initial,
            after_retraining=after,
            active_learning_result=al_result,
            model_promoted=promoted,
            run_id=run.run_id,
            run_directory=str(run.path),
            record_path=str(run.path / "results.json"),
        )
        all_explorations = [initial, *([after] if after is not None else [])]
        artifact_hashes: dict[str, str] = {}
        for exploration in all_explorations:
            paths = [candidate.structure_path for candidate in exploration.phase_candidates]
            paths.extend(item.optimized_structure_path for item in exploration.relaxations)
            for path in paths:
                if Path(path).is_file():
                    artifact_hashes[path] = file_sha256(path)
        reasons = []
        if policy.study_context.thermodynamic_stability_required:
            reasons.append("study context explicitly requires thermodynamic-stability analysis")
        if policy.study_context.experimental_decomposition_evidence:
            reasons.append("experimental decomposition evidence was supplied")
        if policy.study_context.phase_separation_suspected:
            reasons.append("phase separation is suspected in the study context")
        if policy.study_context.known_energy_above_hull_eV_per_atom is not None:
            reasons.append("an existing energy-above-hull result was supplied for review")
        if any(candidate.source == "random" for candidate in initial.phase_candidates):
            reasons.append("random pyXtal candidates require verification before stability claims")
        follow_up = HullFollowUpAssessment(
            status="review_required" if reasons else "not_indicated",
            reasons=reasons,
            observations={
                "composition": initial.composition.formula,
                "candidate_counts": initial.candidate_counts,
                "random_candidate_ids": [
                    item.candidate_id
                    for item in initial.phase_candidates
                    if item.source == "random"
                ],
                "ranking_mode": policy.ranking_mode,
                "ranking_available": initial.stability is not None,
                "ranking_failure": initial.ranking_failure,
                "outcome_class": initial.outcome_class,
                "known_competing_phases": policy.study_context.known_competing_phases,
                "existing_reference_set_id": policy.study_context.existing_reference_set_id,
                "known_energy_above_hull_eV_per_atom": (
                    policy.study_context.known_energy_above_hull_eV_per_atom
                ),
                "temperature_K": policy.study_context.temperature_K,
                "pressure_GPa": policy.study_context.pressure_GPa,
                "chemical_potentials_eV_per_atom": (
                    policy.study_context.chemical_potentials_eV_per_atom
                ),
            },
        )
        result.hull_follow_up = follow_up
        record = PhaseExplorationEvidenceRecord(
            run_id=run.run_id,
            context=policy.study_context,
            policy=policy,
            exploration_options=recorded_options,
            omitted_runtime_options=omitted_options,
            structure_sha256=artifact_hashes,
            result=result,
        )
        run.write_json("hull-follow-up.json", follow_up)
        run.write_json("results.json", record)
        run.append_event("run_finished", {"status": "complete"})
        return result
    except Exception as exc:
        partial_result = None
        artifact_hashes = {}
        if initial is not None:
            partial_result = PhaseExplorationWorkflowResult(
                composition=composition,
                initial=initial,
                after_retraining=after,
                active_learning_result=al_result,
                model_promoted=promoted,
                run_id=run.run_id,
                run_directory=str(run.path),
                record_path=str(run.path / "results.json"),
            )
            explorations = [initial, *([after] if after is not None else [])]
            for exploration in explorations:
                paths = [candidate.structure_path for candidate in exploration.phase_candidates]
                paths.extend(item.optimized_structure_path for item in exploration.relaxations)
                for path in paths:
                    if Path(path).is_file():
                        artifact_hashes[path] = file_sha256(path)
        failure_record = PhaseExplorationEvidenceRecord(
            run_id=run.run_id,
            context=policy.study_context,
            policy=policy,
            exploration_options=recorded_options,
            omitted_runtime_options=omitted_options,
            structure_sha256=artifact_hashes,
            status="failed",
            failure_reason=str(exc),
            failure_type=type(exc).__name__,
            result=partial_result,
        )
        run.write_json("results.json", failure_record)
        run.append_event(
            "run_finished",
            {"status": "failed", "failure_type": type(exc).__name__, "failure_reason": str(exc)},
        )
        raise


__all__ = [
    "HullFollowUpAssessment",
    "PhaseExplorationContext",
    "PhaseExplorationEvidenceRecord",
    "PhaseExplorationPolicy",
    "PhaseExplorationWorkflowResult",
    "run_phase_exploration",
]
