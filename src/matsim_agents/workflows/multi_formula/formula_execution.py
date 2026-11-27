"""Bind one campaign formula to phase exploration and active learning."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from matsim_agents.backends.mlip.calculator import make_mlip_calculator
from matsim_agents.backends.mlip.pool import (
    MLIPWorkerPool,
    driver_on_mlip_device,
    release_device_memory,
)
from matsim_agents.backends.mlip.relaxation import (
    CalculatorCache,
    RelaxStructureInput,
    relax_with_mlip,
)
from matsim_agents.core.relaxation import RelaxationResult
from matsim_agents.core.seeding import stream_seed
from matsim_agents.execution.placement import Placement
from matsim_agents.workflows.active_learning.config import ALConfig
from matsim_agents.workflows.active_learning.loop import run_active_learning
from matsim_agents.workflows.exploration.phase import (
    PhaseExplorationWorkflowResult,
    run_phase_exploration,
)
from matsim_agents.workflows.multi_formula.formula_al import (
    _apply_model_override,
    _durable_iteration_states,
    _exploration_kwargs,
    _model_identifier,
    _prepare_al_template,
    _promoted_model_path,
    _summarize_active_learning,
)
from matsim_agents.workflows.multi_formula.formula_config import (
    CampaignDFTRefinementConfig,
    CampaignFormulaExecutionConfig,
    CampaignRetrainingConfig,
    ReferenceStructureSpec,
)
from matsim_agents.workflows.multi_formula.formula_ranking import _run_independent_dft_ranking
from matsim_agents.workflows.multi_formula.formula_stages import _StageTracker
from matsim_agents.workflows.multi_formula.formula_validation import (
    _cross_model_scores,
    _perturbation_robustness,
    _score_relaxed_candidate_uncertainty,
)
from matsim_agents.workflows.multi_formula.state import CampaignState
from matsim_agents.workflows.relaxation import (
    ScientificRelaxationConfig,
    ScientificRelaxationResult,
    run_relaxation,
)


def _mlip_only_evidence(
    exploration: Any,
    al_cfg: ALConfig,
    candidate_uncertainty: dict[str, float],
    cross_model: Any,
    robustness: Any,
    *,
    node_hours: float,
    stage_manifest_path: Path,
) -> dict[str, Any]:
    """Evidence record for MLIP-only execution with any backend: labels without DFT."""
    candidate_by_path = {
        candidate.structure_path: candidate for candidate in exploration.phase_candidates
    }
    labels = []
    for relaxation in exploration.relaxations:
        candidate = candidate_by_path.get(relaxation.structure_path)
        atom_count = candidate.num_atoms if candidate is not None else None
        labels.append(
            {
                "candidate_id": candidate.candidate_id if candidate is not None else None,
                "structure_path": relaxation.structure_path,
                "optimized_structure_path": relaxation.optimized_structure_path,
                "converged": relaxation.converged,
                "final_energy_eV": relaxation.final_energy_eV,
                "energy_per_atom_eV": (
                    relaxation.final_energy_eV / atom_count if atom_count else None
                ),
                "residual_force_eV_per_A": relaxation.final_max_force_eV_per_A,
                "uncertainty": (
                    candidate_uncertainty.get(candidate.candidate_id)
                    if candidate is not None and candidate.candidate_id is not None
                    else None
                ),
            }
        )
    return {
        "execution_mode": "mlip_only",
        "evidence_level": "mlip_prediction",
        "backend": al_cfg.mlip.backend,
        "model_identifier": al_cfg.mlip.model_reference,
        "n_dft_calculations": 0,
        "n_active_learning_iterations": 0,
        "node_hours": node_hours,
        "model_promoted": False,
        "candidate_uncertainty": candidate_uncertainty,
        "mlip_labels": labels,
        "cross_model_validation": cross_model,
        "perturbation_robustness": robustness,
        "campaign_stages_path": str(stage_manifest_path),
    }


def _formula_mlip_pool(
    al_cfg: ALConfig, formula_root: Path
) -> tuple[MLIPWorkerPool, Placement | None]:
    """Workers for the template's mlip pool (or in-process relaxations) and its placement."""
    if "mlip" not in al_cfg.resources.pools:
        return MLIPWorkerPool(), None
    placement = al_cfg.resources.place(formula_root)
    return MLIPWorkerPool.from_placement(placement), placement


def run_formula_with_active_learning(
    formula: str,
    output_dir: str,
    *,
    config: CampaignFormulaExecutionConfig,
    random_seed: int,
    al_runner: Callable[[ALConfig], None] = run_active_learning,
    phase_runner: Callable[..., PhaseExplorationWorkflowResult] = run_phase_exploration,
    relaxation_runner: Callable[
        [ScientificRelaxationConfig], ScientificRelaxationResult
    ] = run_relaxation,
    mlip_relaxation_runner: Callable[[RelaxStructureInput], RelaxationResult] = relax_with_mlip,
    calculator_factory: Callable[..., Any] = make_mlip_calculator,
) -> PhaseExplorationWorkflowResult:
    """Explore and label one formula using a fresh copy of the AL template.

    *random_seed* is the attempt's seed; generation, AL, uncertainty scoring,
    perturbations, and DFT candidate selection use named streams of it.
    """

    formula_root = Path(output_dir)
    al_root = formula_root / "active_learning"
    al_cfg = _prepare_al_template(formula, config, al_root, random_seed=random_seed)
    stage_manifest_path = formula_root / "campaign_stages.json"
    tracker = _StageTracker(stage_manifest_path, formula, config)

    def active_learning_runner(
        composition: str,
        phase_output_dir: str,
        retrain: bool,
        promote_model: bool,
        promotion_approved: bool,
    ) -> dict[str, Any]:
        if composition != formula:
            raise ValueError(f"phase runner requested {composition!r}, expected {formula!r}")
        if retrain != al_cfg.trainer.enabled:
            raise ValueError("phase retraining policy and AL trainer configuration disagree")
        if (
            promote_model != al_cfg.trainer.promote_model
            or promotion_approved != al_cfg.trainer.promotion_approved
        ):
            raise ValueError("phase promotion policy and AL trainer configuration disagree")
        tracker.set(
            "initial_mlip_exploration",
            "completed",
            output_dir=str(Path(phase_output_dir) / formula),
        )
        tracker.set(
            "active_learning_labels_and_training",
            "in_progress",
            training_enabled=al_cfg.trainer.enabled,
            promotion_requested=al_cfg.trainer.promote_model,
        )
        try:
            al_runner(al_cfg)
        except Exception as exc:
            partial_states = _durable_iteration_states(al_root)
            partial_dft = sum(
                int(state.get("n_dft_converged", 0)) + int(state.get("n_dft_failed", 0))
                for state in partial_states
            )
            partial_seconds = sum(
                float(state.get("timings_sec", {}).get("total", 0.0)) for state in partial_states
            )
            tracker.set(
                "active_learning_labels_and_training",
                "failed",
                failure_reason=f"{type(exc).__name__}: {exc}",
                n_dft_calculations=partial_dft,
                n_active_learning_iterations=len(partial_states),
                node_hours=partial_seconds * config.compute_nodes / 3600.0,
                iteration_states=partial_states,
            )
            raise
        result = _summarize_active_learning(al_root, al_cfg, config.compute_nodes, phase_output_dir)
        states = result["iteration_states"]
        promoted = [state for state in states if state.get("model_promoted")]
        tracker.set(
            "active_learning_labels_and_training",
            "completed",
            training_enabled=al_cfg.trainer.enabled,
            training_status=(states[-1].get("training_status") if states else None),
            training_deferred_reasons=(
                states[-1].get("training_deferred_reasons", []) if states else []
            ),
            model_promoted=bool(promoted),
            training_data=result["training_data"],
            held_out_validation_data=result["held_out_validation_data"],
            n_dft_calculations=result["n_dft_calculations"],
            n_active_learning_iterations=result["n_active_learning_iterations"],
            node_hours=result["node_hours"],
            candidate_model_path=(str(promoted[-1].get("new_logdir")) if promoted else None),
        )
        return result

    phase_started = time.monotonic()
    phase_policy = config.phase_policy.model_copy(
        update={
            "promote_model": al_cfg.trainer.promote_model,
            "promotion_approved": al_cfg.trainer.promotion_approved,
        }
    )
    tracker.set(
        "initial_mlip_exploration",
        "in_progress",
        model_identifier=_model_identifier(al_cfg),
    )
    mlip_pool, mlip_placement = _formula_mlip_pool(al_cfg, formula_root)

    def active_learning_after_exploration(*args: Any) -> dict[str, Any]:
        # Exploration models must not stay resident while AL samples, labels with
        # DFT, and trains on the same devices; reevaluation restarts the workers.
        mlip_pool.close()
        release_device_memory()
        return active_learning_runner(*args)

    try:
        result = phase_runner(
            formula,
            policy=phase_policy,
            output_dir=str(formula_root / "phase_exploration"),
            exploration_kwargs={
                **_exploration_kwargs(
                    al_cfg,
                    config.exploration_kwargs,
                    generation_seed=stream_seed(random_seed, "generation"),
                ),
                "mlip_pool": mlip_pool,
            },
            active_learning_runner=active_learning_after_exploration,
        )
    except Exception as exc:
        stage_name = (
            "active_learning_labels_and_training"
            if tracker.status("active_learning_labels_and_training") == "in_progress"
            else "initial_mlip_exploration"
        )
        tracker.set(
            stage_name,
            "failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        mlip_pool.close()
    phase_seconds = time.monotonic() - phase_started
    exploration = result.after_retraining or result.initial
    effective_al_cfg = al_cfg.model_copy(deep=True)
    promoted_model = _promoted_model_path(result)
    if promoted_model is not None:
        _apply_model_override(effective_al_cfg, promoted_model)
    execution_evidence = result.active_learning_result or {}
    execution_evidence["mlip_backend"] = effective_al_cfg.mlip.backend
    execution_evidence["model_identifier"] = _model_identifier(effective_al_cfg)
    execution_evidence["campaign_stages_path"] = str(stage_manifest_path)
    result.active_learning_result = execution_evidence
    if tracker.status("initial_mlip_exploration") == "in_progress":
        tracker.set(
            "initial_mlip_exploration",
            "completed",
            model_identifier=_model_identifier(al_cfg),
            candidate_count=len(result.initial.phase_candidates),
            relaxation_count=len(result.initial.relaxations),
        )
    if result.after_retraining is not None:
        tracker.set(
            "post_training_mlip_exploration",
            "completed",
            model_identifier=_model_identifier(effective_al_cfg),
            candidate_count=len(result.after_retraining.phase_candidates),
            relaxation_count=len(result.after_retraining.relaxations),
        )
    elif config.phase_policy.reevaluate_after_retraining:
        tracker.set(
            "post_training_mlip_exploration",
            "skipped",
            reason=(
                "training deferred for insufficient cumulative data; retaining incumbent"
                if execution_evidence.get("training_deferred")
                else "no model promoted; explicitly retaining incumbent MLIP exploration"
            ),
            model_identifier=_model_identifier(effective_al_cfg),
        )
    # Driver-side scoring, cross-model checks, and robustness relaxations run on the
    # mlip pool's device (and budget), not on device 0.
    with driver_on_mlip_device(mlip_placement):
        candidate_uncertainty = _score_relaxed_candidate_uncertainty(exploration, effective_al_cfg)
        validation_configs = [
            effective_al_cfg,
            *(ALConfig.from_yaml(path) for path in config.validation_configs),
        ]
        cross_model = _cross_model_scores(
            exploration,
            validation_configs,
            config_paths=[None, *config.validation_configs],
            validation_python=config.validation_python,
            validation_pythons=(
                [
                    None,
                    *(
                        config.validation_pythons * len(config.validation_configs)
                        if len(config.validation_pythons) == 1
                        else config.validation_pythons
                    ),
                ]
                if config.validation_pythons
                else None
            ),
            reference_manifest=config.surrogate_reference_structures,
            unary_max_steps=config.surrogate_unary_max_steps,
            unary_fmax_eV_per_A=config.surrogate_unary_fmax_eV_per_A,
            unary_maxstep_A=config.surrogate_unary_maxstep_A,
            minimum_unique_unary=config.surrogate_minimum_unique_unary,
            calculator_factory=calculator_factory,
        )
        robustness = _perturbation_robustness(
            exploration,
            effective_al_cfg,
            config,
            formula_root / "robustness",
            relaxation_runner=(
                functools.partial(relax_with_mlip, calculators=CalculatorCache())
                if mlip_relaxation_runner is relax_with_mlip
                else mlip_relaxation_runner
            ),
        )
    if config.execution_mode == "mlip_only":
        result.active_learning_result = _mlip_only_evidence(
            exploration,
            al_cfg,
            candidate_uncertainty,
            cross_model,
            robustness,
            node_hours=phase_seconds * config.compute_nodes / 3600.0,
            stage_manifest_path=stage_manifest_path,
        )
        return result
    if result.active_learning_result is not None:
        result.active_learning_result["candidate_uncertainty"] = candidate_uncertainty
    return _run_independent_dft_ranking(
        formula,
        formula_root,
        config,
        al_cfg=al_cfg,
        effective_al_cfg=effective_al_cfg,
        result=result,
        exploration=exploration,
        execution_evidence=execution_evidence,
        tracker=tracker,
        relaxation_runner=relaxation_runner,
    )


def make_formula_runner(
    config: CampaignFormulaExecutionConfig,
) -> Callable[..., PhaseExplorationWorkflowResult]:
    def runner(
        formula: str,
        output_dir: str,
        dft_allowance: int | None = None,
        *,
        random_seed: int,
        mlip_allowance: int | None = None,
        on_mlip_relaxation_attempt: Callable[[], None] | None = None,
    ) -> PhaseExplorationWorkflowResult:
        exploration_kwargs = dict(config.exploration_kwargs)
        if mlip_allowance is not None:
            prior_limit = exploration_kwargs.get("max_relaxations")
            exploration_kwargs["max_relaxations"] = (
                min(prior_limit, mlip_allowance) if prior_limit is not None else mlip_allowance
            )
        if on_mlip_relaxation_attempt is not None:
            exploration_kwargs["on_relaxation_attempt"] = on_mlip_relaxation_attempt
        effective_config = config.model_copy(
            update={
                "max_dft_calculations": dft_allowance,
                "exploration_kwargs": exploration_kwargs,
            }
        )
        try:
            result = run_formula_with_active_learning(
                formula, output_dir, config=effective_config, random_seed=random_seed
            )
        except Exception:
            # A promotion committed before a later stage failed is durable; the next
            # formula must use it, as a restart from the campaign state would.
            promoted = [
                state["new_logdir"]
                for state in _durable_iteration_states(Path(output_dir) / "active_learning")
                if state.get("model_promoted") and state.get("new_logdir")
            ]
            if promoted:
                config.model_override = str(promoted[-1])
            raise
        if result.model_promoted and result.active_learning_result:
            promoted_path = _promoted_model_path(result)
            if promoted_path:
                config.model_override = str(promoted_path)
        return result

    return runner


def latest_promoted_model(campaign: CampaignState) -> str | None:
    """Return the newest promoted checkpoint recorded in durable campaign evidence.

    Uses each formula's durable ``promoted_checkpoint``, ordered by the campaign-wide
    ``promotion_sequence``, so retries that promote nothing do not hide it.
    """

    promoted = [
        record
        for record in campaign.formula_runs.values()
        if record.promoted_checkpoint is not None and record.promotion_sequence is not None
    ]
    if not promoted:
        return None
    return max(promoted, key=lambda record: record.promotion_sequence).promoted_checkpoint


__all__ = [
    "CampaignDFTRefinementConfig",
    "CampaignFormulaExecutionConfig",
    "CampaignRetrainingConfig",
    "ReferenceStructureSpec",
    "latest_promoted_model",
    "make_formula_runner",
    "run_formula_with_active_learning",
]
