"""Composable phase exploration built from relaxation and active learning."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from matsim_agents.discovery.wrapper import CompositionExplorationResult, explore_composition
from matsim_agents.execution.contracts import ApprovalPolicy, ComputeBudget


class PhaseExplorationPolicy(BaseModel):
    relax_structures: bool = True
    active_learning: bool = False
    retrain_mlip: bool = False
    promote_model: bool = False
    reevaluate_after_retraining: bool = False
    continue_on_promotion_rejection: bool = False
    ranking_mode: str = "relative_phase_ranking"
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    approvals: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    dft_approved: bool = False
    retraining_approved: bool = False
    promotion_approved: bool = False

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

    kwargs = dict(exploration_kwargs or {})
    limits = [
        limit
        for limit in (policy.budget.max_mlip_relaxations, kwargs.get("max_relaxations"))
        if limit is not None
    ]
    relaxation_allowance = min(limits) if limits else None
    if relaxation_allowance is not None:
        kwargs["max_relaxations"] = relaxation_allowance
    if not policy.relax_structures:
        # Seed-only exploration is explicit and uses a runner that records no
        # fake relaxation result. The existing wrapper still owns generation.
        from matsim_agents.discovery.composition import parse_composition
        from matsim_agents.discovery.seeds import generate_seeds

        parsed = parse_composition(composition)
        if parsed is None:
            raise ValueError(f"Could not parse composition {composition!r}")
        candidates = generate_seeds(
            parsed,
            str(Path(output_dir) / parsed.formula / "seeds"),
            n_random=kwargs.get("n_random", 50),
            random_seed=kwargs.get("random_seed", 0),
        )
        initial = CompositionExplorationResult(
            composition=parsed,
            phase_candidates=candidates,
            outcome_class="seed_only" if candidates else "generation_failure",
        )
    else:
        initial = explore_composition(composition, output_dir=output_dir, **kwargs)

    al_result = None
    promoted = False
    after = None
    if policy.active_learning:
        if active_learning_runner is None:
            raise ValueError("active_learning=True requires active_learning_runner")
        al_result = active_learning_runner(
            composition,
            output_dir,
            policy.retrain_mlip,
            policy.promote_model,
            policy.promotion_approved,
        )
        promoted = bool(al_result.get("model_promoted", False))
        if promoted and not policy.promote_model:
            raise RuntimeError("active learning promoted a model without promotion being requested")
        if promoted and policy.approvals.before_model_promotion and not policy.promotion_approved:
            raise PermissionError("model promotion requires explicit approval")
        if policy.reevaluate_after_retraining and not promoted:
            if not policy.continue_on_promotion_rejection:
                raise RuntimeError("cannot reevaluate: active learning did not promote a model")
            logging.getLogger(__name__).warning(
                "No model promoted for %s; retaining incumbent MLIP exploration", composition
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
                output_dir=str(Path(output_dir) / "after_retraining"),
                **updated,
            )
    return PhaseExplorationWorkflowResult(
        composition=composition,
        initial=initial,
        after_retraining=after,
        active_learning_result=al_result,
        model_promoted=promoted,
    )


__all__ = ["PhaseExplorationPolicy", "PhaseExplorationWorkflowResult", "run_phase_exploration"]
