"""Composable phase exploration built from relaxation and active learning."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from matsim_agents.discovery.wrapper import CompositionExplorationResult, explore_composition
from matsim_agents.execution.contracts import ComputeBudget


class PhaseExplorationPolicy(BaseModel):
    relax_structures: bool = True
    active_learning: bool = False
    retrain_mlip: bool = False
    reevaluate_after_retraining: bool = False
    ranking_mode: str = "relative_phase_ranking"
    budget: ComputeBudget = Field(default_factory=ComputeBudget)

    @model_validator(mode="after")
    def _consistent_options(self) -> PhaseExplorationPolicy:
        if self.retrain_mlip and not self.active_learning:
            raise ValueError("retrain_mlip requires active_learning")
        if self.reevaluate_after_retraining and not self.retrain_mlip:
            raise ValueError("reevaluate_after_retraining requires retrain_mlip")
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
    active_learning_runner: Callable[[str, str, bool], dict[str, Any]] | None = None,
) -> PhaseExplorationWorkflowResult:
    """Run exploration, optional AL, and optional post-promotion reevaluation.

    The AL callback receives ``(composition, output_dir, retrain)`` and must
    return a mapping containing ``model_promoted`` plus any provenance.  This
    keeps the workflow independent of facility-specific launch mechanics.
    """

    kwargs = dict(exploration_kwargs or {})
    n_random = int(kwargs.get("n_random", 50))
    if policy.budget.max_candidates is not None:
        kwargs["n_random"] = min(n_random, policy.budget.max_candidates)
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
        initial = CompositionExplorationResult(composition=parsed, phase_candidates=candidates)
    else:
        initial = explore_composition(composition, output_dir=output_dir, **kwargs)

    al_result = None
    promoted = False
    after = None
    if policy.active_learning:
        if active_learning_runner is None:
            raise ValueError("active_learning=True requires active_learning_runner")
        al_result = active_learning_runner(composition, output_dir, policy.retrain_mlip)
        promoted = bool(al_result.get("model_promoted", False))
        if policy.reevaluate_after_retraining:
            if not promoted:
                raise RuntimeError("cannot reevaluate: active learning did not promote a model")
            updated = dict(kwargs)
            updated.update(dict(al_result.get("exploration_kwargs", {})))
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
