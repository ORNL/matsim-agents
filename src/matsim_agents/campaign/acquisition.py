"""Deterministic exploration--exploitation selection for campaign formulas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FormulaAcquisitionMetrics(BaseModel):
    """Normalized evidence used to prioritize one formula."""

    predicted_hull_proximity: float = Field(0.0, ge=0.0, le=1.0)
    mlip_uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    structural_novelty: float = Field(0.0, ge=0.0, le=1.0)
    structural_diversity: float = Field(0.0, ge=0.0, le=1.0)
    composition_coverage: float = Field(0.0, ge=0.0, le=1.0)
    hull_uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    llm_disagreement: float = Field(0.0, ge=0.0, le=1.0)
    llm_falsification_priority: float = Field(0.0, ge=0.0, le=1.0)
    convergence: float = Field(1.0, ge=0.0, le=1.0)


class AcquisitionWeights(BaseModel):
    predicted_hull_proximity: float = Field(0.30, ge=0.0)
    mlip_uncertainty: float = Field(0.25, ge=0.0)
    structural_novelty: float = Field(0.15, ge=0.0)
    structural_diversity: float = Field(0.10, ge=0.0)
    composition_coverage: float = Field(0.10, ge=0.0)
    hull_uncertainty: float = Field(0.05, ge=0.0)
    llm_disagreement: float = Field(0.025, ge=0.0)
    llm_falsification_priority: float = Field(0.025, ge=0.0)


class CampaignAcquisitionPolicy(BaseModel):
    """Versioned controls for selecting formulas within each campaign batch."""

    enabled: bool = False
    mode: Literal["random", "exploitation", "exploration", "adaptive"] = "adaptive"
    random_seed: int = 0
    lambda_initial: float = Field(0.5, ge=0.0, le=1.0)
    lambda_minimum: float = Field(0.2, ge=0.0, le=1.0)
    lambda_maximum: float = Field(0.8, ge=0.0, le=1.0)
    update_rate: float = Field(0.15, ge=0.0)
    minimum_exploitation_fraction: float = Field(0.2, ge=0.0, le=1.0)
    minimum_exploration_fraction: float = Field(0.2, ge=0.0, le=1.0)
    weights: AcquisitionWeights = Field(default_factory=AcquisitionWeights)

    @model_validator(mode="after")
    def _validate_bounds(self) -> CampaignAcquisitionPolicy:
        if self.lambda_minimum > self.lambda_initial:
            raise ValueError("lambda_minimum must be <= lambda_initial")
        if self.lambda_initial > self.lambda_maximum:
            raise ValueError("lambda_initial must be <= lambda_maximum")
        if self.minimum_exploitation_fraction + self.minimum_exploration_fraction > 1.0:
            raise ValueError("minimum acquisition fractions must sum to at most one")
        return self


class FormulaAcquisitionScore(BaseModel):
    formula: str
    exploitation_score: float
    exploration_score: float
    combined_score: float
    assigned_branch: Literal["exploitation", "exploration", "random"]
    metrics: FormulaAcquisitionMetrics


class AcquisitionSelectionRecord(BaseModel):
    iteration: int
    policy_version: str = "formula-acquisition-v1"
    mode: str
    lambda_value: float
    eligible_formulas: list[str]
    selected_formulas: list[str]
    scores: dict[str, FormulaAcquisitionScore]


class CampaignAcquisitionState(BaseModel):
    lambda_value: float | None = None
    formula_metrics: dict[str, FormulaAcquisitionMetrics] = Field(default_factory=dict)
    selection_history: list[AcquisitionSelectionRecord] = Field(default_factory=list)
    exploitation_attempts: int = 0
    exploitation_useful: int = 0
    exploration_attempts: int = 0
    exploration_useful: int = 0


def _scores(
    formula: str,
    metrics: FormulaAcquisitionMetrics,
    policy: CampaignAcquisitionPolicy,
    lambda_value: float,
) -> FormulaAcquisitionScore:
    weights = policy.weights
    exploitation = metrics.convergence * (
        weights.predicted_hull_proximity * metrics.predicted_hull_proximity
        + weights.mlip_uncertainty * (1.0 - metrics.mlip_uncertainty)
    )
    exploration = (
        weights.mlip_uncertainty * metrics.mlip_uncertainty
        + weights.structural_novelty * metrics.structural_novelty
        + weights.structural_diversity * metrics.structural_diversity
        + weights.composition_coverage * metrics.composition_coverage
        + weights.hull_uncertainty * metrics.hull_uncertainty
        + weights.llm_disagreement * metrics.llm_disagreement
        + weights.llm_falsification_priority * metrics.llm_falsification_priority
    )
    combined = lambda_value * exploitation + (1.0 - lambda_value) * exploration
    return FormulaAcquisitionScore(
        formula=formula,
        exploitation_score=exploitation,
        exploration_score=exploration,
        combined_score=combined,
        assigned_branch="exploitation" if exploitation >= exploration else "exploration",
        metrics=metrics,
    )


def select_formula_batch(
    *,
    eligible: list[str],
    batch_size: int,
    iteration: int,
    policy: CampaignAcquisitionPolicy,
    state: CampaignAcquisitionState,
) -> AcquisitionSelectionRecord:
    """Select a reproducible batch while preserving minimum branch quotas."""
    import random

    lambda_value = state.lambda_value
    if lambda_value is None:
        lambda_value = policy.lambda_initial
        state.lambda_value = lambda_value
    metrics_by_formula = {
        formula: state.formula_metrics.get(formula, FormulaAcquisitionMetrics())
        for formula in eligible
    }
    scores = {
        formula: _scores(formula, metrics, policy, lambda_value)
        for formula, metrics in metrics_by_formula.items()
    }
    limit = min(batch_size, len(eligible))
    if policy.mode == "random":
        ordered = list(sorted(eligible))
        random.Random(policy.random_seed + iteration).shuffle(ordered)
        selected = ordered[:limit]
        for formula in selected:
            scores[formula].assigned_branch = "random"
    else:
        exploit_order = sorted(
            eligible, key=lambda formula: (-scores[formula].exploitation_score, formula)
        )
        explore_order = sorted(
            eligible, key=lambda formula: (-scores[formula].exploration_score, formula)
        )
        if policy.mode == "exploitation":
            selected = exploit_order[:limit]
            for formula in selected:
                scores[formula].assigned_branch = "exploitation"
        elif policy.mode == "exploration":
            selected = explore_order[:limit]
            for formula in selected:
                scores[formula].assigned_branch = "exploration"
        else:
            n_exploit = min(limit, int(limit * policy.minimum_exploitation_fraction + 0.999999))
            n_explore = min(
                limit - n_exploit,
                int(limit * policy.minimum_exploration_fraction + 0.999999),
            )
            selected = exploit_order[:n_exploit]
            for formula in selected:
                scores[formula].assigned_branch = "exploitation"
            for formula in explore_order:
                if formula not in selected and len(selected) < n_exploit + n_explore:
                    selected.append(formula)
                    scores[formula].assigned_branch = "exploration"
            combined_order = sorted(
                eligible, key=lambda formula: (-scores[formula].combined_score, formula)
            )
            for formula in combined_order:
                if formula not in selected and len(selected) < limit:
                    selected.append(formula)
    record = AcquisitionSelectionRecord(
        iteration=iteration,
        mode=policy.mode,
        lambda_value=lambda_value,
        eligible_formulas=list(eligible),
        selected_formulas=selected,
        scores=scores,
    )
    state.selection_history.append(record)
    return record


def update_adaptive_lambda(
    state: CampaignAcquisitionState,
    policy: CampaignAcquisitionPolicy,
) -> float:
    """Update allocation from cumulative useful-outcome rates."""
    current = state.lambda_value if state.lambda_value is not None else policy.lambda_initial
    exploit_yield = (
        state.exploitation_useful / state.exploitation_attempts
        if state.exploitation_attempts
        else 0.0
    )
    explore_yield = (
        state.exploration_useful / state.exploration_attempts
        if state.exploration_attempts
        else 0.0
    )
    updated = current + policy.update_rate * (exploit_yield - explore_yield)
    state.lambda_value = min(policy.lambda_maximum, max(policy.lambda_minimum, updated))
    return state.lambda_value