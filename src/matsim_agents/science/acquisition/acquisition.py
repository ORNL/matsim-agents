"""Deterministic exploration--exploitation selection for campaign formulas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.core.seeding import python_rng


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

    # Unknown keys fail, so removed seed fields cannot be silently ignored.
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: Literal["random", "exploitation", "exploration", "adaptive"] = "adaptive"
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
    random_seed: int,
) -> AcquisitionSelectionRecord:
    """Select a reproducible batch while preserving minimum branch quotas.

    Random mode draws from the iteration's stream of *random_seed* (the campaign seed).
    """
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
        python_rng(random_seed, "formula-acquisition", iteration).shuffle(ordered)
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
            prior_branches = [
                record.scores[formula].assigned_branch
                for record in state.selection_history
                for formula in record.selected_formulas
            ]
            prior_total = len(prior_branches)
            cumulative_total = prior_total + limit
            required_exploit = int(cumulative_total * policy.minimum_exploitation_fraction + 1e-12)
            required_explore = int(cumulative_total * policy.minimum_exploration_fraction + 1e-12)
            n_exploit = min(
                limit,
                max(0, required_exploit - prior_branches.count("exploitation")),
            )
            n_explore = min(
                limit - n_exploit,
                max(0, required_explore - prior_branches.count("exploration")),
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
        state.exploration_useful / state.exploration_attempts if state.exploration_attempts else 0.0
    )
    updated = current + policy.update_rate * (exploit_yield - explore_yield)
    state.lambda_value = min(policy.lambda_maximum, max(policy.lambda_minimum, updated))
    return state.lambda_value


# ---------------------------------------------------------------------------
# Candidate-level (calculation) acquisition for the hierarchical campaign
# ---------------------------------------------------------------------------

CandidateDecision = Literal["selected", "rejected", "not_selected"]
CANDIDATE_POLICY_VERSION = "candidate-acquisition-v1"
NOT_YET_ESTABLISHED = "NOT_YET_ESTABLISHED"


class CandidateAcquisitionInputs(BaseModel):
    """Evidence about one proposed calculation; every score field is normalised to [0, 1]."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    calculation_id: str
    region: str = "default"
    relevance: float = Field(0.0, ge=0.0, le=1.0)
    newly_covered: frozenset[str] = frozenset()
    uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    transfer_gain: float | None = Field(None, ge=0.0, le=1.0)
    diversity: float = Field(0.0, ge=0.0, le=1.0)
    magnetic_relevance: float = Field(0.0, ge=0.0, le=1.0)
    defect_relevance: float = Field(0.0, ge=0.0, le=1.0)
    connectivity: float = Field(1.0, ge=0.0, le=1.0)
    compatible: bool = True
    incompatibility_reason: str = ""
    cost_node_hours: float = Field(..., gt=0.0)


class CandidateAcquisitionWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: float = Field(0.20, ge=0.0)
    missing_interaction: float = Field(0.20, ge=0.0)
    uncertainty: float = Field(0.15, ge=0.0)
    transfer_gain: float = Field(0.20, ge=0.0)
    diversity: float = Field(0.10, ge=0.0)
    magnetic_relevance: float = Field(0.075, ge=0.0)
    defect_relevance: float = Field(0.075, ge=0.0)


class CandidateAcquisitionPolicy(BaseModel):
    """Cost-aware, deterministic selection of calculations; value is never cost alone."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(8, ge=1)
    max_batch_node_hours: float | None = Field(None, gt=0.0)
    cost_floor_node_hours: float = Field(1.0, gt=0.0)
    # A candidate must carry at least this fraction of the best value to be chosen on cost.
    min_value_fraction: float = Field(0.25, ge=0.0, le=1.0)
    saturation_limit: int = Field(25, ge=1)
    min_connectivity: float = Field(0.2, ge=0.0, le=1.0)
    diversity_penalty: float = Field(0.5, ge=0.0, le=1.0)
    weights: CandidateAcquisitionWeights = Field(default_factory=CandidateAcquisitionWeights)


class CandidateScore(BaseModel):
    candidate_id: str
    calculation_id: str
    decision: CandidateDecision
    reason: str
    value: float = 0.0
    value_per_node_hour: float = 0.0
    cost_node_hours: float
    basis: str = "transfer-guided"
    components: dict[str, float] = Field(default_factory=dict)
    newly_covered: list[str] = Field(default_factory=list)
    rank: int | None = None


class CandidateSelection(BaseModel):
    policy_version: str = CANDIDATE_POLICY_VERSION
    selected: list[str]
    total_node_hours: float
    scores: list[CandidateScore]

    def by_id(self) -> dict[str, CandidateScore]:
        return {s.candidate_id: s for s in self.scores}


def expected_transfer_gain(
    registry_status: str, relative_gain: float | None = None, scale: float = 1.0
) -> float | None:
    """Map a transfer verdict to [0, 1]; unknown evidence stays ``None`` (never invented)."""
    if registry_status == NOT_YET_ESTABLISHED or registry_status == "pending":
        return None
    if registry_status == "unsupported":
        return 0.0
    if registry_status == "validated" and relative_gain is not None:
        return float(min(1.0, max(0.0, relative_gain / max(scale, 1e-12))))
    return None


def _candidate_value(
    item: CandidateAcquisitionInputs,
    policy: CandidateAcquisitionPolicy,
    covered_in_batch: set[str],
    region_counts: dict[str, int],
    n_missing: int,
) -> tuple[float, str, dict[str, float]]:
    w = policy.weights
    marginal = len(item.newly_covered - covered_in_batch)
    coverage = marginal / n_missing if n_missing else 0.0
    components = {
        "relevance": item.relevance,
        "missing_interaction": min(1.0, coverage),
        "uncertainty": item.uncertainty,
        "diversity": item.diversity
        * (1.0 - policy.diversity_penalty) ** region_counts.get(item.region, 0),
        "magnetic_relevance": item.magnetic_relevance,
        "defect_relevance": item.defect_relevance,
    }
    weights = {
        "relevance": w.relevance,
        "missing_interaction": w.missing_interaction,
        "uncertainty": w.uncertainty,
        "diversity": w.diversity,
        "magnetic_relevance": w.magnetic_relevance,
        "defect_relevance": w.defect_relevance,
    }
    if item.transfer_gain is None:
        # Unknown gain: its weight moves, visibly, onto coverage and uncertainty.
        weights["missing_interaction"] += w.transfer_gain / 2.0
        weights["uncertainty"] += w.transfer_gain / 2.0
        basis = "coverage+uncertainty fallback"
    else:
        components["transfer_gain"] = item.transfer_gain
        weights["transfer_gain"] = w.transfer_gain
        basis = "transfer-guided"
    total = sum(weights.values())
    value = sum(weights[k] * components[k] for k in weights) / total if total else 0.0
    return value, basis, components


def select_candidate_batch(
    candidates: Sequence[CandidateAcquisitionInputs],
    policy: CandidateAcquisitionPolicy | None = None,
    *,
    existing_calculation_ids: Iterable[str] = (),
    region_counts: Mapping[str, int] | None = None,
    missing_interactions: Iterable[str] = (),
) -> CandidateSelection:
    """Greedy, deterministic selection by value per node-hour with explicit rejections."""
    policy = policy or CandidateAcquisitionPolicy()
    seen = set(existing_calculation_ids)
    regions: dict[str, int] = dict(region_counts or {})
    n_missing = len(set(missing_interactions))
    scores: dict[str, CandidateScore] = {}
    eligible: list[CandidateAcquisitionInputs] = []
    for item in sorted(candidates, key=lambda c: c.candidate_id):
        reason = ""
        if item.calculation_id in seen:
            reason = "duplicate_calculation"
        elif not item.compatible:
            reason = f"incompatible: {item.incompatibility_reason or 'unspecified'}"
        elif item.connectivity < policy.min_connectivity:
            reason = "poorly_connected"
        elif regions.get(item.region, 0) >= policy.saturation_limit:
            reason = "saturated_region"
        if reason:
            scores[item.candidate_id] = CandidateScore(
                candidate_id=item.candidate_id,
                calculation_id=item.calculation_id,
                decision="rejected",
                reason=reason,
                cost_node_hours=item.cost_node_hours,
            )
            continue
        seen.add(item.calculation_id)
        eligible.append(item)

    selected: list[str] = []
    covered: set[str] = set()
    spent = 0.0
    best_value: float | None = None
    remaining = list(eligible)
    while remaining and len(selected) < policy.batch_size:
        scored = []
        for item in remaining:
            value, basis, comps = _candidate_value(item, policy, covered, regions, n_missing)
            per_hour = value / max(item.cost_node_hours, policy.cost_floor_node_hours)
            scored.append((per_hour, value, basis, comps, item))
        if best_value is None:
            best_value = max(s[1] for s in scored)
        floor = policy.min_value_fraction * best_value
        viable = [s for s in scored if s[1] >= floor and s[1] > 0.0]
        if policy.max_batch_node_hours is not None:
            viable = [
                s for s in viable if spent + s[4].cost_node_hours <= policy.max_batch_node_hours
            ]
        if not viable:
            break
        per_hour, value, basis, comps, item = min(
            viable, key=lambda s: (-s[0], -s[1], s[4].candidate_id)
        )
        fresh = sorted(item.newly_covered - covered)
        selected.append(item.candidate_id)
        spent += item.cost_node_hours
        covered |= item.newly_covered
        regions[item.region] = regions.get(item.region, 0) + 1
        scores[item.candidate_id] = CandidateScore(
            candidate_id=item.candidate_id,
            calculation_id=item.calculation_id,
            decision="selected",
            reason="highest value per node-hour",
            value=value,
            value_per_node_hour=per_hour,
            cost_node_hours=item.cost_node_hours,
            basis=basis,
            components=comps,
            newly_covered=fresh,
            rank=len(selected),
        )
        remaining = [r for r in remaining if r.candidate_id != item.candidate_id]

    for item in remaining:
        value, basis, comps = _candidate_value(item, policy, covered, regions, n_missing)
        below = best_value is not None and value < policy.min_value_fraction * best_value
        scores[item.candidate_id] = CandidateScore(
            candidate_id=item.candidate_id,
            calculation_id=item.calculation_id,
            decision="not_selected",
            reason="below_value_floor" if below else "batch_or_budget_limit",
            value=value,
            value_per_node_hour=value / max(item.cost_node_hours, policy.cost_floor_node_hours),
            cost_node_hours=item.cost_node_hours,
            basis=basis,
            components=comps,
        )
    ordered = [scores[c.candidate_id] for c in sorted(candidates, key=lambda c: c.candidate_id)]
    return CandidateSelection(selected=selected, total_node_hours=spent, scores=ordered)
