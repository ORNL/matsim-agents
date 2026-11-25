"""Choose fusion PFC calculations by expected decision improvement per node-hour (F8).

Decision confidence is ``P(top) * n_complete / n_ranked``: ``P(top)`` is the
Monte-Carlo probability that the point-estimate leader stays first when every
scored observable is drawn from its value and one-sigma uncertainty, and the
second factor counts candidates that cannot be ranked because an objective is
missing. Nothing is zero-filled: an incomplete candidate is simply not ranked.

The expected improvement of a proposal is preposterior: draw the truth from the
current value (or from a declared, labelled prior for an observable with no
value), simulate the proposal's declared uncertainty, combine by inverse
variance, and recompute the confidence. Draws are seeded per proposal. A
proposal for an observable with neither a value nor a declared prior is
``unquantified``. Proxies (any observable not scored for the objective) are
refused. Selection stays within the property category's real remaining
node-hours in the :class:`BudgetLedger`, minus the validation reserve, and a
proposal whose prerequisite is neither done nor selected is ``blocked``. A plan
dispatches nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.campaign.budget import BudgetLedger
from matsim_agents.campaign.property_acquisition import BUDGET_CATEGORY, BudgetReport, _budget
from matsim_agents.execution.contracts import EvidenceLevel
from matsim_agents.fusion.policy import FUSION_PFC_OBJECTIVES, FusionScoringPolicy
from matsim_agents.fusion.scoring import BenchmarkSet, ConstraintStatus, ScoringResult
from matsim_agents.seeding import numpy_rng

ACQUISITION_VERSION = "fusion-pfc-acquisition-v1"
Decision = Literal["selected", "not_selected", "blocked", "unquantified", "duplicate", "refused"]


class FusionCalculation(BaseModel):
    """One proposed calculation of a scored observable for one candidate."""

    model_config = ConfigDict(extra="forbid")

    calc_id: str = Field(min_length=1)
    candidate_id: str
    objective: str
    observable: str
    backend: str
    cost_node_hours: float = Field(gt=0)
    expected_uncertainty: float = Field(gt=0, description="Declared by the method, same unit.")
    resulting_level: EvidenceLevel = EvidenceLevel.CONVERGED_DFT
    requires: tuple[str, ...] = Field((), description="Calc ids that must run first.")
    prior_value: float | None = None
    prior_uncertainty: float | None = Field(None, gt=0)
    prior_source: str | None = None

    @model_validator(mode="after")
    def _prior(self) -> FusionCalculation:
        given = [self.prior_value is not None, self.prior_uncertainty is not None]
        if any(given) and not (all(given) and self.prior_source):
            raise ValueError("a prior needs value, uncertainty and source")
        if self.objective not in FUSION_PFC_OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}")
        return self


class FusionCalculationScore(BaseModel):
    calc_id: str
    candidate_id: str
    objective: str
    observable: str
    backend: str
    decision: Decision
    reason: str
    cost_node_hours: float
    confidence_before: float
    expected_confidence_after: float | None = None
    expected_improvement: float | None = None
    improvement_per_node_hour: float | None = None
    used_prior: bool = False
    rank: int | None = None


class FusionAcquisitionPlan(BaseModel):
    version: str = ACQUISITION_VERSION
    score_digest: str
    policy_version: str
    confidence_before: float
    items: list[FusionCalculationScore]
    selected: list[str]
    budget: BudgetReport
    observed_cost: dict[str, float] = Field(default_factory=dict)
    requires_approval: bool = True
    dispatched: bool = False
    limitations: list[str]
    plan_digest: str = ""


# (candidate, objective, observable) -> (value, sigma)
_State = dict[tuple[str, str, str], tuple[float, float]]


def _state(result: ScoringResult) -> tuple[_State, list[str]]:
    state: _State = {}
    ranked = []
    for c in result.candidates:
        if c.feasibility == ConstraintStatus.VIOLATED:
            continue
        ranked.append(c.candidate_id)
        for o in c.objectives:
            for x in o.observables:
                if x.value is not None:
                    state[(c.candidate_id, o.objective, x.observable)] = (
                        x.value,
                        x.uncertainty or 0.0,
                    )
    return state, ranked


def decision_confidence(
    state: _State,
    ranked: Sequence[str],
    policy: FusionScoringPolicy,
    benchmarks: BenchmarkSet,
    *,
    samples: int = 400,
    seed: int = 0,
) -> float:
    """``P(top) * n_complete / n_ranked`` for the given observable state."""
    complete = [
        cid
        for cid in ranked
        if all(
            (cid, n, obs) in state
            for n in FUSION_PFC_OBJECTIVES
            for obs in benchmarks.objectives[n].observables
        )
    ]
    if not ranked or not complete:
        return 0.0
    if len(complete) == 1:
        return 1.0 / len(ranked)
    rng = numpy_rng(seed, "fusion-acquisition", "confidence")
    totals = np.zeros((len(complete), samples))
    point = np.zeros(len(complete))
    for n in FUSION_PFC_OBJECTIVES:
        w = policy.weights[n]
        obs_d = np.ones((len(complete), samples))
        obs_p = np.ones(len(complete))
        for obs, bench in benchmarks.objectives[n].observables.items():
            lo, hi = bench.worst, bench.target
            for i, cid in enumerate(complete):
                mu, sd = state[(cid, n, obs)]
                draw = mu + sd * rng.standard_normal(samples)
                obs_d[i] = np.minimum(obs_d[i], np.clip((draw - lo) / (hi - lo), 0, 1))
                obs_p[i] = min(obs_p[i], bench.desirability(mu))
        totals += w * obs_d
        point += w * obs_p
    leader = int(np.argmax(point))
    p_top = float(np.mean(np.argmax(totals, axis=0) == leader))
    return p_top * len(complete) / len(ranked)


def _expected_after(
    state: _State,
    ranked: Sequence[str],
    item: FusionCalculation,
    mu: float,
    sd: float,
    policy: FusionScoringPolicy,
    benchmarks: BenchmarkSet,
    draws: int,
    samples: int,
    seed: int,
) -> float:
    rng = numpy_rng(seed, "fusion-acquisition", item.calc_id)
    s = item.expected_uncertainty
    key = (item.candidate_id, item.objective, item.observable)
    total = 0.0
    for _ in range(draws):
        truth = mu + sd * rng.standard_normal()
        measured = truth + s * rng.standard_normal()
        if sd > 0:
            post_sd = (1.0 / sd**2 + 1.0 / s**2) ** -0.5
            post = (mu / sd**2 + measured / s**2) * post_sd**2
        else:
            post_sd, post = s, measured
        trial = {**state, key: (post, post_sd)}
        total += decision_confidence(trial, ranked, policy, benchmarks, samples=samples, seed=seed)
    return total / draws


def plan_fusion_acquisition(
    result: ScoringResult,
    policy: FusionScoringPolicy,
    benchmarks: BenchmarkSet,
    proposals: Sequence[FusionCalculation],
    budget: BudgetLedger,
    *,
    observed_cost: Mapping[str, float] | None = None,
    blockers: Mapping[str, str] | None = None,
    completed_calc_ids: Collection[str] = (),
    draws: int = 12,
    samples: int = 400,
    seed: int = 0,
    min_improvement: float = 0.0,
) -> FusionAcquisitionPlan:
    """Rank proposals by expected decision-confidence improvement per node-hour."""
    if draws < 1 or samples < 10:
        raise ValueError("draws must be >= 1 and samples >= 10")
    if result.policy_digest != policy.digest() or result.benchmark_digest != benchmarks.digest():
        raise ValueError("policy or benchmarks differ from those used for scoring")
    blockers = dict(blockers or {})
    state, ranked = _state(result)
    before = decision_confidence(state, ranked, policy, benchmarks, samples=samples, seed=seed)
    done = set(completed_calc_ids)
    seen: set[str] = set(done)
    targets: set[tuple[str, str, str]] = set()
    known = {c.candidate_id for c in result.candidates}
    scored: list[FusionCalculationScore] = []
    proposals_by_id = {p.calc_id: p for p in proposals}

    for item in sorted(proposals, key=lambda p: p.calc_id):

        def row(decision, reason, after=None, used_prior=False, item=item):
            gain = None if after is None else after - before
            return FusionCalculationScore(
                calc_id=item.calc_id,
                candidate_id=item.candidate_id,
                objective=item.objective,
                observable=item.observable,
                backend=item.backend,
                decision=decision,
                reason=reason,
                cost_node_hours=item.cost_node_hours,
                confidence_before=before,
                expected_confidence_after=after,
                expected_improvement=gain,
                improvement_per_node_hour=None if gain is None else gain / item.cost_node_hours,
                used_prior=used_prior,
            )

        key = (item.candidate_id, item.objective, item.observable)
        if item.calc_id in seen or key in targets:
            scored.append(row("duplicate", "calculation or target already planned or done"))
            continue
        seen.add(item.calc_id)
        targets.add(key)
        bench = benchmarks.objectives.get(item.objective)
        if bench is None or item.observable not in bench.observables:
            scored.append(
                row("refused", f"{item.observable} is not scored for {item.objective} (proxy)")
            )
            continue
        if item.candidate_id not in known:
            scored.append(row("refused", "unknown candidate"))
            continue
        if item.candidate_id not in ranked:
            scored.append(row("refused", "candidate violates a constraint"))
            continue
        if item.backend in blockers:
            scored.append(row("blocked", f"backend {item.backend}: {blockers[item.backend]}"))
            continue
        unknown_req = [r for r in item.requires if r not in done and r not in proposals_by_id]
        if unknown_req:
            scored.append(row("blocked", f"prerequisite not planned or done: {unknown_req}"))
            continue
        if key in state:
            mu, sd = state[key]
            used_prior = False
        elif item.prior_value is not None:
            mu, sd = item.prior_value, float(item.prior_uncertainty)  # type: ignore[arg-type]
            used_prior = True
        else:
            scored.append(
                row("unquantified", "no current value or declared prior; gain is not invented")
            )
            continue
        after = _expected_after(
            state, ranked, item, mu, sd, policy, benchmarks, draws, samples, seed
        )
        scored.append(row("not_selected", "", after, used_prior))

    spendable = _budget(budget, 0.0).spendable
    spent = 0.0
    chosen: list[str] = []
    ranked_items = sorted(
        (s for s in scored if s.decision == "not_selected"),
        key=lambda s: (-(s.improvement_per_node_hour or 0.0), s.calc_id),
    )
    for s in ranked_items:
        prereq = proposals_by_id[s.calc_id].requires
        pending = [r for r in prereq if r not in done and r not in chosen]
        if pending:
            s.decision = "blocked"
            s.reason = f"prerequisite {pending} is not done or selected ahead of it"
        elif (s.expected_improvement or 0.0) <= min_improvement:
            s.reason = "no expected improvement in decision confidence"
        elif spent + s.cost_node_hours > spendable + 1e-9:
            s.reason = f"exceeds spendable budget ({spendable:.1f} node-hours)"
        else:
            spent += s.cost_node_hours
            chosen.append(s.calc_id)
            s.decision, s.rank = "selected", len(chosen)
            s.reason = "highest expected decision improvement per node-hour within budget"
    limitations = [
        "Improvements are estimated one proposal at a time; interactions are not.",
        "Costs and expected uncertainties are the proposals' declared values.",
        "A plan is advice: it dispatches nothing and every item needs approval.",
    ]
    if any(s.used_prior for s in scored):
        limitations.append("Some improvements rest on declared priors, labelled per item.")
    if result.synthetic_benchmarks:
        limitations.append("Benchmarks are SYNTHETIC; the plan is a demonstration only.")
    plan = FusionAcquisitionPlan(
        score_digest=result.digest(),
        policy_version=policy.policy_version,
        confidence_before=before,
        items=scored,
        selected=chosen,
        budget=_budget(budget, spent),
        observed_cost={k: float(v) for k, v in (observed_cost or {}).items() if _num(v)},
        limitations=limitations,
    )
    payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    plan.plan_digest = "facq-" + hashlib.sha256(payload.encode()).hexdigest()[:24]
    return plan


def _num(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def reserve_selected(
    plan: FusionAcquisitionPlan,
    budget: BudgetLedger,
    approved: Sequence[str],
    *,
    authorized_by: str,
) -> list[str]:
    """Reserve (not submit) the named selected items; a person must approve them."""
    if not authorized_by.strip():
        raise ValueError("reserving needs the name of the person who approved it")
    by_id = {s.calc_id: s for s in plan.items}
    for calc_id in approved:
        if calc_id not in by_id or by_id[calc_id].decision != "selected":
            raise ValueError(f"{calc_id} is not a selected item of this plan")
    for calc_id in approved:
        budget.reserve(calc_id, BUDGET_CATEGORY, by_id[calc_id].cost_node_hours)
    return list(approved)


def render_fusion_plan(plan: FusionAcquisitionPlan) -> str:
    b = plan.budget
    out = [
        f"Fusion PFC acquisition plan ({plan.version}, {plan.plan_digest}); scores "
        f"{plan.score_digest}; policy {plan.policy_version}",
        f"Decision confidence now: {plan.confidence_before:.3f}. Requires approval; nothing "
        "dispatched.",
        f"Budget (node-hours): property available {b.category_available:g}, remaining "
        f"{b.remaining:g}, validation reserve left {b.validation_reserve_remaining:g}, "
        f"spendable {b.spendable:g}, after plan {b.remaining_after_plan:g}",
    ]
    if plan.observed_cost:
        out.append(
            "Observed cost so far: "
            + ", ".join(f"{k} {v:g}" for k, v in sorted(plan.observed_cost.items()))
        )
    out.append("calc | candidate | objective/observable | decision | cost | gain/node-h | reason")
    for s in plan.items:
        per = (
            "n/a" if s.improvement_per_node_hour is None else f"{s.improvement_per_node_hour:+.5f}"
        )
        prior = " [prior]" if s.used_prior else ""
        out.append(
            f"{s.calc_id} | {s.candidate_id} | {s.objective}/{s.observable} | {s.decision} | "
            f"{s.cost_node_hours:g} | {per}{prior} | {s.reason}"
        )
    out += [f"limitation: {x}" for x in plan.limitations]
    return "\n".join(out)


__all__ = [
    "ACQUISITION_VERSION",
    "FusionAcquisitionPlan",
    "FusionCalculation",
    "FusionCalculationScore",
    "decision_confidence",
    "plan_fusion_acquisition",
    "render_fusion_plan",
    "reserve_selected",
]
