"""Choose property calculations by expected gain in selection confidence per node-hour.

Selection confidence of a candidate is the confidence of its constraint and Pareto decisions,
``mean(max(p, 1 - p))`` over the probabilities the Monte-Carlo propagation produced, times the
share of its scored values that rest on measured or calculated evidence. A candidate that cannot
be scored (missing property) has confidence 0. The campaign value is the mean over candidates.

The expected gain of a proposed calculation is a preposterior estimate: draw the true value from
the current value and uncertainty, simulate a measurement with the proposal's declared
uncertainty, combine the two by inverse variance, re-run the selection, and average. Draws are
seeded and shared between before and after, so the difference is not Monte-Carlo noise.

A proposal for a property with no current value has no prior; its gain is reported as ``None``
and it is never selected on an invented number. Nothing here dispatches work: a plan is advice,
and only :func:`reserve_approved` touches the budget, for ids a person names, as a reservation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.core.contracts import EvidenceLevel
from matsim_agents.core.seeding import numpy_rng
from matsim_agents.execution.budget import BudgetLedger
from matsim_agents.science.acquisition.selection import (
    CandidateEvidence,
    SelectionResult,
    SelectionSpec,
    _pick,
    select,
)
from matsim_agents.science.properties.calculators import PropertyStatus

ACQUISITION_VERSION = "property-acquisition-v1"
BUDGET_CATEGORY = "property"
Decision = Literal["selected", "not_selected", "blocked", "unquantified", "duplicate"]


class ProposedCalculation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calc_id: str = Field(min_length=1)
    candidate_id: str
    property: str
    temperature_K: float | None = Field(None, gt=0)
    backend: str
    cost_node_hours: float = Field(gt=0)
    expected_uncertainty: float = Field(gt=0, description="Declared by the method, same unit.")
    resulting_level: EvidenceLevel = EvidenceLevel.CONVERGED_DFT


class CalculationScore(BaseModel):
    calc_id: str
    candidate_id: str
    property: str
    backend: str
    decision: Decision
    reason: str
    cost_node_hours: float
    confidence_before: float
    expected_confidence_after: float | None = None
    expected_gain: float | None = None
    gain_per_node_hour: float | None = None
    rank: int | None = None


class BudgetReport(BaseModel):
    """Real ledger quantities; ``expected`` is the plan's predicted cost, not yet reserved."""

    total: float
    category_allocation: float
    category_available: float
    committed: float
    reserved: float
    consumed: float
    expected: float
    validation_reserve: float
    validation_reserve_remaining: float
    remaining: float
    spendable: float
    remaining_after_plan: float


class AcquisitionPlan(BaseModel):
    version: str = ACQUISITION_VERSION
    score_digest: str
    confidence_before: float
    items: list[CalculationScore]
    selected: list[str]
    budget: BudgetReport
    requires_approval: bool = True
    dispatched: bool = False
    limitations: list[str]
    plan_digest: str = ""


def _decision_confidence(p: float) -> float:
    return max(p, 1.0 - p)


def candidate_confidence(result: SelectionResult) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in result.candidates:
        parts = [
            _decision_confidence(x)
            for x in (c.constraint_probability, c.pareto_robustness)
            if x is not None
        ]
        if c.missing or not parts or not c.values:
            out[c.candidate_id] = 0.0
            continue
        reliable = sum(v.reliable for v in c.values.values()) / len(c.values)
        out[c.candidate_id] = float(np.mean(parts)) * reliable
    return out


def selection_confidence(result: SelectionResult) -> float:
    conf = candidate_confidence(result)
    return float(np.mean(list(conf.values()))) if conf else 0.0


def _budget(ledger: BudgetLedger, expected: float) -> BudgetReport:
    s = ledger.summary()
    spendable = max(
        0.0, min(ledger.available(BUDGET_CATEGORY), s.remaining - s.validation_reserve_remaining)
    )
    return BudgetReport(
        total=s.total,
        category_allocation=ledger.allocation[BUDGET_CATEGORY],
        category_available=ledger.available(BUDGET_CATEGORY),
        committed=s.committed,
        reserved=s.reserved,
        consumed=s.consumed,
        expected=expected,
        validation_reserve=s.validation_reserve,
        validation_reserve_remaining=s.validation_reserve_remaining,
        remaining=s.remaining,
        spendable=spendable,
        remaining_after_plan=s.remaining - expected,
    )


def _replace(
    candidates: Sequence[CandidateEvidence],
    candidate_id: str,
    property_id: str,
    value: float,
    uncertainty: float,
    level: EvidenceLevel,
) -> list[CandidateEvidence]:
    out = []
    for cand in candidates:
        if cand.candidate_id != candidate_id:
            out.append(cand)
            continue
        rows = [
            e.model_copy(
                update={
                    "value": value,
                    "uncertainty": uncertainty,
                    "status": PropertyStatus.CALCULATED,
                    "evidence_level": level,
                    "converged": True,
                }
            )
            if e.property_id() == property_id
            else e
            for e in cand.evidence
        ]
        out.append(CandidateEvidence(candidate_id=candidate_id, evidence=rows))
    return out


def _expected_after(
    spec: SelectionSpec,
    candidates: Sequence[CandidateEvidence],
    item: ProposedCalculation,
    picked,
    draws: int,
) -> float:
    rng = numpy_rng(spec.seed, "property-acquisition", item.calc_id)
    mu, sd = float(picked.value), float(picked.uncertainty or 0.0)
    s = item.expected_uncertainty
    total = 0.0
    for _ in range(draws):
        truth = mu + sd * rng.standard_normal()
        measured = truth + s * rng.standard_normal()
        if sd > 0:
            post_sd = (1.0 / sd**2 + 1.0 / s**2) ** -0.5
            post = (mu / sd**2 + measured / s**2) * post_sd**2
        else:
            post_sd, post = s, measured
        trial = _replace(
            candidates, item.candidate_id, picked.property_id(), post, post_sd, item.resulting_level
        )
        total += selection_confidence(select(spec, trial))
    return total / draws


def plan_property_acquisition(
    spec: SelectionSpec,
    candidates: Sequence[CandidateEvidence],
    proposals: Sequence[ProposedCalculation],
    ledger: BudgetLedger,
    *,
    blockers: Mapping[str, str] | None = None,
    existing_calc_ids: Collection[str] = (),
    n_samples: int = 400,
    draws: int = 12,
    min_gain: float = 0.0,
) -> AcquisitionPlan:
    """Rank proposals by expected confidence gain per node-hour within the real budget."""
    if draws < 1:
        raise ValueError("draws must be at least 1")
    blockers = dict(blockers or {})
    fast = spec.model_copy(update={"n_samples": min(spec.n_samples, max(100, n_samples))})
    base = select(fast, candidates)
    before = selection_confidence(base)
    by_id = {c.candidate_id: c for c in candidates}
    seen: set[str] = set(existing_calc_ids)
    seen_targets: set[tuple[str, str, float | None]] = set()
    scored: list[CalculationScore] = []

    for item in sorted(proposals, key=lambda p: p.calc_id):

        def row(decision, reason, after=None, item=item):
            gain = None if after is None else after - before
            return CalculationScore(
                calc_id=item.calc_id,
                candidate_id=item.candidate_id,
                property=item.property,
                backend=item.backend,
                decision=decision,
                reason=reason,
                cost_node_hours=item.cost_node_hours,
                confidence_before=before,
                expected_confidence_after=after,
                expected_gain=gain,
                gain_per_node_hour=None if gain is None else gain / item.cost_node_hours,
            )

        target = (item.candidate_id, item.property, item.temperature_K)
        if item.calc_id in seen or target in seen_targets:
            scored.append(row("duplicate", "calculation or target already planned or done"))
            continue
        seen.add(item.calc_id)
        seen_targets.add(target)
        if item.backend in blockers:
            scored.append(row("blocked", f"backend {item.backend}: {blockers[item.backend]}"))
            continue
        cand = by_id.get(item.candidate_id)
        picked = None
        if cand is not None:
            picked, _, _ = _pick(cand.evidence, item.property, item.temperature_K)
        if picked is None:
            scored.append(
                row("unquantified", "no current value: the gain has no prior and is not invented")
            )
            continue
        scored.append(
            row("not_selected", "", _expected_after(fast, candidates, item, picked, draws))
        )

    spendable = _budget(ledger, 0.0).spendable
    spent = 0.0
    chosen: list[str] = []
    ranked = sorted(
        (s for s in scored if s.decision == "not_selected"),
        key=lambda s: (-(s.gain_per_node_hour or 0.0), s.calc_id),
    )
    for s in ranked:
        if (s.expected_gain or 0.0) <= min_gain:
            s.reason = "no expected gain in selection confidence"
        elif spent + s.cost_node_hours > spendable + 1e-9:
            s.reason = f"exceeds spendable budget ({spendable:.1f} node-hours)"
        else:
            spent += s.cost_node_hours
            chosen.append(s.calc_id)
            s.decision, s.rank = "selected", len(chosen)
            s.reason = "highest expected confidence gain per node-hour within budget"
    plan = AcquisitionPlan(
        score_digest=base.score_digest,
        confidence_before=before,
        items=scored,
        selected=chosen,
        budget=_budget(ledger, spent),
        limitations=[
            "Gains are estimated one proposal at a time; interactions between proposals are not.",
            "Expected uncertainties and costs are the proposals' declared values.",
            "A plan is advice: it dispatches nothing and every item needs approval.",
            "Unquantified items need a first value (e.g. a surrogate) before they can be ranked.",
        ],
    )
    payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    plan.plan_digest = "acq-" + hashlib.sha256(payload.encode()).hexdigest()[:24]
    return plan


def reserve_approved(
    plan: AcquisitionPlan, ledger: BudgetLedger, approved: Sequence[str], *, authorized_by: str
) -> list[str]:
    """Reserve (not submit) the named plan items; only a person's explicit ids are honoured."""
    if not authorized_by.strip():
        raise ValueError("reserving needs the name of the person who approved it")
    by_id = {s.calc_id: s for s in plan.items}
    for calc_id in approved:
        item = by_id.get(calc_id)
        if item is None or item.decision != "selected":
            raise ValueError(f"{calc_id} is not a selected item of this plan")
    for calc_id in approved:
        ledger.reserve(calc_id, BUDGET_CATEGORY, by_id[calc_id].cost_node_hours)
    return list(approved)


def render_plan(plan: AcquisitionPlan) -> str:
    b = plan.budget
    out = [
        f"Property acquisition plan ({plan.version}, {plan.plan_digest}); "
        f"scores {plan.score_digest}",
        f"Selection confidence now: {plan.confidence_before:.3f}. Requires approval; nothing "
        "dispatched.",
        "Budget (node-hours): "
        f"total {b.total:g}, property allocation {b.category_allocation:g}, "
        f"available {b.category_available:g}, committed {b.committed:g}, reserved {b.reserved:g}, "
        f"consumed {b.consumed:g}, expected by plan {b.expected:g}, validation reserve "
        f"{b.validation_reserve:g} ({b.validation_reserve_remaining:g} left), remaining "
        f"{b.remaining:g}, spendable {b.spendable:g}, remaining after plan "
        f"{b.remaining_after_plan:g}",
        "calc | candidate | property | backend | decision | cost | gain | gain/node-h | reason",
    ]
    for s in plan.items:
        gain = "n/a" if s.expected_gain is None else f"{s.expected_gain:+.4f}"
        per = "n/a" if s.gain_per_node_hour is None else f"{s.gain_per_node_hour:+.5f}"
        out.append(
            f"{s.calc_id} | {s.candidate_id} | {s.property} | {s.backend} | {s.decision} | "
            f"{s.cost_node_hours:g} | {gain} | {per} | {s.reason}"
        )
    return "\n".join(out)


__all__ = [
    "ACQUISITION_VERSION",
    "AcquisitionPlan",
    "BudgetReport",
    "CalculationScore",
    "ProposedCalculation",
    "candidate_confidence",
    "plan_property_acquisition",
    "render_plan",
    "reserve_approved",
    "selection_confidence",
]
