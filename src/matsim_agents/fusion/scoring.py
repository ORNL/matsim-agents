"""Deterministic plasma-facing-component scoring (F5, F6).

Stage A applies user constraints (feasible / violated / unknown). Stage B
forms a Pareto front over objective desirabilities; it is authoritative only
when every candidate has all eight objectives. Stage C computes
``S = sum_i w_i d_i`` with desirabilities from fixed, user-supplied
benchmarks, never from the candidate set's own min/max. A missing objective
leaves the total undefined: nothing is zero-filled and weights are never
renormalized. Policy weights are never altered; correlations only warn.

Uncertainty is carried as a one-sigma interval on every desirability and on
the score, and pairwise dominance is ``established`` (intervals separate),
``conditional`` (point values dominate but intervals overlap), or
``undetermined``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.fusion.objectives import OBJECTIVES, mechanical_screening, objective
from matsim_agents.fusion.policy import FUSION_PFC_OBJECTIVES, FusionScoringPolicy
from matsim_agents.fusion.scenario import EvaluationScenario, EvidenceContext, FusionEvidence
from matsim_agents.science.acquisition.selection import (
    _QUALIFICATION_ORDER,
    _STATUS_PRIORITY,
    Constraint,
    Correlation,
    _nondominated_mask,
    _pick,
)
from matsim_agents.science.properties.calculators import (
    PropertyQualification,
    PropertyStatus,
)

SCORING_SCHEMA = "fusion-pfc-score-v1"


class ScoringError(ValueError):
    """Inputs cannot be scored as given."""


# --- benchmarks ------------------------------------------------------------


class ObservableBenchmark(BaseModel):
    """Fixed desirability anchors: d=0 at ``worst``, d=1 at ``target``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    worst: float
    target: float
    source: str = Field(min_length=1, description="Where the anchors come from.")

    @model_validator(mode="after")
    def _distinct(self) -> ObservableBenchmark:
        if not (math.isfinite(self.worst) and math.isfinite(self.target)):
            raise ValueError("benchmark anchors must be finite")
        if self.worst == self.target:
            raise ValueError("benchmark worst and target must differ")
        return self

    def desirability(self, value: float) -> float:
        return min(1.0, max(0.0, (value - self.worst) / (self.target - self.worst)))


class ObjectiveBenchmark(BaseModel):
    """Scored observables of one objective; all must be present to score it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observables: dict[str, ObservableBenchmark] = Field(min_length=1)


class BenchmarkSet(BaseModel):
    """Versioned benchmarks; objective d = min over its scored observables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_version: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    synthetic: bool = False
    objectives: dict[str, ObjectiveBenchmark]

    @model_validator(mode="after")
    def _consistent(self) -> BenchmarkSet:
        for name, bench in self.objectives.items():
            spec = objective(name)
            scored = set(bench.observables)
            missing = [o.name for o in spec.required if o.name not in scored]
            if missing:
                raise ValueError(f"{name}: benchmark omits required {', '.join(missing)}")
            if scored <= set(spec.insufficient_alone):
                raise ValueError(f"{name}: {', '.join(sorted(scored))} alone is insufficient")
            for obs_name, b in bench.observables.items():
                obs = spec.observable(obs_name)
                if obs is None:
                    raise ValueError(f"{name}: {obs_name} is not an observable of the objective")
                up = b.target > b.worst
                if up != (obs.direction == "maximize"):
                    raise ValueError(
                        f"{name}/{obs_name}: anchors contradict direction {obs.direction}"
                    )
        return self

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_benchmarks(path: str | Path) -> BenchmarkSet:
    return BenchmarkSet.model_validate(yaml.safe_load(Path(path).read_text()))


# --- results ---------------------------------------------------------------


class ConstraintStatus(StrEnum):
    FEASIBLE = "feasible"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class RankingStatus(StrEnum):
    """Scoring state of a candidate; separate from whether its inputs parsed."""

    COMPLETE_QUALIFIED = "complete_qualified"
    COMPLETE_PROVISIONAL = "complete_provisional"
    INCOMPLETE = "incomplete"
    INFEASIBLE = "infeasible"


class Dominance(StrEnum):
    ESTABLISHED = "established"
    CONDITIONAL = "conditional"
    UNDETERMINED = "undetermined"


def evidence_key(property_id: str, context: EvidenceContext | None) -> str:
    """Identity of a scored record: its property id plus its fusion context."""
    ctx = context.model_dump_json(exclude_none=True) if context is not None else "{}"
    return property_id if ctx == "{}" else f"{property_id}|{ctx}"


class ConstraintResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    property: str
    minimum: float | None = None
    maximum: float | None = None
    temperature_K: float | None = None
    status: ConstraintStatus
    value: float | None = None
    uncertainty: float | None = None
    evidence_status: PropertyStatus | None = None
    qualification: PropertyQualification | None = None
    property_id: str | None = None
    context: EvidenceContext | None = None
    note: str = ""

    @property
    def evidence_key(self) -> str | None:
        return evidence_key(self.property_id, self.context) if self.property_id else None


class ObservableResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    observable: str
    value: float | None = None
    uncertainty: float | None = None
    desirability: float | None = None
    d_low: float | None = None
    d_high: float | None = None
    status: PropertyStatus | None = None
    qualification: PropertyQualification | None = None
    property_id: str | None = None
    context: EvidenceContext | None = None
    excluded: tuple[str, ...] = ()
    reason: str = ""

    @property
    def evidence_key(self) -> str | None:
        return evidence_key(self.property_id, self.context) if self.property_id else None


class ObjectiveResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    objective: str
    weight: float
    desirability: float | None = None
    d_low: float | None = None
    d_high: float | None = None
    observables: tuple[ObservableResult, ...] = ()
    evidence_status: PropertyStatus | None = Field(
        None, description="Weakest status among the observables used."
    )
    qualification: PropertyQualification | None = None
    reason: str = ""


class CandidateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    constraints: tuple[ConstraintResult, ...]
    feasibility: ConstraintStatus
    objectives: tuple[ObjectiveResult, ...]
    missing_objectives: tuple[str, ...]
    score: float | None = None
    score_low: float | None = None
    score_high: float | None = None
    partial_weighted_sum: float = Field(
        description="sum w_i d_i over present objectives only; not a score."
    )
    covered_weight: float
    ranking_status: RankingStatus
    rank: int | None = None
    pareto: bool | None = None
    mechanical_screening: dict[str, float] = Field(default_factory=dict)

    def objective(self, name: str) -> ObjectiveResult:
        return next(o for o in self.objectives if o.objective == name)


class DominancePair(BaseModel):
    model_config = ConfigDict(frozen=True)

    winner: str
    loser: str
    dominance: Dominance


class WeightSensitivity(BaseModel):
    model_config = ConfigDict(frozen=True)

    objective: str
    factor: float
    top_candidate: str | None
    ranking: tuple[str, ...]
    rank_changes: int


class ScoringResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal["fusion-pfc-score-v1"] = SCORING_SCHEMA
    policy_version: str
    policy_digest: str
    benchmark_version: str
    benchmark_digest: str
    synthetic_benchmarks: bool
    scenario_id: str
    scenario: EvaluationScenario
    candidates: tuple[CandidateResult, ...]
    ranking: tuple[str, ...]
    pareto_front: tuple[str, ...]
    pareto_authoritative: bool
    pareto_objectives: tuple[str, ...]
    dominance: tuple[DominancePair, ...]
    correlation_warnings: tuple[str, ...]
    sensitivity: tuple[WeightSensitivity, ...]
    notes: tuple[str, ...] = ()

    def by_id(self, candidate_id: str) -> CandidateResult:
        return next(c for c in self.candidates if c.candidate_id == candidate_id)

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return f"fsc-{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


# --- scoring ---------------------------------------------------------------


def _weakest_status(statuses: list[PropertyStatus]) -> PropertyStatus | None:
    return min(statuses, key=lambda s: _STATUS_PRIORITY[s]) if statuses else None


def _weakest_qualification(
    quals: list[PropertyQualification],
) -> PropertyQualification | None:
    return min(quals, key=lambda q: _QUALIFICATION_ORDER[q]) if quals else None


def _score_observable(
    objective_name: str,
    obs_name: str,
    bench: ObservableBenchmark,
    items: Sequence[FusionEvidence],
    scenario: EvaluationScenario,
) -> ObservableResult:
    proxies = {p.name for p in objective(objective_name).unsupported_proxies}
    usable, excluded = [], []
    for item in items:
        ev = item.evidence
        if ev.property != obs_name:
            continue
        bad = scenario.mismatches(objective_name, item)
        bad += [f"{ev.property}: derived from proxy {s}" for s in ev.derived_from if s in proxies]
        if bad:
            excluded.extend(bad)
        else:
            usable.append(item)
    chosen_item, reason = _pick_item(usable, obs_name)
    if chosen_item is None:
        if excluded and reason == "no evidence record":
            reason = "no comparable evidence"
        return ObservableResult(observable=obs_name, excluded=tuple(excluded), reason=reason)
    chosen = chosen_item.evidence
    if not isinstance(chosen.value, int | float):
        return ObservableResult(observable=obs_name, reason="tensor value cannot be scored")
    v = float(chosen.value)
    sigma = float(chosen.uncertainty or 0.0)
    d_a, d_b = bench.desirability(v - sigma), bench.desirability(v + sigma)
    return ObservableResult(
        observable=obs_name,
        value=v,
        uncertainty=sigma,
        desirability=bench.desirability(v),
        d_low=min(d_a, d_b),
        d_high=max(d_a, d_b),
        status=chosen.status,
        qualification=chosen.qualification,
        property_id=chosen.property_id(),
        context=chosen_item.context,
        excluded=tuple(excluded),
    )


def _pick_item(items: Sequence[FusionEvidence], prop: str) -> tuple[FusionEvidence | None, str]:
    """``_pick`` over fusion records, keeping the context of the chosen one."""
    chosen, reason, _ = _pick([i.evidence for i in items], prop, None)
    if chosen is None:
        return None, reason
    return next(i for i in items if i.evidence is chosen), ""


def _score_objective(
    name: str,
    weight: float,
    benchmarks: BenchmarkSet,
    items: Sequence[FusionEvidence],
    scenario: EvaluationScenario,
) -> ObjectiveResult:
    bench = benchmarks.objectives.get(name)
    if bench is None:
        return ObjectiveResult(objective=name, weight=weight, reason="no benchmark supplied")
    obs = tuple(
        _score_observable(name, o, b, items, scenario) for o, b in sorted(bench.observables.items())
    )
    missing = [o.observable for o in obs if o.desirability is None]
    if missing:
        return ObjectiveResult(
            objective=name,
            weight=weight,
            observables=obs,
            reason=f"missing: {', '.join(missing)}",
        )
    return ObjectiveResult(
        objective=name,
        weight=weight,
        desirability=min(o.desirability for o in obs),  # type: ignore[type-var]
        d_low=min(o.d_low for o in obs),  # type: ignore[type-var]
        d_high=min(o.d_high for o in obs),  # type: ignore[type-var]
        observables=obs,
        evidence_status=_weakest_status([o.status for o in obs if o.status]),
        qualification=_weakest_qualification([o.qualification for o in obs if o.qualification]),
    )


def _check_constraint(
    c: Constraint, items: Sequence[FusionEvidence], scenario: EvaluationScenario
) -> ConstraintResult:
    # Same conditions the property is scored under; temperature only where it applies.
    fields = tuple(
        dict.fromkeys(
            f
            for spec in OBJECTIVES.values()
            if any(o.name == c.property for o in spec.observables())
            for f in spec.compatible_conditions
        )
    ) or ("temperature_K", "irradiation")
    if c.temperature_K is not None:
        scenario = scenario.model_copy(update={"temperature_K": c.temperature_K})
        fields = tuple(dict.fromkeys((*fields, "temperature_K")))
    rows = [
        i
        for i in items
        if i.evidence.property == c.property and not scenario.condition_mismatches(fields, i)
    ]
    spec = {"property": c.property, "minimum": c.minimum, "maximum": c.maximum}
    spec["temperature_K"] = c.temperature_K
    chosen_item, reason = _pick_item(rows, c.property)
    if chosen_item is None or not isinstance(chosen_item.evidence.value, int | float):
        return ConstraintResult(
            **spec, status=ConstraintStatus.UNKNOWN, note=reason or "tensor value"
        )
    chosen = chosen_item.evidence
    v, s = float(chosen.value), float(chosen.uncertainty or 0.0)
    ok = bool(c.satisfied(v))
    margin = ok != bool(c.satisfied(v - s)) or ok != bool(c.satisfied(v + s))
    return ConstraintResult(
        **spec,
        status=ConstraintStatus.FEASIBLE if ok else ConstraintStatus.VIOLATED,
        value=v,
        uncertainty=s,
        evidence_status=chosen.status,
        qualification=chosen.qualification,
        property_id=chosen.property_id(),
        context=chosen_item.context,
        note="within one sigma of the limit" if margin else "",
    )


def _reliable(status: PropertyStatus | None, qual: PropertyQualification | None) -> bool:
    return (
        status in {PropertyStatus.MEASURED, PropertyStatus.CALCULATED}
        and qual is not None
        and qual != PropertyQualification.UNVALIDATED
    )


def _ranking_status(
    feasibility: ConstraintStatus,
    missing: list[str],
    objectives: Sequence[ObjectiveResult],
    constraints: Sequence[ConstraintResult] = (),
) -> RankingStatus:
    if feasibility == ConstraintStatus.VIOLATED:
        return RankingStatus.INFEASIBLE
    if missing:
        return RankingStatus.INCOMPLETE
    reliable = all(_reliable(o.evidence_status, o.qualification) for o in objectives) and all(
        _reliable(c.evidence_status, c.qualification) for c in constraints
    )
    if reliable and feasibility == ConstraintStatus.FEASIBLE:
        return RankingStatus.COMPLETE_QUALIFIED
    return RankingStatus.COMPLETE_PROVISIONAL


def _rank(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda k: (-scores[k], k))


def _dominance(a: CandidateResult, b: CandidateResult, names: Sequence[str]) -> Dominance | None:
    pa = np.array([a.objective(n).desirability for n in names], float)
    pb = np.array([b.objective(n).desirability for n in names], float)
    if not (np.all(pa >= pb) and np.any(pa > pb)):
        return None
    lo_a = np.array([a.objective(n).d_low for n in names], float)
    hi_b = np.array([b.objective(n).d_high for n in names], float)
    if np.all(lo_a >= hi_b) and np.any(lo_a > hi_b):
        return Dominance.ESTABLISHED
    return Dominance.CONDITIONAL


def _correlation_warnings(
    benchmarks: BenchmarkSet, correlations: Sequence[Correlation], threshold: float
) -> list[str]:
    owners: dict[str, list[str]] = {}
    for name, bench in sorted(benchmarks.objectives.items()):
        for obs in sorted(bench.observables):
            owners.setdefault(obs, []).append(name)
    notes = [
        f"{obs} is scored in {', '.join(objs)}; these objectives are not independent"
        for obs, objs in owners.items()
        if len(objs) > 1
    ]
    for c in correlations:
        if abs(c.rho) >= threshold and c.a in owners and c.b in owners:
            notes.append(
                f"{c.a} and {c.b} are correlated (rho={c.rho:+.2f}); "
                f"objectives {sorted(set(owners[c.a] + owners[c.b]))} partly double-count"
            )
    return notes


def _sensitivity(
    policy: FusionScoringPolicy,
    complete: Sequence[CandidateResult],
    base_rank: Sequence[str],
    factors: Sequence[float],
) -> list[WeightSensitivity]:
    out = []
    if len(complete) < 2:
        return out
    for name in FUSION_PFC_OBJECTIVES:
        for f in factors:
            w = dict(policy.weights)
            w[name] *= f
            total = math.fsum(w.values())
            w = {k: v / total for k, v in w.items()}
            scores = {
                c.candidate_id: math.fsum(
                    w[o.objective] * o.desirability  # type: ignore[operator]
                    for o in c.objectives
                )
                for c in complete
            }
            ranking = _rank(scores)
            out.append(
                WeightSensitivity(
                    objective=name,
                    factor=f,
                    top_candidate=ranking[0],
                    ranking=tuple(ranking),
                    rank_changes=sum(a != b for a, b in zip(ranking, base_rank, strict=True)),
                )
            )
    return out


def score_candidates(
    policy: FusionScoringPolicy,
    benchmarks: BenchmarkSet,
    scenario: EvaluationScenario,
    candidates: Mapping[str, Sequence[FusionEvidence]],
    *,
    constraints: Sequence[Constraint] = (),
    correlations: Sequence[Correlation] = (),
    correlation_threshold: float = 0.7,
    sensitivity_factors: Sequence[float] = (0.5, 1.5),
) -> ScoringResult:
    """Score candidates in one scenario; evidence is read, never changed."""
    if not candidates:
        raise ScoringError("no candidates")
    if set(OBJECTIVES) != set(policy.weights):
        raise ScoringError("policy objectives do not match the objective registry")
    results: list[CandidateResult] = []
    for cid in sorted(candidates):
        items = candidates[cid]
        cons = tuple(_check_constraint(c, items, scenario) for c in constraints)
        statuses = {c.status for c in cons}
        feasibility = (
            ConstraintStatus.VIOLATED
            if ConstraintStatus.VIOLATED in statuses
            else ConstraintStatus.UNKNOWN
            if ConstraintStatus.UNKNOWN in statuses
            else ConstraintStatus.FEASIBLE
        )
        objs = tuple(
            _score_objective(n, policy.weights[n], benchmarks, items, scenario)
            for n in FUSION_PFC_OBJECTIVES
        )
        present = [o for o in objs if o.desirability is not None]
        missing = [o.objective for o in objs if o.desirability is None]
        partial = math.fsum(o.weight * o.desirability for o in present)  # type: ignore[operator]
        covered = math.fsum(o.weight for o in present)
        complete = not missing
        results.append(
            CandidateResult(
                candidate_id=cid,
                constraints=cons,
                feasibility=feasibility,
                objectives=objs,
                missing_objectives=tuple(missing),
                score=partial if complete else None,
                score_low=math.fsum(o.weight * o.d_low for o in objs) if complete else None,  # type: ignore[operator]
                score_high=math.fsum(o.weight * o.d_high for o in objs) if complete else None,  # type: ignore[operator]
                partial_weighted_sum=partial,
                covered_weight=covered,
                ranking_status=_ranking_status(feasibility, missing, objs, cons),
                mechanical_screening=mechanical_screening([i.evidence for i in items]),
            )
        )

    alive = [c for c in results if c.feasibility != ConstraintStatus.VIOLATED]
    common = [
        n
        for n in FUSION_PFC_OBJECTIVES
        if alive and all(c.objective(n).desirability is not None for c in alive)
    ]
    authoritative = bool(alive) and len(common) == len(FUSION_PFC_OBJECTIVES)
    front: set[str] = set()
    if alive and common:
        mat = np.array([[c.objective(n).desirability for n in common] for c in alive], float)
        mask = _nondominated_mask(mat[None], np.ones((1, len(alive)), bool))[0]
        front = {c.candidate_id for c, m in zip(alive, mask, strict=True) if m}

    complete_ok = [c for c in alive if c.score is not None]
    base_rank = _rank({c.candidate_id: c.score for c in complete_ok})  # type: ignore[misc]
    rank_of = {cid: i + 1 for i, cid in enumerate(base_rank)}
    results = [
        c.model_copy(
            update={
                "rank": rank_of.get(c.candidate_id),
                "pareto": (c.candidate_id in front) if c in alive and common else None,
            }
        )
        for c in results
    ]
    by_id = {c.candidate_id: c for c in results}

    pairs = []
    alive_ids = [c.candidate_id for c in alive]
    for ia in alive_ids:
        for ib in alive_ids:
            if ia == ib:
                continue
            a, b = by_id[ia], by_id[ib]
            shared = [
                n
                for n in FUSION_PFC_OBJECTIVES
                if a.objective(n).desirability is not None
                and b.objective(n).desirability is not None
            ]
            if not shared:
                continue
            d = _dominance(a, b, shared)
            if d is None:
                continue
            if len(shared) < len(FUSION_PFC_OBJECTIVES):
                # Missing objectives could reverse the comparison.
                d = Dominance.UNDETERMINED
            pairs.append(DominancePair(winner=ia, loser=ib, dominance=d))

    notes = []
    if not authoritative:
        notes.append(
            f"Pareto front is partial: computed on {len(common)}/8 objectives "
            f"({', '.join(common) or 'none'})"
        )
    if benchmarks.synthetic:
        notes.append("benchmarks are SYNTHETIC; scores are demonstrations only")
    incomplete = [c.candidate_id for c in results if c.missing_objectives]
    if incomplete:
        notes.append(f"no total score for incomplete candidate(s): {', '.join(incomplete)}")

    return ScoringResult(
        policy_version=policy.policy_version,
        policy_digest=policy.digest(),
        benchmark_version=benchmarks.benchmark_version,
        benchmark_digest=benchmarks.digest(),
        synthetic_benchmarks=benchmarks.synthetic,
        scenario_id=scenario.scenario_id,
        scenario=scenario,
        candidates=tuple(results),
        ranking=tuple(base_rank),
        pareto_front=tuple(sorted(front)),
        pareto_authoritative=authoritative,
        pareto_objectives=tuple(common),
        dominance=tuple(pairs),
        correlation_warnings=tuple(
            _correlation_warnings(benchmarks, correlations, correlation_threshold)
        ),
        sensitivity=tuple(
            _sensitivity(
                policy, [by_id[c.candidate_id] for c in complete_ok], base_rank, sensitivity_factors
            )
        ),
        notes=tuple(notes),
    )


def render_report(result: ScoringResult) -> str:
    """Human-readable summary of a scoring result."""
    lines = [
        f"Fusion PFC scoring ({result.schema_version})",
        f"policy {result.policy_version} [{result.policy_digest[:12]}]  "
        f"benchmarks {result.benchmark_version} [{result.benchmark_digest[:12]}]  "
        f"scenario {result.scenario_id}",
    ]
    if result.synthetic_benchmarks:
        lines.append("*** SYNTHETIC BENCHMARKS - NOT A MATERIALS CLAIM ***")
    lines.append("")
    lines.append(f"{'candidate':<20} {'status':<22} {'feas':<9} {'score':>16} {'rank':>4}")
    for c in result.candidates:
        score = (
            f"{c.score:.3f} [{c.score_low:.3f},{c.score_high:.3f}]"
            if c.score is not None
            else f"n/a (cov {c.covered_weight:.2f})"
        )
        lines.append(
            f"{c.candidate_id:<20} {c.ranking_status.value:<22} {c.feasibility.value:<9} "
            f"{score:>16} {c.rank or '-':>4}"
        )
        for o in c.objectives:
            if o.desirability is None:
                lines.append(f"    - {o.objective}: MISSING ({o.reason})")
        for k, v in c.mechanical_screening.items():
            lines.append(f"    {k} = {v:g} (screening only, not scored)")
    label = "authoritative" if result.pareto_authoritative else "PARTIAL"
    lines += ["", f"Pareto front ({label}): {', '.join(result.pareto_front) or 'none'}"]
    for p in result.dominance:
        lines.append(f"  {p.winner} dominates {p.loser}: {p.dominance.value}")
    for w in result.correlation_warnings:
        lines.append(f"warning: {w}")
    flips = [
        s for s in result.sensitivity if result.ranking and s.top_candidate != result.ranking[0]
    ]
    if result.sensitivity:
        n = len(result.sensitivity)
        lines.append(f"weight sensitivity: {len(flips)}/{n} perturbations change the top")
        for s in flips:
            lines.append(f"  {s.objective} x{s.factor:g}: top -> {s.top_candidate}")
    lines += [f"note: {n}" for n in result.notes]
    return "\n".join(lines)


__all__ = [
    "SCORING_SCHEMA",
    "BenchmarkSet",
    "CandidateResult",
    "ConstraintResult",
    "ConstraintStatus",
    "Dominance",
    "DominancePair",
    "ObjectiveBenchmark",
    "ObjectiveResult",
    "ObservableBenchmark",
    "ObservableResult",
    "RankingStatus",
    "ScoringError",
    "ScoringResult",
    "WeightSensitivity",
    "load_benchmarks",
    "render_report",
    "score_candidates",
]
