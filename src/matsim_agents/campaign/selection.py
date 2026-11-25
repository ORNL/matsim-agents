"""Deterministic multi-objective material selection from property evidence.

Requirements (objectives and hard constraints) come only from the caller's
:class:`SelectionSpec`; no threshold, weight, or bound is invented here. Candidates are
compared with Pareto dominance by default; a weighted desirability score is an optional,
explicitly parameterised view. Missing, unsupported, and unconverged values are never
scored, and a surrogate or approximated value is ranked only as ``INSUFFICIENT_EVIDENCE``.

Statuses (first match wins):

``INFEASIBLE``
    a scoreable value violates a hard constraint;
``INCOMPLETE``
    a required objective or constraint property has no scoreable value;
``INSUFFICIENT_EVIDENCE``
    every value exists, but one is surrogate-predicted, approximated, below the required
    qualification, or a constraint is met with less confidence than the spec demands;
``QUALIFIED``
    every value is measured or calculated at the required qualification, every constraint
    holds, and the constraint confidence (if demanded) is met.

Thermal and electrical conductivity are physically coupled (Wiedemann-Franz), so a
value derived from another property is sampled with correlation +/-1 to its source, and
correlated objectives share one weight in the desirability score instead of counting twice.
Monte-Carlo propagation (seeded through :mod:`matsim_agents.seeding`) turns property
uncertainty into score spread, constraint probability, rank distribution, and Pareto
robustness. This module never calls an LLM and never alters evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from matsim_agents.execution.contracts import PROPERTY_UNITS, EvidenceLevel
from matsim_agents.execution.properties import (
    PROPERTY_SHAPE,
    PropertyEvidence,
    PropertyQualification,
    PropertyStatus,
)
from matsim_agents.seeding import numpy_rng

SCORING_VERSION = "selection-v1"
_QUALIFICATION_ORDER = {
    PropertyQualification.UNVALIDATED: 0,
    PropertyQualification.NUMERICALLY_VALIDATED: 1,
    PropertyQualification.SCIENTIFICALLY_QUALIFIED: 2,
}
# Higher wins when one property has several scoreable records.
_STATUS_PRIORITY = {
    PropertyStatus.MEASURED: 4,
    PropertyStatus.CALCULATED: 3,
    PropertyStatus.APPROXIMATED: 2,
    PropertyStatus.SURROGATE_PREDICTED: 1,
}
_RELIABLE = frozenset({PropertyStatus.MEASURED, PropertyStatus.CALCULATED})


class SelectionStatus(StrEnum):
    QUALIFIED = "QUALIFIED"
    INCOMPLETE = "INCOMPLETE"
    INFEASIBLE = "INFEASIBLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def _scalar_property(name: str) -> str:
    if name not in PROPERTY_UNITS:
        raise ValueError(f"unknown property {name!r}")
    if PROPERTY_SHAPE.get(name) is not None:
        raise ValueError(f"{name} is a tensor; select on a scalar property")
    return name


class Objective(BaseModel):
    property: str
    direction: Literal["maximize", "minimize"]
    temperature_K: float | None = Field(None, gt=0, description="Evidence must match, if set.")
    weight: float = Field(1.0, gt=0)
    worst: float | None = Field(None, description="Desirability 0 (user-declared).")
    target: float | None = Field(None, description="Desirability 1 (user-declared).")

    @model_validator(mode="after")
    def _valid(self) -> Objective:
        _scalar_property(self.property)
        if (self.worst is None) != (self.target is None):
            raise ValueError("give both worst and target, or neither")
        if self.worst is not None and self.worst == self.target:
            raise ValueError("worst and target must differ")
        if self.worst is not None:
            better_is_higher = self.target > self.worst  # type: ignore[operator]
            if better_is_higher != (self.direction == "maximize"):
                raise ValueError(f"{self.property}: worst/target contradict the direction")
        return self


class Constraint(BaseModel):
    property: str
    minimum: float | None = None
    maximum: float | None = None
    temperature_K: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _valid(self) -> Constraint:
        _scalar_property(self.property)
        if self.minimum is None and self.maximum is None:
            raise ValueError("a constraint needs a minimum or a maximum supplied by the user")
        if None not in (self.minimum, self.maximum) and self.minimum > self.maximum:  # type: ignore[operator]
            raise ValueError("constraint minimum exceeds maximum")
        return self

    def satisfied(self, value):
        ok = np.ones_like(np.asarray(value, dtype=float), dtype=bool)
        if self.minimum is not None:
            ok &= np.asarray(value) >= self.minimum
        if self.maximum is not None:
            ok &= np.asarray(value) <= self.maximum
        return ok


class Correlation(BaseModel):
    a: str
    b: str
    rho: float = Field(ge=-1.0, le=1.0)


class SelectionSpec(BaseModel):
    objectives: list[Objective] = Field(min_length=1)
    constraints: list[Constraint] = Field(default_factory=list)
    correlations: list[Correlation] = Field(default_factory=list)
    weighted_desirability: bool = False
    min_qualification: PropertyQualification = PropertyQualification.NUMERICALLY_VALIDATED
    min_constraint_confidence: float | None = Field(None, gt=0, le=1)
    correlated_weight_threshold: float = Field(0.9, gt=0, le=1)
    n_samples: int = Field(2000, ge=100)
    seed: int

    @model_validator(mode="after")
    def _valid(self) -> SelectionSpec:
        names = [o.property for o in self.objectives]
        if len(set(names)) != len(names):
            raise ValueError("an objective property may appear once")
        if self.weighted_desirability and any(o.worst is None for o in self.objectives):
            raise ValueError("weighted desirability needs worst and target for every objective")
        return self


class CandidateEvidence(BaseModel):
    candidate_id: str = Field(min_length=1)
    evidence: list[PropertyEvidence]


class UsedValue(BaseModel):
    property: str
    value: float
    uncertainty: float
    status: PropertyStatus
    evidence_level: EvidenceLevel
    qualification: PropertyQualification
    property_id: str
    evidence_ids: list[str]
    reliable: bool
    alternatives: list[str] = Field(default_factory=list)


class Distribution(BaseModel):
    mean: float
    std: float
    p05: float
    p95: float


class ObjectiveScore(BaseModel):
    """One objective of one candidate, and its part in the combined score."""

    property: str
    direction: Literal["maximize", "minimize"]
    value: float
    uncertainty: float
    unit: str
    evidence_status: PropertyStatus
    rank: int | None = Field(None, description="1 is best among ranked candidates.")
    best_value: float | None = None
    declared_weight: float
    effective_weight: float = Field(description="After the correlated-property discount.")
    weight_share: float = Field(description="Effective weight / sum of effective weights.")
    desirability: float | None = Field(None, description="d_k in [0, 1]; needs worst/target.")
    log_contribution: float | None = Field(None, description="weight_share * ln(d_k).")
    share_of_penalty: float | None = Field(
        None, description="Fraction of ln(1/D) caused by this objective."
    )


class CandidateScore(BaseModel):
    candidate_id: str
    status: SelectionStatus
    reasons: list[str]
    values: dict[str, UsedValue]
    missing: dict[str, str]
    violated: list[str]
    constraint_probability: float | None = None
    constraint_probability_by_property: dict[str, float] = Field(default_factory=dict)
    pareto: Literal["nondominated", "dominated", "not_ranked"]
    dominated_by: list[str] = Field(default_factory=list)
    best_in: list[str] = Field(default_factory=list)
    worst_in: list[str] = Field(default_factory=list)
    pareto_robustness: float | None = None
    desirability: float | None = None
    objectives: list[ObjectiveScore] = Field(default_factory=list)
    desirability_distribution: Distribution | None = None
    desirability_rank: int | None = None
    rank_distribution: dict[int, float] = Field(default_factory=dict)
    probability_rank_1: float | None = None


class SelectionResult(BaseModel):
    scoring_version: str = SCORING_VERSION
    spec: SelectionSpec
    candidates: list[CandidateScore]
    pareto_front: list[str]
    weight_notes: list[str] = Field(default_factory=list)
    correlation_notes: list[str] = Field(default_factory=list)
    combination: list[str] = Field(default_factory=list)
    limitations: list[str]
    score_digest: str = ""

    def by_id(self, candidate_id: str) -> CandidateScore:
        return next(c for c in self.candidates if c.candidate_id == candidate_id)


def _matches(ev: PropertyEvidence, prop: str, temperature_K: float | None) -> bool:
    if ev.property != prop:
        return False
    return temperature_K is None or ev.conditions.temperature_K == temperature_K


def _pick(
    evidence: Sequence[PropertyEvidence], prop: str, temperature_K: float | None
) -> tuple[PropertyEvidence | None, str, list[str]]:
    """Best scoreable record for one property, the reason if none, and the losers."""
    rows = [e for e in evidence if _matches(e, prop, temperature_K)]
    if not rows:
        return None, "no evidence record", []
    good = [e for e in rows if e.is_scoreable()]
    if not good:
        e = rows[0]
        return None, f"{e.status.value}: {e.reason or 'no value'}", []
    good.sort(
        key=lambda e: (
            -_STATUS_PRIORITY[e.status],
            -_QUALIFICATION_ORDER[e.qualification],
            float(e.uncertainty or 0.0),
            e.property_id(),
        )
    )
    return good[0], "", [e.property_id() for e in good[1:]]


def _used(ev: PropertyEvidence, alternatives: list[str]) -> UsedValue:
    return UsedValue(
        property=ev.property,
        value=float(ev.value),  # type: ignore[arg-type]
        uncertainty=float(ev.uncertainty or 0.0),  # type: ignore[arg-type]
        status=ev.status,
        evidence_level=ev.evidence_level,
        qualification=ev.qualification,
        property_id=ev.property_id(),
        evidence_ids=list(ev.evidence_ids),
        reliable=ev.status in _RELIABLE,
        alternatives=alternatives,
    )


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _correlation_matrix(
    props: list[str], evidence: dict[str, PropertyEvidence], spec: SelectionSpec
) -> np.ndarray:
    n = len(props)
    corr = np.eye(n)
    explicit = {_pair_key(c.a, c.b): c.rho for c in spec.correlations}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = props[i], props[j]
            rho = explicit.get(_pair_key(a, b))
            if rho is None:
                derived = b in evidence[a].derived_from or a in evidence[b].derived_from
                if derived:
                    inverse = (a == "electrical_resistivity") != (b == "electrical_resistivity")
                    rho = -1.0 if inverse else 1.0
            if rho is not None:
                corr[i, j] = corr[j, i] = rho
    return corr


def _factor(corr: np.ndarray) -> np.ndarray:
    """A matrix M with M M^T the nearest valid correlation matrix (handles rho = +/-1)."""
    w, v = np.linalg.eigh(corr)
    m = v * np.sqrt(np.clip(w, 0.0, None))
    norm = np.sqrt((m**2).sum(axis=1))
    return m / np.where(norm > 0, norm, 1.0)[:, None]


def _desirability(values: np.ndarray, objectives: Sequence[Objective], weights: np.ndarray):
    """Weighted geometric mean of linear desirabilities; values[..., k] is objective k."""
    worst = np.array([o.worst for o in objectives], float)
    target = np.array([o.target for o in objectives], float)
    d = np.clip((values - worst) / (target - worst), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        logd = np.where(d > 0, np.log(np.where(d > 0, d, 1.0)), -np.inf)
    total = (weights * logd).sum(axis=-1) / weights.sum()
    return np.exp(total)


def _effective_weights(spec: SelectionSpec, rho_by_pair: dict[tuple[str, str], float]):
    """Objectives whose property pairs are strongly correlated share one weight."""
    objs = spec.objectives
    parent = list(range(len(objs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            rho = rho_by_pair.get(_pair_key(objs[i].property, objs[j].property))
            if rho is not None and abs(rho) >= spec.correlated_weight_threshold:
                parent[find(i)] = find(j)
    sizes: dict[int, int] = {}
    for i in range(len(objs)):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    weights = np.array([o.weight / sizes[find(i)] for i, o in enumerate(objs)])
    notes = []
    groups: dict[int, list[str]] = {}
    for i, o in enumerate(objs):
        groups.setdefault(find(i), []).append(o.property)
    for members in groups.values():
        if len(members) > 1:
            notes.append(
                f"{', '.join(members)} are strongly correlated; their weights are divided by "
                f"{len(members)} so the shared physics is not counted twice."
            )
    return weights, notes


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def _signs(spec: SelectionSpec) -> np.ndarray:
    return np.array([1.0 if o.direction == "maximize" else -1.0 for o in spec.objectives])


def _nondominated_mask(samples: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """samples (s, c, o) in maximise form; alive (s, c). Returns (s, c) nondominated & alive."""
    a = samples[:, :, None, :]
    b = samples[:, None, :, :]
    dom = (a >= b).all(-1) & (a > b).any(-1) & alive[:, :, None]  # [s, j, i]: j dominates i
    return alive & ~dom.any(axis=1)


def select(spec: SelectionSpec, candidates: Sequence[CandidateEvidence]) -> SelectionResult:
    """Score candidates; evidence is read, never changed."""
    ids = [c.candidate_id for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique")
    obj_props = [o.property for o in spec.objectives]
    wanted = [(o.property, o.temperature_K, "objective") for o in spec.objectives] + [
        (c.property, c.temperature_K, "constraint") for c in spec.constraints
    ]
    cons_by_prop: dict[str, list[Constraint]] = {}
    for c in spec.constraints:
        cons_by_prop.setdefault(c.property, []).append(c)

    picked: dict[str, dict[str, UsedValue]] = {}
    chosen: dict[str, dict[str, PropertyEvidence]] = {}
    missing: dict[str, dict[str, str]] = {}
    for cand in candidates:
        picked[cand.candidate_id], chosen[cand.candidate_id], missing[cand.candidate_id] = (
            {},
            {},
            {},
        )
        for prop, temp, _role in wanted:
            if prop in picked[cand.candidate_id] or prop in missing[cand.candidate_id]:
                continue
            ev, why, alternatives = _pick(cand.evidence, prop, temp)
            if ev is None:
                missing[cand.candidate_id][prop] = why
            else:
                picked[cand.candidate_id][prop] = _used(ev, alternatives)
                chosen[cand.candidate_id][prop] = ev

    scores: dict[str, CandidateScore] = {}
    for cand in candidates:
        cid = cand.candidate_id
        values, miss = picked[cid], missing[cid]
        violated = [
            f"{c.property} {values[c.property].value:g} outside "
            f"[{c.minimum if c.minimum is not None else '-inf'}, "
            f"{c.maximum if c.maximum is not None else 'inf'}]"
            for c in spec.constraints
            if c.property in values and not bool(c.satisfied(values[c.property].value))
        ]
        reasons: list[str] = []
        weak = [
            p
            for p, v in values.items()
            if not v.reliable
            or _QUALIFICATION_ORDER[v.qualification] < _QUALIFICATION_ORDER[spec.min_qualification]
        ]
        if violated:
            status = SelectionStatus.INFEASIBLE
            reasons = [f"violates {v}" for v in violated]
        elif miss:
            status = SelectionStatus.INCOMPLETE
            reasons = [f"{p}: {why}" for p, why in sorted(miss.items())]
        elif weak:
            status = SelectionStatus.INSUFFICIENT_EVIDENCE
            reasons = [
                f"{p}: {values[p].status.value}, qualification {values[p].qualification.value}"
                for p in sorted(weak)
            ]
        else:
            status = SelectionStatus.QUALIFIED
        scores[cid] = CandidateScore(
            candidate_id=cid,
            status=status,
            reasons=reasons,
            values=values,
            missing=miss,
            violated=violated,
            pareto="not_ranked",
        )

    rankable = [
        c.candidate_id
        for c in candidates
        if all(p in picked[c.candidate_id] for p in obj_props)
        and scores[c.candidate_id].status != SelectionStatus.INFEASIBLE
    ]
    sign = _signs(spec)
    nominal = {cid: np.array([picked[cid][p].value for p in obj_props]) * sign for cid in rankable}
    for cid in rankable:
        dominators = [o for o in rankable if o != cid and _dominates(nominal[o], nominal[cid])]
        scores[cid].pareto = "dominated" if dominators else "nondominated"
        scores[cid].dominated_by = dominators
    if len(rankable) > 1:
        matrix = np.array([nominal[c] for c in rankable])
        for k, prop in enumerate(obj_props):
            top, bottom = matrix[:, k].max(), matrix[:, k].min()
            if top == bottom:
                continue
            for r, cid in enumerate(rankable):
                if matrix[r, k] == top:
                    scores[cid].best_in.append(prop)
                if matrix[r, k] == bottom:
                    scores[cid].worst_in.append(prop)

    rho_by_pair: dict[tuple[str, str], float] = {
        _pair_key(c.a, c.b): c.rho for c in spec.correlations
    }
    for cid in rankable:
        for a in obj_props:
            for b in obj_props:
                if a < b and _pair_key(a, b) not in rho_by_pair:
                    ev = chosen[cid]
                    if b in ev[a].derived_from or a in ev[b].derived_from:
                        inverse = (a == "electrical_resistivity") != (b == "electrical_resistivity")
                        rho_by_pair[_pair_key(a, b)] = -1.0 if inverse else 1.0
    weights, weight_notes = _effective_weights(spec, rho_by_pair)
    correlation_notes = [
        f"{a} ~ {b}: rho={rho:+.2f}" for (a, b), rho in sorted(rho_by_pair.items())
    ]

    if spec.weighted_desirability:
        for cid in rankable:
            x = np.array([picked[cid][p].value for p in obj_props])
            scores[cid].desirability = float(_desirability(x, spec.objectives, weights))
        order = sorted(rankable, key=lambda c: (-(scores[c].desirability or 0.0), c))
        for rank, cid in enumerate(order, start=1):
            scores[cid].desirability_rank = rank

    _explain_objectives(spec, candidates, scores, picked, rankable, weights)

    _propagate(spec, candidates, scores, picked, chosen, rankable, weights, cons_by_prop)

    front = [c for c in rankable if scores[c].pareto == "nondominated"]
    limitations = [
        "Scores rank evidence; they are not experiments and cannot qualify a material.",
        "Thresholds, weights, and desirability bounds come from the caller's specification.",
        "Elastic stiffness is not strength; a missing strength is INCOMPLETE, never inferred.",
    ]
    result = SelectionResult(
        spec=spec,
        candidates=[scores[c.candidate_id] for c in candidates],
        pareto_front=front,
        weight_notes=weight_notes,
        correlation_notes=correlation_notes,
        combination=_combination_text(spec, weights),
        limitations=limitations,
    )
    payload = json.dumps(result.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    result.score_digest = "sel-" + hashlib.sha256(payload.encode()).hexdigest()[:24]
    return result


def _combination_text(spec: SelectionSpec, weights: np.ndarray) -> list[str]:
    total = float(weights.sum())
    lines = [
        "Objectives are never added in their own units. The Pareto view compares each objective "
        "separately: a candidate is dominated only if another is at least as good in every "
        "objective and strictly better in one."
    ]
    if spec.weighted_desirability:
        lines += [
            "Combined score D = prod_k d_k^(w_k / sum w), the weighted geometric mean of "
            "per-objective desirabilities d_k = clip((x_k - worst_k) / (target_k - worst_k), "
            "0, 1).",
            "worst_k and target_k are declared by the caller. A geometric mean means one "
            "objective at d_k = 0 sets D = 0: a strong objective cannot compensate for a "
            "failed one.",
            "Weights w_k: "
            + "; ".join(
                f"{o.property} declared {o.weight:g}, effective {w:g}, share {w / total:.1%}"
                for o, w in zip(spec.objectives, weights, strict=True)
            ),
        ]
    else:
        lines.append(
            "No weighted score was requested; no worst/target bounds or weights are applied."
        )
    lines.append(
        "Constraints are hard limits applied before ranking and are not traded against objectives."
    )
    return lines


def _explain_objectives(spec, candidates, scores, picked, rankable, weights) -> None:
    """Fill each candidate's per-objective breakdown (value, rank, weight, d_k, contribution)."""
    objectives = spec.objectives
    total = float(weights.sum())
    bounds = spec.weighted_desirability
    best: dict[str, float] = {}
    rank: dict[str, dict[str, int]] = {}
    for o in objectives:
        values = {c: picked[c][o.property].value for c in rankable}
        if not values:
            continue
        sign = 1.0 if o.direction == "maximize" else -1.0
        ordered = sorted(values, key=lambda c: (-sign * values[c], c))
        best[o.property] = values[ordered[0]]
        rank[o.property] = {c: i for i, c in enumerate(ordered, start=1)}
    for cand in candidates:
        cid = cand.candidate_id
        rows: list[ObjectiveScore] = []
        for o, w in zip(objectives, weights, strict=True):
            used = picked[cid].get(o.property)
            if used is None:
                continue
            d = log_c = None
            if bounds:
                d = float(np.clip((used.value - o.worst) / (o.target - o.worst), 0.0, 1.0))
                log_c = float(w / total * np.log(d)) if d > 0 else float("-inf")
            rows.append(
                ObjectiveScore(
                    property=o.property,
                    direction=o.direction,
                    value=used.value,
                    uncertainty=used.uncertainty,
                    unit=PROPERTY_UNITS[o.property],
                    evidence_status=used.status,
                    rank=rank.get(o.property, {}).get(cid),
                    best_value=best.get(o.property),
                    declared_weight=o.weight,
                    effective_weight=float(w),
                    weight_share=float(w / total),
                    desirability=d,
                    log_contribution=log_c if log_c is None or np.isfinite(log_c) else None,
                )
            )
        if bounds:
            lost = [r for r in rows if r.log_contribution is not None]
            penalty = -sum(r.log_contribution for r in lost)
            zero = any(r.desirability == 0.0 for r in rows)
            for r in rows:
                if zero:
                    r.share_of_penalty = 1.0 if r.desirability == 0.0 else 0.0
                elif penalty > 0 and r.log_contribution is not None:
                    r.share_of_penalty = -r.log_contribution / penalty
        scores[cid].objectives = rows


def _propagate(spec, candidates, scores, picked, chosen, rankable, weights, cons_by_prop) -> None:
    """Monte-Carlo propagation of property uncertainty (correlated within a candidate)."""
    obj_props = [o.property for o in spec.objectives]
    n = spec.n_samples
    rng = numpy_rng(spec.seed, "selection-propagation")
    sign = _signs(spec)
    draws: dict[str, dict[str, np.ndarray]] = {}
    for cand in candidates:
        cid = cand.candidate_id
        props = sorted(picked[cid])
        if not props:
            continue
        mean = np.array([picked[cid][p].value for p in props])
        sd = np.array([picked[cid][p].uncertainty for p in props])
        corr = _correlation_matrix(props, chosen[cid], spec)
        z = rng.standard_normal((n, len(props))) @ _factor(corr).T
        x = mean + sd * z
        draws[cid] = {p: x[:, i] for i, p in enumerate(props)}
        feasible = np.ones(n, dtype=bool)
        complete = all(c.property in picked[cid] for c in spec.constraints)
        for prop, cons in cons_by_prop.items():
            if prop not in draws[cid]:
                continue
            ok = np.ones(n, dtype=bool)
            for c in cons:
                ok &= c.satisfied(draws[cid][prop])
            feasible &= ok
            scores[cid].constraint_probability_by_property[prop] = float(ok.mean())
        if spec.constraints and complete:
            scores[cid].constraint_probability = float(feasible.mean())
            need = spec.min_constraint_confidence
            if (
                need is not None
                and scores[cid].status == SelectionStatus.QUALIFIED
                and scores[cid].constraint_probability < need
            ):
                scores[cid].status = SelectionStatus.INSUFFICIENT_EVIDENCE
                scores[cid].reasons.append(
                    f"constraint probability {scores[cid].constraint_probability:.3f} "
                    f"< required {need}"
                )
        draws[cid]["__feasible__"] = feasible
    if not rankable:
        return

    x_obj = np.stack(
        [np.stack([draws[c][p] for p in obj_props], axis=-1) for c in rankable], axis=1
    )  # (s, c, o)
    alive = np.stack([draws[c]["__feasible__"] for c in rankable], axis=1)
    robust = np.zeros(len(rankable))
    chunk = max(1, int(2e6 // max(1, len(rankable) ** 2 * len(obj_props))))
    for lo in range(0, n, chunk):
        sl = slice(lo, lo + chunk)
        robust += _nondominated_mask(x_obj[sl] * sign, alive[sl]).sum(axis=0)
    for i, cid in enumerate(rankable):
        scores[cid].pareto_robustness = float(robust[i] / n)

    if spec.weighted_desirability:
        d = _desirability(x_obj, spec.objectives, weights)  # (s, c)
        d = np.where(alive, d, -1.0)  # an infeasible draw cannot win
        ranks = (-d).argsort(axis=1).argsort(axis=1) + 1
        for i, cid in enumerate(rankable):
            col = d[:, i]
            live = col[alive[:, i]]
            if len(live):
                scores[cid].desirability_distribution = Distribution(
                    mean=float(live.mean()),
                    std=float(live.std()),
                    p05=float(np.quantile(live, 0.05)),
                    p95=float(np.quantile(live, 0.95)),
                )
            counts = np.bincount(ranks[:, i], minlength=len(rankable) + 1)[1:] / n
            scores[cid].rank_distribution = {r + 1: float(p) for r, p in enumerate(counts) if p > 0}
            scores[cid].probability_rank_1 = float(counts[0])


def render_report(result: SelectionResult) -> str:
    """A plain-text report: each objective per candidate, then how they were combined."""
    out = [f"Selection report ({result.scoring_version}, {result.score_digest})", ""]
    out += ["How the objectives are combined", "------------------------------"]
    out += [f"- {line}" for line in result.combination]
    out += [f"- {note}" for note in (*result.weight_notes, *result.correlation_notes)]
    scored = result.spec.weighted_desirability
    for c in result.candidates:
        head = f"\nCandidate {c.candidate_id}: {c.status.value}, Pareto {c.pareto}"
        if scored and c.desirability is not None:
            head += f", D = {c.desirability:.4f} (rank {c.desirability_rank})"
        out.append(head)
        out += [f"  {r}" for r in c.reasons]
        if c.objectives:
            header = "  objective | value +/- unc [unit] | evidence | rank | best"
            if scored:
                header += " | weight share | d_k | share of penalty"
            out.append(header)
        for r in c.objectives:
            row = (
                f"  {r.property} ({r.direction}) | {r.value:.6g} +/- {r.uncertainty:.3g} "
                f"[{r.unit}] | {r.evidence_status.value} | "
                f"{r.rank if r.rank is not None else '-'} | "
                f"{'-' if r.best_value is None else format(r.best_value, '.6g')}"
            )
            if scored:
                share = "-" if r.share_of_penalty is None else f"{r.share_of_penalty:.1%}"
                row += f" | {r.weight_share:.1%} | {r.desirability:.3f} | {share}"
            out.append(row)
        for p, why in sorted(c.missing.items()):
            out.append(f"  {p}: not scored ({why})")
        if c.constraint_probability is not None:
            out.append(f"  probability all constraints hold: {c.constraint_probability:.3f}")
        if c.pareto_robustness is not None:
            out.append(f"  probability on the Pareto front: {c.pareto_robustness:.3f}")
    out += ["", "Limitations", "-----------", *[f"- {x}" for x in result.limitations]]
    return "\n".join(out)


def correlated_pairs_in(result: SelectionResult) -> list[str]:
    """Notes naming the property pairs that were treated as correlated."""
    return list(result.correlation_notes)


__all__ = [
    "SCORING_VERSION",
    "CandidateEvidence",
    "CandidateScore",
    "Constraint",
    "Correlation",
    "Objective",
    "ObjectiveScore",
    "SelectionResult",
    "SelectionSpec",
    "SelectionStatus",
    "correlated_pairs_in",
    "render_report",
    "select",
]
