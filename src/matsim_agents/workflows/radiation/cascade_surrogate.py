"""Uncertainty-aware surrogates for cascade outcomes, and acquisition of further cascades.

Thin adapter over :class:`RidgeEnsemble` and :class:`OODGate`; it adds only the cascade
feature encoding, a compatibility check on target, unit and interatomic potential, and a
ranking of candidate cascades. The model is a linear ridge ensemble: recoil direction and
composition enter linearly, with no crystal-symmetry reduction, so it cannot represent
channelling. A surrogate value is a prediction, never a cascade result, and must not replace
an ensemble yield from simulation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.workflows.hierarchy.property_surrogates import OODGate
from matsim_agents.workflows.hierarchy.transfer import RidgeEnsemble

_COMPOSITION_TOL = 1e-6


class CascadeQuery(BaseModel):
    """Inputs of one cascade; ``direction`` is normalised and must be non-zero."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str = Field(min_length=1)
    composition: dict[str, float]
    recoil_element: str = Field(min_length=1)
    recoil_energy_mev: float = Field(gt=0.0)
    temperature_K: float = Field(gt=0.0)
    direction: tuple[float, float, float]
    transmutation_state: dict[str, float] = Field(default_factory=dict)
    node_hours: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _validate(self) -> CascadeQuery:
        total = math.fsum(self.composition.values())
        if any(v < 0 for v in self.composition.values()) or abs(total - 1.0) > _COMPOSITION_TOL:
            raise ValueError("composition fractions must be non-negative and sum to 1")
        if self.recoil_element not in self.composition:
            raise ValueError("the recoil element must be part of the composition")
        norm = math.sqrt(sum(c * c for c in self.direction))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("direction must be a finite non-zero vector")
        if any(not math.isfinite(v) or v < 0 for v in self.transmutation_state.values()):
            raise ValueError("transmutation_state values must be finite and non-negative")
        return self

    def unit_direction(self) -> tuple[float, float, float]:
        n = math.sqrt(sum(c * c for c in self.direction))
        return (self.direction[0] / n, self.direction[1] / n, self.direction[2] / n)


class CascadeSample(CascadeQuery):
    """A labelled cascade: the outcome ``value`` and where it came from."""

    value: float
    target: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    potential_id: str = Field(min_length=1)
    simulation_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _finite(self) -> CascadeSample:
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        return self


class CascadeTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)  # e.g. "surviving_frenkel_pairs"
    unit: str = Field(min_length=1)
    potential_id: str = Field(min_length=1)
    non_negative: bool = False  # counts and efficiencies are clipped at zero when True


class CascadeFeatureSpec(BaseModel):
    """Fixed feature layout, so train and query vectors always line up."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    elements: tuple[str, ...] = Field(min_length=1)
    transmutation_species: tuple[str, ...] = ()

    def encode(self, q: CascadeQuery) -> list[float]:
        unknown = (set(q.composition) | {q.recoil_element}) - set(self.elements)
        if unknown:
            raise ValueError(f"elements {sorted(unknown)} are not in the feature spec")
        extra = set(q.transmutation_state) - set(self.transmutation_species)
        if extra:
            raise ValueError(f"transmutation species {sorted(extra)} are not in the feature spec")
        d = q.unit_direction()
        return [
            *(q.composition.get(e, 0.0) for e in self.elements),
            *(1.0 if q.recoil_element == e else 0.0 for e in self.elements),
            math.log10(q.recoil_energy_mev),
            q.temperature_K / 1000.0,
            *d,
            *(q.transmutation_state.get(s, 0.0) for s in self.transmutation_species),
        ]


def incompatibilities(sample: CascadeSample, target: CascadeTarget) -> list[str]:
    problems = []
    if sample.target != target.name:
        problems.append(f"target {sample.target!r} != {target.name!r}")
    if sample.unit != target.unit:
        problems.append(f"unit {sample.unit!r} != {target.unit!r}")
    if sample.potential_id != target.potential_id:
        problems.append(f"potential {sample.potential_id!r} != {target.potential_id!r}")
    return problems


@dataclass
class CascadeSurrogate:
    target: CascadeTarget
    spec: CascadeFeatureSpec
    model: RidgeEnsemble
    gate: OODGate
    training_ids: list[str]
    uncertainty_scale: float
    excluded: dict[str, list[str]] = field(default_factory=dict)


def _matrix(spec: CascadeFeatureSpec, rows: Sequence[CascadeQuery]) -> np.ndarray:
    return np.array([spec.encode(r) for r in rows], float)


def train_cascade_surrogate(
    samples: Sequence[CascadeSample],
    target: CascadeTarget,
    spec: CascadeFeatureSpec,
    *,
    seed: int,
    calibration: Sequence[CascadeSample] = (),
    members: int = 16,
    alpha: float = 1e-3,
    min_rows: int = 10,
) -> CascadeSurrogate:
    """Fit on compatible samples; the uncertainty is rescaled on disjoint calibration rows."""
    keep, excluded = [], {}
    for s in samples:
        problems = incompatibilities(s, target)
        if problems:
            excluded[s.query_id] = problems
        else:
            keep.append(s)
    if len(keep) < min_rows:
        raise ValueError(f"only {len(keep)} compatible samples (need {min_rows}): refusing to fit")
    ids = {s.query_id for s in keep}
    if len(ids) != len(keep):
        raise ValueError("duplicate query_id in the training samples")
    overlap = ids & {c.query_id for c in calibration}
    if overlap:
        raise ValueError(f"calibration rows are in the training set: {sorted(overlap)[:3]}")
    X = _matrix(spec, keep)
    y = np.array([s.value for s in keep], float)
    model = RidgeEnsemble(alpha=alpha, members=members, seed=seed, tag=target.name).fit(X, y)
    scale = 1.0
    cal = [c for c in calibration if not incompatibilities(c, target)]
    if len(cal) >= 5:
        pred, sigma = model.predict(_matrix(spec, cal))
        z = np.abs(pred - np.array([c.value for c in cal])) / np.maximum(sigma, 1e-12)
        scale = max(1.0, float(np.sqrt(np.mean(z**2))))
    return CascadeSurrogate(
        target=target,
        spec=spec,
        model=model,
        gate=OODGate(X),
        training_ids=[s.query_id for s in keep],
        uncertainty_scale=scale,
        excluded=excluded,
    )


class CascadePrediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    target: str
    unit: str
    in_distribution: bool
    value: float | None = None
    standard_error: float | None = None
    clipped_at_zero: bool = False
    note: str = "surrogate prediction, not a cascade result"


def predict_cascades(
    surrogate: CascadeSurrogate, queries: Sequence[CascadeQuery]
) -> list[CascadePrediction]:
    """A value and uncertainty per query, or no value when the query is out of distribution."""
    if not queries:
        return []
    X = _matrix(surrogate.spec, queries)
    ok = surrogate.gate.in_distribution(X)
    mean, sigma = surrogate.model.predict(X)
    out = []
    for q, good, m, s in zip(queries, ok, mean, sigma, strict=True):
        if not good:
            out.append(
                CascadePrediction(
                    query_id=q.query_id,
                    target=surrogate.target.name,
                    unit=surrogate.target.unit,
                    in_distribution=False,
                    note="out of distribution: no value given",
                )
            )
            continue
        value, clipped = float(m), False
        if surrogate.target.non_negative and value < 0.0:
            value, clipped = 0.0, True
        out.append(
            CascadePrediction(
                query_id=q.query_id,
                target=surrogate.target.name,
                unit=surrogate.target.unit,
                in_distribution=True,
                value=value,
                standard_error=float(s * surrogate.uncertainty_scale),
                clipped_at_zero=clipped,
            )
        )
    return out


class CascadeSurrogateReport(BaseModel):
    target: str
    n_train: int
    n_test: int
    n_out_of_distribution: int
    mae: float | None
    rmse: float | None
    coverage_95: float | None  # share of in-distribution test rows inside the 1.96 sigma band
    limitations: list[str]


def evaluate_cascade_surrogate(
    surrogate: CascadeSurrogate, test: Sequence[CascadeSample]
) -> CascadeSurrogateReport:
    """Error only on rows that never reached training and share its target, unit, potential."""
    seen = set(surrogate.training_ids)
    if any(t.query_id in seen for t in test):
        raise ValueError("test rows overlap the training set")
    rows = [t for t in test if not incompatibilities(t, surrogate.target)]
    preds = predict_cascades(surrogate, rows)
    pairs = [(r.value, p) for r, p in zip(rows, preds, strict=True) if p.in_distribution]
    n_ood = len(rows) - len(pairs)
    limitations = [
        "linear ridge ensemble: no symmetry reduction, cannot represent channelling",
        "error is measured on the supplied test rows only",
    ]
    if len(pairs) < 5:
        limitations.append("fewer than five in-distribution test rows: statistics withheld")
        return CascadeSurrogateReport(
            target=surrogate.target.name,
            n_train=len(seen),
            n_test=len(rows),
            n_out_of_distribution=n_ood,
            mae=None,
            rmse=None,
            coverage_95=None,
            limitations=limitations,
        )
    err = np.array([p.value - v for v, p in pairs])
    sig = np.array([p.standard_error for _, p in pairs])
    return CascadeSurrogateReport(
        target=surrogate.target.name,
        n_train=len(seen),
        n_test=len(rows),
        n_out_of_distribution=n_ood,
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        coverage_95=float(np.mean(np.abs(err) <= 1.96 * sig)),
        limitations=limitations,
    )


class AcquisitionPick(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    score: float
    standard_error: float | None
    in_distribution: bool
    node_hours: float


def select_cascades_to_run(
    surrogate: CascadeSurrogate,
    candidates: Sequence[CascadeQuery],
    *,
    budget_node_hours: float,
    batch_size: int | None = None,
) -> list[AcquisitionPick]:
    """Greedy ranking of candidate cascades by uncertainty per node-hour, spread across inputs.

    An out-of-distribution candidate scores by its distance past the gate. Each pick shrinks
    the score of candidates close to it, so a batch does not repeat one input. This is a
    policy, not a result shown to beat random selection. Ties break by ``query_id``.
    """
    if not budget_node_hours > 0:
        raise ValueError("budget_node_hours must be positive")
    if not candidates:
        return []
    X = _matrix(surrogate.spec, candidates) / surrogate.gate.scale
    preds = predict_cascades(surrogate, candidates)
    thr = surrogate.gate.threshold
    base = []
    for p, d in zip(
        preds, surrogate.gate.distance(_matrix(surrogate.spec, candidates)), strict=True
    ):
        base.append(p.standard_error if p.in_distribution else max(d / thr, 1.0))
    pool = list(range(len(candidates)))
    picked: list[int] = []
    out: list[AcquisitionPick] = []
    used = 0.0
    while pool and (batch_size is None or len(out) < batch_size):
        best = None
        for i in pool:
            c = candidates[i]
            if used + c.node_hours > budget_node_hours:
                continue
            if picked:
                gap = min(float(np.linalg.norm(X[i] - X[j])) for j in picked)
                spread = min(1.0, gap / thr)
            else:
                spread = 1.0
            score = base[i] * spread / c.node_hours
            key = (-score, c.query_id)
            if best is None or key < best[0]:
                best = (key, i, score)
        if best is None:
            break
        _, i, score = best
        c = candidates[i]
        out.append(
            AcquisitionPick(
                query_id=c.query_id,
                score=score,
                standard_error=preds[i].standard_error,
                in_distribution=preds[i].in_distribution,
                node_hours=c.node_hours,
            )
        )
        picked.append(i)
        used += c.node_hours
        pool.remove(i)
    return out


__all__ = [
    "AcquisitionPick",
    "CascadeFeatureSpec",
    "CascadePrediction",
    "CascadeQuery",
    "CascadeSample",
    "CascadeSurrogate",
    "CascadeSurrogateReport",
    "CascadeTarget",
    "evaluate_cascade_surrogate",
    "incompatibilities",
    "predict_cascades",
    "select_cascades_to_run",
    "train_cascade_surrogate",
]
