"""Transferable property surrogates trained on lower-component data.

Extends :mod:`matsim_agents.workflows.hierarchy.transfer` (splits, baselines, verdicts) with what a
property surrogate needs before it may inform a decision:

- *compatibility tracking*: a source record is reusable only if its property, unit,
  method signature, temperature, and defect state match the target's;
- *independent test sets*: error is reported only on rows that never reached training;
- *transfer evaluation*: MAE/RMSE overall and by composition, ordering, temperature, and
  defect state; uncertainty calibration; negative transfer against a target-only model;
- *out-of-distribution (OOD) refusal*: queries far from the training data get no value;
- surrogate predictions become ``SURROGATE_PREDICTED`` :class:`PropertyEvidence` that is
  never DFT-level and never ``CALCULATED``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from matsim_agents.core.contracts import PROPERTY_UNITS, EvidenceLevel
from matsim_agents.science.properties.calculators import (
    PropertyConditions,
    PropertyEvidence,
    PropertyStatus,
)
from matsim_agents.workflows.hierarchy.transfer import RidgeEnsemble


class SourceRecord(BaseModel):
    """One labelled row and the provenance that decides whether it is reusable."""

    record_id: str
    property: str
    unit: str
    method_signature: str
    temperature_K: float | None = None
    defect_state: str | None = None
    composition_id: str
    ordering_id: str = "default"
    n_components: int = Field(ge=1)
    features: list[float]
    value: float
    dataset_id: str

    def check_finite(self) -> bool:
        return math.isfinite(self.value) and all(math.isfinite(f) for f in self.features)


class TargetSpec(BaseModel):
    property: str
    method_signature: str
    temperature_K: float | None = None
    temperature_tolerance_K: float = Field(25.0, ge=0)
    defect_state: str | None = None


def incompatibilities(record: SourceRecord, target: TargetSpec) -> list[str]:
    problems = []
    if not record.check_finite():
        problems.append("non-finite value or feature")
    if record.property != target.property:
        problems.append(f"property {record.property} != {target.property}")
    elif record.unit != PROPERTY_UNITS.get(target.property):
        problems.append(
            f"unit {record.unit!r} is not canonical {PROPERTY_UNITS[target.property]!r}"
        )
    if record.method_signature != target.method_signature:
        problems.append(f"method {record.method_signature!r} != {target.method_signature!r}")
    if target.temperature_K is not None and (
        record.temperature_K is None
        or abs(record.temperature_K - target.temperature_K) > target.temperature_tolerance_K
    ):
        problems.append(f"temperature {record.temperature_K} K vs {target.temperature_K} K")
    if record.defect_state != target.defect_state:
        problems.append(f"defect state {record.defect_state!r} != {target.defect_state!r}")
    return problems


def split_by_compatibility(
    records: Sequence[SourceRecord], target: TargetSpec
) -> tuple[list[SourceRecord], dict[str, list[str]]]:
    """Compatible records, and the reasons every other record was excluded."""
    keep: list[SourceRecord] = []
    excluded: dict[str, list[str]] = {}
    for r in records:
        problems = incompatibilities(r, target)
        if problems:
            excluded[r.record_id] = problems
        else:
            keep.append(r)
    return keep, excluded


def _matrix(records: Sequence[SourceRecord]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([r.features for r in records], float),
        np.array([r.value for r in records], float),
    )


class OODGate:
    """Refuses queries whose nearest training neighbour is farther than seen in training."""

    def __init__(self, X: np.ndarray, quantile: float = 0.95, margin: float = 1.5) -> None:
        if len(X) < 3:
            raise ValueError("the OOD gate needs at least three training rows")
        self.scale = np.maximum(X.std(axis=0), 1e-12)
        self.train = X / self.scale
        d = np.linalg.norm(self.train[:, None] - self.train[None], axis=2)
        np.fill_diagonal(d, np.inf)
        self.threshold = float(np.quantile(d.min(axis=1), quantile) * margin)

    def distance(self, X: np.ndarray) -> np.ndarray:
        q = X / self.scale
        return np.linalg.norm(q[:, None] - self.train[None], axis=2).min(axis=1)

    def in_distribution(self, X: np.ndarray) -> np.ndarray:
        return self.distance(X) <= self.threshold


@dataclass
class PropertySurrogate:
    target: TargetSpec
    model: RidgeEnsemble
    gate: OODGate
    training_ids: list[str]
    uncertainty_scale: float
    excluded: dict[str, list[str]]

    def training_digest(self) -> str:
        return hashlib.sha256(json.dumps(sorted(self.training_ids)).encode()).hexdigest()[:24]


def train_surrogate(
    records: Sequence[SourceRecord],
    target: TargetSpec,
    *,
    seed: int,
    calibration: Sequence[SourceRecord] = (),
    members: int = 16,
    alpha: float = 1e-3,
) -> PropertySurrogate:
    """Fit on compatible records only; calibrate the uncertainty on disjoint rows if given."""
    keep, excluded = split_by_compatibility(records, target)
    if len(keep) < 5:
        raise ValueError(f"only {len(keep)} compatible training records: refusing to fit")
    held = {r.record_id for r in calibration}
    overlap = held & {r.record_id for r in keep}
    if overlap:
        raise ValueError(f"calibration rows are in the training set: {sorted(overlap)[:3]}")
    X, y = _matrix(keep)
    model = RidgeEnsemble(alpha=alpha, members=members, seed=seed, tag=target.property).fit(X, y)
    scale = 1.0
    cal_ok, _ = split_by_compatibility(calibration, target)
    if len(cal_ok) >= 5:
        Xc, yc = _matrix(cal_ok)
        pred, sigma = model.predict(Xc)
        z = np.abs(pred - yc) / np.maximum(sigma, 1e-12)
        scale = max(1.0, float(np.sqrt(np.mean(z**2))))
    return PropertySurrogate(
        target=target,
        model=model,
        gate=OODGate(X),
        training_ids=[r.record_id for r in keep],
        uncertainty_scale=scale,
        excluded=excluded,
    )


class TransferReport(BaseModel):
    property: str
    n_train: int
    n_test: int
    n_excluded_incompatible: int
    mae: float
    rmse: float
    coverage_1sigma: float
    coverage_2sigma: float
    calibration_error: float
    ood_fraction: float
    mae_in_distribution: float | None
    mae_by_composition: dict[str, float]
    mae_by_ordering: dict[str, float]
    mae_by_temperature: dict[str, float]
    mae_by_defect: dict[str, float]
    target_only_mae: float | None = None
    negative_transfer: bool | None = None
    independent: bool = True
    limitations: list[str] = Field(default_factory=list)


def _by(keys: Sequence[str], err: np.ndarray) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for k, e in zip(keys, err, strict=True):
        groups[k].append(float(e))
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def evaluate_transfer(
    surrogate: PropertySurrogate,
    test: Sequence[SourceRecord],
    *,
    target_only: PropertySurrogate | None = None,
    min_test_rows: int = 10,
) -> TransferReport:
    """Error on an independent test set; the test must share no id or composition with training."""
    train_ids = set(surrogate.training_ids)
    if {r.record_id for r in test} & train_ids:
        raise ValueError("test rows overlap the training set: not an independent test")
    usable, excluded = split_by_compatibility(test, surrogate.target)
    if len(usable) < min_test_rows:
        raise ValueError(
            f"{len(usable)} compatible test rows < {min_test_rows}: transfer not evaluable"
        )
    X, y = _matrix(usable)
    pred, sigma = surrogate.model.predict(X)
    sigma = sigma * surrogate.uncertainty_scale
    err = np.abs(pred - y)
    z = err / np.maximum(sigma, 1e-12)
    c1, c2 = float(np.mean(z <= 1.0)), float(np.mean(z <= 2.0))
    inside = surrogate.gate.in_distribution(X)
    limitations = ["Surrogate estimates are screening-level and not DFT evidence."]
    if bool(np.all(inside)) is False:
        limitations.append("Some test rows are out of distribution and would be refused.")
    report = TransferReport(
        property=surrogate.target.property,
        n_train=len(train_ids),
        n_test=len(usable),
        n_excluded_incompatible=len(excluded),
        mae=float(err.mean()),
        rmse=float(np.sqrt(np.mean(err**2))),
        coverage_1sigma=c1,
        coverage_2sigma=c2,
        calibration_error=float((abs(c1 - 0.6827) + abs(c2 - 0.9545)) / 2),
        ood_fraction=float(1 - inside.mean()),
        mae_in_distribution=float(err[inside].mean()) if inside.any() else None,
        mae_by_composition=_by([r.composition_id for r in usable], err),
        mae_by_ordering=_by([f"{r.composition_id}/{r.ordering_id}" for r in usable], err),
        mae_by_temperature=_by([str(r.temperature_K) for r in usable], err),
        mae_by_defect=_by([str(r.defect_state) for r in usable], err),
        limitations=limitations,
    )
    if target_only is not None:
        other, _ = target_only.model.predict(X)
        report.target_only_mae = float(np.abs(other - y).mean())
        report.negative_transfer = report.mae > report.target_only_mae
        if report.negative_transfer:
            report.limitations.append(
                "Negative transfer: the target-only model is more accurate on the same test rows."
            )
    return report


def predict_property_evidence(
    surrogate: PropertySurrogate,
    *,
    structure_id: str,
    composition: str,
    features: Sequence[float],
    conditions: PropertyConditions,
    report: TransferReport | None = None,
) -> PropertyEvidence:
    """A SURROGATE_PREDICTED value, or UNSUPPORTED when the query is out of distribution."""
    spec = surrogate.target
    X = np.asarray([features], float)
    if spec.temperature_K is not None and conditions.temperature_K != spec.temperature_K:
        raise ValueError("the surrogate was trained for a different temperature")
    if not surrogate.gate.in_distribution(X)[0]:
        return PropertyEvidence(
            structure_id=structure_id,
            composition=composition,
            property=spec.property,
            status=PropertyStatus.UNSUPPORTED,
            unit=PROPERTY_UNITS[spec.property],
            conditions=conditions,
            reason=(
                "out of distribution: the query is farther from the training data than the "
                "training rows are from each other"
            ),
        )
    mean, sigma = surrogate.model.predict(X)
    limitations = ["Surrogate prediction: not DFT evidence."]
    if report is not None:
        limitations.append(
            f"independent-test MAE {report.mae:.4g}, "
            f"calibration error {report.calibration_error:.3f}"
        )
        if report.negative_transfer:
            limitations.append("negative transfer was detected against a target-only model")
    return PropertyEvidence(
        structure_id=structure_id,
        composition=composition,
        property=spec.property,
        status=PropertyStatus.SURROGATE_PREDICTED,
        value=float(mean[0]),
        unit=PROPERTY_UNITS[spec.property],
        conditions=conditions,
        method="transfer_surrogate",
        backend="ridge_ensemble",
        evidence_ids=[f"surrogate-{surrogate.training_digest()}"],
        evidence_level=EvidenceLevel.MLIP_PREDICTION,
        uncertainty=float(sigma[0] * surrogate.uncertainty_scale),
        limitations=limitations,
    )


__all__ = [
    "OODGate",
    "PropertySurrogate",
    "SourceRecord",
    "TargetSpec",
    "TransferReport",
    "evaluate_transfer",
    "incompatibilities",
    "predict_property_evidence",
    "split_by_compatibility",
    "train_surrogate",
]
