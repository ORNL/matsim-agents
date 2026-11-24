"""MLIP validation protocol: compare predictions with a DFT reference set.

This module computes error statistics and a pass/fail/insufficient verdict. It runs no model and
no DFT; callers supply paired arrays. A verdict never promotes anything by itself: the written
report is an artifact that ``radiation_qualification.record_evidence`` can cite.

Rules:

* a reference set smaller than ``min_samples`` is ``insufficient``, never ``pass``;
* non-finite values anywhere make the verdict ``invalid``;
* the verdict is ``pass`` only if every metric's upper confidence bound is within its
  threshold, so a lucky small sample cannot pass on its mean alone;
* ``synthetic`` is carried into the report so synthetic runs cannot be mistaken for evidence.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.seeding import numpy_rng

REPORT_SCHEMA = "matsim-mlip-validation/1"
Verdict = Literal["pass", "fail", "insufficient", "invalid"]


class ValidationThresholds(BaseModel):
    """Upper limits on the confidence bound of each error metric."""

    model_config = ConfigDict(frozen=True)

    energy_mae_meV_per_atom: float = Field(default=10.0, gt=0)
    force_mae_meV_per_A: float = Field(default=100.0, gt=0)
    barrier_abs_error_meV: float = Field(default=100.0, gt=0)
    min_samples: int = Field(default=20, ge=2)
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    n_bootstrap: int = Field(default=1000, ge=100)


class MetricResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    unit: str
    n: int
    mean: float
    ci_low: float
    ci_high: float
    maximum: float
    threshold: float
    within_threshold: bool


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = Field(default=REPORT_SCHEMA, alias="schema")
    model_id: str
    reference_id: str
    held_out: bool
    synthetic: bool
    seed: int
    verdict: Verdict
    metrics: tuple[MetricResult, ...] = ()
    reasons: tuple[str, ...] = ()


def _absolute_errors(predicted: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    p = np.asarray(predicted, dtype=float).ravel()
    r = np.asarray(reference, dtype=float).ravel()
    if p.shape != r.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {r.shape}")
    return np.abs(p - r)


def _metric(
    name: str,
    unit: str,
    errors: np.ndarray,
    threshold: float,
    thresholds: ValidationThresholds,
    seed: int,
) -> MetricResult:
    rng = numpy_rng(seed, "mlip-validation", name)
    idx = rng.integers(0, errors.size, size=(thresholds.n_bootstrap, errors.size))
    means = errors[idx].mean(axis=1)
    alpha = (1.0 - thresholds.confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return MetricResult(
        name=name,
        unit=unit,
        n=int(errors.size),
        mean=float(errors.mean()),
        ci_low=float(low),
        ci_high=float(high),
        maximum=float(errors.max()),
        threshold=threshold,
        within_threshold=bool(high <= threshold),
    )


def evaluate_validation(
    *,
    model_id: str,
    reference_id: str,
    held_out: bool,
    synthetic: bool,
    seed: int,
    energy_pred_eV_per_atom: Sequence[float] = (),
    energy_ref_eV_per_atom: Sequence[float] = (),
    force_pred_eV_per_A: Sequence[float] = (),
    force_ref_eV_per_A: Sequence[float] = (),
    barrier_pred_eV: Sequence[float] = (),
    barrier_ref_eV: Sequence[float] = (),
    thresholds: ValidationThresholds | None = None,
) -> ValidationReport:
    """Error statistics and a verdict. Inputs are paired, in eV-based units."""
    th = thresholds or ValidationThresholds()

    def report(verdict: Verdict, reasons: list[str], metrics=()) -> ValidationReport:
        return ValidationReport(
            model_id=model_id,
            reference_id=reference_id,
            held_out=held_out,
            synthetic=synthetic,
            seed=seed,
            verdict=verdict,
            metrics=tuple(metrics),
            reasons=tuple(reasons),
        )

    pairs = [
        (
            "energy_mae",
            "meV/atom",
            energy_pred_eV_per_atom,
            energy_ref_eV_per_atom,
            th.energy_mae_meV_per_atom,
        ),
        ("force_mae", "meV/A", force_pred_eV_per_A, force_ref_eV_per_A, th.force_mae_meV_per_A),
        ("barrier_abs_error", "meV", barrier_pred_eV, barrier_ref_eV, th.barrier_abs_error_meV),
    ]
    errors: list[tuple[str, str, np.ndarray, float]] = []
    for name, unit, pred, ref, limit in pairs:
        if len(pred) == 0 and len(ref) == 0:
            continue
        arrays = (np.asarray(pred, dtype=float), np.asarray(ref, dtype=float))
        if not all(np.isfinite(a).all() for a in arrays):
            return report("invalid", [f"{name}: non-finite values"])
        try:
            err = 1000.0 * _absolute_errors(pred, ref)
        except ValueError as exc:
            return report("invalid", [f"{name}: {exc}"])
        errors.append((name, unit, err, limit))
    if not errors:
        return report("insufficient", ["no paired data supplied"])
    if not held_out:
        return report("insufficient", ["the reference set is not held out from training"])

    reasons: list[str] = []
    for name, _unit, err, _limit in errors:
        if err.size < th.min_samples:
            reasons.append(f"{name}: {err.size} samples, need {th.min_samples}")
    if reasons:
        return report("insufficient", reasons)

    metrics = [_metric(n, u, e, lim, th, seed) for n, u, e, lim in errors]
    failed = [m for m in metrics if not m.within_threshold]
    reasons = [
        f"{m.name}: upper bound {m.ci_high:.3g} {m.unit} exceeds {m.threshold:g}" for m in failed
    ]
    return report("fail" if failed else "pass", reasons, metrics)


def write_validation_report(report: ValidationReport, path: str | os.PathLike) -> Path:
    """Atomically write the report; its hash can be cited as evidence."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, staging = tempfile.mkstemp(prefix=".mlip-val-", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(report.model_dump(mode="json", by_alias=True), stream, indent=2)
            stream.write("\n")
        os.replace(staging, target)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "REPORT_SCHEMA",
    "MetricResult",
    "ValidationReport",
    "ValidationThresholds",
    "Verdict",
    "evaluate_validation",
    "write_validation_report",
]
