"""Pre-registered, paired experimental design for campaign comparisons."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from itertools import product
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from matsim_agents.execution.contracts import ComputeBudget


class BenchmarkArm(StrEnum):
    RANDOM = "random"
    EXPLOITATION = "exploitation"
    EXPLORATION = "exploration"
    ADAPTIVE = "adaptive"
    NO_LLM = "no_llm"
    SINGLE_LLM = "single_llm"
    MULTI_LLM_NO_CRITIQUE = "multi_llm_no_critique"
    MULTI_LLM_CROSS_CRITIQUE = "multi_llm_cross_critique"


class BenchmarkMetric(BaseModel):
    name: str
    direction: Literal["maximize", "minimize"]
    success_threshold: float | None = None
    unit: str | None = None


class BenchmarkProtocol(BaseModel):
    """Frozen comparison contract shared by every policy arm and replicate."""

    protocol_id: str
    element_set: list[str] = Field(min_length=1)
    candidate_pool_regime: Literal["fixed", "generative"]
    candidate_pool_manifest: str
    initial_model_identifier: str
    reference_set_identifier: str
    dft_method_signature: str
    budget: ComputeBudget
    random_seeds: list[int] = Field(min_length=2)
    arms: list[BenchmarkArm] = Field(default_factory=lambda: list(BenchmarkArm))
    metrics: list[BenchmarkMetric] = Field(min_length=1)
    confidence_level: float = Field(0.95, gt=0.0, lt=1.0)
    account_failed_dft_against_budget: bool = True
    account_restarts_against_budget: bool = True
    stopping_rule_identifier: str
    decoding_parameters: dict[str, float | int | str | bool] = Field(default_factory=dict)
    protocol_version: str = "campaign-benchmark-v1"

    @model_validator(mode="after")
    def _validate_design(self) -> BenchmarkProtocol:
        if len(self.random_seeds) != len(set(self.random_seeds)):
            raise ValueError("benchmark random_seeds must be unique")
        if len(self.arms) < 2:
            raise ValueError("paired benchmark requires at least two arms")
        if len(self.arms) != len(set(self.arms)):
            raise ValueError("benchmark arms must be unique")
        if not any(
            value is not None
            for value in (
                self.budget.max_dft_calculations,
                self.budget.max_node_hours,
            )
        ):
            raise ValueError("benchmark requires a hard DFT or node-hour budget")
        if self.candidate_pool_regime == "fixed" and not self.candidate_pool_manifest:
            raise ValueError("fixed-pool benchmark requires candidate_pool_manifest")
        return self

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def run_matrix(self) -> list[BenchmarkRunSpec]:
        digest = self.digest()
        return [
            BenchmarkRunSpec(
                run_id=f"{self.protocol_id}-{arm.value}-seed-{seed}",
                protocol_id=self.protocol_id,
                protocol_digest=digest,
                arm=arm,
                random_seed=seed,
                candidate_pool_manifest=self.candidate_pool_manifest,
                budget=self.budget.model_copy(deep=True),
            )
            for seed in self.random_seeds
            for arm in self.arms
        ]


class BenchmarkRunSpec(BaseModel):
    """One immutable arm/seed cell in a paired benchmark matrix."""

    run_id: str
    protocol_id: str
    protocol_digest: str
    arm: BenchmarkArm
    random_seed: int
    candidate_pool_manifest: str
    budget: ComputeBudget


class BenchmarkObservation(BaseModel):
    protocol_digest: str
    arm: BenchmarkArm
    random_seed: int
    metrics: dict[str, float]
    dft_calculations_attempted: int = Field(ge=0)
    node_hours_consumed: float = Field(ge=0.0)
    completed: bool
    failure_reason: str | None = None


class PairedComparison(BaseModel):
    metric: str
    treatment: BenchmarkArm
    control: BenchmarkArm
    n_pairs: int
    mean_difference: float
    confidence_level: float
    confidence_interval: tuple[float, float]
    two_sided_p_value: float
    favorable: bool


def paired_metric_differences(
    observations: list[BenchmarkObservation],
    *,
    treatment: BenchmarkArm,
    control: BenchmarkArm,
    metric: str,
) -> list[float]:
    """Return treatment-minus-control values matched by protocol and seed."""
    indexed: dict[tuple[str, int, BenchmarkArm], BenchmarkObservation] = {}
    for item in observations:
        key = (item.protocol_digest, item.random_seed, item.arm)
        if key in indexed:
            raise ValueError(
                "duplicate benchmark observation for "
                f"protocol_digest={item.protocol_digest!r}, "
                f"random_seed={item.random_seed}, arm={item.arm.value!r}"
            )
        indexed[key] = item
    indexed = {
        key: item for key, item in indexed.items() if item.completed and metric in item.metrics
    }
    pairs: list[float] = []
    keys = sorted({(item.protocol_digest, item.random_seed) for item in observations})
    for digest, seed in keys:
        treatment_item = indexed.get((digest, seed, treatment))
        control_item = indexed.get((digest, seed, control))
        if treatment_item is not None and control_item is not None:
            pairs.append(treatment_item.metrics[metric] - control_item.metrics[metric])
    return pairs


def compare_paired_metric(
    observations: list[BenchmarkObservation],
    *,
    treatment: BenchmarkArm,
    control: BenchmarkArm,
    metric: BenchmarkMetric,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 10_000,
    random_seed: int = 0,
) -> PairedComparison:
    """Estimate a paired effect, bootstrap interval, and sign-flip p-value."""
    differences = np.asarray(
        paired_metric_differences(
            observations,
            treatment=treatment,
            control=control,
            metric=metric.name,
        ),
        dtype=float,
    )
    if differences.size < 2:
        raise ValueError("paired comparison requires at least two complete seed pairs")
    mean_difference = float(np.mean(differences))
    rng = np.random.default_rng(random_seed)
    draws = rng.choice(differences, size=(bootstrap_samples, differences.size), replace=True)
    bootstrap_means = np.mean(draws, axis=1)
    alpha = 1.0 - confidence_level
    interval = tuple(
        float(value) for value in np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    )
    if differences.size <= 16:
        null_means = np.asarray(
            [
                np.mean(differences * np.asarray(signs, dtype=float))
                for signs in product((-1.0, 1.0), repeat=differences.size)
            ]
        )
    else:
        signs = rng.choice((-1.0, 1.0), size=(bootstrap_samples, differences.size))
        null_means = np.mean(signs * differences, axis=1)
    p_value = float(np.mean(np.abs(null_means) >= abs(mean_difference)))
    favorable_difference = mean_difference if metric.direction == "maximize" else -mean_difference
    threshold = metric.success_threshold or 0.0
    return PairedComparison(
        metric=metric.name,
        treatment=treatment,
        control=control,
        n_pairs=int(differences.size),
        mean_difference=mean_difference,
        confidence_level=confidence_level,
        confidence_interval=interval,
        two_sided_p_value=p_value,
        favorable=favorable_difference >= threshold,
    )
