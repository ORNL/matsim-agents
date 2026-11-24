"""Model-agnostic magnetic surrogate and active-learning interface (M17).

Any model that implements ``MagneticSurrogate`` plugs in. Training is refused when it would leak
validation information: shared ids, shared content (same structure hash, or same canonical
symmetry key) or an overlap of the label source. ``StratumMeanSurrogate`` is the deterministic
fallback, and accuracy is reported only from a held-out set of sufficient size, otherwise as
``None`` rather than as an invented number.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from matsim_agents.discovery.sampling import Selection, sample_adaptive

MIN_VALIDATION = 10


@dataclass(frozen=True)
class LabeledExample:
    candidate_id: str
    stratum: str
    target: float  # e.g. relative energy or moment; a DFT-derived label
    content_key: str  # structure hash or symmetry key, used only for leakage checks
    label_source: str = "dft"  # "dft" for real evidence, "synthetic" for software tests


@dataclass(frozen=True)
class Prediction:
    mean: float
    std: float


@runtime_checkable
class MagneticSurrogate(Protocol):
    name: str

    def fit(self, examples: Sequence[LabeledExample]) -> None: ...

    def predict(self, candidate_id: str, stratum: str) -> Prediction: ...


class LeakageError(ValueError):
    pass


def check_leakage(train: Sequence[LabeledExample], validation: Sequence[LabeledExample]) -> None:
    """Raise ``LeakageError`` listing every kind of overlap found."""
    problems: list[str] = []
    ids = {e.candidate_id for e in train} & {e.candidate_id for e in validation}
    if ids:
        problems.append(f"{len(ids)} shared candidate ids")
    content = {e.content_key for e in train} & {e.content_key for e in validation}
    if content:
        problems.append(f"{len(content)} shared structure or symmetry keys")
    if problems:
        raise LeakageError("train/validation leakage: " + "; ".join(problems))


def fit_without_leakage(
    model: MagneticSurrogate,
    train: Sequence[LabeledExample],
    validation: Sequence[LabeledExample],
) -> None:
    check_leakage(train, validation)
    model.fit(train)


class StratumMeanSurrogate:
    """Deterministic fallback: per-stratum mean and standard deviation; global stats for
    unseen strata with an inflated spread so they rank as uncertain."""

    name = "stratum_mean"

    def __init__(self, unseen_penalty: float = 2.0) -> None:
        self.unseen_penalty = unseen_penalty
        self._stats: dict[str, tuple[float, float]] = {}
        self._global: tuple[float, float] = (0.0, 1.0)
        self._fitted = False

    def fit(self, examples: Sequence[LabeledExample]) -> None:
        if not examples:
            raise ValueError("cannot fit on zero examples")
        groups: dict[str, list[float]] = defaultdict(list)
        for e in examples:
            groups[e.stratum].append(e.target)
        self._stats = {s: _mean_std(v) for s, v in groups.items()}
        self._global = _mean_std([e.target for e in examples])
        self._fitted = True

    def predict(self, candidate_id: str, stratum: str) -> Prediction:
        if not self._fitted:
            raise RuntimeError("surrogate is not fitted")
        if stratum in self._stats:
            mean, std = self._stats[stratum]
            return Prediction(mean, std)
        mean, std = self._global
        return Prediction(mean, max(std, 1e-12) * self.unseen_penalty)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 1.0  # a single sample gives no spread; stay maximally uncertain
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


@dataclass(frozen=True)
class Evaluation:
    n: int
    rmse: float | None
    mae: float | None
    note: str


def evaluate(
    model: MagneticSurrogate,
    validation: Sequence[LabeledExample],
    *,
    min_n: int = MIN_VALIDATION,
) -> Evaluation:
    """Held-out error. Below ``min_n`` examples, or with synthetic labels, no figure is given."""
    if any(e.label_source != "dft" for e in validation):
        return Evaluation(len(validation), None, None, "validation labels are not DFT evidence")
    if len(validation) < min_n:
        return Evaluation(len(validation), None, None, f"fewer than {min_n} validation examples")
    errors = [model.predict(e.candidate_id, e.stratum).mean - e.target for e in validation]
    rmse = math.sqrt(sum(x * x for x in errors) / len(errors))
    mae = sum(abs(x) for x in errors) / len(errors)
    return Evaluation(len(errors), rmse, mae, "held-out error on independent validation set")


def select_for_labeling(
    model: MagneticSurrogate,
    pool: Mapping[str, str],
    n: int,
    *,
    seed: int,
    round_index: int,
    exploration_fraction: float = 0.2,
) -> list[Selection]:
    """Uncertainty-driven active-learning picks over ``pool`` (id -> stratum), which must not
    contain validation candidates."""
    scores = {cid: model.predict(cid, stratum).std for cid, stratum in pool.items()}
    return sample_adaptive(
        scores,
        pool,
        n,
        seed=seed,
        round_index=round_index,
        exploration_fraction=exploration_fraction,
    )


def deterministic_fallback(
    model: MagneticSurrogate | None, train: Sequence[LabeledExample]
) -> MagneticSurrogate:
    """Return ``model`` when it is usable, else a fitted ``StratumMeanSurrogate``."""
    if model is not None and isinstance(model, MagneticSurrogate):
        return model
    fallback = StratumMeanSurrogate()
    fallback.fit(train)
    return fallback
