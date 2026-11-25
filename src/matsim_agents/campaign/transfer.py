"""Hierarchical transfer experiments with leakage-safe splits, baselines A-E and verdicts.

Transfer is judged per property from held-out error on paired seeds; it is never assumed. With
too few seeds or held-out rows the verdict is ``NOT_YET_ESTABLISHED``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy import stats

from matsim_agents.campaign.registry import NOT_YET_ESTABLISHED
from matsim_agents.seeding import numpy_rng

STRATEGIES = ("A", "B", "C", "D", "E")
STRATEGY_NAMES = {
    "A": "source only",
    "B": "source + sparse target (systematic)",
    "C": "target only",
    "D": "source + random target acquisition",
    "E": "source + transfer-guided acquisition",
}
Verdict = (
    str  # useful_transfer | negative_transfer | no_detectable_transfer | insufficient_evidence
)


class LeakageError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class TransferData:
    """Labelled rows for ONE property; every sequence aligns with ``X`` and ``y``."""

    property: str
    ids: tuple[str, ...]
    X: np.ndarray
    y: np.ndarray
    system: tuple[str, ...]
    composition_id: tuple[str, ...]
    ordering_id: tuple[str, ...]
    environment: tuple[str, ...]
    sro_regime: tuple[str, ...]
    magnetic: tuple[str, ...]
    cost_node_hours: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.y)
        lengths = {
            len(self.ids),
            len(self.X),
            len(self.system),
            len(self.composition_id),
            len(self.ordering_id),
            len(self.environment),
            len(self.sro_regime),
            len(self.magnetic),
            len(self.cost_node_hours),
        }
        if lengths != {n}:
            raise ValueError("TransferData columns must have equal length")
        if len(set(self.ids)) != n:
            raise ValueError("TransferData ids must be unique")
        if not np.all(np.isfinite(self.y)) or not np.all(np.isfinite(self.X)):
            raise ValueError("TransferData needs finite features and labels")
        if np.any(np.asarray(self.cost_node_hours) < 0):
            raise ValueError("costs must be non-negative")

    def __len__(self) -> int:
        return len(self.y)

    def order(self, i: int) -> int:
        return len(self.system[i].split("-"))

    def orders(self) -> np.ndarray:
        return np.array([self.order(i) for i in range(len(self))])


class RidgeEnsemble:
    """Bootstrap ridge ensemble; the spread across members is the uncertainty."""

    def __init__(self, *, alpha: float = 1e-3, members: int = 12, seed: int, tag: str = ""):
        self.alpha, self.members, self.seed, self.tag = alpha, members, seed, tag
        self._w: np.ndarray | None = None

    def _solve(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        Xa = np.hstack([X, np.ones((len(X), 1))])
        reg = self.alpha * np.eye(Xa.shape[1])
        reg[-1, -1] = 0.0
        return np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeEnsemble:
        if len(X) == 0:
            raise ValueError("cannot fit on zero rows")
        rng = numpy_rng(self.seed, "ridge-ensemble", self.tag)
        weights = [self._solve(X, y)]
        for _ in range(self.members - 1):
            pick = rng.integers(0, len(X), len(X))
            weights.append(self._solve(X[pick], y[pick]))
        self._w = np.asarray(weights)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._w is None:
            raise RuntimeError("model is not fitted")
        Xa = np.hstack([X, np.ones((len(X), 1))])
        preds = Xa @ self._w.T
        return preds[:, 0], preds[:, 1:].std(axis=1) if self.members > 1 else np.zeros(len(X))


class TransferExperiment(BaseModel):
    name: str
    source_orders: tuple[int, ...]
    target_order: int
    n_new: tuple[int, ...] = (20,)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    eval_fraction: float = Field(0.25, gt=0.0, lt=1.0)
    reserve_fraction: float = Field(0.1, ge=0.0, lt=1.0)
    min_seeds: int = Field(3, ge=2)
    min_eval_rows: int = Field(20, ge=1)
    min_relative_gain: float = Field(0.05, ge=0.0)
    confidence: float = Field(0.95, gt=0.5, lt=1.0)

    @model_validator(mode="after")
    def _valid(self) -> TransferExperiment:
        if self.target_order in self.source_orders:
            raise ValueError("the target order must not be among the source orders")
        if not self.source_orders or not self.n_new or any(n < 1 for n in self.n_new):
            raise ValueError("need source orders and positive n_new")
        if self.eval_fraction + self.reserve_fraction >= 1.0:
            raise ValueError("eval and reserve fractions must leave a training pool")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        return self


def standard_experiments(**overrides) -> list[TransferExperiment]:
    """binary->ternary, binary+ternary->quaternary, ->quinary, ->six-component."""
    return [
        TransferExperiment(name="2->3", source_orders=(2,), target_order=3, **overrides),
        TransferExperiment(name="2,3->4", source_orders=(2, 3), target_order=4, **overrides),
        TransferExperiment(name="2,3,4->5", source_orders=(2, 3, 4), target_order=5, **overrides),
        TransferExperiment(
            name="2,3,4,5->6", source_orders=(2, 3, 4, 5), target_order=6, **overrides
        ),
    ]


@dataclass(frozen=True)
class Split:
    source: np.ndarray
    pool: np.ndarray
    evaluation: np.ndarray
    reserve: np.ndarray


def split_experiment(data: TransferData, exp: TransferExperiment, seed: int) -> Split:
    """Group by composition: held-out compositions (all orderings), reserve and training pool."""
    orders = data.orders()
    source = np.flatnonzero(np.isin(orders, exp.source_orders))
    target = np.flatnonzero(orders == exp.target_order)
    comps = sorted({data.composition_id[i] for i in target})
    rng = numpy_rng(seed, "transfer-split", exp.name, data.property)
    shuffled = [comps[k] for k in rng.permutation(len(comps))]
    n_eval = max(1, round(len(comps) * exp.eval_fraction)) if comps else 0
    n_reserve = round(len(comps) * exp.reserve_fraction)
    eval_c = set(shuffled[:n_eval])
    reserve_c = set(shuffled[n_eval : n_eval + n_reserve])
    pick = lambda group: np.array(  # noqa: E731
        [i for i in target if data.composition_id[i] in group], dtype=int
    )
    pool_c = set(comps) - eval_c - reserve_c
    split = Split(source, pick(pool_c), pick(eval_c), pick(reserve_c))
    check_leakage(data, split)
    return split


def check_leakage(data: TransferData, split: Split) -> None:
    """No held-out or reserved composition/ordering/id may appear in anything trainable."""
    trainable = np.concatenate([split.source, split.pool]).astype(int)
    for name, held in (("evaluation", split.evaluation), ("reserve", split.reserve)):
        held_ids = {data.ids[i] for i in held}
        held_comps = {data.composition_id[i] for i in held}
        held_orderings = {(data.composition_id[i], data.ordering_id[i]) for i in held}
        for i in trainable:
            if data.ids[i] in held_ids:
                raise LeakageError(f"{name} row {data.ids[i]!r} is trainable")
            if data.composition_id[i] in held_comps:
                raise LeakageError(f"{name} composition {data.composition_id[i]!r} is trainable")
            if (data.composition_id[i], data.ordering_id[i]) in held_orderings:
                raise LeakageError(f"{name} parent ordering of {data.ids[i]!r} is trainable")
    if {data.ids[i] for i in split.evaluation} & {data.ids[i] for i in split.reserve}:
        raise LeakageError("evaluation and reserve sets overlap")
    systems_target = {data.system[i] for i in split.pool} | {
        data.system[i] for i in split.evaluation
    }
    if systems_target & {data.system[i] for i in split.source}:
        raise LeakageError("source and target share a chemical system")


Selector = Callable[[RidgeEnsemble, np.ndarray, np.ndarray, np.ndarray, int], list[int]]


def uncertainty_diversity_selector(
    model: RidgeEnsemble,
    X_pool: np.ndarray,
    cost_pool: np.ndarray,
    X_train: np.ndarray,
    n: int,
    weights: np.ndarray | None = None,
) -> list[int]:
    """Greedy uncertainty x distance-to-selected; ties break on index. Cost-blind by design.

    ``weights`` optionally scales each pool row's score (e.g. magnetic or defect relevance).
    """
    _, sigma = model.predict(X_pool)
    if weights is not None:
        sigma = sigma * np.asarray(weights, dtype=float)
    scale = np.maximum(X_pool.std(axis=0), 1e-12)
    anchors = [row for row in X_train / scale]
    chosen: list[int] = []
    for _ in range(min(n, len(X_pool))):
        best, best_score = -1, -math.inf
        for k in range(len(X_pool)):
            if k in chosen:
                continue
            row = X_pool[k] / scale
            distance = min((float(np.linalg.norm(row - a)) for a in anchors), default=1.0)
            score = float(sigma[k]) * distance
            if score > best_score + 1e-15:
                best, best_score = k, score
        chosen.append(best)
        anchors.append(X_pool[best] / scale)
    return chosen


class StrategyResult(BaseModel):
    strategy: str
    seed: int
    n_budget: int
    n_new: int
    target_fraction: float
    node_hours: float
    mae: float
    rmse: float
    coverage_1sigma: float
    coverage_2sigma: float
    calibration_error: float
    mae_by_environment: dict[str, float]
    mae_by_sro_regime: dict[str, float]
    mae_by_magnetic: dict[str, float]
    mae_by_composition: dict[str, float]
    selected_ids: list[str]


def _group_mae(keys: Sequence[str], errors: np.ndarray) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for k, e in zip(keys, errors, strict=True):
        groups[k].append(e)
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def score_model(
    model: RidgeEnsemble,
    data: TransferData,
    evaluation: np.ndarray,
    *,
    strategy: str,
    seed: int,
    selected: Sequence[int],
    pool_size: int,
    n_budget: int,
) -> StrategyResult:
    pred, sigma = model.predict(data.X[evaluation])
    err = np.abs(pred - data.y[evaluation])
    positive = sigma > 0
    z = np.where(positive, err / np.where(positive, sigma, 1.0), np.inf)
    c1, c2 = float(np.mean(z <= 1.0)), float(np.mean(z <= 2.0))
    take = lambda column: [column[i] for i in evaluation]  # noqa: E731
    return StrategyResult(
        strategy=strategy,
        seed=seed,
        n_budget=n_budget,
        n_new=len(selected),
        target_fraction=len(selected) / pool_size if pool_size else 0.0,
        node_hours=float(data.cost_node_hours[list(selected)].sum()) if len(selected) else 0.0,
        mae=float(err.mean()),
        rmse=float(np.sqrt(np.mean(err**2))),
        coverage_1sigma=c1,
        coverage_2sigma=c2,
        calibration_error=float((abs(c1 - 0.6827) + abs(c2 - 0.9545)) / 2),
        mae_by_environment=_group_mae(take(data.environment), err),
        mae_by_sro_regime=_group_mae(take(data.sro_regime), err),
        mae_by_magnetic=_group_mae(take(data.magnetic), err),
        mae_by_composition=_group_mae(take(data.composition_id), err),
        selected_ids=[data.ids[i] for i in selected],
    )


class PairedDifference(BaseModel):
    mean: float
    ci_low: float
    ci_high: float
    n_seeds: int


def paired_difference(values: Sequence[float], confidence: float) -> PairedDifference:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return PairedDifference(
            mean=float(arr.mean()) if len(arr) else math.nan,
            ci_low=-math.inf,
            ci_high=math.inf,
            n_seeds=len(arr),
        )
    se = arr.std(ddof=1) / math.sqrt(len(arr))
    half = float(stats.t.ppf(0.5 + confidence / 2, len(arr) - 1) * se)
    return PairedDifference(
        mean=float(arr.mean()),
        ci_low=float(arr.mean() - half),
        ci_high=float(arr.mean() + half),
        n_seeds=len(arr),
    )


class StrategySummary(BaseModel):
    strategy: str
    n_new: int
    mae_mean: float
    mae_std: float
    rmse_mean: float
    node_hours_mean: float
    target_fraction_mean: float
    calibration_error_mean: float
    gain_per_calculation: float | None
    gain_per_node_hour: float | None


class TransferVerdict(BaseModel):
    property: str
    experiment: str
    n_new: int
    status: Verdict
    reason: str
    benefit_vs_target_only: PairedDifference | None = None
    n_eval_rows: int = 0

    @property
    def registry_status(self) -> str:
        return {
            "useful_transfer": "validated",
            "negative_transfer": "unsupported",
            "no_detectable_transfer": "unsupported",
            "insufficient_evidence": NOT_YET_ESTABLISHED,
        }[self.status]


class ExperimentResult(BaseModel):
    experiment: TransferExperiment
    property: str
    results: list[StrategyResult]
    summary: list[StrategySummary]
    verdicts: list[TransferVerdict]


def _verdict(
    exp: TransferExperiment,
    prop: str,
    n_new: int,
    rows: dict[tuple[str, int], StrategyResult],
    n_eval: int,
) -> TransferVerdict:
    seeds = sorted({s for (_, s) in rows})
    base = dict(property=prop, experiment=exp.name, n_new=n_new, n_eval_rows=n_eval)
    if len(seeds) < exp.min_seeds:
        return TransferVerdict(
            **base, status="insufficient_evidence", reason=f"{len(seeds)} seeds < {exp.min_seeds}"
        )
    if n_eval < exp.min_eval_rows:
        return TransferVerdict(
            **base,
            status="insufficient_evidence",
            reason=f"{n_eval} held-out rows < {exp.min_eval_rows}",
        )
    diffs = [rows[("C", s)].mae - rows[("D", s)].mae for s in seeds]
    diff = paired_difference(diffs, exp.confidence)
    reference = float(np.mean([rows[("C", s)].mae for s in seeds]))
    threshold = exp.min_relative_gain * reference
    if diff.ci_low > threshold:
        status, reason = "useful_transfer", "source data lowers held-out MAE beyond the margin"
    elif diff.ci_high < 0:
        status, reason = "negative_transfer", "source data raises held-out MAE (CI below zero)"
    elif diff.ci_low <= 0 <= diff.ci_high or diff.mean <= threshold:
        status, reason = "no_detectable_transfer", "difference is within noise or the margin"
    else:
        status, reason = "insufficient_evidence", "difference positive but CI crosses the margin"
    return TransferVerdict(**base, status=status, reason=reason, benefit_vs_target_only=diff)


def run_transfer_experiment(
    data: TransferData,
    exp: TransferExperiment,
    *,
    selector: Selector = uncertainty_diversity_selector,
    alpha: float = 1e-3,
) -> ExperimentResult:
    """Run baselines A-E for every (n_new, seed); strategies share each seed's split."""
    results: list[StrategyResult] = []
    verdicts: list[TransferVerdict] = []
    for n_new in exp.n_new:
        rows: dict[tuple[str, int], StrategyResult] = {}
        n_eval = 0
        for seed in exp.seeds:
            split = split_experiment(data, exp, seed)
            n_eval = len(split.evaluation)
            pool = split.pool
            if len(split.source) == 0 or len(pool) < n_new or n_eval == 0:
                raise ValueError(f"{exp.name}: not enough source/pool/eval rows for n_new={n_new}")
            rng = numpy_rng(seed, "transfer-acquire", exp.name, data.property, n_new)
            random_pick = list(rng.choice(len(pool), n_new, replace=False))
            stride = np.linspace(0, len(pool) - 1, n_new).round().astype(int)
            systematic = sorted(set(stride.tolist()))
            while len(systematic) < n_new:
                systematic.append(next(k for k in range(len(pool)) if k not in systematic))

            def fit(rows_idx: np.ndarray, tag: str, seed=seed) -> RidgeEnsemble:
                return RidgeEnsemble(alpha=alpha, seed=seed, tag=tag).fit(
                    data.X[rows_idx], data.y[rows_idx]
                )

            source_only = fit(split.source, "A")
            guided = selector(
                source_only, data.X[pool], data.cost_node_hours[pool], data.X[split.source], n_new
            )
            plan = {
                "A": ([], split.source),
                "B": (systematic, np.concatenate([split.source, pool[systematic]])),
                "C": (random_pick, pool[random_pick]),
                "D": (random_pick, np.concatenate([split.source, pool[random_pick]])),
                "E": (guided, np.concatenate([split.source, pool[guided]])),
            }
            for strategy, (picked, train_rows) in plan.items():
                model = source_only if strategy == "A" else fit(np.asarray(train_rows), strategy)
                result = score_model(
                    model,
                    data,
                    split.evaluation,
                    strategy=strategy,
                    seed=seed,
                    selected=[int(pool[k]) for k in picked],
                    pool_size=len(pool),
                    n_budget=n_new,
                )
                rows[(strategy, seed)] = result
                results.append(result)
        verdicts.append(_verdict(exp, data.property, n_new, rows, n_eval))
    return ExperimentResult(
        experiment=exp,
        property=data.property,
        results=results,
        summary=_summarise(results),
        verdicts=verdicts,
    )


def _summarise(results: list[StrategyResult]) -> list[StrategySummary]:
    by_key: dict[tuple[int, str], list[StrategyResult]] = defaultdict(list)
    for r in results:
        by_key[(r.n_budget, r.strategy)].append(r)
    out = []
    for (n_new, strategy), group in sorted(by_key.items()):
        mae_a = np.mean([r.mae for r in by_key[(n_new, "A")]])
        mae = np.mean([r.mae for r in group])
        hours = float(np.mean([r.node_hours for r in group]))
        gain = float(mae_a - mae)
        out.append(
            StrategySummary(
                strategy=strategy,
                n_new=n_new,
                mae_mean=float(mae),
                mae_std=float(np.std([r.mae for r in group], ddof=1)) if len(group) > 1 else 0.0,
                rmse_mean=float(np.mean([r.rmse for r in group])),
                node_hours_mean=hours,
                target_fraction_mean=float(np.mean([r.target_fraction for r in group])),
                calibration_error_mean=float(np.mean([r.calibration_error for r in group])),
                gain_per_calculation=gain / n_new if strategy != "A" else None,
                gain_per_node_hour=gain / hours if strategy != "A" and hours > 0 else None,
            )
        )
    return out


def evaluate_properties(
    datasets: dict[str, TransferData], exp: TransferExperiment, required: Sequence[str]
) -> dict[str, list[TransferVerdict]]:
    """Per-property verdicts; a required property without data is NOT_YET_ESTABLISHED."""
    out: dict[str, list[TransferVerdict]] = {}
    for prop in required:
        if prop not in datasets:
            out[prop] = [
                TransferVerdict(
                    property=prop,
                    experiment=exp.name,
                    n_new=0,
                    status="insufficient_evidence",
                    reason="no labelled data for this property",
                )
            ]
        else:
            out[prop] = run_transfer_experiment(datasets[prop], exp).verdicts
    return out
