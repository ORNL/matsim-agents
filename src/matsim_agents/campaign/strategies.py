"""Cost-normalised comparison of six-component data-acquisition strategies.

A direct six-component | B pretrain + random augmentation | C pretrain + stratified augmentation |
D transfer-guided | E transfer-guided + magnetic/defect relevance.

All strategies share each seed's split, so they are scored on the identical held-out rows, and
differences are paired across seeds with a confidence interval. Without enough seeds or held-out
rows nothing is claimed: the verdict is ``NOT_YET_ESTABLISHED``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

from matsim_agents.campaign.registry import NOT_YET_ESTABLISHED
from matsim_agents.campaign.transfer import (
    PairedDifference,
    RidgeEnsemble,
    StrategyResult,
    TransferData,
    TransferExperiment,
    paired_difference,
    score_model,
    split_experiment,
    uncertainty_diversity_selector,
)
from matsim_agents.seeding import numpy_rng

COMPARISON_STRATEGIES = ("A", "B", "C", "D", "E")
COMPARISON_NAMES = {
    "A": "direct six-component",
    "B": "pretrain + random augmentation",
    "C": "pretrain + stratified augmentation",
    "D": "transfer-guided",
    "E": "transfer-guided + magnetic/defect relevance",
}


class StrategyOutcome(BaseModel):
    strategy: str
    name: str
    mae_mean: float
    mae_std: float
    node_hours_mean: float
    gain_over_direct_per_node_hour: float | None
    gain_over_pretrained_random_per_node_hour: float | None
    n_seeds: int


class PairedComparison(BaseModel):
    better: str
    baseline: str
    mae_improvement: PairedDifference
    significant: bool


class StrategyComparison(BaseModel):
    experiment: str
    n_new: int
    n_eval_rows: int
    outcomes: list[StrategyOutcome]
    comparisons: list[PairedComparison]
    status: str
    best_strategy: str | None
    reason: str
    results: list[StrategyResult] = Field(default_factory=list)


def _stratified(
    data: TransferData, pool: np.ndarray, n: int, rng: np.random.Generator
) -> list[int]:
    """Round-robin over (environment, SRO regime, magnetic) strata; positions within ``pool``."""
    strata: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for k, i in enumerate(pool):
        strata[(data.environment[i], data.sro_regime[i], data.magnetic[i])].append(k)
    queues = {
        key: [members[j] for j in rng.permutation(len(members))] for key, members in strata.items()
    }
    order = sorted(queues)
    chosen: list[int] = []
    while len(chosen) < min(n, len(pool)):
        for key in order:
            if queues[key] and len(chosen) < n:
                chosen.append(queues[key].pop(0))
    return chosen


def default_relevance(data: TransferData, magnetic_boost: float = 1.0) -> np.ndarray:
    """Weight rows in magnetic environments up; callers may pass any defect-aware array."""
    return np.array([1.0 + magnetic_boost * (m != "nm") for m in data.magnetic], dtype=float)


def compare_strategies(
    data: TransferData,
    exp: TransferExperiment,
    n_new: int,
    *,
    relevance: np.ndarray | None = None,
    alpha: float = 1e-3,
    min_relative_gain: float | None = None,
) -> StrategyComparison:
    """Run A-E on identical held-out rows over ``exp.seeds`` and compare them cost-normalised."""
    relevance = relevance if relevance is not None else default_relevance(data)
    min_gain = exp.min_relative_gain if min_relative_gain is None else min_relative_gain
    results: list[StrategyResult] = []
    n_eval = 0
    per_seed: dict[tuple[str, int], StrategyResult] = {}
    for seed in exp.seeds:
        split = split_experiment(data, exp, seed)
        pool, source = split.pool, split.source
        n_eval = len(split.evaluation)
        if len(pool) == 0 or len(source) == 0 or n_eval == 0:
            continue
        k = min(n_new, len(pool))
        rng = numpy_rng(seed, "strategy-compare", exp.name)
        random_pick = [int(j) for j in rng.permutation(len(pool))[:k]]
        stratified = _stratified(data, pool, k, rng)

        def fit(rows: np.ndarray, tag: str, seed: int = seed) -> RidgeEnsemble:
            return RidgeEnsemble(alpha=alpha, seed=seed, tag=tag).fit(data.X[rows], data.y[rows])

        pretrained = fit(source, "pretrain")
        args = (pretrained, data.X[pool], data.cost_node_hours[pool], data.X[source], k)
        guided = uncertainty_diversity_selector(*args)
        guided_rel = uncertainty_diversity_selector(*args, weights=relevance[pool])
        plan = {
            "A": (random_pick, pool[random_pick]),
            "B": (random_pick, np.concatenate([source, pool[random_pick]])),
            "C": (stratified, np.concatenate([source, pool[stratified]])),
            "D": (guided, np.concatenate([source, pool[guided]])),
            "E": (guided_rel, np.concatenate([source, pool[guided_rel]])),
        }
        for strategy, (picked, rows) in plan.items():
            model = fit(np.asarray(rows), strategy)
            res = score_model(
                model,
                data,
                split.evaluation,
                strategy=strategy,
                seed=seed,
                selected=[int(pool[j]) for j in picked],
                pool_size=len(pool),
                n_budget=n_new,
            )
            per_seed[(strategy, seed)] = res
            results.append(res)

    seeds = sorted({s for _, s in per_seed})
    if len(seeds) < exp.min_seeds or n_eval < exp.min_eval_rows:
        return StrategyComparison(
            experiment=exp.name,
            n_new=n_new,
            n_eval_rows=n_eval,
            outcomes=[],
            comparisons=[],
            status=NOT_YET_ESTABLISHED,
            best_strategy=None,
            reason=(
                f"{len(seeds)} usable seed(s) (need {exp.min_seeds}) and {n_eval} held-out "
                f"row(s) (need {exp.min_eval_rows}); no strategy claim is made"
            ),
            results=results,
        )

    def column(strategy: str, attr: str) -> np.ndarray:
        return np.array([getattr(per_seed[(strategy, s)], attr) for s in seeds])

    mae = {s: column(s, "mae") for s in COMPARISON_STRATEGIES}
    hours = {s: column(s, "node_hours") for s in COMPARISON_STRATEGIES}

    def per_hour(strategy: str, base: str) -> float | None:
        spent = float(hours[strategy].mean())
        return float((mae[base].mean() - mae[strategy].mean()) / spent) if spent > 0 else None

    outcomes = [
        StrategyOutcome(
            strategy=s,
            name=COMPARISON_NAMES[s],
            mae_mean=float(mae[s].mean()),
            mae_std=float(mae[s].std(ddof=1)),
            node_hours_mean=float(hours[s].mean()),
            gain_over_direct_per_node_hour=None if s == "A" else per_hour(s, "A"),
            gain_over_pretrained_random_per_node_hour=(
                None if s in ("A", "B") else per_hour(s, "B")
            ),
            n_seeds=len(seeds),
        )
        for s in COMPARISON_STRATEGIES
    ]
    comparisons = []
    for better, baseline in (
        ("B", "A"),
        ("C", "B"),
        ("D", "B"),
        ("D", "C"),
        ("E", "D"),
        ("E", "B"),
    ):
        diff = paired_difference(list(mae[baseline] - mae[better]), exp.confidence)
        comparisons.append(
            PairedComparison(
                better=better,
                baseline=baseline,
                mae_improvement=diff,
                significant=diff.ci_low > 0 and diff.mean >= min_gain * float(mae[baseline].mean()),
            )
        )
    winners = [c for c in comparisons if c.significant]
    best = min(outcomes, key=lambda o: (o.mae_mean, o.node_hours_mean, o.strategy))
    supported = {c.better for c in winners}
    if best.strategy in supported or (best.strategy == "A" and not winners):
        status, chosen = "evidence", best.strategy
        reason = f"{best.strategy} has the lowest held-out MAE; {len(winners)} paired claim(s) hold"
    else:
        status, chosen = NOT_YET_ESTABLISHED, None
        reason = "lowest-MAE strategy is not significantly better than its baseline"
    return StrategyComparison(
        experiment=exp.name,
        n_new=n_new,
        n_eval_rows=n_eval,
        outcomes=outcomes,
        comparisons=comparisons,
        status=status,
        best_strategy=chosen,
        reason=reason,
        results=results,
    )


def comparison_over_budgets(
    data: TransferData, exp: TransferExperiment, budgets: Sequence[int], **kw
) -> list[StrategyComparison]:
    return [compare_strategies(data, exp, n, **kw) for n in budgets]
