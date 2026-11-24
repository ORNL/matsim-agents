"""Transfer-driven acquisition for gas and transmutation configurations.

Source data (the original alloy and core mechanisms) may or may not help a target family
(hydrogen, helium, transmutation, gas-vacancy clusters). This module measures that instead of
assuming it:

* ``inventory_existing_data`` counts what exists per family and names what is missing.
* ``assess_gas_transfer`` compares five strategies at matched node-hour budgets on identical
  held-out target rows, with paired confidence intervals over seeds:
  ``direct`` (target data only, no transfer), ``random``, ``stratified``, ``uncertainty_only``
  and ``transfer_guided`` (uncertainty x distance x relevance).
* A fraction of every budget is held back for the original alloy and core mechanisms. It cannot
  be spent on the target family, and the held-out core rows are scored so that a regression on
  the core is reported.
* Negative transfer is flagged overall and per environment group. Transfer value is reported
  per node-hour spent.

Verdicts need enough seeds and held-out rows; otherwise they are ``NOT_YET_ESTABLISHED``.
Results on synthetic data carry the qualification ``SYNTHETIC_ONLY`` and say nothing about the
real alloy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field, model_validator

from matsim_agents.campaign.registry import NOT_YET_ESTABLISHED
from matsim_agents.campaign.strategies import _stratified, default_relevance
from matsim_agents.campaign.transfer import (
    PairedDifference,
    RidgeEnsemble,
    TransferData,
    _group_mae,
    paired_difference,
    uncertainty_diversity_selector,
)
from matsim_agents.seeding import numpy_rng

CORE_FAMILIES = ("original_alloy", "core_mechanism")
GAS_FAMILIES = ("transmutation", "hydrogen", "helium", "gas_vacancy_cluster")
STRATEGY_IDS = ("direct", "random", "stratified", "uncertainty_only", "transfer_guided")
SYNTHETIC_ONLY = "SYNTHETIC_ONLY"


class GasTransferError(ValueError):
    """The assessment cannot be run on the supplied data."""


@dataclass(frozen=True, eq=False)
class FamilyData:
    """``TransferData`` rows labelled with the family each configuration belongs to."""

    data: TransferData
    families: tuple[str, ...]
    synthetic: bool = False

    def __post_init__(self) -> None:
        if len(self.families) != len(self.data):
            raise ValueError("families must align with the data rows")

    def rows(self, families: Sequence[str]) -> np.ndarray:
        wanted = set(families)
        return np.array([i for i, f in enumerate(self.families) if f in wanted], dtype=int)

    def group(self, i: int) -> tuple[str, str]:
        return (self.data.composition_id[i], self.data.ordering_id[i])


# --------------------------------------------------------------------------- #
# Inventory                                                                   #
# --------------------------------------------------------------------------- #


class FamilyInventory(BaseModel):
    family: str
    n_rows: int
    n_compositions: int
    n_parent_orderings: int
    node_hours: float
    environments: dict[str, int]
    magnetic: dict[str, int]


class DataInventory(BaseModel):
    property: str
    families: list[FamilyInventory]
    total_rows: int
    total_node_hours: float
    missing_families: list[str]
    synthetic: bool
    notes: list[str]


def inventory_existing_data(
    fd: FamilyData, expected: Sequence[str] = (*CORE_FAMILIES, *GAS_FAMILIES)
) -> DataInventory:
    """Rows, cost and coverage per family; an expected family with no rows is listed missing."""
    data = fd.data
    out: list[FamilyInventory] = []
    for family in sorted({*fd.families, *expected}):
        rows = fd.rows([family])
        out.append(
            FamilyInventory(
                family=family,
                n_rows=len(rows),
                n_compositions=len({data.composition_id[i] for i in rows}),
                n_parent_orderings=len({fd.group(i) for i in rows}),
                node_hours=float(data.cost_node_hours[rows].sum()) if len(rows) else 0.0,
                environments=dict(sorted(Counter(data.environment[i] for i in rows).items())),
                magnetic=dict(sorted(Counter(data.magnetic[i] for i in rows).items())),
            )
        )
    missing = sorted(f.family for f in out if f.n_rows == 0 and f.family in set(expected))
    notes = []
    if missing:
        notes.append(f"no data for: {', '.join(missing)}; transfer there is NOT_YET_ESTABLISHED")
    if fd.synthetic:
        notes.append("synthetic data: software test only")
    return DataInventory(
        property=data.property,
        families=out,
        total_rows=len(data),
        total_node_hours=float(data.cost_node_hours.sum()) if len(data) else 0.0,
        missing_families=missing,
        synthetic=fd.synthetic,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Configuration and split                                                     #
# --------------------------------------------------------------------------- #


class GasTransferConfig(BaseModel):
    name: str = Field(min_length=1)
    target_family: str = Field(min_length=1)
    source_families: tuple[str, ...] = CORE_FAMILIES
    core_families: tuple[str, ...] = CORE_FAMILIES
    budgets_node_hours: tuple[float, ...]
    reserve_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    eval_fraction: float = Field(0.25, gt=0.0, lt=1.0)
    core_eval_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    reserve_pool_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    min_seeds: int = Field(3, ge=2)
    min_eval_rows: int = Field(10, ge=1)
    min_relative_gain: float = Field(0.05, ge=0.0)
    confidence: float = Field(0.95, gt=0.5, lt=1.0)
    alpha: float = Field(1e-3, gt=0.0)
    core_regression_tolerance: float = Field(0.1, ge=0.0)

    @model_validator(mode="after")
    def _valid(self) -> GasTransferConfig:
        if self.target_family in self.source_families or self.target_family in self.core_families:
            raise ValueError("the target family must not be a source or core family")
        if not self.budgets_node_hours or any(b <= 0 for b in self.budgets_node_hours):
            raise ValueError("budgets must be positive node-hours")
        if not self.core_families:
            raise ValueError("a reserve needs at least one core family")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        return self


@dataclass(frozen=True)
class GasSplit:
    source: np.ndarray
    pool: np.ndarray
    evaluation: np.ndarray
    reserve_pool: np.ndarray
    core_eval: np.ndarray
    excluded_for_leakage: int


def _split_groups(
    fd: FamilyData, rows: np.ndarray, fractions: Sequence[float], rng: np.random.Generator
) -> list[np.ndarray]:
    """Split ``rows`` by parent-ordering group into len(fractions)+1 disjoint parts."""
    groups = sorted({fd.group(i) for i in rows})
    shuffled = [groups[k] for k in rng.permutation(len(groups))]
    parts: list[set] = []
    start = 0
    for frac in fractions:
        n = max(1, round(len(groups) * frac)) if groups else 0
        parts.append(set(shuffled[start : start + n]))
        start += n
    parts.append(set(shuffled[start:]))
    return [np.array([i for i in rows if fd.group(i) in p], dtype=int) for p in parts]


def split_gas_experiment(fd: FamilyData, cfg: GasTransferConfig, seed: int) -> GasSplit:
    rng = numpy_rng(seed, "gas-transfer-split", cfg.name, fd.data.property)
    target = fd.rows([cfg.target_family])
    evaluation, pool = _split_groups(fd, target, [cfg.eval_fraction], rng)
    core = fd.rows(cfg.core_families)
    core_eval, reserve_pool, core_source = _split_groups(
        fd, core, [cfg.core_eval_fraction, cfg.reserve_pool_fraction], rng
    )
    other = [
        i
        for i in fd.rows(cfg.source_families)
        if fd.families[i] not in set(cfg.core_families) and fd.families[i] != cfg.target_family
    ]
    source_rows = np.array(sorted({*core_source.tolist(), *other}), dtype=int)
    held_target = {fd.group(i) for i in evaluation}
    held_core = {fd.group(i) for i in core_eval}
    held = held_target | held_core
    keep = [i for i in source_rows if fd.group(i) not in held]
    keep_pool = [i for i in pool if fd.group(i) not in held_core]
    keep_reserve = [i for i in reserve_pool if fd.group(i) not in held]
    excluded = (
        (len(source_rows) - len(keep))
        + (len(pool) - len(keep_pool))
        + (len(reserve_pool) - len(keep_reserve))
    )
    split = GasSplit(
        source=np.array(keep, dtype=int),
        pool=np.array(keep_pool, dtype=int),
        evaluation=evaluation,
        reserve_pool=np.array(keep_reserve, dtype=int),
        core_eval=core_eval,
        excluded_for_leakage=excluded,
    )
    _check_split(fd, split)
    return split


def _check_split(fd: FamilyData, split: GasSplit) -> None:
    trainable = [*split.source, *split.pool, *split.reserve_pool]
    for name, held in (("evaluation", split.evaluation), ("core_eval", split.core_eval)):
        groups = {fd.group(i) for i in held}
        ids = {fd.data.ids[i] for i in held}
        for i in trainable:
            if fd.data.ids[i] in ids or fd.group(i) in groups:
                raise GasTransferError(f"{name} leaks into trainable row {fd.data.ids[i]!r}")


# --------------------------------------------------------------------------- #
# Matched-budget acquisition                                                  #
# --------------------------------------------------------------------------- #


def fill_budget(order: Sequence[int], costs: np.ndarray, budget: float) -> list[int]:
    """Walk ``order`` and take every row that still fits; positions index ``costs``."""
    chosen: list[int] = []
    remaining = budget
    for k in order:
        c = float(costs[k])
        if c <= remaining + 1e-12:
            chosen.append(int(k))
            remaining -= c
    return chosen


def _orders(
    fd: FamilyData,
    split: GasSplit,
    pretrained: RidgeEnsemble,
    seed: int,
    cfg: GasTransferConfig,
    spendable: float,
) -> dict[str, list[int]]:
    data, pool = fd.data, split.pool
    costs = data.cost_node_hours[pool]
    rng = numpy_rng(seed, "gas-transfer-order", cfg.name, data.property)
    random_order = [int(k) for k in rng.permutation(len(pool))]
    stratified = _stratified(data, pool, len(pool), numpy_rng(seed, "gas-strata", cfg.name))
    _, sigma = pretrained.predict(data.X[pool])
    unc = sorted(range(len(pool)), key=lambda k: (-float(sigma[k]), k))
    positive = costs[costs > 0]
    cap = len(pool) if len(positive) < len(costs) else min(len(pool), int(spendable / costs.min()))
    guided = uncertainty_diversity_selector(
        pretrained,
        data.X[pool],
        costs,
        data.X[split.source],
        max(cap, 1),
        weights=default_relevance(data)[pool],
    )
    return {
        "direct": random_order,
        "random": random_order,
        "stratified": [int(k) for k in stratified],
        "uncertainty_only": unc,
        "transfer_guided": [int(k) for k in guided],
    }


class _Run(BaseModel):
    seed: int
    strategy: str
    budget: float
    spent: float
    n_selected: int
    target_mae: float
    core_mae: float | None
    group_mae: dict[str, float]
    reserve_spent: float
    reserve_n: int


def _mae(model: RidgeEnsemble, data: TransferData, rows: np.ndarray) -> tuple[float, np.ndarray]:
    pred, _ = model.predict(data.X[rows])
    err = np.abs(pred - data.y[rows])
    return float(err.mean()), err


# --------------------------------------------------------------------------- #
# Result records                                                              #
# --------------------------------------------------------------------------- #


class StrategyBudgetOutcome(BaseModel):
    strategy: str
    budget_node_hours: float
    n_seeds: int
    target_mae_mean: float
    target_mae_std: float
    core_mae_mean: float | None
    spent_node_hours_mean: float
    utilization_mean: float
    n_selected_mean: float
    gain_vs_direct: PairedDifference | None
    transfer_value_per_node_hour: float | None
    gain_vs_random: PairedDifference | None
    verdict: str
    reason: str
    negative_transfer_groups: list[str]
    core_regression: bool


class BudgetAssessment(BaseModel):
    budget_node_hours: float
    reserve_node_hours: float
    spendable_node_hours: float
    reserve_spent_node_hours_mean: float
    reserve_funded: bool
    n_eval_rows: int
    outcomes: list[StrategyBudgetOutcome]
    best_by_mae: str | None
    best_by_value_per_node_hour: str | None
    status: str
    reason: str


class GasTransferAssessment(BaseModel):
    config: GasTransferConfig
    inventory: DataInventory
    budgets: list[BudgetAssessment]
    status: str
    qualification: str
    reason: str
    notes: list[str]


def _classify(diff: PairedDifference, reference: float, margin: float) -> tuple[str, str]:
    threshold = margin * reference
    if diff.ci_low > threshold:
        return "useful_transfer", "held-out MAE falls beyond the margin"
    if diff.ci_high < 0:
        return "negative_transfer", "held-out MAE rises (confidence interval below zero)"
    if diff.ci_low <= 0 <= diff.ci_high or diff.mean <= threshold:
        return "no_detectable_transfer", "difference is within noise or the margin"
    return "insufficient_evidence", "difference positive but the interval crosses the margin"


# --------------------------------------------------------------------------- #
# Assessment                                                                  #
# --------------------------------------------------------------------------- #


def _not_established(cfg: GasTransferConfig, inv: DataInventory, reason: str):
    return GasTransferAssessment(
        config=cfg,
        inventory=inv,
        budgets=[],
        status=NOT_YET_ESTABLISHED,
        qualification=NOT_YET_ESTABLISHED,
        reason=reason,
        notes=[reason],
    )


def assess_gas_transfer(fd: FamilyData, cfg: GasTransferConfig) -> GasTransferAssessment:
    """Run the five strategies at every budget on identical held-out rows."""
    inv = inventory_existing_data(fd, expected=(*cfg.core_families, cfg.target_family))
    if cfg.target_family in inv.missing_families:
        return _not_established(cfg, inv, f"no data for target family {cfg.target_family!r}")
    if any(f in inv.missing_families for f in cfg.core_families):
        return _not_established(cfg, inv, "a core family has no data, so no reserve can be held")

    data = fd.data
    runs: dict[tuple[float, str], list[_Run]] = defaultdict(list)
    source_core: dict[int, float | None] = {}
    n_eval = 0
    usable_seeds: list[int] = []
    for seed in cfg.seeds:
        split = split_gas_experiment(fd, cfg, seed)
        n_eval = len(split.evaluation)
        if min(len(split.source), len(split.pool), n_eval) == 0:
            continue
        usable_seeds.append(seed)
        pretrained = RidgeEnsemble(alpha=cfg.alpha, seed=seed, tag="gas-pretrain").fit(
            data.X[split.source], data.y[split.source]
        )
        source_core[seed] = (
            _mae(pretrained, data, split.core_eval)[0] if len(split.core_eval) else None
        )
        for budget in cfg.budgets_node_hours:
            reserve_budget = cfg.reserve_fraction * budget
            spendable = budget - reserve_budget
            reserve_rows = np.array([], dtype=int)
            if len(split.reserve_pool):
                order = _stratified(
                    data,
                    split.reserve_pool,
                    len(split.reserve_pool),
                    numpy_rng(seed, "gas-reserve", cfg.name),
                )
                picked = fill_budget(
                    order, data.cost_node_hours[split.reserve_pool], reserve_budget
                )
                reserve_rows = split.reserve_pool[picked]
            reserve_spent = float(data.cost_node_hours[reserve_rows].sum())
            orders = _orders(fd, split, pretrained, seed, cfg, spendable)
            pool_costs = data.cost_node_hours[split.pool]
            for strategy in STRATEGY_IDS:
                chosen = fill_budget(orders[strategy], pool_costs, spendable)
                if not chosen:
                    continue
                target_rows = split.pool[chosen]
                train = (
                    target_rows
                    if strategy == "direct"
                    else np.concatenate([split.source, target_rows, reserve_rows])
                )
                model = RidgeEnsemble(alpha=cfg.alpha, seed=seed, tag=strategy).fit(
                    data.X[train], data.y[train]
                )
                mae, err = _mae(model, data, split.evaluation)
                core = (
                    _mae(model, data, split.core_eval)[0]
                    if len(split.core_eval) and strategy != "direct"
                    else None
                )
                runs[(budget, strategy)].append(
                    _Run(
                        seed=seed,
                        strategy=strategy,
                        budget=budget,
                        spent=float(pool_costs[chosen].sum()),
                        n_selected=len(chosen),
                        target_mae=mae,
                        core_mae=core,
                        group_mae=_group_mae([data.environment[i] for i in split.evaluation], err),
                        reserve_spent=reserve_spent,
                        reserve_n=len(reserve_rows),
                    )
                )

    notes: list[str] = []
    if fd.synthetic:
        notes.append("synthetic data: results qualify the software only")
    if len(usable_seeds) < cfg.min_seeds or n_eval < cfg.min_eval_rows:
        reason = (
            f"{len(usable_seeds)} usable seed(s) (need {cfg.min_seeds}) and {n_eval} held-out "
            f"row(s) (need {cfg.min_eval_rows}); no transfer claim is made"
        )
        out = _not_established(cfg, inv, reason)
        return out.model_copy(update={"notes": [*out.notes, *notes]})

    budgets = [
        _assess_budget(cfg, budget, runs, usable_seeds, source_core, n_eval)
        for budget in cfg.budgets_node_hours
    ]
    verdicts = {o.verdict for b in budgets for o in b.outcomes if o.strategy != "direct"}
    status = (
        "negative_transfer_detected"
        if "negative_transfer" in verdicts
        else "useful_transfer_detected"
        if "useful_transfer" in verdicts
        else NOT_YET_ESTABLISHED
    )
    qualification = SYNTHETIC_ONLY if fd.synthetic else status
    return GasTransferAssessment(
        config=cfg,
        inventory=inv,
        budgets=budgets,
        status=status,
        qualification=qualification,
        reason="paired seeds on identical held-out target rows at matched node-hour budgets",
        notes=notes,
    )


def _paired(a: dict[int, float], b: dict[int, float], confidence: float) -> PairedDifference:
    seeds = sorted(set(a) & set(b))
    return paired_difference([a[s] - b[s] for s in seeds], confidence)


def _assess_budget(
    cfg: GasTransferConfig,
    budget: float,
    runs: dict[tuple[float, str], list[_Run]],
    seeds: list[int],
    source_core: dict[int, float | None],
    n_eval: int,
) -> BudgetAssessment:
    by = {s: {r.seed: r for r in runs.get((budget, s), [])} for s in STRATEGY_IDS}
    reserve = cfg.reserve_fraction * budget
    spendable = budget - reserve
    if not by["direct"] or not by["random"]:
        return BudgetAssessment(
            budget_node_hours=budget,
            reserve_node_hours=reserve,
            spendable_node_hours=spendable,
            reserve_spent_node_hours_mean=0.0,
            reserve_funded=False,
            n_eval_rows=n_eval,
            outcomes=[],
            best_by_mae=None,
            best_by_value_per_node_hour=None,
            status=NOT_YET_ESTABLISHED,
            reason="budget funds no target row; no comparison is possible",
        )
    mae = {s: {k: r.target_mae for k, r in by[s].items()} for s in STRATEGY_IDS}
    outcomes: list[StrategyBudgetOutcome] = []
    for s in STRATEGY_IDS:
        rows = list(by[s].values())
        if not rows:
            continue
        values = [r.target_mae for r in rows]
        spent = float(np.mean([r.spent for r in rows]))
        gain = None if s == "direct" else _paired(mae["direct"], mae[s], cfg.confidence)
        vs_random = (
            None if s in ("direct", "random") else _paired(mae["random"], mae[s], cfg.confidence)
        )
        if gain is None:
            verdict, reason = "reference", "direct target-only reference"
        elif len(gain_seeds := sorted(set(mae["direct"]) & set(mae[s]))) < cfg.min_seeds:
            verdict, reason = "insufficient_evidence", f"{len(gain_seeds)} paired seeds"
        else:
            verdict, reason = _classify(
                gain, float(np.mean(list(mae["direct"].values()))), cfg.min_relative_gain
            )
        core = [r.core_mae for r in rows if r.core_mae is not None]
        core_ref = [source_core[r.seed] for r in rows if r.core_mae is not None]
        regression = bool(
            core
            and s != "direct"
            and np.mean(core)
            > np.mean([c for c in core_ref if c is not None])
            * (1.0 + cfg.core_regression_tolerance)
        )
        outcomes.append(
            StrategyBudgetOutcome(
                strategy=s,
                budget_node_hours=budget,
                n_seeds=len(rows),
                target_mae_mean=float(np.mean(values)),
                target_mae_std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                core_mae_mean=float(np.mean(core)) if core else None,
                spent_node_hours_mean=spent,
                utilization_mean=spent / spendable,
                n_selected_mean=float(np.mean([r.n_selected for r in rows])),
                gain_vs_direct=gain,
                transfer_value_per_node_hour=(
                    None if gain is None or spent <= 0 else gain.mean / spent
                ),
                gain_vs_random=vs_random,
                verdict=verdict,
                reason=reason,
                negative_transfer_groups=(
                    [] if s == "direct" else _negative_groups(cfg, by["direct"], by[s])
                ),
                core_regression=regression,
            )
        )
    eligible = [o for o in outcomes if o.verdict != "negative_transfer"]
    best_mae = min(eligible, key=lambda o: (o.target_mae_mean, o.strategy), default=None)
    valued = [o for o in eligible if o.transfer_value_per_node_hour is not None]
    best_value = max(
        valued, key=lambda o: (o.transfer_value_per_node_hour or 0.0, o.strategy), default=None
    )
    guided = next((o for o in outcomes if o.strategy == "transfer_guided"), None)
    supported = bool(
        guided
        and guided.gain_vs_random is not None
        and guided.gain_vs_random.ci_low
        > cfg.min_relative_gain * float(np.mean(list(mae["random"].values())))
    )
    return BudgetAssessment(
        budget_node_hours=budget,
        reserve_node_hours=reserve,
        spendable_node_hours=spendable,
        reserve_spent_node_hours_mean=float(
            np.mean([r.reserve_spent for r in by["random"].values()])
        ),
        reserve_funded=any(r.reserve_n > 0 for r in by["random"].values()),
        n_eval_rows=n_eval,
        outcomes=outcomes,
        best_by_mae=best_mae.strategy if best_mae else None,
        best_by_value_per_node_hour=best_value.strategy if best_value else None,
        status="guided_acquisition_supported" if supported else NOT_YET_ESTABLISHED,
        reason=(
            "transfer-guided beats random beyond the margin"
            if supported
            else "transfer-guided acquisition is not shown to beat random at this budget"
        ),
    )


def _negative_groups(
    cfg: GasTransferConfig, direct: dict[int, _Run], other: dict[int, _Run]
) -> list[str]:
    per_group: dict[str, list[float]] = defaultdict(list)
    for seed in sorted(set(direct) & set(other)):
        for group, value in direct[seed].group_mae.items():
            if group in other[seed].group_mae:
                per_group[group].append(value - other[seed].group_mae[group])
    flagged = []
    for group, diffs in sorted(per_group.items()):
        if len(diffs) >= cfg.min_seeds and paired_difference(diffs, cfg.confidence).ci_high < 0:
            flagged.append(group)
    return flagged


def format_assessment(result: GasTransferAssessment) -> str:
    """Plain-text table of every strategy at every budget."""
    lines = [f"status={result.status} qualification={result.qualification}"]
    for b in result.budgets:
        lines.append(
            f"budget={b.budget_node_hours:g} nh reserve={b.reserve_node_hours:g} "
            f"spendable={b.spendable_node_hours:g} reserve_funded={b.reserve_funded} "
            f"best_mae={b.best_by_mae} best_value={b.best_by_value_per_node_hour}"
        )
        for o in b.outcomes:
            value = (
                "-"
                if o.transfer_value_per_node_hour is None
                else f"{o.transfer_value_per_node_hour:.3g}"
            )
            lines.append(
                f"  {o.strategy:<17} mae={o.target_mae_mean:.4g} "
                f"spent={o.spent_node_hours_mean:.3g} "
                f"value/nh={value} verdict={o.verdict}"
                + (
                    " NEGATIVE_GROUPS=" + ",".join(o.negative_transfer_groups)
                    if o.negative_transfer_groups
                    else ""
                )
                + (" CORE_REGRESSION" if o.core_regression else "")
            )
    return "\n".join(lines)


__all__ = [
    "CORE_FAMILIES",
    "GAS_FAMILIES",
    "STRATEGY_IDS",
    "SYNTHETIC_ONLY",
    "BudgetAssessment",
    "DataInventory",
    "FamilyData",
    "GasSplit",
    "GasTransferAssessment",
    "GasTransferConfig",
    "GasTransferError",
    "StrategyBudgetOutcome",
    "assess_gas_transfer",
    "fill_budget",
    "format_assessment",
    "inventory_existing_data",
    "split_gas_experiment",
]
