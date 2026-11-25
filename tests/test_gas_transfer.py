"""Gas/transmutation transfer-driven acquisition on deterministic synthetic data."""

import numpy as np
import pytest

from matsim_agents.workflows.candidate_registry import NOT_YET_ESTABLISHED
from matsim_agents.workflows.hierarchy.transfer import TransferData
from matsim_agents.workflows.radiation.gas_transfer import (
    CORE_FAMILIES,
    STRATEGY_IDS,
    SYNTHETIC_ONLY,
    FamilyData,
    GasTransferConfig,
    GasTransferError,
    assess_gas_transfer,
    fill_budget,
    format_assessment,
    inventory_existing_data,
    split_gas_experiment,
)

W = np.array([1.0, -2.0, 0.5])


def make(
    target_weights=W,
    n_core=60,
    n_target=40,
    noise=0.01,
    target_family="hydrogen",
    shared_groups=0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    rows = []
    for fam, n, weights in (
        ("original_alloy", n_core // 2, W),
        ("core_mechanism", n_core - n_core // 2, W),
        (target_family, n_target, np.asarray(target_weights)),
    ):
        for k in range(n):
            x = rng.normal(size=3)
            env = ("bulk", "gb")[k % 2]
            rows.append((fam, f"{fam}-{k}", x, float(x @ weights + noise * rng.normal()), env, k))
    fams, ids, X, y, env, k = zip(*rows, strict=True)
    n = len(rows)
    comp = tuple(f"c{i % shared_groups}" if shared_groups else f"c{i}" for i in range(n))
    order = tuple(f"o{i % shared_groups}" if shared_groups else f"o{i}" for i in range(n))
    cost = np.array([1.0 + 0.5 * (j % 3) for j in range(n)])
    data = TransferData(
        property="E_form",
        ids=ids,
        X=np.array(X),
        y=np.array(y),
        system=tuple(f"s{i}" for i in range(n)),
        composition_id=comp,
        ordering_id=order,
        environment=env,
        sro_regime=tuple("sro" for _ in range(n)),
        magnetic=tuple(("fm", "nm")[j % 2] for j in range(n)),
        cost_node_hours=cost,
    )
    return FamilyData(data=data, families=fams, synthetic=True)


def cfg(**kw):
    base = dict(
        name="test",
        target_family="hydrogen",
        budgets_node_hours=(8.0, 20.0),
        seeds=(0, 1, 2, 3, 4),
        min_eval_rows=5,
    )
    base.update(kw)
    return GasTransferConfig(**base)


def test_inventory_counts_families_and_names_missing():
    inv = inventory_existing_data(make(), expected=(*CORE_FAMILIES, "hydrogen", "helium"))
    by = {f.family: f for f in inv.families}
    assert by["original_alloy"].n_rows == 30 and by["hydrogen"].n_rows == 40
    assert by["helium"].n_rows == 0
    assert inv.missing_families == ["helium"]
    assert inv.total_rows == 100 and inv.synthetic
    assert any("NOT_YET_ESTABLISHED" in n for n in inv.notes)
    assert by["hydrogen"].environments == {"bulk": 20, "gb": 20}


def test_split_has_no_leakage_and_excludes_shared_groups():
    fd = make(shared_groups=12)
    split = split_gas_experiment(fd, cfg(), 0)
    held = {fd.group(i) for i in split.evaluation}
    assert all(fd.group(i) not in held for i in (*split.source, *split.pool, *split.reserve_pool))
    assert split.excluded_for_leakage > 0


def test_split_is_deterministic_and_separates_core_eval():
    fd = make()
    a, b = split_gas_experiment(fd, cfg(), 3), split_gas_experiment(fd, cfg(), 3)
    assert (a.evaluation == b.evaluation).all() and (a.pool == b.pool).all()
    assert not set(a.core_eval) & set(a.source) and not set(a.core_eval) & set(a.reserve_pool)
    assert set(fd.families[i] for i in a.core_eval) <= set(CORE_FAMILIES)


def test_fill_budget_never_exceeds_budget():
    costs = np.array([3.0, 1.0, 2.0, 1.0])
    chosen = fill_budget([0, 1, 2, 3], costs, 4.0)
    assert chosen == [0, 1] and costs[chosen].sum() <= 4.0
    assert fill_budget([0], costs, 1.0) == []


def test_matched_budget_all_strategies_stay_within_spendable():
    result = assess_gas_transfer(make(), cfg())
    assert result.qualification in (SYNTHETIC_ONLY, NOT_YET_ESTABLISHED) and result.budgets
    for b in result.budgets:
        assert b.reserve_node_hours == pytest.approx(0.2 * b.budget_node_hours)
        assert {o.strategy for o in b.outcomes} == set(STRATEGY_IDS)
        for o in b.outcomes:
            assert o.spent_node_hours_mean <= b.spendable_node_hours + 1e-9
            assert 0 < o.utilization_mean <= 1.0 + 1e-9
        assert b.reserve_spent_node_hours_mean <= b.reserve_node_hours + 1e-9
        assert b.reserve_funded


def test_transferable_target_shows_useful_transfer_with_value_per_node_hour():
    result = assess_gas_transfer(
        make(noise=0.05, n_target=60), cfg(budgets_node_hours=(6.0,), seeds=tuple(range(10)))
    )
    out = {o.strategy: o for o in result.budgets[0].outcomes}
    assert out["transfer_guided"].target_mae_mean < out["direct"].target_mae_mean
    assert out["transfer_guided"].verdict == "useful_transfer"
    assert out["transfer_guided"].transfer_value_per_node_hour > 0
    assert out["direct"].transfer_value_per_node_hour is None
    assert result.status == "useful_transfer_detected"
    assert result.qualification == SYNTHETIC_ONLY


def test_negative_transfer_is_detected_when_source_physics_differs():
    result = assess_gas_transfer(
        make(target_weights=-W, noise=0.01, n_core=200, n_target=60),
        cfg(budgets_node_hours=(12.0,)),
    )
    out = {o.strategy: o for o in result.budgets[0].outcomes}
    assert out["random"].verdict == "negative_transfer"
    assert out["random"].transfer_value_per_node_hour < 0
    assert result.status == "negative_transfer_detected"
    assert out["random"].negative_transfer_groups


def test_negative_transfer_strategies_are_not_recommended():
    result = assess_gas_transfer(
        make(target_weights=-W, n_core=200, n_target=60), cfg(budgets_node_hours=(12.0,))
    )
    budget = result.budgets[0]
    assert budget.best_by_mae == "direct"
    assert budget.best_by_value_per_node_hour in (None, "direct") or all(
        o.verdict != "negative_transfer"
        for o in budget.outcomes
        if o.strategy == budget.best_by_value_per_node_hour
    )


def test_missing_target_family_is_not_yet_established():
    result = assess_gas_transfer(make(), cfg(target_family="helium"))
    assert result.status == NOT_YET_ESTABLISHED and result.budgets == []
    assert "no data for target family" in result.reason


def test_too_few_seeds_or_eval_rows_gives_no_claim():
    few_seeds = assess_gas_transfer(make(), cfg(seeds=(0, 1)))
    assert few_seeds.status == NOT_YET_ESTABLISHED and "no transfer claim" in few_seeds.reason
    few_rows = assess_gas_transfer(make(n_target=8), cfg(min_eval_rows=50))
    assert few_rows.status == NOT_YET_ESTABLISHED


def test_budget_too_small_funds_nothing():
    result = assess_gas_transfer(make(), cfg(budgets_node_hours=(0.5,)))
    assert result.status == NOT_YET_ESTABLISHED
    assert result.budgets[0].outcomes == [] and "funds no target row" in result.budgets[0].reason


def test_assessment_is_deterministic():
    one = assess_gas_transfer(make(), cfg())
    two = assess_gas_transfer(make(), cfg())
    assert one == two


def test_config_rejects_target_in_sources_and_bad_budgets():
    with pytest.raises(ValueError, match="target family"):
        cfg(target_family="original_alloy")
    with pytest.raises(ValueError, match="positive node-hours"):
        cfg(budgets_node_hours=(0.0,))
    with pytest.raises(ValueError, match="distinct"):
        cfg(seeds=(1, 1, 2))


def test_family_data_must_align():
    fd = make()
    with pytest.raises(ValueError, match="align"):
        FamilyData(data=fd.data, families=fd.families[:-1])


def test_format_lists_every_strategy_and_flags():
    text = format_assessment(assess_gas_transfer(make(), cfg(budgets_node_hours=(8.0,))))
    for name in STRATEGY_IDS:
        assert name in text
    assert "SYNTHETIC_ONLY" in text and "reserve_funded=True" in text
    _ = GasTransferError
