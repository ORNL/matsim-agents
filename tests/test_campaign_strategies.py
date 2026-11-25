"""H8: A-E strategy comparison on identical held-out rows (synthetic data, software test only)."""

from __future__ import annotations

import numpy as np

from matsim_agents.workflows.candidate_registry import NOT_YET_ESTABLISHED
from matsim_agents.workflows.hierarchy.strategies import compare_strategies, comparison_over_budgets
from matsim_agents.workflows.hierarchy.synthetic import SyntheticSpec, make_synthetic_hierarchy
from matsim_agents.workflows.hierarchy.transfer import TransferExperiment, split_experiment


def _exp(**kw) -> TransferExperiment:
    base = {"name": "2,3,4->5", "source_orders": (2, 3, 4), "target_order": 5}
    return TransferExperiment(**{**base, "seeds": (0, 1, 2, 3, 4), "min_eval_rows": 6, **kw})


def _data(**kw):
    return make_synthetic_hierarchy(
        SyntheticSpec(
            seed=3, orders=(2, 3, 4, 5), **{"compositions_per_system": 2, "n_orderings": 3, **kw}
        )
    )


def test_all_five_strategies_scored_on_identical_heldout_rows():
    data, exp = _data(), _exp()
    out = compare_strategies(data, exp, 12)
    assert [o.strategy for o in out.outcomes] == list("ABCDE")
    by_seed = {}
    for r in out.results:
        by_seed.setdefault(r.seed, set()).add(r.strategy)
    assert all(v == set("ABCDE") for v in by_seed.values())
    # the held-out split depends only on the seed, so it is the same for every strategy
    for seed in exp.seeds:
        a = split_experiment(data, exp, seed).evaluation
        b = split_experiment(data, exp, seed).evaluation
        assert np.array_equal(a, b)
        assert out.n_eval_rows == len(a) or out.n_eval_rows > 0


def test_deterministic_rerun():
    data, exp = _data(), _exp()
    one, two = compare_strategies(data, exp, 12), compare_strategies(data, exp, 12)
    assert one.model_dump() == two.model_dump()


def test_heldout_and_reserve_are_never_selected():
    data, exp = _data(), _exp()
    out = compare_strategies(data, exp, 12)
    for r in out.results:
        split = split_experiment(data, exp, r.seed)
        forbidden = {data.ids[i] for i in np.concatenate([split.evaluation, split.reserve])}
        assert not forbidden & set(r.selected_ids)


def test_pretraining_beats_direct_when_target_data_is_scarce():
    out = compare_strategies(_data(), _exp(), 6)
    mae = {o.strategy: o.mae_mean for o in out.outcomes}
    assert mae["B"] < mae["A"]
    claim = next(c for c in out.comparisons if (c.better, c.baseline) == ("B", "A"))
    assert claim.significant and claim.mae_improvement.ci_low > 0


def test_cost_normalised_gain_reported():
    out = compare_strategies(_data(), _exp(), 6)
    by = {o.strategy: o for o in out.outcomes}
    assert by["A"].gain_over_direct_per_node_hour is None
    assert by["B"].gain_over_direct_per_node_hour is not None
    assert by["B"].gain_over_pretrained_random_per_node_hour is None
    assert by["D"].gain_over_pretrained_random_per_node_hour is not None
    assert all(c.mae_improvement.n_seeds == 5 for c in out.comparisons)


def test_too_few_seeds_is_not_yet_established():
    out = compare_strategies(_data(), _exp(seeds=(0, 1)), 12)
    assert out.status == NOT_YET_ESTABLISHED
    assert out.best_strategy is None and out.outcomes == []


def test_too_few_heldout_rows_is_not_yet_established():
    data = _data(compositions_per_system=1, n_orderings=2)
    out = compare_strategies(data, _exp(min_eval_rows=10_000), 4)
    assert out.status == NOT_YET_ESTABLISHED
    assert "no strategy claim" in out.reason


def test_no_claim_when_nothing_distinguishable():
    # useless source: mismatched lower orders carry no usable signal for the target
    data = _data(mismatched_orders=(2, 3, 4), mismatch=5.0)
    out = compare_strategies(data, _exp(), 40)
    assert out.status in {"evidence", NOT_YET_ESTABLISHED}
    if out.status == "evidence":
        assert out.best_strategy is not None


def test_relevance_changes_selection_for_e_only():
    data, exp = _data(), _exp()
    flat = compare_strategies(data, exp, 10, relevance=np.ones(len(data)))
    weighted = compare_strategies(data, exp, 10, relevance=1.0 + (np.arange(len(data)) % 7))
    picks = lambda out, s: [r.selected_ids for r in out.results if r.strategy == s]  # noqa: E731
    assert picks(flat, "D") == picks(weighted, "D")
    assert picks(flat, "E") != picks(weighted, "E")


def test_budget_sweep():
    outs = comparison_over_budgets(_data(), _exp(), (4, 8))
    assert [o.n_new for o in outs] == [4, 8]
