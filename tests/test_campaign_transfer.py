from __future__ import annotations

import numpy as np
import pytest

from matsim_agents.workflows.hierarchy.synthetic import SyntheticSpec, make_synthetic_hierarchy
from matsim_agents.workflows.hierarchy.transfer import (
    LeakageError,
    Split,
    TransferExperiment,
    check_leakage,
    evaluate_properties,
    paired_difference,
    run_transfer_experiment,
    split_experiment,
    standard_experiments,
)

SMALL = dict(
    elements=("Cr", "Mo", "Ta", "V", "W"),
    magnetic_element="Cr",
    orders=(2, 3, 4),
    compositions_per_system=4,
)


def _exp(**kw):
    base = dict(name="2->3", source_orders=(2,), target_order=3, n_new=(10,), min_eval_rows=10)
    base.update(kw)
    return TransferExperiment(**base)


@pytest.fixture(scope="module")
def good():
    return make_synthetic_hierarchy(SyntheticSpec(seed=1, **SMALL))


@pytest.fixture(scope="module")
def bad():
    return make_synthetic_hierarchy(
        SyntheticSpec(seed=1, mismatched_orders=(2,), mismatch=4.0, **SMALL)
    )


def test_standard_experiments_cover_the_hierarchy():
    names = [e.name for e in standard_experiments()]
    assert names == ["2->3", "2,3->4", "2,3,4->5", "2,3,4,5->6"]
    with pytest.raises(ValueError):
        TransferExperiment(name="x", source_orders=(3,), target_order=3)


def test_splits_hold_out_compositions_and_never_leak(good):
    exp = _exp()
    for seed in range(4):
        s = split_experiment(good, exp, seed)
        held = {good.composition_id[i] for i in s.evaluation}
        assert held and not held & {good.composition_id[i] for i in s.pool}
        assert not {good.system[i] for i in s.source} & {good.system[i] for i in s.pool}
        assert split_experiment(good, exp, seed).evaluation.tolist() == s.evaluation.tolist()


def test_leakage_checker_detects_each_kind_of_overlap(good):
    s = split_experiment(good, _exp(), 0)
    leaked_row = Split(s.source, np.append(s.pool, s.evaluation[0]), s.evaluation, s.reserve)
    with pytest.raises(LeakageError):
        check_leakage(good, leaked_row)
    sibling = next(
        i for i in s.pool if good.composition_id[i] != good.composition_id[s.evaluation[0]]
    )
    assert sibling is not None
    same_comp = [
        i
        for i in range(len(good))
        if good.composition_id[i] == good.composition_id[s.evaluation[0]]
        and i not in set(s.evaluation.tolist())
    ]
    if same_comp:
        with pytest.raises(LeakageError):
            check_leakage(
                good, Split(s.source, np.append(s.pool, same_comp[0]), s.evaluation, s.reserve)
            )
    with pytest.raises(LeakageError):
        check_leakage(good, Split(s.source, s.pool, s.evaluation, s.evaluation[:1]))


def test_reserve_is_disjoint_and_never_trainable(good):
    exp = _exp(reserve_fraction=0.2)
    s = split_experiment(good, exp, 0)
    assert len(s.reserve) > 0
    reserve_ids = {good.ids[i] for i in s.reserve}
    assert not reserve_ids & {good.ids[i] for i in np.concatenate([s.source, s.pool, s.evaluation])}


def test_useful_transfer_is_detected(good):
    result = run_transfer_experiment(good, _exp())
    verdict = result.verdicts[0]
    assert verdict.status == "useful_transfer" and verdict.registry_status == "validated"
    by = {r.strategy: r for r in result.summary}
    assert by["D"].mae_mean < by["C"].mae_mean
    assert set(by) == {"A", "B", "C", "D", "E"}
    assert by["A"].gain_per_calculation is None and by["A"].n_new == 10
    for k in "BCDE":
        assert np.isfinite(by[k].gain_per_calculation) and np.isfinite(by[k].gain_per_node_hour)
    assert by["D"].gain_per_calculation == pytest.approx((by["A"].mae_mean - by["D"].mae_mean) / 10)
    assert by["A"].node_hours_mean == 0 and by["E"].node_hours_mean > 0


def test_negative_transfer_is_detected_and_reported_unsupported(bad):
    verdict = run_transfer_experiment(bad, _exp(n_new=(60,))).verdicts[0]
    assert verdict.status == "negative_transfer"
    assert verdict.registry_status == "unsupported"
    assert verdict.status != "useful_transfer"


def test_insufficient_evidence_with_few_seeds_or_rows(good):
    few = run_transfer_experiment(good, _exp(seeds=(0, 1), min_seeds=3)).verdicts[0]
    assert few.status == "insufficient_evidence" and few.registry_status == "NOT_YET_ESTABLISHED"
    rows = run_transfer_experiment(good, _exp(min_eval_rows=10_000)).verdicts[0]
    assert rows.status == "insufficient_evidence"


def test_property_without_data_is_not_yet_established(good):
    out = evaluate_properties({"bulk": good}, _exp(), ["bulk", "barrier"])
    assert out["bulk"][0].status == "useful_transfer"
    assert out["barrier"][0].registry_status == "NOT_YET_ESTABLISHED"


def test_results_report_required_metrics(good):
    r = run_transfer_experiment(good, _exp(seeds=(0, 1, 2))).results[0]
    for field in (
        "mae",
        "rmse",
        "coverage_1sigma",
        "calibration_error",
        "target_fraction",
        "node_hours",
    ):
        assert getattr(r, field) >= 0
    assert r.mae_by_environment and r.mae_by_sro_regime and r.mae_by_magnetic


def test_same_seed_reproduces_exactly(good):
    a = run_transfer_experiment(good, _exp(seeds=(0, 1, 2)))
    b = run_transfer_experiment(good, _exp(seeds=(0, 1, 2)))
    assert [r.mae for r in a.results] == [r.mae for r in b.results]


def test_paired_difference_interval_and_degenerate_cases():
    d = paired_difference([1.0, 1.1, 0.9, 1.0], 0.95)
    assert d.ci_low > 0 and d.n_seeds == 4
    single = paired_difference([1.0], 0.95)
    assert single.ci_low == -np.inf and single.ci_high == np.inf
