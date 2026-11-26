import math

import numpy as np
import pytest

from matsim_agents.science.defects.cascade_ensemble import (
    EnsembleError,
    Realization,
    Stratum,
    one_value_per_cascade,
    plan_stratified,
    stratified_estimate,
)
from matsim_agents.science.defects.ensemble import weighted_estimate

STRATA = [Stratum("low", 0.9), Stratum("high", 0.1, {"energy_eV": 1e5})]


def _draw(rng, n_low, n_high, mu=(2.0, 50.0), sd=(1.0, 5.0)):
    out = []
    for name, n, m, s in (("low", n_low, mu[0], sd[0]), ("high", n_high, mu[1], sd[1])):
        for i in range(n):
            out.append(
                Realization(
                    f"{name}-{i}", name, 1000 * (name == "high") + i, float(rng.normal(m, s))
                )
            )
    return out


def test_plan_has_reproducible_distinct_seeds_and_requires_two_per_stratum():
    plan = plan_stratified(STRATA, {"low": 5, "high": 5}, base_seed=3)
    assert plan == plan_stratified(STRATA, {"low": 5, "high": 5}, base_seed=3)
    assert len({p[2] for p in plan}) == 10
    assert plan != plan_stratified(STRATA, {"low": 5, "high": 5}, base_seed=4)
    with pytest.raises(EnsembleError, match="at least 2"):
        plan_stratified(STRATA, {"low": 5, "high": 1}, base_seed=3)
    with pytest.raises(EnsembleError, match="sum to 1"):
        plan_stratified([Stratum("a", 0.5)], {"a": 3}, base_seed=0)


def test_oversampled_rare_stratum_keeps_physical_weight():
    rng = np.random.default_rng(0)
    real = _draw(rng, 20, 20)  # high stratum is 50% of the samples but 10% of the physics
    est = stratified_estimate(STRATA, real)
    truth = 0.9 * 2.0 + 0.1 * 50.0
    assert abs(est.mean - truth) < 4 * est.standard_error
    naive = float(np.mean([r.value for r in real]))
    assert abs(naive - truth) > 10  # the unweighted mean is badly biased
    assert est.ci_low < est.mean < est.ci_high and est.unsampled_probability == 0.0


def test_coverage_of_interval_is_close_to_nominal():
    rng = np.random.default_rng(1)
    truth = 0.9 * 2.0 + 0.1 * 50.0
    hits = 0
    trials = 400
    for _ in range(trials):
        est = stratified_estimate(STRATA, _draw(rng, 12, 12))
        hits += est.ci_low <= truth <= est.ci_high
    assert 0.92 <= hits / trials <= 0.985


def test_reduces_to_ordinary_estimate_with_one_stratum():
    rng = np.random.default_rng(2)
    vals = rng.normal(5, 2, 15)
    one = [Stratum("all", 1.0)]
    est = stratified_estimate(
        one, [Realization(f"r{i}", "all", i, float(v)) for i, v in enumerate(vals)]
    )
    ref = weighted_estimate(vals, method="t")
    assert est.mean == pytest.approx(ref.mean) and est.standard_error == pytest.approx(
        ref.standard_error
    )
    assert est.ci_low == pytest.approx(ref.ci_low)


def test_correlated_frames_are_not_counted_as_realizations():
    frames = [("c1", t, 1.0 + t) for t in range(5)] + [("c2", t, 3.0 + t) for t in range(5)]
    vals, dropped = one_value_per_cascade(frames)
    assert vals == {"c1": 5.0, "c2": 7.0} and dropped == 8
    dup = [Realization("c1", "all", 1, 1.0), Realization("c1", "all", 2, 2.0)]
    with pytest.raises(EnsembleError, match="duplicate realization_id"):
        stratified_estimate([Stratum("all", 1.0)], dup)


def test_shared_seed_unknown_stratum_missing_stratum_and_nonfinite_are_refused():
    s1 = [Stratum("all", 1.0)]
    with pytest.raises(EnsembleError, match="share a seed"):
        stratified_estimate(s1, [Realization("a", "all", 1, 1.0), Realization("b", "all", 1, 2.0)])
    with pytest.raises(EnsembleError, match="unknown stratum"):
        stratified_estimate(s1, [Realization("a", "x", 1, 1.0), Realization("b", "x", 2, 1.0)])
    with pytest.raises(EnsembleError, match="at least 2"):
        stratified_estimate(STRATA, _draw(np.random.default_rng(0), 5, 1))
    with pytest.raises(EnsembleError, match="finite"):
        stratified_estimate(
            s1, [Realization("a", "all", 1, math.nan), Realization("b", "all", 2, 1.0)]
        )


def test_zero_probability_stratum_may_be_unsampled_and_is_reported():
    strata = [Stratum("a", 1.0), Stratum("rare", 0.0)]
    est = stratified_estimate(
        strata, [Realization("x", "a", 1, 1.0), Realization("y", "a", 2, 3.0)]
    )
    assert est.mean == pytest.approx(2.0) and "rare" not in est.per_stratum
    one = [Realization("x", "a", 1, 1.0), Realization("y", "a", 2, 3.0)]
    est = stratified_estimate(strata, [*one, Realization("z", "rare", 3, 9.0)])
    assert est.mean == pytest.approx(2.0) and math.isfinite(est.dof)


def test_negative_probabilities_are_refused_by_the_estimator():
    strata = [Stratum("a", 2.0), Stratum("b", -1.0)]
    real = [Realization(f"{s}{i}", s, i + 10 * (s == "b"), 1.0 + i) for s in "ab" for i in (0, 1)]
    with pytest.raises(EnsembleError, match="nonnegative"):
        stratified_estimate(strata, real)
