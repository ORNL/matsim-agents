import numpy as np
import pytest
from scipy import stats

from matsim_agents.science.defects.ensemble import convergence_curve, weighted_estimate


def test_equal_weights_match_an_independent_t_interval():
    x = np.array([3.0, 5.0, 4.0, 6.0, 2.0, 5.0, 7.0, 4.0])
    est = weighted_estimate(x)
    low, high = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    assert est.mean == pytest.approx(x.mean())
    assert est.standard_error == pytest.approx(stats.sem(x))
    assert (est.ci_low, est.ci_high) == pytest.approx((low, high))
    assert est.effective_n == pytest.approx(len(x))


def test_weights_are_normalized_so_scaling_changes_nothing():
    x = [1.0, 4.0, 2.0, 8.0, 5.0]
    w = [1.0, 2.0, 0.5, 3.0, 1.5]
    base = weighted_estimate(x, w)
    scaled = weighted_estimate(x, [1000.0 * v for v in w])
    assert base == scaled


def test_weighting_moves_the_mean_to_the_weighted_average():
    x, w = [0.0, 10.0, 4.0], [1.0, 3.0, 1.0]
    est = weighted_estimate(x, w)
    assert est.mean == pytest.approx((0 * 1 + 10 * 3 + 4 * 1) / 5)
    assert est.effective_n == pytest.approx(25.0 / 11.0)
    assert est.mean != pytest.approx(np.mean(x))


def test_zero_weight_realization_does_not_contribute():
    x, w = [2.0, 4.0, 6.0, 1000.0], [1.0, 1.0, 1.0, 0.0]
    assert weighted_estimate(x, w).mean == pytest.approx(4.0)
    assert weighted_estimate(x, w).mean == pytest.approx(weighted_estimate(x[:3]).mean)


def test_bootstrap_is_reproducible_and_seed_dependent():
    x = np.random.default_rng(1).poisson(6.0, 40).astype(float)
    a = weighted_estimate(x, method="bootstrap", seed=5)
    assert a == weighted_estimate(x, method="bootstrap", seed=5)
    assert a != weighted_estimate(x, method="bootstrap", seed=6)
    assert a.seed == 5 and a.ci_low < a.mean < a.ci_high
    t = weighted_estimate(x)
    assert a.ci_high - a.ci_low == pytest.approx(t.ci_high - t.ci_low, rel=0.25)


def test_interval_coverage_is_near_nominal_for_independent_samples():
    rng = np.random.default_rng(7)
    truth, hits, trials = 5.0, 0, 600
    for _ in range(trials):
        sample = rng.normal(truth, 2.0, 20)
        est = weighted_estimate(sample)
        hits += est.ci_low <= truth <= est.ci_high
    assert 0.92 < hits / trials < 0.98


def test_half_width_shrinks_like_inverse_sqrt_of_ensemble_size():
    x = np.random.default_rng(3).normal(10.0, 3.0, 4000)
    curve = convergence_curve(x, [100, 400, 1600])
    widths = [e.ci_high - e.ci_low for e in curve]
    assert widths[0] > widths[1] > widths[2]
    assert widths[0] / widths[1] == pytest.approx(2.0, rel=0.2)
    assert widths[1] / widths[2] == pytest.approx(2.0, rel=0.2)
    assert [e.n for e in curve] == [100, 400, 1600]


def test_invalid_inputs_are_rejected():
    for bad in (
        lambda: weighted_estimate([1.0]),
        lambda: weighted_estimate([1.0, np.nan]),
        lambda: weighted_estimate([1.0, 2.0], [1.0, -1.0]),
        lambda: weighted_estimate([1.0, 2.0], [0.0, 0.0]),
        lambda: weighted_estimate([1.0, 2.0], [1.0, 0.0]),
        lambda: weighted_estimate([1.0, 2.0], [1.0]),
        lambda: weighted_estimate([1.0, 2.0], confidence=1.0),
        lambda: weighted_estimate([1.0, 2.0], method="bogus"),
        lambda: weighted_estimate([1.0, 2.0, 3.0], method="bootstrap", n_bootstrap=10),
        lambda: convergence_curve([1.0, 2.0, 3.0], [1]),
        lambda: convergence_curve([1.0, 2.0, 3.0], [5]),
    ):
        with pytest.raises(ValueError):
            bad()
