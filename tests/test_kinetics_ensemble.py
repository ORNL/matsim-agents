"""KMC ensembles (synthetic networks)."""

import math

import pytest

from matsim_agents.kinetics.ensemble import run_ensemble, wilson_interval
from matsim_agents.kinetics.graph import TransitionNetwork
from tests.test_kinetics_graph import PROV, _state
from tests.test_kinetics_kmc import _decay, _edge


def _ens(net, start, **kw):
    kw.setdefault("require", "exploratory")
    return run_ensemble(net, start.state_id, 300.0, **kw)


def test_ensemble_is_reproducible_and_seed_dependent():
    net, a, _ = _decay(2.0)
    one = _ens(net, a, n_trajectories=50, seed=3)
    assert one == _ens(net, a, n_trajectories=50, seed=3)
    assert (
        one.first_passage_times_s != _ens(net, a, n_trajectories=50, seed=4).first_passage_times_s
    )


def test_first_passage_mean_matches_analytic_and_outcome_probability_is_one():
    net, a, _ = _decay(2.0)
    res = _ens(net, a, n_trajectories=3000, seed=11)
    assert res.mean_first_passage_s == pytest.approx(0.5, rel=0.06)
    assert res.median_first_passage_s == pytest.approx(math.log(2) / 2.0, rel=0.08)
    assert res.outcomes["recombined"].probability == 1.0
    assert res.n_censored == 0 and not res.first_passage_biased and not res.warnings


def test_horizon_censoring_keeps_denominator_and_flags_bias():
    net, a, _ = _decay(1.0)
    res = _ens(net, a, n_trajectories=2000, seed=5, horizon_s=1.0)
    expected = 1 - math.exp(-1.0)
    prob = res.outcomes["recombined"]
    assert prob.probability == pytest.approx(expected, abs=0.03)
    assert prob.wilson_low <= expected <= prob.wilson_high
    assert res.n_censored == res.n_trajectories - prob.count
    assert res.first_passage_biased and res.warnings
    assert res.mean_first_passage_s < 1.0 / 1.0
    assert all(t == 1.0 for t in res.censoring_times_s)


def test_competing_absorbing_outcomes_sum_to_one():
    net = TransitionNetwork(PROV)
    a = _state("a")
    sink = _state("sink", absorbing=True, absorbing_kind="sink")
    rec = _state("rec", absorbing=True, absorbing_kind="recombined")
    for s in (a, sink, rec):
        net.add_state(s)
    _edge(net, a, sink, 1.0, atoms=(1,))
    _edge(net, a, rec, 1.0, atoms=(2,))
    res = _ens(net, a, n_trajectories=2000, seed=2)
    total = sum(o.count for o in res.outcomes.values())
    assert total == 2000
    assert res.outcomes["sink"].probability == pytest.approx(0.5, abs=0.04)


def test_all_censored_ensemble_has_no_first_passage_statistics():
    net, a, _ = _decay(1e-12)
    res = _ens(net, a, n_trajectories=20, seed=1, horizon_s=1.0)
    assert res.outcomes == {} and res.mean_first_passage_s is None
    assert res.median_first_passage_s is None and res.censored_fraction == 1.0


def test_wilson_interval_properties_and_argument_checks():
    low, high = wilson_interval(0, 50)
    assert low == 0.0 and 0 < high < 0.1
    low, high = wilson_interval(50, 50)
    assert high == 1.0 and low > 0.9
    assert wilson_interval(25, 100)[0] < 0.25 < wilson_interval(25, 100)[1]
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    net, a, _ = _decay(1.0)
    with pytest.raises(ValueError):
        _ens(net, a, n_trajectories=0, seed=1)
