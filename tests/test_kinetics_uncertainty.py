"""C2.4: uncertainty and sensitivity (synthetic networks)."""

import pytest

from matsim_agents.kinetics.graph import TransitionNetwork
from matsim_agents.kinetics.rates import htst_rate
from matsim_agents.kinetics.uncertainty import (
    barrier_sensitivity,
    pathway_keys,
    perturb_network,
    propagate_uncertainty,
)
from tests.test_kinetics_graph import PROV, _event, _state
from tests.test_kinetics_kmc import _evidence

T = 300.0
NU = 1e13


def _competing(b_fast=0.60, b_slow=0.62):
    net = TransitionNetwork(PROV)
    a = _state("a")
    fast = _state("fast", absorbing=True, absorbing_kind="sink")
    slow = _state("slow", absorbing=True, absorbing_kind="recombined")
    for s in (a, fast, slow):
        net.add_state(s)
    for dst, atoms, barrier in ((fast, (1,), b_fast), (slow, (2,), b_slow)):
        net.add_transition(
            _event(
                a,
                dst,
                atoms=atoms,
                evidence=_evidence(barrier),
                qualification="unqualified",
                attempt_frequency_hz=NU,
            )
        )
    return net, a


def _barrier(net, state_id):
    return next(e.barrier_eV for e in net.events if e.target_id == state_id)


def test_perturb_shifts_scales_and_drops_without_touching_the_original():
    net, a = _competing()
    keys = pathway_keys(net)
    assert len(keys) == 2
    out = perturb_network(net, barrier_shift_eV={keys[0]: 0.1}, prefactor_scale={keys[1]: 2.0})
    before = {e.target_id: e for e in net.events}
    after = {e.target_id: e for e in out.events}
    moved = [t for t in before if after[t].barrier_eV != pytest.approx(before[t].barrier_eV)]
    scaled = [t for t in before if after[t].attempt_frequency_hz != NU]
    assert len(moved) == 1 and len(scaled) == 1 and moved != scaled
    assert after[moved[0]].barrier_eV == pytest.approx(before[moved[0]].barrier_eV + 0.1)
    assert after[scaled[0]].attempt_frequency_hz == pytest.approx(2 * NU)
    assert sorted(e.barrier_eV for e in net.events) == [0.6, 0.62]
    assert len(perturb_network(net, drop={keys[0]}).events) == 1


def test_shift_never_makes_a_barrier_negative():
    net, _ = _competing(b_fast=0.05)
    key = pathway_keys(net)[0]
    out = perturb_network(net, barrier_shift_eV={key: -1.0})
    assert all(e.barrier_eV >= 0 for e in out.events)


def test_sensitivity_sign_and_size_match_the_analytic_branching_ratio():
    net, a = _competing(0.60, 0.62)
    rows = barrier_sensitivity(
        net,
        a.state_id,
        T,
        "sink",
        delta_eV=0.02,
        n_trajectories=4000,
        seed=1,
        require="exploratory",
    )
    k_fast, k_slow = htst_rate(0.60, T, NU), htst_rate(0.62, T, NU)
    base = k_fast / (k_fast + k_slow)
    assert rows[0].baseline_probability == pytest.approx(base, abs=0.03)
    fast_key = next(
        k
        for k in pathway_keys(net)
        if net.state(k[1]).absorbing_kind == "sink" or net.state(k[0]).absorbing_kind == "sink"
    )
    row = next(r for r in rows if r.pathway == fast_key)
    other = next(r for r in rows if r.pathway != fast_key)
    assert row.slope_per_eV < 0 < other.slope_per_eV
    kb_t = 8.617333262e-5 * T
    analytic = -base * (1 - base) / kb_t
    assert row.slope_per_eV == pytest.approx(analytic, rel=0.25)


def test_uncertainty_spreads_with_sigma_and_is_reproducible():
    net, a = _competing()
    kw = {
        "barrier_correlation": 0.0,
        "n_samples": 30,
        "n_trajectories": 200,
        "seed": 4,
        "require": "exploratory",
    }
    none = propagate_uncertainty(net, a.state_id, T, "sink", barrier_sigma_eV=0.0, **kw)
    wide = propagate_uncertainty(net, a.state_id, T, "sink", barrier_sigma_eV=0.1, **kw)
    assert wide == propagate_uncertainty(net, a.state_id, T, "sink", barrier_sigma_eV=0.1, **kw)
    assert wide.std > none.std
    assert wide.p05 <= wide.mean <= wide.p95


def test_common_shift_leaves_the_branching_ratio_nearly_unchanged():
    net, a = _competing()
    kw = {"n_samples": 30, "n_trajectories": 300, "seed": 9, "require": "exploratory"}
    indep = propagate_uncertainty(
        net, a.state_id, T, "sink", barrier_sigma_eV=0.1, barrier_correlation=0.0, **kw
    )
    common = propagate_uncertainty(
        net, a.state_id, T, "sink", barrier_sigma_eV=0.1, barrier_correlation=1.0, **kw
    )
    assert common.std < indep.std


def test_dropping_channels_changes_outcome_probability():
    net, a = _competing()
    res = propagate_uncertainty(
        net,
        a.state_id,
        T,
        "sink",
        barrier_sigma_eV=0.0,
        barrier_correlation=0.0,
        drop_probability=0.5,
        n_samples=30,
        n_trajectories=100,
        seed=2,
        require="exploratory",
    )
    assert res.std > 0.1


def test_argument_validation():
    net, a = _competing()
    base = {"n_samples": 5, "n_trajectories": 5, "seed": 1, "require": "exploratory"}
    for bad in (
        {"barrier_correlation": 1.5, "barrier_sigma_eV": 0.1},
        {"barrier_correlation": 0.0, "barrier_sigma_eV": -0.1},
        {"barrier_correlation": 0.0, "barrier_sigma_eV": 0.1, "drop_probability": 1.0},
    ):
        with pytest.raises(ValueError):
            propagate_uncertainty(net, a.state_id, T, "sink", **bad, **base)
    with pytest.raises(ValueError):
        barrier_sensitivity(net, a.state_id, T, "sink", delta_eV=0.0, n_trajectories=5, seed=1)
