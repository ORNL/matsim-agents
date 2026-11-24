"""Gillespie KMC against analytic results (synthetic networks)."""

import pytest

from matsim_agents.kinetics.contracts import BarrierEvidence
from matsim_agents.kinetics.graph import TransitionNetwork
from matsim_agents.kinetics.kmc import run_kmc
from matsim_agents.kinetics.rates import RateError
from tests.test_kinetics_graph import PROV, _event, _state


def _evidence(barrier=0.0, **over):
    base = {
        "method_id": "synthetic-fixture",
        "neb_state": "unconverged",
        "forward_barrier_eV": barrier,
        "reverse_barrier_eV": barrier,
        "n_images": 6,
        "synthetic": True,
    }
    base.update(over)
    return BarrierEvidence(**base)


def _edge(net, src, dst, nu, atoms=(3,), mechanism="vacancy_hop"):
    net.add_transition(
        _event(
            src,
            dst,
            atoms=atoms,
            mechanism=mechanism,
            evidence=_evidence(),
            qualification="unqualified",
            attempt_frequency_hz=nu,
        )
    )


def _decay(rate):
    net = TransitionNetwork(PROV)
    a = _state("a")
    done = _state("done", absorbing=True, absorbing_kind="recombined")
    net.add_state(a)
    net.add_state(done)
    _edge(net, a, done, rate)
    return net, a, done


def _run(net, start, seed=1, stream=(), **kw):
    kw.setdefault("require", "exploratory")
    return run_kmc(net, start.state_id, 300.0, seed=seed, stream=stream, **kw)


def test_single_channel_mean_waiting_time_matches_one_over_rate():
    net, a, done = _decay(2.0)
    times = []
    for i in range(3000):
        traj = _run(net, a, stream=(i,), record_steps=False)
        assert traj.termination == "absorbed" and traj.absorbing_kind == "recombined"
        times.append(traj.final_time_s)
    mean = sum(times) / len(times)
    assert mean == pytest.approx(0.5, rel=0.06)


def test_branching_ratio_matches_rate_ratio():
    net = TransitionNetwork(PROV)
    a = _state("a")
    fast = _state("fast", absorbing=True, absorbing_kind="sink")
    slow = _state("slow", absorbing=True, absorbing_kind="recombined")
    for s in (a, fast, slow):
        net.add_state(s)
    _edge(net, a, fast, 3.0, atoms=(1,))
    _edge(net, a, slow, 1.0, atoms=(2,))
    hits = sum(
        _run(net, a, stream=(i,), record_steps=False).final_state_id == fast.state_id
        for i in range(3000)
    )
    assert hits / 3000 == pytest.approx(0.75, abs=0.03)


def test_same_seed_and_stream_reproduce_and_different_streams_differ():
    net, a, _ = _decay(1.0)
    one = _run(net, a, seed=7, stream=(0,))
    assert one == _run(net, a, seed=7, stream=(0,))
    assert one.final_time_s != _run(net, a, seed=7, stream=(1,)).final_time_s
    assert one.final_time_s != _run(net, a, seed=8, stream=(0,)).final_time_s


def test_horizon_censors_without_reporting_absorption():
    net, a, _ = _decay(1e-9)
    traj = _run(net, a, horizon_s=1.0)
    assert traj.termination == "horizon" and traj.final_time_s == 1.0
    assert traj.absorbing_kind is None and traj.final_state_id == a.state_id


def test_max_events_caps_a_reversible_chain():
    net = TransitionNetwork(PROV)
    a, b = _state("a"), _state("b")
    net.add_state(a)
    net.add_state(b)
    _edge(net, a, b, 1.0)
    _edge(net, b, a, 1.0)
    traj = _run(net, a, max_events=25)
    assert traj.termination == "max_events" and traj.n_events == 25 and len(traj.steps) == 25
    times = [s.time_s for s in traj.steps]
    assert times == sorted(times) and times[0] > 0
    assert all(s.source_id != s.target_id for s in traj.steps)


def test_state_without_outgoing_events_is_stuck_not_absorbed():
    net = TransitionNetwork(PROV)
    a = _state("a")
    net.add_state(a)
    traj = _run(net, a)
    assert traj.termination == "stuck" and traj.absorbing_kind is None and traj.final_time_s == 0


def test_absorbing_initial_state_ends_immediately():
    net, _, done = _decay(1.0)
    traj = _run(net, done)
    assert traj.termination == "absorbed" and traj.n_events == 0


def test_qualified_policy_refuses_unqualified_network_and_bad_arguments():
    net, a, _ = _decay(1.0)
    with pytest.raises(RateError):
        run_kmc(net, a.state_id, 300.0, seed=1)
    with pytest.raises(ValueError):
        _run(net, a, horizon_s=0.0)
    with pytest.raises(ValueError):
        _run(net, a, max_events=0)
