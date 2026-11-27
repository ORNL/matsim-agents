"""B pathways -> network -> evidence -> rates -> KMC, with analytic checks.

Every energy here is a synthetic fixture; nothing is a calculated barrier.
"""

import math

import pytest

from matsim_agents.science.defects.bcc import generate_frenkel_pair
from matsim_agents.science.kinetics.ensemble import run_ensemble
from matsim_agents.science.kinetics.evidence import attach_neb_result
from matsim_agents.science.kinetics.kmc import run_kmc
from matsim_agents.science.kinetics.rates import KB_EV_PER_K, htst_rate
from matsim_agents.science.kinetics.transitions import add_candidate_transition
from matsim_agents.science.kinetics.validation import validate_network
from matsim_agents.science.neb.endpoints import generate_frenkel_recombination
from tests.test_kinetics_evidence import _result
from tests.test_kinetics_transitions import PARENT, A, _hop, _network

NU = 1e13
T = 600.0


def _hop_network():
    net, transition = _network(), _hop().transition
    forward, _ = add_candidate_transition(net, transition)
    attach_neb_result(net, transition, _result(), attempt_frequency_hz=NU, synthetic=True)
    return net, forward.source_id, forward.target_id


def test_reversible_hop_reaches_the_boltzmann_occupancy():
    net, a, b = _hop_network()
    traj = run_kmc(net, a, T, seed=1, max_events=40_000, require="qualified")
    assert traj.termination == "max_events"
    spent = {a: 0.0, b: 0.0}
    state, last = a, 0.0
    for step in traj.steps:
        spent[state] += step.time_s - last
        state, last = step.target_id, step.time_s
    k_f, k_r = htst_rate(0.8, T, NU), htst_rate(1.0, T, NU)
    assert spent[b] / (spent[a] + spent[b]) == pytest.approx(k_f / (k_f + k_r), abs=0.03)
    ratio = math.exp((1.0 - 0.8) / (KB_EV_PER_K * T))
    assert k_f / k_r == pytest.approx(ratio)


def test_recombination_first_passage_follows_the_synthetic_barrier():
    initial, config = generate_frenkel_pair(PARENT, "p", 0, 9, A)
    pair = generate_frenkel_recombination(initial, config, PARENT)
    net = _network()
    (event,) = add_candidate_transition(net, pair.transition)
    result = _result(
        forward_barrier_eV=0.3,
        reverse_barrier_eV=0.5,
        relative_energies_eV=(0.0, 0.15, 0.3, 0.2, 0.0, -0.2),
    )
    attach_neb_result(net, pair.transition, result, attempt_frequency_hz=NU, synthetic=True)
    rate = htst_rate(0.3, T, NU)
    res = run_ensemble(net, event.source_id, T, n_trajectories=2000, seed=5, require="qualified")
    assert res.outcomes["recombined"].probability == 1.0
    assert res.mean_first_passage_s == pytest.approx(1.0 / rate, rel=0.07)


def test_candidate_network_cannot_drive_a_qualified_run_and_report_says_why():
    net, transition = _network(), _hop().transition
    forward, _ = add_candidate_transition(net, transition)
    report = validate_network(net, forward.source_id)
    assert not report.ready_for_exploratory_kmc and not report.ready_for_qualified_kmc
    with pytest.raises(Exception, match="no barrier"):
        run_kmc(net, forward.source_id, T, seed=1)


def test_synthetic_qualified_network_is_still_not_qualified_ready():
    net, a, _ = _hop_network()
    report = validate_network(net, a)
    assert report.ready_for_exploratory_kmc and not report.ready_for_qualified_kmc
    assert report.n_synthetic_evidence == report.n_events == 2
