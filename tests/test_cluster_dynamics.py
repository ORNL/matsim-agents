"""Cluster-dynamics bridge: analytic solutions, inventory accounting, rejection rules."""

import math

import pytest

from matsim_agents.kinetics.cluster_dynamics import (
    FUTURE_WORK,
    ClusterError,
    ClusterNetwork,
    ClusterReaction,
    ClusterState,
    FutureWorkError,
    SinkTerm,
    SourceTerm,
    capabilities,
    import_transition_network,
    reject_future_work,
)
from matsim_agents.kinetics.contracts import (
    BarrierEvidence,
    DefectState,
    NetworkProvenance,
    TransitionEvent,
)
from matsim_agents.kinetics.gas_mechanisms import OpenSystemTerm
from matsim_agents.kinetics.graph import TransitionNetwork

T = 600.0


def cs(v, he, h, env="bulk"):
    return ClusterState(n_vacancies=v, n_helium=he, n_hydrogen=h, environment=env)


def table(source, target, rate, mechanism="helium_trapping", **kw):
    return ClusterReaction(
        mechanism=mechanism,
        source_key=source.key,
        target_key=target.key,
        evidence_ref="synthetic-table",
        qualification="synthetic",
        rate_per_s=rate,
        table_temperature_K=T,
        **kw,
    )


def two_state(rate=2.0):
    net = ClusterNetwork(T)
    free, trapped = cs(1, 1, 0, "free"), cs(1, 1, 0, "trapped")
    net.add_state(free)
    net.add_state(trapped)
    net.add_reaction(table(free, trapped, rate))
    return net, free, trapped


def test_state_key_includes_environment_and_inventory():
    assert cs(2, 3, 1, "gb").key == "V2He3H1@gb"
    assert cs(2, 3, 1, "gb").inventory == {"V": 2, "He": 3, "H": 1}
    assert cs(0, 0, 0).inventory == {}
    assert cs(1, 0, 0, "a").key != cs(1, 0, 0, "b").key


def test_first_order_decay_matches_analytic_solution():
    net, free, trapped = two_state(rate=2.0)
    out = net.evolve({free.key: 100.0}, 0.7)
    assert out[free.key] == pytest.approx(100.0 * math.exp(-1.4), rel=1e-9)
    assert out[trapped.key] == pytest.approx(100.0 * (1 - math.exp(-1.4)), rel=1e-9)


def test_closed_network_conserves_inventory_exactly():
    net, free, trapped = two_state()
    before = net.inventory({free.key: 10.0})
    after = net.inventory(net.evolve({free.key: 10.0}, 3.0))
    for species in before:
        assert after[species] == pytest.approx(before[species], rel=1e-12)


def test_reversible_pair_reaches_detailed_balance_ratio():
    net, free, trapped = two_state(rate=3.0)
    net.add_reaction(table(trapped, free, 1.0, mechanism="helium_detrapping"))
    out = net.evolve({free.key: 40.0}, 50.0)
    assert out[trapped.key] / out[free.key] == pytest.approx(3.0, rel=1e-6)


def test_closed_reaction_that_changes_inventory_is_rejected():
    net = ClusterNetwork(T)
    a, b = cs(2, 2, 0), cs(2, 3, 0)
    net.add_state(a)
    net.add_state(b)
    with pytest.raises(ClusterError, match="without explicit source/sink"):
        net.add_reaction(table(a, b, 1.0, mechanism="gas_cluster_association"))


def test_open_reaction_needs_matching_terms_and_is_reported_in_balance():
    net = ClusterNetwork(T)
    a, b = cs(1, 2, 0), cs(1, 3, 0)
    net.add_state(a)
    net.add_state(b)
    term = OpenSystemTerm(species="He", delta=1, reservoir="implanter", basis="synthetic flux")
    net.add_reaction(table(a, b, 0.5, mechanism="gas_cluster_association", open_terms=(term,)))
    balance = net.inventory_balance({a.key: 4.0})
    assert balance["open_reactions"] == {"He": pytest.approx(2.0)}
    assert balance["total"]["He"] == pytest.approx(2.0)
    wrong = OpenSystemTerm(species="He", delta=2, reservoir="implanter", basis="b")
    other = cs(1, 3, 0, "x")
    net.add_state(other)
    with pytest.raises(ClusterError, match="do not equal"):
        net.add_reaction(table(b, other, 1.0, open_terms=(wrong,)))


def test_source_and_sink_reach_analytic_steady_state():
    net, free, trapped = two_state(rate=2.0)
    net.add_source(SourceTerm(state_key=free.key, rate_per_s=5.0, reservoir="beam", basis="b"))
    net.add_sink(SinkTerm(state_key=trapped.key, rate_per_s=0.5, sink="surface", basis="b"))
    steady = net.steady_state()
    assert steady[free.key] == pytest.approx(5.0 / 2.0)
    assert steady[trapped.key] == pytest.approx(5.0 / 0.5)
    late = net.evolve({}, 200.0)
    assert late[trapped.key] == pytest.approx(10.0, rel=1e-6)


def test_inventory_rate_balances_when_sources_and_sinks_cancel():
    net, free, trapped = two_state(rate=2.0)
    net.add_source(SourceTerm(state_key=free.key, rate_per_s=5.0, reservoir="beam", basis="b"))
    net.add_sink(SinkTerm(state_key=trapped.key, rate_per_s=0.5, sink="surface", basis="b"))
    balance = net.inventory_balance(net.steady_state())
    for species, rate in balance["total"].items():
        assert rate == pytest.approx(0.0, abs=1e-12), species


def test_sourced_network_without_sink_has_no_steady_state():
    net, free, _ = two_state()
    net.add_source(SourceTerm(state_key=free.key, rate_per_s=1.0, reservoir="beam", basis="b"))
    with pytest.raises(ClusterError, match="no steady state"):
        net.steady_state()


def test_reaction_without_rate_evidence_is_rejected():
    a, b = cs(1, 1, 0, "x"), cs(1, 1, 0, "y")
    with pytest.raises(ValueError, match="table rate or barrier"):
        ClusterReaction(
            mechanism="helium_trapping",
            source_key=a.key,
            target_key=b.key,
            evidence_ref="r",
            qualification="exploratory",
        )
    with pytest.raises(ValueError, match="attempt frequency"):
        ClusterReaction(
            mechanism="helium_trapping",
            source_key=a.key,
            target_key=b.key,
            evidence_ref="r",
            qualification="exploratory",
            barrier_eV=0.3,
        )
    with pytest.raises(ValueError, match="tabulated"):
        ClusterReaction(
            mechanism="helium_trapping",
            source_key=a.key,
            target_key=b.key,
            evidence_ref="r",
            qualification="exploratory",
            rate_per_s=1.0,
        )


def test_table_rate_at_other_temperature_is_rejected():
    net = ClusterNetwork(900.0)
    a, b = cs(1, 1, 0, "x"), cs(1, 1, 0, "y")
    net.add_state(a)
    net.add_state(b)
    with pytest.raises(ClusterError, match="tabulated|table rate is for"):
        net.add_reaction(table(a, b, 1.0))


def test_barrier_reaction_uses_htst_with_explicit_frequency():
    net = ClusterNetwork(T)
    a, b = cs(1, 1, 0, "x"), cs(1, 1, 0, "y")
    net.add_state(a)
    net.add_state(b)
    net.add_reaction(
        ClusterReaction(
            mechanism="helium_trapping",
            source_key=a.key,
            target_key=b.key,
            evidence_ref="neb-1",
            qualification="synthetic",
            barrier_eV=0.2,
            attempt_frequency_hz=1e12,
            multiplicity=2,
        )
    )
    expected = 2e12 * math.exp(-0.2 / (8.617333262e-5 * T))
    assert net.reaction_rate(net.reactions[0]) == pytest.approx(expected, rel=1e-12)


def test_unsupported_mechanism_and_unknown_state_rejected():
    net = ClusterNetwork(T)
    a, b = cs(1, 1, 0, "x"), cs(1, 1, 0, "y")
    net.add_state(a)
    net.add_state(b)
    with pytest.raises(ClusterError, match="unsupported mechanism"):
        net.add_reaction(table(a, b, 1.0, mechanism="vacancy_hop"))
    with pytest.raises(ClusterError, match="unknown cluster state"):
        net.add_source(SourceTerm(state_key="nope", rate_per_s=1.0, reservoir="r", basis="b"))


@pytest.mark.parametrize("name", sorted(FUTURE_WORK))
def test_future_work_is_rejected_and_reported(name):
    with pytest.raises(FutureWorkError, match="future work"):
        reject_future_work(name)
    net = ClusterNetwork(T)
    a, b = cs(1, 1, 0, "x"), cs(1, 1, 0, "y")
    net.add_state(a)
    net.add_state(b)
    with pytest.raises(FutureWorkError):
        net.add_reaction(table(a, b, 1.0, mechanism=name))
    assert capabilities()[name] == "FUTURE_WORK"


def test_negative_or_unknown_populations_rejected():
    net, free, _ = two_state()
    with pytest.raises(ClusterError):
        net.evolve({free.key: -1.0}, 1.0)
    with pytest.raises(ClusterError, match="unknown states"):
        net.evolve({"nope": 1.0}, 1.0)
    with pytest.raises(ClusterError):
        net.evolve({free.key: 1.0}, -1.0)


def test_uncertainty_is_deterministic_and_zero_width_without_errors():
    net, free, trapped = two_state(rate=2.0)

    def observable(pop):
        return pop[trapped.key]

    kw = dict(table_log10_sigma=0.0, barrier_sigma_eV=0.0, source_log10_sigma=0.0, seed=7)
    flat = net.propagate_uncertainty({free.key: 1.0}, 0.5, observable, n_samples=8, **kw)
    assert flat.std == pytest.approx(0.0, abs=1e-12)
    assert flat.mean == pytest.approx(1 - math.exp(-1.0), rel=1e-9)
    kw["table_log10_sigma"] = 0.3
    one = net.propagate_uncertainty({free.key: 1.0}, 0.5, observable, n_samples=64, **kw)
    two = net.propagate_uncertainty({free.key: 1.0}, 0.5, observable, n_samples=64, **kw)
    assert one == two
    assert one.std > 0 and one.p05 < one.mean < one.p95
    assert "missing channels and states" in one.not_propagated


def test_source_uncertainty_scales_steady_population():
    net, free, trapped = two_state(rate=2.0)
    net.add_source(SourceTerm(state_key=free.key, rate_per_s=5.0, reservoir="beam", basis="b"))
    net.add_sink(SinkTerm(state_key=trapped.key, rate_per_s=0.5, sink="s", basis="b"))
    res = net.propagate_uncertainty(
        {},
        500.0,
        lambda pop: pop[trapped.key],
        barrier_sigma_eV=0.0,
        table_log10_sigma=0.0,
        source_log10_sigma=0.2,
        n_samples=128,
        seed=3,
    )
    assert res.std > 0.1 * res.mean * 0.2
    assert res.mean == pytest.approx(10.0, rel=0.25)


# -- transition-network bridge --------------------------------------------- #


def tn_pair(synthetic=True, qualification="unqualified"):
    prov = NetworkProvenance(network_id="n", parent_id="p", synthetic=True)
    network = TransitionNetwork(prov)
    for sid in ("s1", "s2"):
        network.add_state(
            DefectState(state_id=sid, parent_id="p", n_atoms=17, provenance={"origin": "test"})
        )
    evidence = BarrierEvidence(
        method_id="synthetic",
        neb_state="converged_unqualified",
        forward_barrier_eV=0.3,
        reverse_barrier_eV=0.4,
        n_images=7,
        synthetic=synthetic,
    )
    network.add_transition(
        TransitionEvent(
            transition_id="t1",
            source_id="s1",
            target_id="s2",
            mechanism="helium_trapping",
            moving_atom_ids=(16,),
            attempt_frequency_hz=1e12,
            evidence=evidence,
            qualification=qualification,
            provenance={"origin": "test"},
        )
    )
    return network


def test_bridge_imports_evidence_backed_events_only():
    tn = tn_pair()
    net = ClusterNetwork(T)
    mapping = {"s1": cs(1, 1, 0, "free"), "s2": cs(1, 1, 0, "trapped")}
    report = import_transition_network(net, tn, mapping, require="exploratory")
    assert report.reactions_added == ("t1",)
    assert net.reactions[0].qualification == "synthetic"
    assert net.reactions[0].barrier_eV == 0.3 and net.reactions[0].evidence_ref == "t1"


def test_bridge_refuses_unqualified_under_default_policy():
    net = ClusterNetwork(T)
    mapping = {"s1": cs(1, 1, 0, "free"), "s2": cs(1, 1, 0, "trapped")}
    with pytest.raises(ClusterError, match="cannot import t1"):
        import_transition_network(net, tn_pair(), mapping)


def test_bridge_requires_state_mapping_and_reports_intra_state_events():
    net = ClusterNetwork(T)
    with pytest.raises(ClusterError, match="no cluster-state mapping"):
        import_transition_network(net, tn_pair(), {"s1": cs(1, 1, 0)}, require="exploratory")
    same = cs(1, 1, 0, "same")
    report = import_transition_network(
        ClusterNetwork(T), tn_pair(), {"s1": same, "s2": same}, require="exploratory"
    )
    assert report.intra_state_events == ("t1",) and report.reactions_added == ()
