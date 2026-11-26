"""Software verification of the generalised states and event library (constructed data only)."""

import math

import pytest

from matsim_agents.science.kinetics.cluster_dynamics import (
    FUTURE_WORK,
    ClusterError,
    ClusterNetwork,
    ClusterReaction,
    ClusterState,
    FutureWorkError,
    SinkTerm,
    SourceTerm,
    reject_future_work,
)
from matsim_agents.science.kinetics.event_library import EVENT_IDS, EVENTS
from matsim_agents.science.kinetics.gas_mechanisms import OpenSystemTerm
from matsim_agents.science.kinetics.transitions import get_mechanism

T = 600.0


def st(v=0, i=0, he=0, h=0, env="bulk", **solutes):
    return ClusterState(
        n_vacancies=v,
        n_interstitials=i,
        n_helium=he,
        n_hydrogen=h,
        environment=env,
        solutes=solutes,
    )


def rx(mechanism, a, b, rate=1.0, **kw):
    return ClusterReaction(
        mechanism=mechanism,
        source_key=a.key,
        target_key=b.key,
        evidence_ref="synthetic-table",
        qualification="synthetic",
        rate_per_s=rate,
        table_temperature_K=T,
        **kw,
    )


def net(*states):
    n = ClusterNetwork(T)
    for s in states:
        n.add_state(s)
    return n


def term(species, delta):
    return OpenSystemTerm(species=species, delta=delta, reservoir="r", basis="b")


def test_legacy_key_and_inventory_are_unchanged():
    s = st(2, 0, 1, 3)
    assert s.key == "V2He1H3@bulk"
    assert s.inventory == {"V": 2, "He": 1, "H": 3}


def test_interstitial_and_solute_state_identity_and_inventory():
    s = st(1, 2, 0, 0, "gb", Re=1, Os=2)
    assert s.key == "V1He0H0I2+Os2+Re1@gb"
    assert s.inventory == {"V": 1, "I": 2, "Os": 2, "Re": 1}
    assert st(0, Re=1, Os=2).key == st(0, Os=2, Re=1).key


@pytest.mark.parametrize("bad", [{"V": 1}, {"He": 1}, {"Xx": 1}, {"Re": 0}, {"Re": 1.5}])
def test_unusable_solutes_are_rejected(bad):
    with pytest.raises(ValueError):
        st(**bad)


def test_every_event_is_registered_with_a_policy():
    for name in EVENT_IDS:
        assert get_mechanism(name).reversible == EVENTS[name].reversible


def test_migration_conserves_and_is_not_hardcoded_to_vacancy_hops():
    a, b = st(1, env="bulk"), st(1, env="near-sink")
    n = net(a, b)
    n.add_reaction(rx("vacancy_migration", a, b))
    c, d = st(i=1, env="bulk"), st(i=1, env="near-sink")
    for s in (c, d):
        n.add_state(s)
    n.add_reaction(rx("interstitial_migration", c, d))
    assert len(n.reactions) == 2
    with pytest.raises(ClusterError, match="needs at least 1 I"):
        n.add_reaction(rx("interstitial_migration", a, b))


def test_migration_cannot_change_inventory():
    a, b = st(1), st(2)
    with pytest.raises(ClusterError, match="closed-system"):
        net(a, b).add_reaction(rx("vacancy_migration", a, b))


def test_recombination_removes_equal_numbers():
    a, b = st(2, 2), st(1, 1)
    n = net(a, b)
    n.add_reaction(rx("frenkel_recombination", a, b))
    with pytest.raises(ClusterError, match="equal numbers"):
        n2 = net(st(2, 1), st(1, 1))
        n2.add_reaction(rx("frenkel_recombination", st(2, 1), st(1, 1)))
    with pytest.raises(ClusterError, match="equal numbers"):
        n3 = net(st(1, 1), st(2, 2))
        n3.add_reaction(rx("frenkel_recombination", st(1, 1), st(2, 2)))


def test_generation_adds_a_pair_and_not_a_single_defect():
    a, b = st(0, 0), st(1, 1)
    net(a, b).add_reaction(rx("frenkel_generation", a, b))
    c = st(1, 0)
    with pytest.raises(ClusterError, match="equal numbers"):
        net(a, c).add_reaction(rx("frenkel_generation", a, c))


def test_closed_event_rejects_open_terms():
    a, b = st(1, env="x"), st(1, env="y")
    with pytest.raises(ClusterError, match="cannot carry open-system"):
        net(a, b).add_reaction(rx("vacancy_migration", a, b, open_terms=(term("H", 1),)))


def test_gas_production_and_release_need_correct_signs_and_terms():
    a, b = st(1), st(1, he=1)
    n = net(a, b)
    n.add_reaction(rx("gas_production", a, b, open_terms=(term("He", 1),)))
    with pytest.raises(ClusterError, match="needs explicit"):
        net(a, b).add_reaction(rx("gas_production", a, b))
    with pytest.raises(ClusterError, match="must be a gain"):
        net(b, a).add_reaction(rx("gas_production", b, a, open_terms=(term("He", -1),)))
    n.add_reaction(rx("gas_release", b, a, open_terms=(term("He", -1),)))
    with pytest.raises(ClusterError, match="cannot exchange"):
        net(a, st(1, Re=1)).add_reaction(
            rx("gas_release", a, st(1, Re=1), open_terms=(term("Re", 1),))
        )


def test_term_total_must_equal_the_inventory_change():
    a, b = st(1), st(1, he=2)
    with pytest.raises(ClusterError, match="do not equal"):
        net(a, b).add_reaction(rx("gas_implantation", a, b, open_terms=(term("He", 1),)))


def test_transmutation_product_appears_with_explicit_term():
    a, b = st(1), st(1, Re=1)
    n = net(a, b)
    n.add_reaction(rx("transmutation", a, b, open_terms=(term("Re", 1),)))
    with pytest.raises(ClusterError, match="needs explicit"):
        net(a, b).add_reaction(rx("transmutation", a, b))


def test_sink_absorption_is_a_named_loss():
    a, b = st(i=2), st(i=1)
    n = net(a, b)
    n.add_reaction(rx("sink_absorption", a, b, open_terms=(term("I", -1),)))
    with pytest.raises(ClusterError, match="must be a loss"):
        net(b, a).add_reaction(rx("sink_absorption", b, a, open_terms=(term("I", 1),)))


def test_association_and_trapping_conserve_the_extended_inventory():
    mobile, trapped = st(he=1, Re=1), st(he=1, env="trap", Re=1)
    n = net(mobile, trapped)
    n.add_reaction(rx("defect_trapping", mobile, trapped))
    n.add_reaction(rx("defect_detrapping", trapped, mobile))
    a, b = st(i=1), st(i=2)
    with pytest.raises(ClusterError, match="closed-system"):
        net(a, b).add_reaction(rx("cluster_association", a, b))
    assert n.inventory({mobile.key: 1.0, trapped.key: 2.0}) == {"He": 3.0, "Re": 3.0}


def test_unknown_mechanism_and_precipitation_are_rejected():
    a, b = st(1), st(2)
    with pytest.raises(ClusterError, match="unsupported mechanism"):
        net(a, b).add_reaction(rx("vacancy_hop", a, b))
    assert "precipitate_formation" in FUTURE_WORK
    with pytest.raises(FutureWorkError, match="future work"):
        reject_future_work("precipitate_formation")
    with pytest.raises(FutureWorkError):
        net(a, b).add_reaction(rx("precipitate_formation", a, b))


def test_interstitial_population_balances_source_and_sink():
    i = st(0, 1)
    n = net(i)
    n.add_sink(SinkTerm(state_key=i.key, rate_per_s=3.0, sink="dislocations", basis="b"))
    n.add_source(SourceTerm(state_key=i.key, rate_per_s=1.0, reservoir="cascade", basis="b"))
    assert n.steady_state()[i.key] == pytest.approx(1.0 / 3.0)
    balance = n.inventory_balance({i.key: 1.0})
    assert balance["sources"]["I"] == pytest.approx(1.0)
    assert balance["sinks"]["I"] == pytest.approx(-3.0)
    assert balance["total"]["I"] == pytest.approx(-2.0)


def test_recombination_evolves_a_pair_population_exponentially():
    pair, nothing = st(1, 1), st(0, 0)
    n = net(pair, nothing)
    n.add_reaction(rx("frenkel_recombination", pair, nothing, rate=2.0))
    out = n.evolve({pair.key: 1.0}, 1.0)
    assert out[nothing.key] == pytest.approx(1.0 - math.exp(-2.0))
