"""Software verification of rate construction and the equilibrium detailed-balance check."""

import math

import pytest

from matsim_agents.science.kinetics.cluster_dynamics import (
    ClusterError,
    ClusterNetwork,
    ClusterState,
    SourceTerm,
)
from matsim_agents.science.kinetics.harmonic_tst import vibrational_free_energy_eV
from matsim_agents.science.kinetics.rate_construction import (
    PrefactorBasis,
    add_reaction_pair,
    check_equilibrium_detailed_balance,
)
from matsim_agents.science.kinetics.rates import KB_EV_PER_K

T = 700.0
KT = KB_EV_PER_K * T
MIN_A = (4.0e12, 5.0e12, 6.0e12)
MIN_B = (3.0e12, 4.5e12, 5.5e12)
SADDLE = (3.5e12, 5.0e12)
CONSTANT = PrefactorBasis(kind="user_supplied", attempt_frequency_hz=1e13, citation="cited value")


def cs(env, v=1):
    return ClusterState(n_vacancies=v, n_helium=0, n_hydrogen=0, environment=env)


def pair(basis=CONSTANT, net=None, **kw):
    net = net or ClusterNetwork(T)
    a, b = cs("a"), cs("b")
    args = dict(
        forward_mechanism="vacancy_migration",
        reverse_mechanism="vacancy_migration",
        forward_barrier_eV=0.9,
        basis=basis,
        evidence_ref="synthetic",
        evidence_level="synthetic",
        reaction_energy_eV=0.2,
    )
    args.update(kw)
    return net, a, b, add_reaction_pair(net, a, b, **args)


def harmonic():
    return PrefactorBasis(
        kind="harmonic",
        source_freqs_hz=MIN_A,
        target_freqs_hz=MIN_B,
        saddle_freqs_hz=SADDLE,
    )


def test_reverse_barrier_follows_from_the_reaction_energy():
    _, _, _, report = pair()
    assert report.reverse.barrier_eV == pytest.approx(0.7)
    assert report.reverse_barrier_source == "from_reaction_energy"
    f, r = report.forward.reaction, report.reverse.reaction
    forward_rate = f.attempt_frequency_hz * math.exp(-0.9 / KT)
    reverse_rate = r.attempt_frequency_hz * math.exp(-0.7 / KT)
    assert math.log(forward_rate / reverse_rate) == pytest.approx(-0.2 / KT)


def test_constant_prefactor_pair_satisfies_detailed_balance():
    net, a, b, _ = pair()
    report = check_equilibrium_detailed_balance(net, {a.key: 0.0, b.key: 0.2})
    assert report.consistent and len(report.checked_pairs) == 1
    assert report.max_abs_log_error < 1e-9


def test_wrong_free_energy_is_reported():
    net, a, b, _ = pair()
    report = check_equilibrium_detailed_balance(net, {a.key: 0.0, b.key: 0.05})
    assert not report.consistent and "kf/kr" in report.violations[0]


def test_harmonic_pair_balances_against_vibrational_free_energies():
    net, a, b, report = pair(harmonic())
    ga = 0.0 + vibrational_free_energy_eV(MIN_A, T)
    gb = 0.2 + vibrational_free_energy_eV(MIN_B, T)
    out = check_equilibrium_detailed_balance(net, {a.key: ga, b.key: gb})
    assert out.consistent, out.violations
    assert any("harmonic TST" in n for n in report.forward.assumptions)
    bare = check_equilibrium_detailed_balance(net, {a.key: 0.0, b.key: 0.2})
    assert not bare.consistent


def test_harmonic_rate_depends_on_temperature_through_the_prefactor():
    rates = []
    for temp in (500.0, 900.0):
        net = ClusterNetwork(temp)
        add_reaction_pair(
            net,
            cs("a"),
            cs("b"),
            forward_mechanism="vacancy_migration",
            reverse_mechanism="vacancy_migration",
            forward_barrier_eV=0.9,
            basis=harmonic(),
            evidence_ref="s",
            evidence_level="synthetic",
            reaction_energy_eV=0.2,
        )
        rates.append(net.reaction_rate(net.reactions[0]))
    assert rates[1] > rates[0]


def test_independent_reverse_barrier_can_be_inconsistent_and_is_detected():
    net, a, b, report = pair(reaction_energy_eV=None, reverse_barrier_eV_supplied=0.5)
    assert report.reverse_barrier_source == "supplied"
    out = check_equilibrium_detailed_balance(net, {a.key: 0.0, b.key: 0.2})
    assert not out.consistent


def test_missing_reverse_information_is_rejected():
    with pytest.raises(ClusterError, match="reaction energy or an independent reverse"):
        pair(reaction_energy_eV=None)
    with pytest.raises(ClusterError, match="negative reverse barrier"):
        pair(reaction_energy_eV=1.5)


def test_fallback_prefactor_is_labelled_and_never_qualified():
    basis = PrefactorBasis(
        kind="documented_fallback", attempt_frequency_hz=1e13, citation="assumed"
    )
    _, _, _, report = pair(basis, evidence_level="qualified")
    assert report.qualification == "exploratory"
    assert report.forward.reaction.qualification == "exploratory"
    assert any("assumed prefactor" in n for n in report.forward.assumptions)


def test_prefactor_basis_requires_citation_and_modes():
    with pytest.raises(ValueError, match="citation"):
        PrefactorBasis(kind="user_supplied", attempt_frequency_hz=1e13)
    with pytest.raises(ValueError, match="needs attempt_frequency_hz"):
        PrefactorBasis(kind="documented_fallback", citation="x")
    with pytest.raises(ValueError, match="modes"):
        PrefactorBasis(kind="harmonic", source_freqs_hz=MIN_A)


def test_uncertainty_is_none_when_not_supplied_and_flagged():
    _, _, _, report = pair()
    assert report.forward.log_rate_sigma is None
    assert any("unknown, not zero" in n for n in report.forward.assumptions)
    _, _, _, with_sigma = pair(barrier_sigma_eV=0.05)
    assert with_sigma.forward.log_rate_sigma == pytest.approx(0.05 / KT)


def test_pair_must_conserve_inventory():
    net = ClusterNetwork(T)
    with pytest.raises(ClusterError, match="equal inventories"):
        add_reaction_pair(
            net,
            cs("a", 1),
            cs("b", 2),
            forward_mechanism="vacancy_migration",
            reverse_mechanism="vacancy_migration",
            forward_barrier_eV=0.9,
            basis=CONSTANT,
            evidence_ref="s",
            evidence_level="synthetic",
            reaction_energy_eV=0.1,
        )


def test_driven_terms_and_open_events_are_excluded_not_tested():
    net, a, b, _ = pair()
    net.add_source(SourceTerm(state_key=a.key, rate_per_s=1.0, reservoir="cascade", basis="b"))
    out = check_equilibrium_detailed_balance(net, {a.key: 0.0, b.key: 0.2})
    assert out.consistent and out.excluded_driven_terms
    assert "not tested" in out.scope


def test_missing_free_energy_and_unpaired_reactions():
    net, a, b, _ = pair()
    with pytest.raises(ClusterError, match="missing free energy"):
        check_equilibrium_detailed_balance(net, {a.key: 0.0})
    one_way = ClusterNetwork(T)
    c = cs("c")
    one_way.add_state(cs("a"))
    one_way.add_state(c)
    from matsim_agents.science.kinetics.cluster_dynamics import ClusterReaction

    one_way.add_reaction(
        ClusterReaction(
            mechanism="vacancy_migration",
            source_key=cs("a").key,
            target_key=c.key,
            evidence_ref="s",
            qualification="synthetic",
            rate_per_s=1.0,
            table_temperature_K=T,
        )
    )
    out = check_equilibrium_detailed_balance(one_way, {})
    assert out.unpaired and not out.checked_pairs
