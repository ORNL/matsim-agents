"""Harmonic TST rates: analytic limits and the evidence gate (synthetic fixtures)."""

import math

import pytest

from matsim_agents.science.kinetics.evidence import attach_neb_result
from matsim_agents.science.kinetics.harmonic_tst import (
    H_EV_S,
    harmonic_event_rate,
    harmonic_tst_rate,
    reverse_barrier_eV,
    vibrational_free_energy_eV,
    vineyard_prefactor_hz,
)
from matsim_agents.science.kinetics.rates import KB_EV_PER_K, RateError, htst_rate
from tests.test_kinetics_evidence import _result, _setup

MIN = (4e12, 5e12, 6e12)
SADDLE = (3e12, 7e12)


def test_high_temperature_limit_is_the_vineyard_rate():
    t = 5.0e5
    got = harmonic_tst_rate(0.4, MIN, SADDLE, t, multiplicity=3)
    vine = htst_rate(0.4, t, vineyard_prefactor_hz(MIN, SADDLE), 3)
    assert got.rate_per_s == pytest.approx(vine, rel=2e-3)


def test_low_temperature_correction_is_the_zero_point_energy_difference():
    t = 5.0
    zpe = 0.5 * H_EV_S * (sum(SADDLE) - sum(MIN))
    got = harmonic_tst_rate(0.4, MIN, SADDLE, t)
    assert got.vibrational_correction_eV == pytest.approx(zpe, rel=1e-9)
    assert got.activation_free_energy_eV == pytest.approx(0.4 + zpe, rel=1e-9)


def test_prefactor_depends_on_temperature_unlike_a_constant():
    cold = harmonic_tst_rate(0.4, MIN, SADDLE, 100.0)
    hot = harmonic_tst_rate(0.4, MIN, SADDLE, 1000.0)
    assert cold.effective_prefactor_hz != pytest.approx(hot.effective_prefactor_hz, rel=1e-3)


def test_free_energy_is_checked_against_a_direct_sum():
    f, t = (2e12, 9e12), 400.0
    kt = KB_EV_PER_K * t
    direct = sum(0.5 * H_EV_S * x + kt * math.log(1 - math.exp(-H_EV_S * x / kt)) for x in f)
    assert vibrational_free_energy_eV(f, t) == pytest.approx(direct, rel=1e-12)
    assert vibrational_free_energy_eV(f, t, include_zpe=False) < vibrational_free_energy_eV(f, t)


def test_uncertainty_is_none_unless_given_and_combines_in_quadrature():
    assert harmonic_tst_rate(0.4, MIN, SADDLE, 300.0).log_rate_sigma is None
    t, kt = 300.0, KB_EV_PER_K * 300.0
    r = harmonic_tst_rate(0.4, MIN, SADDLE, t, barrier_sigma_eV=0.05, prefactor_log_sigma=0.5)
    assert r.log_rate_sigma == pytest.approx(math.hypot(0.05 / kt, 0.5))
    assert harmonic_tst_rate(0.4, MIN, SADDLE, t, barrier_sigma_eV=0.0).log_rate_sigma == 0.0


def test_huge_barrier_underflows_to_zero():
    assert harmonic_tst_rate(500.0, MIN, SADDLE, 100.0).rate_per_s == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"barrier_eV": -0.1},
        {"minimum_freqs_hz": ()},
        {"saddle_freqs_hz": (3e12, 7e12, 1e12)},
        {"saddle_freqs_hz": (3e12, -7e12)},
        {"temperature_K": 0.0},
        {"multiplicity": 0},
        {"barrier_sigma_eV": -1.0},
        {"prefactor_log_sigma": float("nan")},
    ],
)
def test_invalid_inputs_are_rejected(kwargs):
    base = dict(
        barrier_eV=0.4,
        minimum_freqs_hz=MIN,
        saddle_freqs_hz=SADDLE,
        temperature_K=300.0,
    )
    with pytest.raises(RateError):
        harmonic_tst_rate(**{**base, **kwargs})


def test_detailed_balance_with_modes_of_both_states():
    # Forward and reverse share the saddle, so k_f/k_r = (Z_B/Z_A) exp(-dE/kT) in the high-T limit.
    t, de = 3.0e5, 0.2
    a, b = (4e12, 5e12, 6e12), (3.5e12, 5.5e12, 6.5e12)
    fwd = harmonic_tst_rate(0.6, a, SADDLE, t)
    rev = harmonic_tst_rate(reverse_barrier_eV(0.6, de), b, SADDLE, t)
    expected = math.prod(a) / math.prod(b) * math.exp(-de / (KB_EV_PER_K * t))
    assert fwd.rate_per_s / rev.rate_per_s == pytest.approx(expected, rel=5e-3)


def test_reverse_barrier_rejects_inconsistent_energies():
    assert reverse_barrier_eV(0.8, -0.2) == pytest.approx(1.0)
    with pytest.raises(RateError):
        reverse_barrier_eV(0.1, 0.5)


def test_event_gate_follows_the_evidence_policy_and_records_replacement():
    net, transition = _setup()
    forward, _ = attach_neb_result(
        net, transition, _result(), attempt_frequency_hz=1e13, synthetic=True
    )
    rate, harm = harmonic_event_rate(forward, MIN, SADDLE, 600.0)
    assert rate.rate_per_s == pytest.approx(harm.rate_per_s) and rate.rate_per_s > 0
    assert any("replaced" in a for a in rate.assumptions)
    assert any("synthetic" in a for a in rate.assumptions)

    net2, transition2 = _setup()
    weak, _ = attach_neb_result(
        net2,
        transition2,
        _result(state="unconverged", endpoints_qualified=False),
        attempt_frequency_hz=1e13,
    )
    with pytest.raises(RateError, match="not qualified"):
        harmonic_event_rate(weak, MIN, SADDLE, 600.0)
    rate, _ = harmonic_event_rate(weak, MIN, SADDLE, 600.0, require="exploratory")
    assert rate.qualification == "unqualified"
    assert any("exploratory" in a for a in rate.assumptions)
