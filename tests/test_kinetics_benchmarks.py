"""Kinetic-model analytical benchmarks (synthetic networks)."""

import pytest

from matsim_agents.science.kinetics.benchmarks import (
    check_arrhenius,
    check_branching,
    check_first_passage,
    run_kinetic_benchmarks,
)


def test_standard_benchmarks_pass_and_scope_is_limited():
    report = run_kinetic_benchmarks(seed=7, n_trajectories=1500)
    assert report.verdict == "pass", [r for r in report.results if r.verdict == "fail"]
    assert report.supports == ("analytical-benchmarks",)
    assert all(r.synthetic for r in report.results)
    assert any("attempt frequencies" in c for c in report.caveats)


def test_benchmarks_are_deterministic():
    assert run_kinetic_benchmarks(seed=3, n_trajectories=300) == run_kinetic_benchmarks(
        seed=3, n_trajectories=300
    )


def test_first_passage_negative_control_fails():
    wrong = check_first_passage(2.0, n_trajectories=1500, seed=1, expected_override=1.0)
    assert wrong.verdict == "fail"


def test_branching_negative_control_fails():
    wrong = check_branching(1.0, 3.0, n_trajectories=1500, seed=1, expected_override=0.5)
    assert wrong.verdict == "fail"


def test_branching_interval_contains_exact_ratio():
    out = check_branching(1.0, 3.0, n_trajectories=2000, seed=5)
    assert out.expected == pytest.approx(0.25)
    assert out.low <= 0.25 <= out.high and out.verdict == "pass"


def test_arrhenius_matches_hand_value():
    out = check_arrhenius(0.0, 300.0, 1.0e13)
    assert out.verdict == "pass" and out.expected == pytest.approx(1.0e13)


@pytest.mark.parametrize(
    "call",
    [
        lambda: check_first_passage(0.0, n_trajectories=100, seed=1),
        lambda: check_first_passage(1.0, n_trajectories=5, seed=1),
        lambda: check_first_passage(1.0, n_trajectories=100, seed=1, z=0.0),
        lambda: check_branching(1.0, -1.0, n_trajectories=100, seed=1),
        lambda: check_branching(1.0, 1.0, n_trajectories=5, seed=1),
        lambda: check_arrhenius(0.5, 0.0, 1.0e13),
    ],
)
def test_invalid_inputs_are_rejected(call):
    with pytest.raises(ValueError):
        call()
