"""Targeted-SRO generation (A2.5): bounded greedy swaps on a fixed BCC lattice."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from matsim_agents.discovery.alloys import (
    AlloySpec,
    SROTarget,
    generate_alloy,
    generate_targeted_sro,
    warren_cowley_sro,
)

A = 3.16
SPEC = AlloySpec(counts={"Mo": 64, "W": 64}, repeats=(4, 4, 4), seed=3)
ORDERING_TARGET = [SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.5)]


@pytest.fixture(scope="module")
def start():
    return generate_alloy(SPEC)


@pytest.fixture(scope="module")
def result(start):
    return generate_targeted_sro(start, A, ORDERING_TARGET, seed=7, max_attempts=3000)


def test_preserves_counts_sites_and_input(start, result):
    before = start.get_chemical_symbols()
    assert Counter(result.atoms.get_chemical_symbols()) == Counter(before)
    assert np.array_equal(result.atoms.positions, start.positions)
    assert np.array_equal(result.atoms.cell, start.cell)
    assert result.atoms.get_chemical_symbols() != before
    assert start.get_chemical_symbols() == before


def test_deterministic_for_fixed_seed_and_seed_is_reported(start, result):
    again = generate_targeted_sro(start, A, ORDERING_TARGET, seed=7, max_attempts=3000)
    assert again.atoms.get_chemical_symbols() == result.atoms.get_chemical_symbols()
    assert (again.objective, again.attempts) == (result.objective, result.attempts)
    assert result.seed == 7
    other = generate_targeted_sro(start, A, ORDERING_TARGET, seed=8, max_attempts=3000)
    assert other.atoms.get_chemical_symbols() != result.atoms.get_chemical_symbols()


def test_feasible_target_reduces_objective_and_reports_matching_sro(start, result):
    initial_alpha = warren_cowley_sro(start, A).first_shell.alpha[0, 1]
    assert result.objective < 0.1 * result.initial_objective
    achieved = result.sro.first_shell.alpha[0, 1]
    assert abs(achieved + 0.5) < abs(initial_alpha + 0.5)
    assert result.objective == pytest.approx((achieved + 0.5) ** 2, abs=1e-12)
    assert result.converged == (result.objective <= result.tolerance)
    assert result.accepted_swaps > 0


def test_symmetric_pair_order_gives_identical_result(start, result):
    reversed_target = [SROTarget(shell=1, pair=("W", "Mo"), alpha=-0.5)]
    mirrored = generate_targeted_sro(start, A, reversed_target, seed=7, max_attempts=3000)
    assert mirrored.atoms.get_chemical_symbols() == result.atoms.get_chemical_symbols()


def test_second_shell_target_leaves_untargeted_shell_unconstrained(start):
    targets = [SROTarget(shell=2, pair=("Mo", "Mo"), alpha=0.3)]
    out = generate_targeted_sro(start, A, targets, seed=1, max_attempts=3000)
    assert out.objective < out.initial_objective
    assert out.sro.second_shell.alpha[0, 0] > warren_cowley_sro(start, A).second_shell.alpha[0, 0]


def test_stops_at_attempt_limit_for_unreachable_tolerance(start):
    out = generate_targeted_sro(start, A, ORDERING_TARGET, seed=0, tolerance=0.0, max_attempts=40)
    assert out.attempts == 40
    assert not out.converged
    assert out.objective == pytest.approx((out.sro.first_shell.alpha[0, 1] + 0.5) ** 2)


def test_zero_attempts_returns_start_unchanged(start):
    out = generate_targeted_sro(start, A, ORDERING_TARGET, seed=0, max_attempts=0)
    assert out.attempts == 0
    assert out.atoms.get_chemical_symbols() == start.get_chemical_symbols()
    assert out.objective == out.initial_objective


def test_met_target_stops_immediately(start):
    measured = warren_cowley_sro(start, A).first_shell.alpha[0, 1]
    target = [SROTarget(shell=1, pair=("Mo", "W"), alpha=float(measured))]
    out = generate_targeted_sro(start, A, target, seed=0, tolerance=1e-9)
    assert out.converged
    assert out.attempts == 0


def test_perfect_b2_is_reachable_target_and_never_exceeds_limit(start):
    target = [SROTarget(shell=1, pair=("Mo", "W"), alpha=-1.0)]
    out = generate_targeted_sro(start, A, target, seed=2, tolerance=1e-12, max_attempts=500)
    assert out.attempts <= 500
    assert out.objective <= out.initial_objective


@pytest.mark.parametrize(
    "targets",
    [
        [SROTarget(shell=1, pair=("Mo", "W"), alpha=-1.5)],
        [SROTarget(shell=1, pair=("Mo", "W"), alpha=1.5)],
        [SROTarget(shell=1, pair=("Mo", "Ta"), alpha=0.0)],
        [
            SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.2),
            SROTarget(shell=1, pair=("W", "Mo"), alpha=-0.4),
        ],
        [
            SROTarget(shell=1, pair=("Mo", "Mo"), alpha=0.2),
            SROTarget(shell=1, pair=("Mo", "W"), alpha=0.2),
        ],
        [],
    ],
    ids=["below-range", "above-one", "unknown-species", "conflict", "row-sum", "empty"],
)
def test_rejects_inconsistent_targets(start, targets):
    with pytest.raises(ValueError):
        generate_targeted_sro(start, A, targets, seed=0, max_attempts=10)


@pytest.mark.parametrize(
    "kwargs",
    [{"seed": -1}, {"seed": True}, {"max_attempts": -1}, {"tolerance": float("nan")}],
)
def test_rejects_invalid_search_parameters(start, kwargs):
    params = {"seed": 0, "max_attempts": 10} | kwargs
    with pytest.raises(ValueError):
        generate_targeted_sro(start, A, ORDERING_TARGET, **params)


def test_rejects_invalid_structures(start):
    pure = generate_alloy(AlloySpec(counts={"W": 8}, repeats=(2, 2, 1), seed=0))
    with pytest.raises(ValueError):
        generate_targeted_sro(pure, A, ORDERING_TARGET, seed=0, max_attempts=10)
    open_cell = start.copy()
    open_cell.pbc = False
    with pytest.raises(ValueError):
        generate_targeted_sro(open_cell, A, ORDERING_TARGET, seed=0, max_attempts=10)
    distorted = start.copy()
    distorted.positions[0] += 0.5
    with pytest.raises(ValueError):
        generate_targeted_sro(distorted, A, ORDERING_TARGET, seed=0, max_attempts=10)


def test_invalid_shell_is_rejected_by_target_model():
    with pytest.raises(ValueError):
        SROTarget(shell=3, pair=("Mo", "W"), alpha=0.0)
    with pytest.raises(ValueError):
        SROTarget(shell=1, pair=("Mo", "W"), alpha=float("inf"))
    with pytest.raises(ValueError):
        SROTarget(shell=1, pair=("Mo", "W"), alpha=0.0, weight=0.0)
