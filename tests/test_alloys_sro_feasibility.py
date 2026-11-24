"""Target feasibility and achieved-ordering validation (A2.6) on small periodic BCC cells."""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
import pytest

from matsim_agents.discovery.alloys import (
    AlloySpec,
    InvalidSROTargetError,
    SROTarget,
    generate_alloy,
    generate_targeted_sro,
    validate_targeted_sro,
    warren_cowley_sro,
)

A = 3.16
BINARY = AlloySpec(counts={"Mo": 32, "W": 32}, repeats=(4, 4, 2), seed=5)
TERNARY = AlloySpec(counts={"Mo": 24, "Ta": 16, "W": 24}, repeats=(4, 4, 2), seed=2)
TARGET = [SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.5)]


@pytest.fixture(scope="module")
def start():
    return generate_alloy(BINARY)


@pytest.fixture(scope="module")
def result(start):
    return generate_targeted_sro(start, A, TARGET, seed=4, max_attempts=2000)


def test_valid_target_runs_and_validates(start, result):
    check = validate_targeted_sro(start, result, A, TARGET)
    assert check.reproducibility_checked
    assert check.status == result.status
    assert len(check.comparisons) == 1
    entry = check.comparisons[0]
    assert (entry.shell, entry.pair, entry.target) == (1, ("Mo", "W"), -0.5)
    assert entry.achieved == pytest.approx(result.sro.first_shell.alpha[0, 1])
    assert entry.deviation == pytest.approx(entry.achieved + 0.5)


@pytest.mark.parametrize(
    "target",
    [
        SROTarget(shell=1, pair=("Mo", "W"), alpha=-1.2),
        SROTarget(shell=1, pair=("Mo", "W"), alpha=1.2),
        SROTarget(shell=2, pair=("Mo", "Mo"), alpha=1.5),
    ],
    ids=["below", "above", "diagonal-above"],
)
def test_out_of_bounds_target_is_invalid(start, target):
    with pytest.raises(InvalidSROTargetError):
        generate_targeted_sro(start, A, [target], seed=0, max_attempts=10)


def test_composition_bound_is_tighter_than_unit_interval():
    skewed = generate_alloy(AlloySpec(counts={"Mo": 48, "W": 16}, repeats=(4, 4, 2), seed=1))
    # P(Mo|Mo) >= 1 - c_W / c_Mo = 2/3 forces alpha_MoMo <= 1 - (2/3)/0.75 = 1/9.
    with pytest.raises(InvalidSROTargetError):
        generate_targeted_sro(
            skewed, A, [SROTarget(shell=1, pair=("Mo", "Mo"), alpha=0.5)], seed=0, max_attempts=10
        )
    generate_targeted_sro(
        skewed, A, [SROTarget(shell=1, pair=("Mo", "Mo"), alpha=0.1)], seed=0, max_attempts=10
    )


def test_unknown_species_and_nonfinite_values(start):
    with pytest.raises(InvalidSROTargetError):
        generate_targeted_sro(
            start, A, [SROTarget(shell=1, pair=("Mo", "Ta"), alpha=0.0)], seed=0, max_attempts=10
        )
    with pytest.raises(ValueError):
        SROTarget(shell=1, pair=("Mo", "W"), alpha=float("nan"))
    with pytest.raises(ValueError):
        SROTarget(shell=4, pair=("Mo", "W"), alpha=0.0)


def test_contradictory_reciprocal_targets_are_invalid(start):
    targets = [
        SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.2),
        SROTarget(shell=1, pair=("W", "Mo"), alpha=-0.4),
    ]
    with pytest.raises(InvalidSROTargetError, match="contradictory"):
        generate_targeted_sro(start, A, targets, seed=0, max_attempts=10)


def test_consistent_reciprocal_targets_are_accepted(start):
    targets = [
        SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.3),
        SROTarget(shell=1, pair=("W", "Mo"), alpha=-0.3),
    ]
    out = generate_targeted_sro(start, A, targets, seed=0, max_attempts=50)
    assert validate_targeted_sro(start, out, A, targets).comparisons[0].target == -0.3


def test_complete_row_must_be_normalized(start):
    # Row Mo: c_Mo alpha_MoMo + c_W alpha_MoW = 0 requires alpha_MoMo = -alpha_MoW for 50/50.
    bad = [
        SROTarget(shell=1, pair=("Mo", "Mo"), alpha=0.2),
        SROTarget(shell=1, pair=("Mo", "W"), alpha=0.2),
    ]
    with pytest.raises(InvalidSROTargetError, match="sum_j"):
        generate_targeted_sro(start, A, bad, seed=0, max_attempts=10)
    good = [
        SROTarget(shell=1, pair=("Mo", "Mo"), alpha=0.2),
        SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.2),
        SROTarget(shell=1, pair=("W", "W"), alpha=0.2),
    ]
    out = generate_targeted_sro(start, A, good, seed=0, max_attempts=50)
    assert len(validate_targeted_sro(start, out, A, good).comparisons) == 3


def test_partial_rows_are_supported_without_constraining_other_pairs():
    ternary = generate_alloy(TERNARY)
    targets = [SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.3)]
    out = generate_targeted_sro(ternary, A, targets, seed=1, max_attempts=1500)
    check = validate_targeted_sro(ternary, out, A, targets)
    assert [(c.shell, c.pair) for c in check.comparisons] == [(1, ("Mo", "W"))]
    # Unspecified pairs are left free to move.
    before = warren_cowley_sro(ternary, A).first_shell.alpha
    assert not np.allclose(out.sro.first_shell.alpha, before)


def test_overfull_partial_row_is_invalid():
    ternary = generate_alloy(TERNARY)
    # Specified P(j|Mo) = c_j (1 - alpha) = (0.375 + 0.25 + 0.375) * 2 > 1.
    targets = [
        SROTarget(shell=1, pair=("Mo", "Mo"), alpha=-1.0),
        SROTarget(shell=1, pair=("Mo", "Ta"), alpha=-1.0),
    ]
    with pytest.raises(InvalidSROTargetError):
        generate_targeted_sro(ternary, A, targets, seed=0, max_attempts=10)


def test_exactly_achievable_target_from_existing_configuration():
    reference = generate_alloy(BINARY)
    symbols = reference.get_chemical_symbols()
    goal = float(warren_cowley_sro(reference, A).first_shell.alpha[0, 1])
    nearby = reference.copy()
    # One Mo/W swap that changes alpha, so the goal is reachable by construction.
    mo = [k for k, s in enumerate(symbols) if s == "Mo"]
    w = [k for k, s in enumerate(symbols) if s == "W"]
    for a, b in ((a, b) for a in mo for b in w):
        swapped = list(symbols)
        swapped[a], swapped[b] = swapped[b], swapped[a]
        nearby.set_chemical_symbols(swapped)
        if not np.isclose(warren_cowley_sro(nearby, A).first_shell.alpha[0, 1], goal):
            break
    targets = [SROTarget(shell=1, pair=("Mo", "W"), alpha=goal)]
    assert not np.isclose(warren_cowley_sro(nearby, A).first_shell.alpha[0, 1], goal)
    out = generate_targeted_sro(nearby, A, targets, seed=0, tolerance=1e-12, max_attempts=5000)
    assert out.converged and out.status == "converged"
    assert out.objective <= 1e-12
    check = validate_targeted_sro(nearby, out, A, targets)
    assert check.status == "converged"
    assert check.max_abs_deviation < 1e-6


def test_plausible_target_not_reached_is_not_invalid(start):
    hard = [SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.95)]
    out = generate_targeted_sro(start, A, hard, seed=0, tolerance=1e-12, max_attempts=5)
    assert not out.converged
    assert out.status == "not_converged"
    assert out.attempts == 5
    check = validate_targeted_sro(start, out, A, hard)
    assert check.status == "not_converged"
    assert check.max_abs_deviation > 0


def test_reported_alpha_and_objective_match_remeasurement(start, result):
    fresh = warren_cowley_sro(result.atoms, A, result.sro.species)
    assert np.allclose(fresh.first_shell.alpha, result.sro.first_shell.alpha, atol=1e-12)
    assert np.allclose(fresh.second_shell.alpha, result.sro.second_shell.alpha, atol=1e-12)
    expected = (fresh.first_shell.alpha[0, 1] + 0.5) ** 2
    assert result.objective == pytest.approx(expected, abs=1e-12)
    check = validate_targeted_sro(start, result, A, TARGET)
    assert check.objective == pytest.approx(result.objective, abs=1e-12)


def test_composition_sites_and_cell_preserved_exactly(start, result):
    assert Counter(result.atoms.get_chemical_symbols()) == Counter(start.get_chemical_symbols())
    assert np.array_equal(result.atoms.positions, start.positions)
    assert np.array_equal(result.atoms.cell, start.cell)
    assert result.atoms.pbc.all()


def test_identical_seed_is_deterministic(start, result):
    again = generate_targeted_sro(start, A, TARGET, seed=4, max_attempts=2000)
    assert again.atoms.get_chemical_symbols() == result.atoms.get_chemical_symbols()
    assert (again.objective, again.attempts, again.status) == (
        result.objective,
        result.attempts,
        result.status,
    )


def test_validation_detects_inconsistent_results(start, result):
    symbols = result.atoms.get_chemical_symbols()
    other = result.atoms.copy()
    swapped = list(symbols)
    a = symbols.index("Mo")
    b = symbols.index("W")
    swapped[a], swapped[b] = swapped[b], swapped[a]
    other.set_chemical_symbols(swapped)
    with pytest.raises(ValueError, match="does not match"):
        validate_targeted_sro(start, dataclasses.replace(result, atoms=other), A, TARGET)
    with pytest.raises(ValueError, match="objective"):
        validate_targeted_sro(
            start, dataclasses.replace(result, objective=result.objective + 0.1), A, TARGET
        )
    with pytest.raises(ValueError, match="convergence"):
        validate_targeted_sro(
            start, dataclasses.replace(result, converged=not result.converged), A, TARGET
        )
    with pytest.raises(ValueError, match="reproducible"):
        validate_targeted_sro(start, dataclasses.replace(result, seed=result.seed + 1), A, TARGET)


def test_validation_detects_tampered_composition_and_conditional_probability(start, result):
    bad_composition = dataclasses.replace(
        result.sro, composition=result.sro.composition[::-1] * 0.9
    )
    with pytest.raises(ValueError, match="composition"):
        validate_targeted_sro(start, dataclasses.replace(result, sro=bad_composition), A, TARGET)
    shell = result.sro.first_shell
    bad_shell = dataclasses.replace(
        shell, conditional_probability=shell.conditional_probability + 0.01
    )
    bad_sro = dataclasses.replace(result.sro, first_shell=bad_shell)
    with pytest.raises(ValueError, match="first_shell"):
        validate_targeted_sro(start, dataclasses.replace(result, sro=bad_sro), A, TARGET)


@pytest.mark.parametrize("seed", [2, 4, 5])
def test_already_met_target_converges_at_zero_tolerance_on_skewed_composition(seed):
    spec = AlloySpec(counts={"Mo": 3, "W": 13}, repeats=(2, 2, 2), seed=seed)
    atoms = generate_alloy(spec)
    measured = warren_cowley_sro(atoms, A).first_shell
    assert measured.pair_counts[0, 1] == 20
    target = [SROTarget(shell=1, pair=("Mo", "W"), alpha=float(measured.alpha[0, 1]))]
    result = generate_targeted_sro(atoms, A, target, seed=0, tolerance=0.0, max_attempts=0)
    assert result.objective == 0.0 and result.converged
    assert validate_targeted_sro(atoms, result, A, target).status == "converged"


def test_validation_detects_changed_counts_sites_and_cell(start, result):
    mutated = result.atoms.copy()
    mutated.set_chemical_symbols(["W"] * len(mutated))
    with pytest.raises(ValueError, match="counts"):
        validate_targeted_sro(start, dataclasses.replace(result, atoms=mutated), A, TARGET)
    moved = result.atoms.copy()
    moved.positions[0] += 0.01
    with pytest.raises(ValueError, match="positions"):
        validate_targeted_sro(start, dataclasses.replace(result, atoms=moved), A, TARGET)
    scaled = result.atoms.copy()
    scaled.set_cell(scaled.cell * 1.0 + np.eye(3) * 1e-3, scale_atoms=False)
    with pytest.raises(ValueError, match="cell"):
        validate_targeted_sro(start, dataclasses.replace(result, atoms=scaled), A, TARGET)
