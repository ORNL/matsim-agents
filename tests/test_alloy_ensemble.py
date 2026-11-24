"""Independent configurational ensembles (A2.7): random and targeted-SRO, small BCC cells."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from matsim_agents.discovery.alloy_ensemble import (
    generate_alloy_ensemble,
    occupation_fingerprint,
)
from matsim_agents.discovery.alloys import (
    AlloySpec,
    SROTarget,
    generate_alloy,
    generate_targeted_sro,
    warren_cowley_sro,
)

A = 3.16
SMALL = AlloySpec(counts={"Mo": 16, "W": 16}, repeats=(4, 2, 2), seed=0)
TERNARY = AlloySpec(counts={"Mo": 12, "Ta": 8, "W": 12}, repeats=(4, 2, 2), seed=0)
REFERENCE = AlloySpec(counts={"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}, seed=0)
TARGET = [SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.5)]


@pytest.fixture(scope="module")
def random_ensemble():
    return generate_alloy_ensemble(SMALL, n_configurations=6, mode="random", seed=11)


@pytest.fixture(scope="module")
def targeted_ensemble():
    return generate_alloy_ensemble(
        SMALL,
        n_configurations=3,
        mode="targeted_sro",
        seed=5,
        sro_targets=TARGET,
        max_swap_attempts=1500,
    )


def test_requested_number_and_ids(random_ensemble):
    assert len(random_ensemble.records) == len(random_ensemble.structures) == 6
    assert [r.config_id for r in random_ensemble.records] == [f"random-{k:04d}" for k in range(6)]
    assert random_ensemble.spec.n_configurations == 6


def test_size_is_not_capped_at_five():
    out = generate_alloy_ensemble(SMALL, n_configurations=9, mode="random", seed=0)
    assert len(out.records) == 9


def test_composition_and_geometry_preserved(random_ensemble, targeted_ensemble):
    lattice = generate_alloy(SMALL)
    for ensemble in (random_ensemble, targeted_ensemble):
        for record, atoms in zip(ensemble.records, ensemble.structures, strict=True):
            assert Counter(atoms.get_chemical_symbols()) == Counter(SMALL.counts)
            assert record.counts == dict(SMALL.counts)
            assert np.array_equal(atoms.positions, lattice.positions)
            assert np.array_equal(atoms.cell, lattice.cell)
            assert atoms.pbc.all()


def test_structures_are_distinct_objects(random_ensemble):
    structures = random_ensemble.structures
    assert len({id(atoms) for atoms in structures}) == 6
    assert len({atoms.positions.ctypes.data for atoms in structures}) == 6
    assert len({atoms.numbers.ctypes.data for atoms in structures}) == 6


def test_deterministic_replay(random_ensemble):
    again = generate_alloy_ensemble(SMALL, n_configurations=6, mode="random", seed=11)
    assert [a.get_chemical_symbols() for a in again.structures] == [
        a.get_chemical_symbols() for a in random_ensemble.structures
    ]
    assert [r.seed for r in again.records] == [r.seed for r in random_ensemble.records]
    assert json.dumps(again.metadata()) == json.dumps(random_ensemble.metadata())


def test_replay_does_not_touch_global_numpy_state():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    generate_alloy_ensemble(SMALL, n_configurations=3, mode="random", seed=1)
    assert np.array_equal(np.random.get_state()[1], before)


def test_configurations_use_different_streams(random_ensemble):
    seeds = [r.seed for r in random_ensemble.records]
    assert len(set(seeds)) == len(seeds)
    assert random_ensemble.n_unique > 1
    other = generate_alloy_ensemble(SMALL, n_configurations=6, mode="random", seed=12)
    assert [r.seed for r in other.records] != seeds


def test_fingerprints_follow_occupation(random_ensemble):
    for record, atoms in zip(random_ensemble.records, random_ensemble.structures, strict=True):
        assert record.fingerprint == occupation_fingerprint(atoms)
        assert len(record.fingerprint) == 64
    swapped = random_ensemble.structures[0].copy()
    symbols = swapped.get_chemical_symbols()
    a, b = symbols.index("Mo"), symbols.index("W")
    symbols[a], symbols[b] = symbols[b], symbols[a]
    swapped.set_chemical_symbols(symbols)
    assert occupation_fingerprint(swapped) != random_ensemble.records[0].fingerprint
    same = random_ensemble.structures[0].copy()
    assert occupation_fingerprint(same) == random_ensemble.records[0].fingerprint


def test_exact_duplicates_are_detected_and_reported():
    # 1 Mo + 1 W on a 2-site cell has only two occupations, so 8 configurations must repeat.
    tiny = AlloySpec(counts={"Mo": 1, "W": 1}, repeats=(1, 1, 1), seed=0)
    out = generate_alloy_ensemble(tiny, n_configurations=8, mode="random", seed=0)
    assert out.n_unique <= 2
    assert out.duplicate_groups
    grouped = {config_id for group in out.duplicate_groups for config_id in group}
    for group in out.duplicate_groups:
        fingerprints = {r.fingerprint for r in out.records if r.config_id in group}
        assert len(fingerprints) == 1
    assert grouped <= {r.config_id for r in out.records}
    assert len(out.records) == 8


def test_no_duplicates_reported_when_all_distinct(random_ensemble):
    assert random_ensemble.n_unique == len(random_ensemble.records)
    assert random_ensemble.duplicate_groups == ()


def test_sro_metadata_matches_remeasurement(random_ensemble, targeted_ensemble):
    for ensemble in (random_ensemble, targeted_ensemble):
        for record, atoms in zip(ensemble.records, ensemble.structures, strict=True):
            fresh = warren_cowley_sro(atoms, A, record.sro.species)
            for shell in ("first_shell", "second_shell"):
                got, want = getattr(record.sro, shell), getattr(fresh, shell)
                assert np.array_equal(got.pair_counts, want.pair_counts)
                assert np.allclose(got.alpha, want.alpha, atol=1e-12)
    data = random_ensemble.metadata()["configurations"][0]
    assert data["species"] == ["Mo", "W"]
    assert np.array(data["first_shell"]["alpha"]).shape == (2, 2)
    assert np.array(data["second_shell"]["pair_counts"]).shape == (2, 2)
    assert "positions" not in json.dumps(data)


def test_targeted_records_report_objective_and_status(targeted_ensemble):
    for record, atoms in zip(targeted_ensemble.records, targeted_ensemble.structures, strict=True):
        assert record.mode == "targeted_sro"
        assert record.optimizer_seed is not None
        alpha = warren_cowley_sro(atoms, A, record.sro.species).first_shell.alpha[0, 1]
        assert record.objective == pytest.approx((alpha + 0.5) ** 2, abs=1e-12)
        assert record.converged == (record.objective <= targeted_ensemble.spec.tolerance)
        assert 0 <= record.attempts <= 1500


def test_random_records_have_no_optimizer_fields(random_ensemble):
    for record in random_ensemble.records:
        assert record.converged is None
        assert record.objective is None
        assert record.attempts is None
        assert record.optimizer_seed is None


def test_targeted_replay_is_deterministic(targeted_ensemble):
    again = generate_alloy_ensemble(
        SMALL,
        n_configurations=3,
        mode="targeted_sro",
        seed=5,
        sro_targets=TARGET,
        max_swap_attempts=1500,
    )
    assert [r.fingerprint for r in again.records] == [
        r.fingerprint for r in targeted_ensemble.records
    ]
    assert json.dumps(again.metadata()) == json.dumps(targeted_ensemble.metadata())


def test_nonconvergence_is_reported_not_raised():
    out = generate_alloy_ensemble(
        SMALL,
        n_configurations=2,
        mode="targeted_sro",
        seed=3,
        sro_targets=[SROTarget(shell=1, pair=("Mo", "W"), alpha=-0.9)],
        max_swap_attempts=2,
        tolerance=1e-12,
    )
    assert all(r.converged is False for r in out.records)
    assert all(r.attempts == 2 for r in out.records)


def test_convergence_reported_when_target_already_met():
    out = generate_alloy_ensemble(
        SMALL,
        n_configurations=2,
        mode="targeted_sro",
        seed=3,
        sro_targets=TARGET,
        max_swap_attempts=0,
        tolerance=10.0,
    )
    assert all(r.converged is True and r.attempts == 0 for r in out.records)


def test_ternary_ensemble_composition_preserved():
    out = generate_alloy_ensemble(TERNARY, n_configurations=3, mode="random", seed=2)
    for atoms in out.structures:
        assert Counter(atoms.get_chemical_symbols()) == Counter(TERNARY.counts)
    assert out.records[0].sro.species == ("Mo", "Ta", "W")


@pytest.mark.parametrize("n", [0, -1, True, 2.0, "3"])
def test_invalid_ensemble_sizes_are_rejected(n):
    with pytest.raises(ValueError):
        generate_alloy_ensemble(SMALL, n_configurations=n, mode="random", seed=0)


@pytest.mark.parametrize("mode", ["sqs", "clustered", "", "Random"])
def test_unsupported_modes_are_rejected(mode):
    with pytest.raises(ValueError):
        generate_alloy_ensemble(SMALL, n_configurations=2, mode=mode, seed=0)


def test_mode_and_target_consistency_and_seed_validation():
    with pytest.raises(ValueError):
        generate_alloy_ensemble(SMALL, n_configurations=2, mode="targeted_sro", seed=0)
    with pytest.raises(ValueError):
        generate_alloy_ensemble(
            SMALL, n_configurations=2, mode="random", seed=0, sro_targets=TARGET
        )
    with pytest.raises(ValueError):
        generate_alloy_ensemble(SMALL, n_configurations=2, mode="random", seed=-1)
    with pytest.raises(ValueError):
        generate_alloy_ensemble(
            SMALL, n_configurations=2, mode="random", seed=0, max_swap_attempts=-1
        )


def test_invalid_targets_are_rejected_before_generation():
    with pytest.raises(ValueError):
        generate_alloy_ensemble(
            SMALL,
            n_configurations=2,
            mode="targeted_sro",
            seed=0,
            sro_targets=[SROTarget(shell=1, pair=("Mo", "W"), alpha=-1.5)],
        )


def test_overlap_matrix_is_symmetric_with_unit_diagonal(random_ensemble):
    matrix = random_ensemble.overlap_matrix()
    assert matrix.shape == (6, 6)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.array_equal(matrix, matrix.T)
    a, b = random_ensemble.structures[0], random_ensemble.structures[1]
    assert matrix[0, 1] == pytest.approx(float((a.numbers == b.numbers).mean()))


def test_overlap_of_duplicates_is_one_and_summary_is_serializable():
    tiny = AlloySpec(counts={"Mo": 1, "W": 1}, repeats=(1, 1, 1), seed=0)
    out = generate_alloy_ensemble(tiny, n_configurations=8, mode="random", seed=0)
    first_group = out.duplicate_groups[0]
    ids = [r.config_id for r in out.records]
    i, j = ids.index(first_group[0]), ids.index(first_group[1])
    assert out.overlap_matrix()[i, j] == 1.0
    assert json.loads(json.dumps(out.metadata()))["overlap"]["max"] == 1.0


def test_random_overlap_is_consistent_with_expectation_and_single_config_has_none():
    spec = AlloySpec(counts={"Mo": 128, "W": 128}, repeats=(8, 4, 4), seed=0)
    out = generate_alloy_ensemble(spec, n_configurations=6, mode="random", seed=3)
    summary = out.overlap_summary()
    assert out.expected_random_overlap == pytest.approx(0.5)
    # sd of one pair is about 0.03 for 256 sites; 6 sigma over 15 pairs is a safe bound.
    assert abs(summary["mean"] - 0.5) < 0.05
    assert 0.3 < summary["min"] <= summary["mean"] <= summary["max"] < 0.7
    single = generate_alloy_ensemble(spec, n_configurations=1, mode="random", seed=3)
    assert single.overlap_summary()["mean"] is None


def test_reference_1024_atom_random_ensemble():
    out = generate_alloy_ensemble(REFERENCE, n_configurations=3, mode="random", seed=0)
    assert len(out.structures) == 3
    for atoms in out.structures:
        assert len(atoms) == 1024
        assert Counter(atoms.get_chemical_symbols()) == Counter(REFERENCE.counts)
    assert out.n_unique == 3
    sro = out.records[0].sro
    assert sro.first_shell.pair_counts.shape == (6, 6)
    assert sro.first_shell.pair_counts.sum() == 1024 * 8
    assert sro.second_shell.pair_counts.sum() == 1024 * 6


def test_record_seeds_replay_through_the_public_generators(targeted_ensemble):
    for record, atoms in zip(targeted_ensemble.records, targeted_ensemble.structures, strict=True):
        start = generate_alloy(SMALL.model_copy(update={"seed": record.seed}))
        replay = generate_targeted_sro(
            start,
            A,
            TARGET,
            seed=record.optimizer_seed,
            tolerance=targeted_ensemble.spec.tolerance,
            max_attempts=targeted_ensemble.spec.max_swap_attempts,
        )
        assert replay.atoms.get_chemical_symbols() == atoms.get_chemical_symbols()


def test_existing_generation_is_unchanged_by_topology_reuse():
    atoms = generate_alloy(SMALL)
    assert warren_cowley_sro(atoms, A).first_shell.pair_counts.sum() == 32 * 8
