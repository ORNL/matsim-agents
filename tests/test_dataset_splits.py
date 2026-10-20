"""Objective-aware dataset splits: strategies, manifests, and leakage checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from matsim_agents.datasets.governance import DatasetValidationSummary, write_dataset_manifest
from matsim_agents.datasets.splits import (
    SPLIT_MANIFEST,
    SplitConfig,
    TestDomainConfig,
    load_split,
    make_split,
)


def _frame(symbols, *, trajectory, step, iteration, structure, temperature, shift=0.0):
    n = len(symbols)
    rng = np.random.default_rng(abs(hash((trajectory, step))) % 2**32)
    positions = np.array([[i * 2.1 + shift, 0.0, 0.0] for i in range(n)])
    positions += 0.05 * step + 0.02 * rng.standard_normal((n, 3))
    atoms = Atoms(symbols, positions=positions, cell=[2.1 * n + 4, 6, 6], pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=-float(n) - 0.01 * step, forces=np.zeros((n, 3))
    )
    atoms.info.update(
        trajectory_id=trajectory,
        time_index=step,
        al_iteration=iteration,
        initial_structure_id=structure,
        temperature_K=temperature,
    )
    return atoms


def _write(path: Path, frames, *, backend="qe", signature="pbe-test"):
    write(path, frames, format="extxyz")
    write_dataset_manifest(
        path,
        dft_backend=backend,
        energy_reference=f"{backend}:native_total_energy",
        validation=DatasetValidationSummary(accepted=len(frames)),
        method_signature=signature,
    )
    return path


@pytest.fixture
def dataset(tmp_path):
    """8 trajectories: 2 compositions x 2 temperatures x 2 seeds, 10 steps, 4 iterations."""
    frames = []
    trajectory = 0
    for symbols in (["Nb", "O", "O"], ["Ta", "O", "O"]):
        for temperature in (300.0, 900.0):
            for seed in range(2):
                for step in range(10):
                    frames.append(
                        _frame(
                            symbols,
                            trajectory=f"traj-{trajectory}",
                            step=step,
                            iteration=trajectory // 2,
                            structure=f"{''.join(symbols)}-seed{seed}",
                            temperature=temperature,
                            shift=0.3 * trajectory,
                        )
                    )
                trajectory += 1
    return _write(tmp_path / "dataset.extxyz", frames)


def _roles(manifest, field):
    return {
        role: {getattr(entry, field) for entry in manifest.frames if entry.role == role}
        for role in ("train", "validation", "test")
    }


def test_random_frame_is_reproducible_and_keeps_the_old_test_frames(dataset, tmp_path):
    cfg = SplitConfig(strategy="random_frame", train_fraction=0.8, test_fraction=0.2, seed=7)
    first = make_split(dataset, cfg, tmp_path / "a", objective="interpolation")
    second = make_split(dataset, cfg, tmp_path / "b", objective="interpolation")
    assert first.roles["test"].config_ids == second.roles["test"].config_ids
    # The previous finetune_eval split: shuffle once, the first round(0.2 n) are test.
    order = np.arange(80)
    np.random.default_rng(7).shuffle(order)
    expected = {first.frames[i].config_id for i in order[:16].tolist()}
    assert set(first.roles["test"].config_ids) == expected
    assert any(
        c.name == "correlated_frames_across_roles" and c.status == "warn" for c in first.checks
    )
    assert any("interpolation among similar samples" in note for note in first.limitations)


THREE_ROLES = {"train_fraction": 0.5, "validation_fraction": 0.25, "test_fraction": 0.25}
TWO_ROLES = {"train_fraction": 0.5, "test_fraction": 0.5}


@pytest.mark.parametrize(
    ("strategy", "field", "extra"),
    [
        ("trajectory_holdout", "trajectory_id", THREE_ROLES),
        ("structure_holdout", "initial_structure_id", THREE_ROLES),
        ("composition_holdout", "composition", TWO_ROLES),
        (
            "condition_holdout",
            "group_id",
            {**TWO_ROLES, "conditions": [{"name": "temperature_K", "unit": "K"}]},
        ),
        ("generic_group", "group_id", {**THREE_ROLES, "group_by": "initial_structure_id"}),
    ],
)
def test_group_strategies_keep_each_group_in_one_role(dataset, tmp_path, strategy, field, extra):
    cfg = SplitConfig(strategy=strategy, seed=3, **extra)
    manifest = make_split(dataset, cfg, tmp_path / strategy, objective=f"{strategy} transfer")
    roles = _roles(manifest, field)
    assert all(roles[role] for role in manifest.roles)
    assert not (roles["train"] & roles["test"] or roles["train"] & roles["validation"])
    assert not manifest.failed_checks
    for role, membership in manifest.roles.items():
        assert len(read(membership.path, index=":")) == membership.n_frames
        sidecar = json.loads(Path(membership.path + ".manifest.json").read_text())
        assert sidecar["method_signature"] == "pbe-test"
        assert sidecar["split_role"] == {"train": "training_pool"}.get(role, role)


def test_composition_bins_group_nearby_alloys(dataset, tmp_path):
    cfg = SplitConfig(
        strategy="composition_holdout",
        train_fraction=0.5,
        test_fraction=0.5,
        composition={"mode": "bin", "bin_width": 0.5},
    )
    # Nb O2 and Ta O2 fall in different element bins, so there are two groups.
    manifest = make_split(dataset, cfg, tmp_path / "bins", objective="composition transfer")
    assert sorted(manifest.roles["train"].group_ids + manifest.roles["test"].group_ids) == [
        "Nb[0,0.5)O[0.5,1)",
        "O[0.5,1)Ta[0,0.5)",
    ]


def test_al_iteration_holdout_is_chronological(dataset, tmp_path):
    cfg = SplitConfig(
        strategy="al_iteration_holdout",
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
    )
    manifest = make_split(dataset, cfg, tmp_path / "al", objective="later batches")
    iterations = _roles(manifest, "al_iteration")
    assert max(iterations["train"]) < min(iterations["validation"])
    assert max(iterations["validation"]) < min(iterations["test"])
    assert next(c for c in manifest.checks if c.name == "chronological_order").status == "pass"


def test_temporal_blocks_follow_time_with_a_purge_gap(dataset, tmp_path):
    cfg = SplitConfig(
        strategy="temporal_block",
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        purge_gap_frames=1,
    )
    manifest = make_split(dataset, cfg, tmp_path / "time", objective="forward in time")
    assert len(manifest.purged) == 8 * 2  # two gaps per trajectory
    for trajectory in {entry.trajectory_id for entry in manifest.frames}:
        steps = {
            role: [
                e.time_index
                for e in manifest.frames
                if e.trajectory_id == trajectory and e.role == role
            ]
            for role in ("train", "validation", "test")
        }
        assert max(steps["train"]) + 1 < min(steps["validation"])
        assert max(steps["validation"]) + 1 < min(steps["test"])
    assert next(c for c in manifest.checks if c.name == "chronological_order").status == "pass"


def test_missing_metadata_fails_instead_of_falling_back(tmp_path):
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.3, 1.3, 1.3]], cell=[5, 5, 5], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((2, 3)))
    plain = _write(tmp_path / "plain.extxyz", [atoms] * 4)
    with pytest.raises(ValueError, match="needs info\\['trajectory_id'\\]"):
        make_split(plain, SplitConfig(strategy="trajectory_holdout"), tmp_path / "s", objective="x")


def test_too_few_groups_are_rejected(dataset, tmp_path):
    cfg = SplitConfig(
        strategy="composition_holdout",
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    with pytest.raises(ValueError, match="2 independent group\\(s\\) cannot fill the 3 roles"):
        make_split(dataset, cfg, tmp_path / "few", objective="composition transfer")


def test_explicit_groups_must_be_complete_and_disjoint(dataset, tmp_path):
    groups = ["NbOO-seed0", "NbOO-seed1", "TaOO-seed0", "TaOO-seed1"]
    with pytest.raises(ValueError, match="unassigned"):
        make_split(
            dataset,
            SplitConfig(
                strategy="structure_holdout",
                train_fraction=0.5,
                test_fraction=0.5,
                explicit_groups={"train": groups[:2], "test": groups[2:3]},
            ),
            tmp_path / "partial",
            objective="new seeds",
        )
    with pytest.raises(ValueError, match="more than one role"):
        make_split(
            dataset,
            SplitConfig(
                strategy="structure_holdout",
                train_fraction=0.5,
                test_fraction=0.5,
                explicit_groups={"train": groups[:3], "test": groups[2:]},
            ),
            tmp_path / "contaminated",
            objective="new seeds",
        )


def _domain(tmp_path, name, frames, **kwargs):
    return _write(tmp_path / f"{name}.extxyz", frames, **kwargs)


def test_test_domains_must_be_unseen(dataset, tmp_path):
    cfg = SplitConfig(
        strategy="composition_holdout",
        train_fraction=0.5,
        test_fraction=0.5,
        explicit_groups={"train": ["NbO2"], "test": ["O2Ta"]},
    )
    fresh = [
        _frame(
            ["Nb", "Nb", "O"],
            trajectory="new",
            step=s,
            iteration=9,
            structure="new",
            temperature=300.0,
        )
        for s in range(3)
    ]
    leaking = [
        _frame(
            ["Nb", "O", "O"],
            trajectory="leak",
            step=s,
            iteration=9,
            structure="leak",
            temperature=300.0,
            shift=5.0,
        )
        for s in range(3)
    ]
    good = TestDomainConfig(
        name="new_alloys",
        dataset=_domain(tmp_path, "new_alloys", fresh),
        membership="composition_holdout",
    )
    manifest = make_split(
        dataset, cfg, tmp_path / "ok", objective="composition transfer", test_domains=[good]
    )
    assert manifest.test_domains[0].group_ids == ["Nb2O"]
    bad = TestDomainConfig(
        name="seen_composition",
        dataset=_domain(tmp_path, "seen", leaking),
        membership="composition_holdout",
    )
    with pytest.raises(ValueError, match="groups_unseen_seen_composition"):
        make_split(
            dataset, cfg, tmp_path / "bad", objective="composition transfer", test_domains=[bad]
        )
    # The failed split is still written for inspection.
    assert (tmp_path / "bad" / SPLIT_MANIFEST).is_file()


def test_domains_overlap_only_when_declared_and_share_the_dft_method(dataset, tmp_path):
    cfg = SplitConfig(strategy="trajectory_holdout", train_fraction=0.75, test_fraction=0.25)
    frames = [
        _frame(
            ["Nb", "O", "O"],
            trajectory="x",
            step=s,
            iteration=9,
            structure="x",
            temperature=1500.0,
            shift=9.0,
        )
        for s in range(3)
    ]
    path = _domain(tmp_path, "hot", frames)
    first = TestDomainConfig(name="hot", dataset=path, membership="trajectory_holdout")
    second = TestDomainConfig(name="hot_again", dataset=path, membership="trajectory_holdout")
    with pytest.raises(ValueError, match="domains_hot_hot_again"):
        make_split(dataset, cfg, tmp_path / "o1", objective="t", test_domains=[first, second])
    allowed = second.model_copy(update={"allow_overlap_with": ["hot"]})
    make_split(dataset, cfg, tmp_path / "o2", objective="t", test_domains=[first, allowed])
    other_method = TestDomainConfig(
        name="vasp",
        dataset=_domain(tmp_path, "vasp", frames, backend="vasp", signature="pbe-vasp"),
        membership="trajectory_holdout",
    )
    with pytest.raises(ValueError, match="dft_method_vasp"):
        make_split(dataset, cfg, tmp_path / "o3", objective="t", test_domains=[other_method])


def test_splits_are_immutable_and_verified_on_load(dataset, tmp_path):
    cfg = SplitConfig(strategy="trajectory_holdout", train_fraction=0.75, test_fraction=0.25)
    out = tmp_path / "split"
    manifest = make_split(dataset, cfg, out, objective="new trajectories")
    assert make_split(dataset, cfg, out, objective="new trajectories").content_sha256 == (
        manifest.content_sha256
    )
    with pytest.raises(ValueError, match="already holds a different split"):
        make_split(dataset, cfg.model_copy(update={"seed": 5}), out, objective="new trajectories")
    loaded = load_split(out)
    assert loaded.roles["test"].config_ids == manifest.roles["test"].config_ids
    test_file = Path(manifest.roles["test"].path)
    test_file.write_text(test_file.read_text().replace("traj", "TRAJ", 1))
    with pytest.raises(ValueError, match="test partition .* changed"):
        load_split(out)


def test_near_duplicates_across_roles_are_reported(tmp_path):
    frames = [
        _frame(
            ["Si", "Si"], trajectory=f"t{i}", step=0, iteration=0, structure="s", temperature=300.0
        )
        for i in range(4)
    ]
    for atoms in frames:  # identical geometry up to a rigid translation
        atoms.positions = np.array([[0.0, 0, 0], [2.1, 0, 0]]) + atoms.info["time_index"]
    for index, atoms in enumerate(frames):
        atoms.info["config_id"] = f"frame-{index}"
    path = _write(tmp_path / "dupes.extxyz", frames)
    manifest = make_split(
        path,
        SplitConfig(strategy="trajectory_holdout", train_fraction=0.5, test_fraction=0.5),
        tmp_path / "split",
        objective="new trajectories",
    )
    check = next(c for c in manifest.checks if c.name == "near_duplicates_across_roles")
    assert check.status == "warn"


def test_strategy_settings_are_validated():
    with pytest.raises(ValueError, match="sum to 1"):
        SplitConfig(strategy="random_frame", train_fraction=0.8, test_fraction=0.3)
    with pytest.raises(ValueError, match="requires group_by"):
        SplitConfig(strategy="generic_group")
    with pytest.raises(ValueError, match="purge_gap_frames applies to temporal_block only"):
        SplitConfig(strategy="random_frame", purge_gap_frames=3)
    with pytest.raises(ValueError, match="requires at least one condition"):
        SplitConfig(strategy="condition_holdout")
