"""Per-group evaluation reports, report aggregation, and uncertainty calibration."""

from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from matsim_agents.active_learning.eval_report import (
    collect_runs,
    group_report,
    matrix,
    paired_difference,
)
from matsim_agents.active_learning.uncertainty import calibrate_uncertainty


def _parity(values):
    values = np.asarray(values, dtype=float)
    return {
        "frame_force_mae_eV_per_A": values,
        "frame_force_vector_mae_eV_per_A": values * 1.5,
        "frame_energy_abs_err_eV_per_atom": values / 10,
        "frame_formation_abs_err_eV_per_atom": np.full(len(values), np.nan),
    }


def test_group_report_resamples_groups_and_names_the_worst():
    groups = ["a"] * 4 + ["b"] * 4 + ["c"] * 2
    report = group_report(_parity([0.1] * 4 + [0.2] * 4 + [0.9] * 2), groups, seed=1)
    force = report["metrics"]["force_mae_eV_per_A"]
    assert report["n_groups"] == 3 and report["n_frames"] == 10
    assert force["worst_group"]["group"] == "c" and force["worst_group"]["n_frames"] == 2
    assert force["frame_quantiles"]["max"] == pytest.approx(0.9)
    low, high = force["group_bootstrap_95"]
    assert low <= force["frame_mean"] <= high
    # A metric without labels is omitted rather than reported as zero.
    assert "formation_energy_mae_eV_per_atom" not in report["metrics"]
    single = group_report(_parity([0.1, 0.2]), ["only", "only"])
    assert single["metrics"]["force_mae_eV_per_A"]["group_bootstrap_95"] is None


def test_paired_differences_use_identical_groups():
    groups = ["a", "a", "b", "b"]
    before = group_report(_parity([0.2, 0.2, 0.4, 0.4]), groups)
    after = group_report(_parity([0.1, 0.1, 0.5, 0.5]), groups)
    diff = paired_difference(after, before)["force_mae_eV_per_A"]
    assert diff["mean_difference"] == pytest.approx(0.0)
    assert diff["groups_improved"] == 1 and diff["n_groups"] == 2


def test_matrix_tabulates_selected_models_by_backend_and_strategy(tmp_path):
    for backend, strategy, value in (
        ("uma", "trajectory_holdout", 0.05),
        ("mace", "trajectory_holdout", 0.07),
        ("uma", "composition_holdout", 0.11),
    ):
        report = {
            "backend": backend,
            "split": {"strategy": strategy, "objective": "x"},
            "selection": {"selected": "finetuned"},
            "evaluations": {
                "test": {
                    "finetuned": {
                        "metrics": {"force_mae_eV_per_A": value},
                        "groups": {
                            "n_groups": 3,
                            "metrics": {
                                "force_mae_eV_per_A": {"group_bootstrap_95": [value - 0.01, value]}
                            },
                        },
                    }
                }
            },
        }
        path = tmp_path / f"{backend}-{strategy}" / "eval" / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report))
    table = matrix(collect_runs([tmp_path]), "force_mae_eV_per_A")
    assert "| backend | composition_holdout | trajectory_holdout |" in table
    assert "| uma | 0.11 [0.1, 0.11] (3 groups) | 0.05 [0.04, 0.05] (3 groups) |" in table
    assert "| mace | - | 0.07" in table


def test_uncertainty_calibration_is_validation_only_and_empirical():
    rng = np.random.default_rng(0)
    scores = rng.random(40)
    errors = scores * 0.2 + 0.01 * rng.random(40)
    report = calibrate_uncertainty(scores, errors, error_threshold=0.1, partition="validation")
    assert report["spearman_score_vs_error"] > 0.9
    assert report["screening"]["precision"] >= 0.5
    assert "not a calibrated error probability" in report["note"]
    with pytest.raises(ValueError, match="never on test"):
        calibrate_uncertainty(scores, errors, error_threshold=0.1, partition="test")


def test_finetuners_take_the_validation_partition(tmp_path):
    from matsim_agents.training.finetune_mace import _write_xyz_splits
    from matsim_agents.training.finetune_uma import _load_labeled_frames

    frames = []
    for index in range(5):
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1.5 + 0.1 * index, 0, 0]])
        atoms.info["energy"] = -float(index)
        atoms.arrays["forces"] = np.zeros((2, 3))
        frames.append(atoms)
    train_xyz, val_xyz, n_train, n_val, _ = _write_xyz_splits(
        frames, [-1.0, -2.0, -3.0, -4.0, -5.0], tmp_path, seed=0, n_validation=2
    )
    assert (n_train, n_val) == (3, 2)
    assert [a.info["REF_energy"] for a in read(val_xyz, index=":")] == [-4.0, -5.0]
    one = tmp_path / "one.extxyz"
    from ase.io import write

    write(one, frames[:1])
    assert len(_load_labeled_frames(one, minimum=1)) == 1
    with pytest.raises(ValueError, match="need >=2"):
        _load_labeled_frames(one)
