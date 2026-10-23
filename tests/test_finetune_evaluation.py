"""Role-aware fine-tune evaluation: validation-only selection, then frozen test evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from matsim_agents.active_learning import finetune_eval
from matsim_agents.active_learning.evaluate import EvalMetrics
from matsim_agents.backends.mlip.config import MLIPConfig, UMAConfig


def _frames(n_trajectories=6, steps=4, symbols=("Nb", "O", "O"), shift=0.0):
    frames = []
    for trajectory in range(n_trajectories):
        for step in range(steps):
            atoms = Atoms(
                list(symbols),
                positions=[[0, 0, 0], [1.9 + 0.01 * step, 0, 0], [0, 1.9 + 0.1 * trajectory, 0]],
                cell=[8, 8, 8],
                pbc=True,
            )
            atoms.positions[0, 2] += shift
            atoms.calc = SinglePointCalculator(atoms, energy=-23.0, forces=np.zeros((3, 3)))
            atoms.info.update(
                trajectory_id=f"t{trajectory}{'-x' if shift else ''}",
                time_index=step,
                al_iteration=trajectory // 2,
            )
            frames.append(atoms)
    return frames


@pytest.fixture
def inputs(tmp_path, elemental_manifest, dataset_method_sidecar, monkeypatch):
    reference = elemental_manifest({"Nb": -10.0, "O": -5.0})
    dataset = tmp_path / "dataset.extxyz"
    write(dataset, _frames())
    dataset_method_sidecar(dataset, reference)
    domain = tmp_path / "hot.extxyz"
    write(domain, _frames(n_trajectories=2, shift=0.5))
    dataset_method_sidecar(domain, reference)
    monkeypatch.setattr(finetune_eval, "_enforce_device_visibility", lambda _device: None)
    calls = []

    def fake_endpoints(train, validation, ft_dir, dev, **kwargs):
        calls.append(("fine-tune", Path(train).name, validation and Path(validation).name))
        ft_dir.mkdir(parents=True, exist_ok=True)
        (ft_dir / "inference_ckpt.pt").write_text("weights")
        cfg = MLIPConfig(backend="uma", uma=UMAConfig())
        return finetune_eval._Models(cfg, "uma-s-1p1", cfg, str(ft_dir / "inference_ckpt.pt"))

    def fake_evaluate(cfg, frames, *, model_path, test_set_label, **kwargs):
        finetuned = model_path.endswith(".pt")
        selection_exists = any(Path(tmp_path).rglob("selection.json"))
        calls.append(("evaluate", Path(test_set_label).name, finetuned, selection_exists))
        error = 0.05 if finetuned else 0.1
        n = len(frames)
        metrics = EvalMetrics(
            backend="uma",
            model_path=model_path,
            iteration=kwargs.get("iteration"),
            test_set=test_set_label,
            n_frames_total=n,
            n_frames_evaluated=n,
            n_energy_frames_evaluated=n,
            n_force_frames_evaluated=n,
            n_atoms_total=3 * n,
            energy_mae_eV=error,
            energy_rmse_eV=error,
            energy_mae_eV_per_atom=error,
            energy_rmse_eV_per_atom=error,
            energy_mean_offset_eV_per_atom=0.0,
            force_mae_eV_per_A=error,
            force_rmse_eV_per_A=error,
            formation_energy_mae_eV_per_atom=error,
            formation_energy_rmse_eV_per_atom=error,
        )
        per_frame = np.full(n, error)
        parity = {
            key: per_frame
            for key in (
                "frame_energy_abs_err_eV_per_atom",
                "frame_formation_abs_err_eV_per_atom",
                "frame_force_mae_eV_per_A",
                "frame_force_vector_mae_eV_per_A",
            )
        }
        parity["f_ref_eV_per_A"] = np.zeros(3)
        parity["f_pred_eV_per_A"] = np.zeros(3)
        return metrics, parity

    monkeypatch.setattr(finetune_eval, "_uma_endpoints", fake_endpoints)
    monkeypatch.setattr(finetune_eval, "evaluate_frames", fake_evaluate)
    return dataset, domain, reference, calls


def _evaluation(domain, **extra):
    return finetune_eval.EvaluationConfig.model_validate(
        {
            "objective": "transfer to new trajectories",
            "split": {
                "strategy": "trajectory_holdout",
                "train_fraction": 0.5,
                "validation_fraction": 0.25,
                "test_fraction": 0.25,
            },
            "test_domains": [
                {"name": "hot", "dataset": str(domain), "membership": "trajectory_holdout"}
            ],
            **extra,
        }
    )


def test_selection_is_frozen_on_validation_before_any_test_frame(inputs, tmp_path):
    dataset, domain, reference, calls = inputs
    output = tmp_path / "run"
    finetune_eval.run_finetune_endpoints(
        dataset,
        output,
        backend="uma",
        device="cpu",
        elemental_reference_manifest=reference,
        evaluation=_evaluation(domain),
    )
    assert calls[0] == ("fine-tune", "train.extxyz", "validation.extxyz")
    evaluations = [call for call in calls if call[0] == "evaluate"]
    assert [call[1] for call in evaluations[:2]] == ["validation.extxyz"] * 2
    assert not any(call[3] for call in evaluations[:2])
    # Every test and test-domain evaluation happens after selection.json exists.
    assert {call[1] for call in evaluations[2:]} == {"test.extxyz", "hot.extxyz"}
    assert all(call[3] for call in evaluations[2:])

    selection = json.loads((output / "selection.json").read_text())
    assert selection["selected"] == "finetuned" and selection["decided_on"] == "validation"
    assert selection["models"]["finetuned"]["sha256"]
    report = json.loads((output / "eval" / "report.json").read_text())
    assert set(report["evaluations"]) == {"validation", "test", "domain:hot"}
    assert report["split"]["strategy"] == "trajectory_holdout"
    assert report["split"]["roles"]["test"]["n_groups"] >= 1
    assert "force_vector_mae_eV_per_A" in report["metric_definitions"]
    paired = report["paired_finetuned_minus_baseline"]["test"]["force_mae_eV_per_A"]
    assert paired["mean_difference"] == pytest.approx(-0.05)
    groups = report["evaluations"]["domain:hot"]["finetuned"]["groups"]
    assert groups["n_groups"] == 2 and groups["metrics"]["force_mae_eV_per_A"]["worst_group"]
    # The test partition keeps the iter0 / iter<N> names earlier analyses read.
    assert (output / "eval" / "iter0.json").is_file()
    assert (output / "eval" / "iter3.json").is_file()


def test_without_validation_the_finetuned_model_is_reported_unselected(inputs, tmp_path):
    dataset, _domain, reference, calls = inputs
    output = tmp_path / "random"
    finetune_eval.run_finetune_endpoints(
        dataset,
        output,
        backend="uma",
        device="cpu",
        elemental_reference_manifest=reference,
        evaluation=finetune_eval.EvaluationConfig.model_validate(
            {
                "objective": "interpolation among random frames",
                "split": {"strategy": "random_frame", "train_fraction": 0.8, "test_fraction": 0.2},
            }
        ),
    )
    selection = json.loads((output / "selection.json").read_text())
    assert selection["selected"] == "finetuned" and selection["decided_on"] is None
    report = json.loads((output / "eval" / "report.json").read_text())
    assert report["split"]["strategy"] == "random_frame"
    assert any("interpolation" in note for note in report["split"]["limitations"])


def test_cli_requires_an_evaluation_config(tmp_path, capsys):
    with pytest.raises(SystemExit):
        finetune_eval.main(
            ["--backend", "uma", "--dataset", "d.extxyz", "--output-dir", str(tmp_path)]
        )
    assert "--evaluation-config" in capsys.readouterr().err


def test_repetitions_reuse_one_split_per_split_seed(inputs, tmp_path):
    dataset, domain, reference, _calls = inputs
    output = tmp_path / "repeated"
    finetune_eval.run_finetune_endpoints(
        dataset,
        output,
        backend="uma",
        device="cpu",
        elemental_reference_manifest=reference,
        evaluation=_evaluation(
            domain, repetitions={"split_seeds": [1, 2], "training_seeds": [10, 11]}
        ),
    )
    for split_seed in (1, 2):
        assert (output / f"split-seed-{split_seed}" / "split" / "split-manifest.json").is_file()
        for training_seed in (10, 11):
            run = output / f"split-seed-{split_seed}" / f"train-seed-{training_seed}"
            assert (run / "eval" / "report.json").is_file()
    summary = json.loads((output / "repetitions.json").read_text())
    assert summary["n_repetitions"] == 4
    assert "not independent experiments" in summary["independence"]
    assert summary["metrics"]["test"]["force_mae_eV_per_A"]["mean"] == pytest.approx(0.05)


def test_other_backends_reuse_the_same_split(inputs, tmp_path):
    dataset, domain, reference, _calls = inputs
    first = tmp_path / "uma"
    finetune_eval.run_finetune_endpoints(
        dataset,
        first,
        backend="uma",
        device="cpu",
        elemental_reference_manifest=reference,
        evaluation=_evaluation(domain),
    )
    manifest = first / "split" / "split-manifest.json"
    second = tmp_path / "second"
    finetune_eval.run_finetune_endpoints(
        dataset,
        second,
        backend="uma",
        device="cpu",
        elemental_reference_manifest=reference,
        evaluation=finetune_eval.EvaluationConfig(
            objective="transfer to new trajectories", split_manifest=manifest
        ),
    )
    assert not (second / "split").exists()
    report = json.loads((second / "eval" / "report.json").read_text())
    assert report["split"]["manifest_sha256"] == json.loads(manifest.read_text())["content_sha256"]


def test_evaluation_config_rejects_contradictions(tmp_path):
    with pytest.raises(ValueError, match="exactly one of split and split_manifest"):
        finetune_eval.EvaluationConfig(objective="x")
    with pytest.raises(ValueError, match="already fixes"):
        finetune_eval.EvaluationConfig(
            objective="x",
            split_manifest=tmp_path / "m.json",
            repetitions={"split_seeds": [1, 2]},
        )


@pytest.mark.parametrize(
    "name",
    [
        "random-frame-interpolation",
        "trajectory-holdout",
        "composition-holdout",
        "al-iteration-holdout",
    ],
)
def test_documented_examples_are_valid_evaluation_configs(name):
    root = Path(__file__).resolve().parents[1]
    config = finetune_eval.EvaluationConfig.from_yaml(
        root / "examples" / "evaluation" / f"{name}.yaml"
    )
    assert config.objective and config.split is not None
