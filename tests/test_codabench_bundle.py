from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODEBENCH = ROOT / "benchmarks" / "codabench"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_baselines_load_as_distinct_modules() -> None:
    runner = _load("codabench_run_baselines", CODEBENCH / "run_baselines.py")
    mace = runner.load_baseline_class("mace_mp0")
    hydragnn = runner.load_baseline_class("hydragnn")
    assert mace.__module__ == "matsim_codabench_baseline_mace_mp0"
    assert hydragnn.__module__ == "matsim_codabench_baseline_hydragnn"
    assert mace is not hydragnn


def test_aggregate_baselines_dispatch_incompatible_backends(tmp_path, monkeypatch) -> None:
    runner = _load("codabench_run_dispatch", CODEBENCH / "run_baselines.py")
    base_python = tmp_path / "base-python"
    mace_python = tmp_path / "mace-python"
    base_python.touch()
    mace_python.touch()
    monkeypatch.setenv("MATSIM_BASE_PYTHON", str(base_python))
    monkeypatch.setenv("MATSIM_MACE_PYTHON", str(mace_python))
    monkeypatch.setattr(runner.sys, "argv", ["run_baselines.py", "--model=all", "--device", "cpu"])
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, check: calls.append(command))

    runner._dispatch_aggregate("all")

    assert len(calls) == 4
    assert calls[0][0] == str(mace_python)
    assert "--model=mace" in calls[0]
    assert all(call[0] == str(base_python) for call in calls[1:])
    assert [next(arg for arg in call if arg.startswith("--model=")) for call in calls] == [
        "--model=mace",
        "--model=hydragnn",
        "--model=uma",
        "--model=allscaip",
    ]


def test_submission_packager_rejects_raw_total_energies(tmp_path) -> None:
    package = _load("codabench_package", CODEBENCH / "package_submission.py")
    predictions = tmp_path / "predictions"
    raw = predictions / "formation_energies.csv"
    _write_csv(raw, ["structure_id", "energy_eV"], [{"structure_id": "MATS-0001", "energy_eV": -1}])
    try:
        package.package_submission(predictions, tmp_path / "submission")
    except ValueError as exc:
        assert "Raw model total energies" in str(exc)
    else:
        raise AssertionError("raw energies were accepted as formation energies")


def test_builds_self_contained_release_bundle(tmp_path, monkeypatch) -> None:
    validator = _load("validate_bundle", CODEBENCH / "validate_bundle.py")
    monkeypatch.syspath_prepend(str(CODEBENCH))
    builder = _load("codabench_builder", CODEBENCH / "build_bundle.py")

    public = tmp_path / "public"
    structures = public / "structures"
    structures.mkdir(parents=True)
    (structures / "MATS-0001.xyz").write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    _write_csv(
        public / "structures_metadata.csv",
        ["structure_id", "file_path"],
        [{"structure_id": "MATS-0001", "file_path": "MATS-0001.xyz"}],
    )

    reference = tmp_path / "reference"
    for directory in ("forces", "relaxed"):
        (reference / directory).mkdir(parents=True)
    np.save(reference / "forces" / "MATS-0001.npy", np.zeros((1, 3)))
    (reference / "relaxed" / "MATS-0001.xyz").write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    for name in ("public_ids.txt", "private_ids.txt"):
        (reference / name).write_text("MATS-0001\n", encoding="utf-8")
    (reference / "elemental_energies.json").write_text("{}\n", encoding="utf-8")
    _write_csv(
        reference / "formation_energies.csv",
        ["structure_id", "formation_energy_eV_per_atom", "n_atoms"],
        [{"structure_id": "MATS-0001", "formation_energy_eV_per_atom": -1, "n_atoms": 1}],
    )
    _write_csv(
        reference / "structures_metadata.csv",
        ["structure_id", "material_class", "formula"],
        [{"structure_id": "MATS-0001", "material_class": "test", "formula": "H"}],
    )

    output = builder.build_bundle(tmp_path / "competition.zip", public, reference)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert "competition.yaml" in names
    assert "starting_kit/baselines/hydragnn/model.py" in names
    assert "starting_kit/package_submission.py" in names
    assert "starting_kit/requirements-mace.txt" in names
    assert "starting_kit/requirements-fairchem.txt" in names
    assert "public_data/structures/MATS-0001.xyz" in names
    assert "reference_data/forces/MATS-0001.npy" in names
    assert "reference_data/relaxed/MATS-0001.xyz" in names
    assert validator.validate_bundle(CODEBENCH, release=False) == []


def test_scoring_program_accepts_packaged_submission(tmp_path) -> None:
    input_dir = tmp_path / "input"
    res = input_dir / "res"
    ref = input_dir / "ref"
    out = tmp_path / "out"
    res.mkdir(parents=True)
    (ref / "forces").mkdir(parents=True)
    rows = [
        {"structure_id": "MATS-0001", "formation_energy_eV_per_atom": -1.1, "n_atoms": 1},
        {"structure_id": "MATS-0002", "formation_energy_eV_per_atom": -0.9, "n_atoms": 1},
    ]
    _write_csv(ref / "formation_energies.csv", list(rows[0]), rows)
    _write_csv(
        ref / "structures_metadata.csv",
        ["structure_id", "formula"],
        [
            {"structure_id": "MATS-0001", "formula": "H"},
            {"structure_id": "MATS-0002", "formula": "H"},
        ],
    )
    (ref / "public_ids.txt").write_text("MATS-0001\n", encoding="utf-8")
    _write_csv(res / "task1.csv", list(rows[0]), rows)
    _write_csv(res / "task5.csv", list(rows[0]), rows)
    for sid in ("MATS-0001", "MATS-0002"):
        np.save(ref / "forces" / f"{sid}.npy", np.zeros((1, 3)))

    subprocess.run(
        [sys.executable, str(CODEBENCH / "scoring_program" / "score.py"), str(input_dir), str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    scores = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    assert "public_Task1_energy_MAE_eV_per_atom" in scores
    assert "private_Task1_energy_MAE_eV_per_atom" in scores
