from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import zipfile
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from matsim_agents.training.finetune_mace import MACE_MODELS

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


def test_split_uses_published_scope_and_keeps_singletons_private(tmp_path) -> None:
    _write_csv(
        tmp_path / "public_data" / "structures_metadata.csv",
        ["structure_id", "file_path"],
        [
            {"structure_id": identity, "file_path": f"{identity}.xyz"}
            for identity in ("MATS-0001", "MATS-0002", "MATS-0003")
        ],
    )
    private_metadata = tmp_path / "private.csv"
    _write_csv(
        private_metadata,
        ["anon_id", "formula"],
        [
            {"anon_id": "MATS-0001", "formula": "H"},
            {"anon_id": "MATS-0002", "formula": "H"},
            {"anon_id": "MATS-0003", "formula": "O"},
            {"anon_id": "MATS-0004", "formula": "H"},
        ],
    )
    reference = tmp_path / "reference_data"
    reference.mkdir()
    script = reference / "create_split.py"
    script.write_text((CODEBENCH / "reference_data" / "create_split.py").read_text())
    command = [sys.executable, str(script)]
    environment = {"MATSIM_PRIVATE_METADATA": str(private_metadata)}
    subprocess.run(command, env=environment, check=True, capture_output=True)
    public_file, private_file = reference / "public_ids.txt", reference / "private_ids.txt"
    public = set(public_file.read_text().split())
    private = set(private_file.read_text().split())
    assert len(public) == 1
    assert "MATS-0003" in private
    assert not public & private
    assert public | private == {"MATS-0001", "MATS-0002", "MATS-0003"}
    original = public_file.read_bytes(), private_file.read_bytes()
    subprocess.run(command, env=environment, check=True, capture_output=True)
    assert original == (public_file.read_bytes(), private_file.read_bytes())
    _write_csv(private_metadata, ["anon_id", "formula"], [{"anon_id": "MATS-0001", "formula": "H"}])
    failed = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "cover every benchmark ID" in failed.stderr
    assert original == (public_file.read_bytes(), private_file.read_bytes())


MLIP_SELECTIONS = CODEBENCH / "mlip"
BACKEND_SELECTIONS = {
    "hydragnn": MLIP_SELECTIONS / "hydragnn.yaml",
    "uma": MLIP_SELECTIONS / "uma.yaml",
    "mace": MLIP_SELECTIONS / "materials" / "mace_omat_medium.yaml",
}


def _manifest_args(tmp_path: Path) -> list[str]:
    return ["--elemental-reference-manifest", str(tmp_path / "refs.json")]


def test_baselines_load_as_distinct_modules() -> None:
    runner = _load("codabench_run_baselines", CODEBENCH / "run_baselines.py")
    mlip = runner.load_baseline_class("mlip")
    allscaip = runner.load_baseline_class("allscaip")
    assert mlip.__module__ == "matsim_codabench_baseline_mlip"
    assert allscaip.__module__ == "matsim_codabench_baseline_allscaip"
    assert mlip is not allscaip
    assert sorted(path.name for path in (CODEBENCH / "baselines").iterdir() if path.is_dir()) == [
        "allscaip",
        "mlip",
    ]


@pytest.mark.parametrize("backend", sorted(BACKEND_SELECTIONS))
def test_codabench_mlip_adapter_builds_each_backend(tmp_path, monkeypatch, backend) -> None:
    from matsim_agents.backends.mlip.config import MLIP_BACKEND_CONFIGS

    monkeypatch.setenv("HYDRAGNN_LOGDIR", str(tmp_path / "logdir"))
    built = []

    class Inner:
        def calculate(self, atoms, properties, system_changes):
            self.results = {"energy": np.float64(-2.0), "forces": np.zeros((len(atoms), 3))}

    def build_calculator(self, *, enable_mc_dropout=False):
        built.append(self)
        return Inner()

    monkeypatch.setattr(MLIP_BACKEND_CONFIGS[backend], "build_calculator", build_calculator)
    adapter = _load(f"codabench_mlip_adapter_{backend}", CODEBENCH / "baselines/mlip/model.py")
    selection = tmp_path / "checkpoint"
    selection.mkdir()
    (selection / "mlip.yaml").write_text(BACKEND_SELECTIONS[backend].read_text())

    calculator = adapter.AtomisticCalculator.from_checkpoint(str(selection), device="xpu")
    from_file = adapter.AtomisticCalculator.from_checkpoint(
        str(BACKEND_SELECTIONS[backend]), device="cuda"
    )

    assert [(cfg.backend, cfg.device) for cfg in built] == [(backend, "xpu"), (backend, "cuda")]
    if backend == "hydragnn":
        assert built[0].logdir == tmp_path / "logdir"
    assert from_file._inner is not calculator._inner
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    predictions = calculator.predict_many([atoms, atoms.copy()])
    assert [prediction["energy"] for prediction in predictions] == [-2.0, -2.0]
    assert all(prediction["forces"].dtype == np.float32 for prediction in predictions)
    with pytest.raises(ValueError, match="device"):
        adapter.AtomisticCalculator.from_checkpoint(str(selection), device="tpu")


def test_codabench_mace_catalog_matches_workflow(monkeypatch) -> None:
    from matsim_agents.backends.mlip.config import load_mlip_config

    monkeypatch.delenv("MLIP_DEVICE", raising=False)
    materials = sorted((MLIP_SELECTIONS / "materials").glob("*.yaml"))
    molecular = sorted((MLIP_SELECTIONS / "molecular").glob("*.yaml"))
    catalog = {path.stem: load_mlip_config(path) for path in materials + molecular}

    assert {name: (cfg.family, cfg.model) for name, cfg in catalog.items()} == {
        name: (entry["family"], entry["model"]) for name, entry in MACE_MODELS.items()
    }
    assert len(materials) == 14
    assert all(catalog[path.stem].family == "mace_mp" for path in materials)
    assert len(catalog) == 22
    assert all(cfg.device == "cpu" for cfg in catalog.values())
    assert {name for name, cfg in catalog.items() if cfg.precision != "fp32"} == {"mace_anicc"}
    monkeypatch.setenv("MLIP_DEVICE", "cuda")
    assert load_mlip_config(materials[0]).device == "cuda"


@pytest.mark.parametrize("variant", sorted(MACE_MODELS))
def test_codabench_mace_variants_select_load_and_predict(tmp_path, monkeypatch, variant) -> None:
    runner = _load("codabench_variant_runner", CODEBENCH / "run_baselines.py")
    family = MACE_MODELS[variant]["family"]
    model = MACE_MODELS[variant]["model"]
    group = "materials" if family == "mace_mp" else "molecular"
    monkeypatch.delenv("MLIP_DEVICE", raising=False)
    _args, selections = runner.parse_args(
        [
            "--mlip-config",
            str(MLIP_SELECTIONS / group / f"{variant}.yaml"),
            *_manifest_args(tmp_path),
        ]
    )
    [(label, _path, cfg)] = selections
    assert (label, cfg.backend, cfg.family, cfg.model) == (variant, "mace", family, model)

    calls = []
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])

    class FakeCalculator:
        def calculate(self, atoms, properties, system_changes):
            self.results = {
                "energy": -2.0,
                "forces": np.zeros((len(atoms), 3)),
                "stress": np.zeros(6),
            }

    def loader(name):
        def load(**kwargs):
            calls.append((name, kwargs))
            return FakeCalculator()

        return load

    calculators = ModuleType("mace.calculators")
    calculators.MACECalculator = loader("checkpoint")
    for name in ("mace_mp", "mace_off", "mace_omol", "mace_polar", "mace_anicc"):
        setattr(calculators, name, loader(name))
    mace_module = ModuleType("mace")
    mace_module.calculators = calculators
    monkeypatch.setitem(sys.modules, "mace", mace_module)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    calculator = runner.load_baseline_class("mlip").from_mlip_config(cfg)
    expected_kwargs = (
        {"model_path": None, "device": "cpu"}
        if family == "mace_anicc"
        else {"model": model, "device": "cpu", "default_dtype": "float32"}
    )
    if family == "mace_mp":
        expected_kwargs["dispersion"] = False
    assert calls == [(family, expected_kwargs)]
    predictions = calculator.predict_many([atoms, atoms.copy()])
    assert len(predictions) == 2
    for prediction in predictions:
        assert prediction["energy"] == -2.0
        assert prediction["forces"].shape == (2, 3)
        assert prediction["forces"].dtype == np.float32
        assert np.isfinite(prediction["forces"]).all()
    assert predictions[0]["forces"] is not predictions[1]["forces"]
    assert calculator.results["stress"].shape == (6,)


def test_run_baselines_runs_each_mlip_config_on_its_device(tmp_path, monkeypatch) -> None:
    from matsim_agents.backends.mlip.config import MLIP_BACKEND_CONFIGS

    runner = _load("codabench_run_mlip_configs", CODEBENCH / "run_baselines.py")
    monkeypatch.setenv("HYDRAGNN_LOGDIR", str(tmp_path / "logdir"))
    monkeypatch.setenv("MLIP_DEVICE", "xpu")
    for backend, config_class in MLIP_BACKEND_CONFIGS.items():
        monkeypatch.setattr(
            config_class,
            "build_calculator",
            lambda self, *, enable_mc_dropout=False, backend=backend: SimpleNamespace(
                backend=backend
            ),
        )
    runs = []
    monkeypatch.setattr(
        runner,
        "run_predictions",
        lambda calc, label, device, relax, fmax, steps, **kwargs: runs.append(
            (calc._inner.backend, label, device, relax, fmax, steps, kwargs)
        ),
    )
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected dispatch")
    )
    reference = tmp_path / "refs.json"
    for backend, selection in BACKEND_SELECTIONS.items():
        runner.main(
            [
                "--mlip-config",
                str(selection),
                "--elemental-reference-manifest",
                str(reference),
                "--relax",
                "--fmax",
                "0.1",
            ]
        )
        assert runs.pop() == (
            backend,
            selection.stem,
            "xpu",
            True,
            0.1,
            500,
            {"elemental_reference_manifest": reference},
        )
    materials = sorted(str(path) for path in (MLIP_SELECTIONS / "materials").glob("*.yaml"))
    runner.main(["--mlip-config", *materials, "--elemental-reference-manifest", str(reference)])
    assert [label for _backend, label, *_rest in runs] == [Path(path).stem for path in materials]


def test_run_baselines_rejects_removed_and_invalid_selections(tmp_path, capsys) -> None:
    runner = _load("codabench_run_invalid", CODEBENCH / "run_baselines.py")
    common = _manifest_args(tmp_path)
    with pytest.raises(SystemExit):
        runner.parse_args(common)
    assert "--mlip-config FILE and/or --allscaip" in capsys.readouterr().err
    uma = str(BACKEND_SELECTIONS["uma"])
    for removed in (
        ["--model", "mace"],
        ["--mace-variant", "materials"],
        ["--mace-family", "mace_mp"],
        ["--mace-model", "small"],
        ["--hydragnn-logdir", "x"],
        ["--uma-model", "uma-s-1p2"],
        ["--uma-task", "omat"],
        ["--device", "cpu"],
    ):
        with pytest.raises(SystemExit):
            runner.parse_args(["--mlip-config", uma, *removed, *common])
    with pytest.raises(SystemExit):
        runner.parse_args(["--mlip-config", uma, uma, *common])
    assert "prediction labels must be unique" in capsys.readouterr().err
    missing = tmp_path / "missing.yaml"
    with pytest.raises(SystemExit):
        runner.parse_args(["--mlip-config", str(missing), *common])
    assert f"--mlip-config {missing}" in capsys.readouterr().err


def test_aggregate_baselines_dispatch_incompatible_backends(tmp_path, monkeypatch) -> None:
    runner = _load("codabench_run_dispatch", CODEBENCH / "run_baselines.py")
    base_python = tmp_path / "base-python"
    mace_python = tmp_path / "mace-python"
    base_python.touch()
    mace_python.touch()
    monkeypatch.setenv("MATSIM_BASE_PYTHON", str(base_python))
    monkeypatch.setenv("MATSIM_MACE_PYTHON", str(mace_python))
    monkeypatch.setenv("HYDRAGNN_LOGDIR", str(tmp_path / "logdir"))
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, check: calls.append(command))
    materials = sorted((MLIP_SELECTIONS / "materials").glob("*.yaml"))
    args, selections = runner.parse_args(
        [
            "--mlip-config",
            *map(str, materials),
            str(BACKEND_SELECTIONS["hydragnn"]),
            "--mlip-config",
            str(BACKEND_SELECTIONS["uma"]),
            "--allscaip",
            "--allscaip-device",
            "cuda",
            "--relax",
            *_manifest_args(tmp_path),
        ]
    )

    assert runner.dispatch_environments(args, selections)

    assert [call[0] for call in calls] == [str(mace_python), str(base_python)]

    def configs(call):
        start = call.index("--mlip-config") + 1
        end = next(
            (i for i, arg in enumerate(call[start:], start) if arg.startswith("--")), len(call)
        )
        return [Path(arg).name for arg in call[start:end]]

    assert configs(calls[0]) == [path.name for path in materials]
    assert configs(calls[1]) == ["hydragnn.yaml", "uma.yaml"]
    assert "--allscaip" not in calls[0]
    assert calls[1][calls[1].index("--allscaip-device") + 1] == "cuda"
    assert all("--relax" in call for call in calls)
    assert not runner.dispatch_environments(
        *runner.parse_args(["--allscaip", *_manifest_args(tmp_path)])
    )
    single_family = runner.parse_args(
        ["--mlip-config", str(BACKEND_SELECTIONS["uma"]), "--allscaip", *_manifest_args(tmp_path)]
    )
    assert not runner.dispatch_environments(*single_family)
    assert len(calls) == 2


def test_baseline_prediction_failures_raise(tmp_path, monkeypatch) -> None:
    runner = _load("codabench_run_failure", CODEBENCH / "run_baselines.py")
    structures = tmp_path / "structures"
    structures.mkdir()
    metadata = tmp_path / "structures_metadata.csv"
    _write_csv(
        metadata,
        ["structure_id", "file_path"],
        [{"structure_id": "missing", "file_path": "missing.extxyz"}],
    )
    monkeypatch.setattr(runner, "STRUCT_META", metadata)
    monkeypatch.setattr(runner, "STRUCT_ROOT", structures)
    monkeypatch.setattr(runner, "PRED_ROOT", tmp_path / "predictions")

    with pytest.raises(ValueError, match="elemental reference manifest"):
        runner.run_predictions(SimpleNamespace(), "broken", "cpu")


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


def test_baseline_predicts_references_before_packaging_formation_energies(
    tmp_path, monkeypatch, elemental_manifest
):
    from ase.calculators.calculator import Calculator, all_changes
    from ase.io import write

    runner = _load("codabench_formation_runner", CODEBENCH / "run_baselines.py")
    structures = tmp_path / "structures"
    structures.mkdir()
    write(structures / "compound.extxyz", Atoms("NbO2"))
    metadata = tmp_path / "metadata.csv"
    _write_csv(
        metadata,
        ["structure_id", "file_path"],
        [{"structure_id": "MATS-0001", "file_path": "compound.extxyz"}],
    )
    monkeypatch.setattr(runner, "STRUCT_META", metadata)
    monkeypatch.setattr(runner, "STRUCT_ROOT", structures)
    monkeypatch.setattr(runner, "PRED_ROOT", tmp_path / "predictions")
    calls = []

    class Model(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            symbols = atoms.get_chemical_symbols()
            calls.append(set(symbols))
            baseline = sum({"Nb": 100.0, "O": -20.0}[symbol] for symbol in symbols)
            self.results = {
                "energy": baseline - (3.0 if len(set(symbols)) > 1 else 0.0),
                "forces": np.zeros((len(atoms), 3)),
            }

    runner.run_predictions(
        Model(),
        "test",
        "cpu",
        elemental_reference_manifest=elemental_manifest({"Nb": -10.0, "O": -5.0}),
    )
    assert calls[:2] == [{"Nb"}, {"O"}]
    predictions = tmp_path / "predictions/test"
    with (predictions / "formation_energies.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["formation_energy_eV_per_atom"]) == -1.0
    assert (predictions / "elemental_reference_predictions.json").is_file()
    evaluator = _load("codabench_formation_evaluator", CODEBENCH / "evaluate.py")
    monkeypatch.setattr(
        evaluator,
        "parse_args",
        lambda: Namespace(
            submission=str(tmp_path / "model"),
            structures=metadata,
            struct_dir=structures,
            output=tmp_path / "submitted-predictions",
            device="cpu",
            relax=False,
            elemental_reference_manifest=tmp_path / "elemental-references.json",
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "load_calculator_class",
        lambda _path: SimpleNamespace(
            __name__="SyntheticModel",
            from_checkpoint=lambda *_args, **_kwargs: Model(),
        ),
    )
    calls.clear()
    evaluator.main()
    assert calls[:2] == [{"Nb"}, {"O"}]
    with (tmp_path / "submitted-predictions/formation_energies.csv").open() as handle:
        submitted_rows = list(csv.DictReader(handle))
    assert submitted_rows[0]["structure_id"] == rows[0]["structure_id"]
    assert float(submitted_rows[0]["formation_energy_eV_per_atom"]) == -1.0
    package = _load("codabench_formation_package", CODEBENCH / "package_submission.py")
    package.package_submission(predictions, tmp_path / "submission")
    assert (tmp_path / "submission/task1.csv").read_text() == (
        predictions / "formation_energies.csv"
    ).read_text()


def test_builds_self_contained_release_bundle(tmp_path, monkeypatch, elemental_manifest) -> None:
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
    source_manifest = elemental_manifest({"H": -1.0})
    reference_payload = json.loads(source_manifest.read_text())
    for spec in reference_payload["references"].values():
        source = tmp_path / spec["structure_path"]
        (public / source.name).write_bytes(source.read_bytes())
    (public / "elemental_references.json").write_text(json.dumps(reference_payload))

    reference = tmp_path / "reference"
    for directory in ("forces", "relaxed"):
        (reference / directory).mkdir(parents=True)
    np.save(reference / "forces" / "MATS-0001.npy", np.zeros((1, 3)))
    (reference / "relaxed" / "MATS-0001.xyz").write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    for name in ("public_ids.txt", "private_ids.txt"):
        (reference / name).write_text("MATS-0001\n", encoding="utf-8")
    (reference / "elemental_energies.json").write_text('{"H": -1.0}\n', encoding="utf-8")
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
        extracted = tmp_path / "extracted"
        archive.extractall(extracted)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import evaluate; "
            "assert evaluate.predict_elemental_references.__module__ == 'energy_references'",
        ],
        cwd=extracted / "starting_kit",
        check=True,
        capture_output=True,
        text=True,
    )
    assert validator.validate_bundle(extracted, release=True) == []
    (extracted / "reference_data/elemental_energies.json").write_text('{"H": -2.0}\n')
    assert any(
        "does not match" in error for error in validator.validate_bundle(extracted, release=True)
    )
    (extracted / "reference_data/elemental_energies.json").write_text('{"H": -1.0}\n')
    outside_reference = tmp_path / "elemental-H.extxyz"
    reference_payload["references"]["H"]["structure_path"] = str(outside_reference)
    (extracted / "public_data/elemental_references.json").write_text(json.dumps(reference_payload))
    assert any(
        "bundled inside public_data" in error
        for error in validator.validate_bundle(extracted, release=True)
    )
    assert "competition.yaml" in names
    assert "starting_kit/baselines/mlip/model.py" in names
    assert "starting_kit/mlip/materials/mace_omat_medium.yaml" in names
    assert "starting_kit/package_submission.py" in names
    assert "starting_kit/energy_references.py" in names
    assert "starting_kit/evaluate.py" in names
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
