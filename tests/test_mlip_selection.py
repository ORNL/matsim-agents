"""One MLIP selection format, interchangeable across every backend."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError
from typer.testing import CliRunner

from matsim_agents.active_learning import ALConfig
from matsim_agents.backends.mlip.config import (
    MLIP_BACKENDS,
    HydraGNNConfig,
    MACEConfig,
    MLIPBackendConfig,
    MLIPConfig,
    UMAConfig,
    load_mlip_config,
    parse_mlip_config,
)
from matsim_agents.cli import app

ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = sorted((ROOT / "examples/mlip").glob("*.yaml"))


def _al_yaml(tmp_path: Path) -> Path:
    seed = tmp_path / "seed.extxyz"
    seed.write_text("1\n\nCu 0 0 0\n")
    for name in ("vasp", "wrapper", "INCAR"):
        (tmp_path / name).touch()
    (tmp_path / "potcars").mkdir()
    path = tmp_path / "al.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "mlip": {"backend": "hydragnn", "logdir": "${UNSET_HYDRAGNN_LOGDIR}"},
                "md": {"seed_source": {"kind": "paths", "paths": [str(seed)]}, "n_steps": 5},
                "acquisition": {"strategy": "random", "n_select": 1},
                "dft": {
                    "backend": "vasp",
                    "vasp": {
                        "vasp_bin": str(tmp_path / "vasp"),
                        "vasp_wrapper": str(tmp_path / "wrapper"),
                        "incar_template": str(tmp_path / "INCAR"),
                        "potcar_dir": str(tmp_path / "potcars"),
                    },
                },
                "trainer": {"enabled": False},
                "loop": {"n_iterations": 1, "out_dir": str(tmp_path / "out")},
                "resources": {
                    "preset": "perlmutter-gpu",
                    "pools": {
                        "dft": {"nodes": "all", "share_with": ["training"]},
                        "training": {"share_with": ["dft"]},
                    },
                },
            }
        )
    )
    return path


@pytest.fixture
def hydragnn_env(monkeypatch, tmp_path):
    monkeypatch.setenv("INITIAL_LOGDIR", str(tmp_path / "logdir"))
    monkeypatch.setenv("HYDRAGNN_DIR", str(tmp_path / "HydraGNN"))


@pytest.mark.parametrize("path", SELECTIONS, ids=lambda path: path.stem)
def test_reusable_selections_load_as_their_backend_config(path, hydragnn_env):
    cfg = load_mlip_config(path)
    assert isinstance(cfg, MLIPBackendConfig)
    assert cfg.backend in MLIP_BACKENDS
    assert cfg.model_identifier().startswith(f"{cfg.backend}:")


def test_every_backend_has_a_reusable_selection(hydragnn_env):
    assert {load_mlip_config(path).backend for path in SELECTIONS} == set(MLIP_BACKENDS)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"backend": "hydragnn", "logdir": "/tmp/logdir"}, HydraGNNConfig),
        ({"backend": "uma", "model_name": "uma-s-1p1"}, UMAConfig),
        ({"backend": "mace", "model": "medium"}, MACEConfig),
    ],
)
def test_backend_discriminates_the_flat_selection(raw, expected):
    assert type(parse_mlip_config(raw)) is expected
    assert type(TypeAdapter(MLIPConfig).validate_python(raw)) is expected


@pytest.mark.parametrize(
    "raw",
    [
        {"backend": "mace", "mace": {"model": "medium"}},
        {"backend": "uma", "model_name": "uma-s-1p1", "family": "mace_mp"},
        {"backend": "hydragnn", "logdir": "/tmp/logdir", "mlp_device": "cpu"},
        {"backend": "hydragnn", "logdir": "/tmp/logdir", "hydragnn_branch_mlp_checkpoint": "x"},
        {"backend": "mace", "model": "medium", "ensemble_paths": []},
        {"backend": "orb"},
    ],
    ids=["nested", "foreign-field", "mlp_device", "long-mlp-name", "ensemble_paths", "unknown"],
)
def test_only_the_current_flat_fields_are_accepted(raw):
    with pytest.raises(ValidationError):
        parse_mlip_config(raw)


def test_selection_file_requires_an_mlip_block(tmp_path):
    path = tmp_path / "bare.yaml"
    path.write_text(yaml.safe_dump({"backend": "uma", "model_name": "uma-s-1p1"}))
    with pytest.raises(ValueError, match="mlip block"):
        load_mlip_config(path)


@pytest.mark.parametrize("path", SELECTIONS, ids=lambda path: path.stem)
def test_al_config_runs_with_any_selected_model(path, tmp_path, hydragnn_env):
    selected = load_mlip_config(path)
    cfg = ALConfig.from_yaml(_al_yaml(tmp_path), mlip=selected)
    assert cfg.mlip == selected


def test_al_cli_replaces_the_mlip_block(tmp_path):
    selection = ROOT / "examples/mlip/mace-omat-0-medium.yaml"
    result = CliRunner().invoke(
        app,
        ["al", "validate-config", str(_al_yaml(tmp_path)), "--mlip-config", str(selection)],
    )
    assert result.exit_code == 0, result.output
    assert '"backend": "mace"' in result.output
    assert '"model": "medium-omat-0"' in result.output


def test_fine_tuning_recipe_belongs_to_the_mlip_block(tmp_path):
    path = _al_yaml(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["trainer"]["train_script"] = "finetune_hydragnn.py"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="train_script"):
        ALConfig.from_yaml(path, mlip=UMAConfig(device="cpu"))


@pytest.mark.parametrize("path", SELECTIONS, ids=lambda path: path.stem)
def test_selected_model_brings_its_own_fine_tuner(path, tmp_path, hydragnn_env):
    cfg = ALConfig.from_yaml(_al_yaml(tmp_path), mlip=load_mlip_config(path))
    assert cfg.mlip.fine_tuner_module == f"matsim_agents.training.{cfg.mlip.fine_tuner}"
    assert cfg.mlip.fine_tuner.startswith(f"finetune_{cfg.mlip.backend}")
