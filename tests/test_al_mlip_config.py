"""Unit tests for the multi-backend MLP config schema (HydraGNN, UMA, MACE).

These exercise pure pydantic validation — no torch / fairchem / HydraGNN.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from matsim_agents.active_learning.config import ALConfig
from matsim_agents.backends.mlip.config import (
    HydraGNNConfig,
    MACEConfig,
    UMAConfig,
    parse_mlip_config,
)


def _base_blocks(tmp_path) -> dict:
    """Minimal md/acquisition/dft/trainer/loop blocks shared by the tests."""
    seed = tmp_path / "seed.extxyz"
    seed.write_text("0\n\n")  # content irrelevant; path just needs to exist as a string
    return {
        "md": {
            "seed_source": {"kind": "paths", "paths": [str(seed)]},
        },
        "acquisition": {"strategy": "mc_dropout", "n_select": 8},
        "dft": {
            "backend": "vasp",
            "vasp": {
                "vasp_bin": str(tmp_path / "vasp_std"),
                "vasp_wrapper": str(tmp_path / "wrap.sh"),
                "incar_template": str(tmp_path / "INCAR"),
                "potcar_dir": str(tmp_path / "potcar"),
            },
        },
        "trainer": {"enabled": False},
        "loop": {"out_dir": str(tmp_path / "out")},
        "resources": {"preset": "perlmutter-gpu", "pools": {"dft": {"nodes": "all"}}},
    }


def test_hydragnn_backend_parses(tmp_path) -> None:
    data = _base_blocks(tmp_path)
    data["mlip"] = {
        "backend": "hydragnn",
        "logdir": str(tmp_path / "logdir"),
        "checkpoint": "best_model.pt",
    }
    cfg = ALConfig.model_validate(data)
    assert isinstance(cfg.mlip, HydraGNNConfig)
    assert cfg.mlip.backend == "hydragnn"
    assert str(cfg.mlip.checkpoint) == "best_model.pt"


def test_uma_backend_parses(tmp_path) -> None:
    data = _base_blocks(tmp_path)
    data["mlip"] = {
        "backend": "uma",
        "model_name": "uma-s-1p1",
        "task_name": "omol",
        "dropout": {"enabled": True, "p": 0.15, "target_layers": "linear"},
    }
    cfg = ALConfig.model_validate(data)
    assert isinstance(cfg.mlip, UMAConfig)
    assert cfg.mlip.task_name == "omol"
    assert cfg.mlip.dropout.p == 0.15
    assert cfg.mlip.ensemble_models == []


def test_mace_backend_parses(tmp_path) -> None:
    data = _base_blocks(tmp_path)
    data["mlip"] = {"backend": "mace", "family": "mace_mp", "model": "medium", "device": "cpu"}
    cfg = ALConfig.model_validate(data)
    assert isinstance(cfg.mlip, MACEConfig)
    assert cfg.mlip.model == "medium"
    assert cfg.mlip.device == "cpu"


def test_mlip_backend_requires_its_fields() -> None:
    with pytest.raises(ValidationError, match="logdir"):
        parse_mlip_config({"backend": "hydragnn"})
    with pytest.raises(ValidationError):
        parse_mlip_config({"backend": "unknown"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        parse_mlip_config({"backend": "uma", "logdir": "x"})


def test_uma_ensemble_models_are_ensemble_members() -> None:
    cfg = UMAConfig(model_name="uma-s-1p1", ensemble_models=["uma-s-1p1-seed2"])
    assert cfg.ensemble_models == ["uma-s-1p1-seed2"]
    assert [member.model_name for member in cfg.ensemble_configs()] == [
        "uma-s-1p1",
        "uma-s-1p1-seed2",
    ]


def test_ensemble_strategy_requires_members(tmp_path) -> None:
    data = _base_blocks(tmp_path)
    data["acquisition"]["strategy"] = "ensemble"
    data["mlip"] = {"backend": "uma", "model_name": "uma-s-1p1"}
    with pytest.raises(ValueError, match="requires at least one additional model"):
        ALConfig.model_validate(data)


@pytest.mark.parametrize(
    ("mlip", "module"),
    [
        (
            {"backend": "hydragnn", "logdir": "logdir"},
            "matsim_agents.training.finetune_hydragnn",
        ),
        (
            {
                "backend": "hydragnn",
                "logdir": "logdir",
                "fine_tuner": "finetune_hydragnn_newhead",
            },
            "matsim_agents.training.finetune_hydragnn_newhead",
        ),
        ({"backend": "uma"}, "matsim_agents.training.finetune_uma"),
        ({"backend": "mace"}, "matsim_agents.training.finetune_mace"),
    ],
)
def test_enabled_trainer_uses_the_mlip_blocks_fine_tuner(tmp_path, mlip, module) -> None:
    data = _base_blocks(tmp_path)
    references = tmp_path / "references.json"
    references.write_text("{}")
    data["mlip"] = mlip
    data["trainer"] = {"enabled": True, "validation_reference_set": str(references)}
    data["resources"]["pools"]["training"] = {}
    cfg = ALConfig.model_validate(data)
    assert cfg.mlip.fine_tuner_module == module
    assert "train_script" not in cfg.trainer.model_dump()
