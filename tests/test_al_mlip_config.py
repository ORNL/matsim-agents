"""Unit tests for the multi-backend MLP config schema (HydraGNN <-> UMA).

These exercise pure pydantic validation — no torch / fairchem / HydraGNN.
"""

from __future__ import annotations

import pytest

from matsim_agents.active_learning.config import ALConfig, MLIPConfig


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
    }


def test_legacy_hydragnn_block_is_promoted_to_mlp(tmp_path) -> None:
    """A top-level ``hydragnn:`` block (old schema) must become mlp.backend=hydragnn."""
    data = _base_blocks(tmp_path)
    data["hydragnn"] = {"logdir": str(tmp_path / "logdir"), "checkpoint": "best_model.pt"}
    cfg = ALConfig.model_validate(data)
    assert cfg.mlip.backend == "hydragnn"
    assert cfg.mlip.hydragnn is not None
    assert cfg.mlip.uma is None
    assert str(cfg.mlip.hydragnn.checkpoint) == "best_model.pt"


def test_uma_backend_parses(tmp_path) -> None:
    data = _base_blocks(tmp_path)
    data["mlp"] = {
        "backend": "uma",
        "uma": {
            "model_name": "uma-s-1p1",
            "task_name": "omol",
            "dropout": {"enabled": True, "p": 0.15, "target_layers": "linear"},
        },
    }
    cfg = ALConfig.model_validate(data)
    assert cfg.mlip.backend == "uma"
    assert cfg.mlip.uma is not None
    assert cfg.mlip.uma.task_name == "omol"
    assert cfg.mlip.uma.dropout.p == 0.15
    # No ensemble members -> unified ensemble_paths is empty.
    assert cfg.mlip.ensemble_paths == []


def test_mlip_backend_requires_matching_block() -> None:
    with pytest.raises(ValueError, match="requires an mlp.uma block"):
        MLIPConfig(backend="uma")
    with pytest.raises(ValueError, match="requires an mlp.hydragnn block"):
        MLIPConfig(backend="hydragnn")


def test_uma_ensemble_models_feed_ensemble_paths() -> None:
    cfg = MLIPConfig(
        backend="uma",
        uma={"model_name": "uma-s-1p1", "ensemble_models": ["uma-s-1p1-seed2"]},
    )
    assert cfg.ensemble_paths == ["uma-s-1p1-seed2"]


def test_ensemble_strategy_requires_members(tmp_path) -> None:
    data = _base_blocks(tmp_path)
    data["acquisition"]["strategy"] = "ensemble"
    data["mlp"] = {"backend": "uma", "uma": {"model_name": "uma-s-1p1"}}
    with pytest.raises(ValueError, match="requires at least one additional model"):
        ALConfig.model_validate(data)


def test_trainer_enabled_requires_train_script(tmp_path) -> None:
    data = _base_blocks(tmp_path)
    data["mlp"] = {"backend": "uma", "uma": {"model_name": "uma-s-1p1"}}
    data["trainer"] = {"enabled": True}  # no train_script
    with pytest.raises(ValueError, match="requires trainer.train_script"):
        ALConfig.model_validate(data)
