from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from typer.main import get_command

from matsim_agents.active_learning.calculator import build_hydragnn_calculator
from matsim_agents.active_learning.config import (
    HYDRAGNN_DATASET_HEADS,
    HydraGNNConfig,
    resolve_hydragnn_inference_head,
)
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput
from matsim_agents.chat import DiscoveryChatConfig
from matsim_agents.cli import app

EXPECTED_HEADS = (
    "Alexandria",
    "ANI1x",
    "MPTrj",
    "OC2020",
    "OC2022",
    "OC25",
    "ODAC23",
    "OMat24",
    "OMol25",
    "OMol25-neutral",
    "OMol25-non-neutral",
    "OPoly2026",
    "Nabla2DFT",
    "QCML",
    "QM7X",
    "transition1x",
)


def test_hydragnn_dataset_head_mapping_matches_training_order():
    assert HYDRAGNN_DATASET_HEADS == EXPECTED_HEADS
    assert [resolve_hydragnn_inference_head(name) for name in EXPECTED_HEADS] == list(range(16))
    assert resolve_hydragnn_inference_head("omat24") == 7
    assert resolve_hydragnn_inference_head("15") == 15


@pytest.mark.parametrize("value", [-1, 16, "unknown-dataset"])
def test_hydragnn_dataset_head_rejects_invalid_selection(value):
    with pytest.raises(ValueError):
        resolve_hydragnn_inference_head(value)


def test_pinned_head_does_not_require_branch_weight_mlp(tmp_path):
    logdir = tmp_path / "model"
    logdir.mkdir()
    (logdir / "config.json").write_text("{}", encoding="utf-8")

    config = HydraGNNConfig(logdir=logdir, inference_head="OMat24")
    request = RelaxStructureInput(
        structure_path="structure.extxyz",
        mlip_backend="hydragnn",
        logdir=str(logdir),
        hydragnn_inference_head="OMat24",
    )
    chat = DiscoveryChatConfig(logdir=str(logdir), hydragnn_inference_head="OMat24")

    assert resolve_hydragnn_inference_head(config.inference_head) == 7
    assert request.hydragnn_branch_mlp_checkpoint is None
    assert chat.hydragnn_branch_mlp_checkpoint is None


def test_builder_bypasses_fused_stack_for_selected_head(tmp_path, monkeypatch):
    import matsim_agents.active_learning.calculator as calculator

    logdir = tmp_path / "model"
    logdir.mkdir()
    (logdir / "config.json").write_text("{}", encoding="utf-8")
    inference = ModuleType("inference_random_structures")
    inference.load_config_and_model = lambda *args: (
        SimpleNamespace(num_branches=16),
        {"NeuralNetwork": {"Architecture": {"radius": 6.0, "max_neighbours": 24}}},
        "cuda",
        None,
        "float64",
    )
    monkeypatch.setitem(sys.modules, "inference_random_structures", inference)
    captured = {}

    def selected_builder(model, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(calculator, "_build_selected_head_calculator", selected_builder)

    result = build_hydragnn_calculator(HydraGNNConfig(logdir=logdir, inference_head="OMat24"))

    assert result.head_index == 7
    assert captured["radius"] == 6.0
    assert captured["max_neighbours"] == 24


def test_hydragnn_head_selector_is_exposed_by_public_commands():
    root_command = get_command(app)

    for command_name in ("run", "chat", "supervisor-run"):
        command = root_command.commands[command_name]
        option_names = {
            option for parameter in command.params for option in getattr(parameter, "opts", ())
        }
        assert "--hydragnn-inference-head" in option_names
