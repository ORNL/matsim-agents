from __future__ import annotations

import sys
from contextlib import nullcontext
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
    autocast_ctx = nullcontext()
    inference.load_config_and_model = lambda *args: (
        SimpleNamespace(num_branches=16),
        {"NeuralNetwork": {"Architecture": {"radius": 6.0, "max_neighbours": 24}}},
        "cuda",
        autocast_ctx,
        "float64",
    )
    monkeypatch.setitem(sys.modules, "inference_random_structures", inference)
    monkeypatch.setitem(sys.modules, "torch", None)
    captured = {}

    def selected_builder(model, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(calculator, "_build_selected_head_calculator", selected_builder)

    result = build_hydragnn_calculator(HydraGNNConfig(logdir=logdir, inference_head="OMat24"))

    assert result.head_index == 7
    assert captured["radius"] == 6.0
    assert captured["max_neighbours"] == 24
    assert captured["autocast_ctx"] is autocast_ctx


def test_selected_head_calculator_exposes_model_for_mc_dropout(monkeypatch):
    import matsim_agents.active_learning.calculator as calculator

    torch = ModuleType("torch")
    torch.float32 = "float32"
    torch.tensor = lambda value, dtype: (value, dtype)
    monkeypatch.setitem(sys.modules, "torch", torch)
    model = SimpleNamespace()

    result = calculator._build_selected_head_calculator(
        model,
        autocast_ctx=nullcontext(),
        head_index=7,
        radius=6.0,
        max_neighbours=24,
        param_dtype="float64",
        device="cuda",
        charge=0,
        spin=0,
    )

    assert result.model is model


@pytest.mark.parametrize("precision", ["fp32", "fp64", "bf16"])
def test_selected_head_calculator_preserves_precision_context(monkeypatch, precision):
    import torch
    from ase import Atoms

    import matsim_agents.active_learning.calculator as calculator
    import matsim_agents.backends.mlip.relaxation as relaxation

    dtype = torch.float64 if precision == "fp64" else torch.float32
    autocast_ctx = (
        torch.autocast("cpu", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    )
    calls = []

    class Graph(SimpleNamespace):
        def to(self, _device):
            return self

    def graph(atoms, *_args):
        return Graph(
            pos=torch.tensor(atoms.positions, dtype=dtype),
            cell=None,
            x=torch.ones((len(atoms), 1), dtype=dtype),
        )

    def model(data):
        assert torch.is_autocast_enabled("cpu") == (precision == "bf16")
        assert data.dataset_name.item() == 7
        output = data.pos @ torch.eye(3, dtype=dtype)
        calls.append(output.dtype)
        return [output.sum().reshape(1, 1)]

    original_grad = torch.autograd.grad

    def grad(*args, **kwargs):
        assert torch.is_autocast_enabled("cpu") == (precision == "bf16")
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(relaxation, "_atoms_to_graph", graph)
    monkeypatch.setattr(torch.autograd, "grad", grad)
    calc = calculator._build_selected_head_calculator(
        model,
        autocast_ctx=autocast_ctx,
        head_index=7,
        radius=6.0,
        max_neighbours=24,
        param_dtype=dtype,
        device="cpu",
        charge=0,
        spin=0,
    )
    atoms = Atoms("H", positions=[[1, 2, 3]], calculator=calc)
    for expected_energy in (6.0, 9.0):
        assert atoms.get_potential_energy() == pytest.approx(expected_energy)
        assert atoms.get_forces().tolist() == [[-1.0, -1.0, -1.0]]
        atoms.translate([1, 1, 1])
    assert calls == [torch.bfloat16 if precision == "bf16" else dtype] * 2


def test_hydragnn_head_selector_is_exposed_by_public_commands():
    root_command = get_command(app)

    for command_name in ("run", "chat", "supervisor-run"):
        command = root_command.commands[command_name]
        option_names = {
            option for parameter in command.params for option in getattr(parameter, "opts", ())
        }
        assert "--hydragnn-inference-head" in option_names
