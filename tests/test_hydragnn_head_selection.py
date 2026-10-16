from __future__ import annotations

import json
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


@pytest.mark.parametrize("declared", [False, True])
def test_public_relaxation_inputs_accept_bundled_declared_checkpoint(
    tmp_path, monkeypatch, declared
):
    from matsim_agents.agents import executor
    from matsim_agents.orchestration.state import MatSimState, TaskSpec

    logdir = tmp_path / "model"
    logdir.mkdir()
    if declared:
        (logdir / "energy-convention.json").write_text("{}")
    for env in ("HYDRAGNN_BRANCH_MLP_CHECKPOINT", "MATSIM_HYDRAGNN_INFERENCE_HEAD"):
        monkeypatch.delenv(env, raising=False)
    captured = []

    def run(args):
        captured.append(args)
        return SimpleNamespace(
            structure_path=args.structure_path,
            final_energy_eV=-1,
            final_max_force_eV_per_A=0,
            num_steps=1,
        )

    monkeypatch.setattr(executor, "relax_with_mlip", run)
    state = MatSimState(pending_tasks=[TaskSpec(structure_path="structure.extxyz")])
    if declared:
        chat = DiscoveryChatConfig(logdir=str(logdir))
        request = RelaxStructureInput(structure_path="structure.extxyz", logdir=str(logdir))
        result = executor.executor_node(
            state, {"configurable": {"mlip_backend": "hydragnn", "logdir": str(logdir)}}
        )
        assert chat.hydragnn_branch_mlp_checkpoint is None
        assert request.hydragnn_branch_mlp_checkpoint is None
        assert captured[0].hydragnn_branch_mlp_checkpoint is None
        assert len(result["results"]) == 1
    else:
        with pytest.raises(ValueError, match="BranchWeightMLP"):
            DiscoveryChatConfig(logdir=str(logdir))
        with pytest.raises(ValueError, match="hydragnn_branch_mlp_checkpoint"):
            RelaxStructureInput(structure_path="structure.extxyz", logdir=str(logdir))
        with pytest.raises(ValueError, match="BranchWeightMLP"):
            executor.executor_node(
                state, {"configurable": {"mlip_backend": "hydragnn", "logdir": str(logdir)}}
            )
        assert not captured


@pytest.mark.parametrize("active_branches", [1, 2])
def test_relaxation_uq_uses_active_routed_branches(tmp_path, monkeypatch, active_branches):
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.io import write

    import matsim_agents.active_learning.calculator as factory
    import matsim_agents.backends.mlip.relaxation as relaxation

    class RoutedCalculator(Calculator):
        implemented_properties = ["energy", "forces"]
        model = SimpleNamespace(num_branches=16)
        active_branch_count = active_branches
        last_branch_weights = np.array([1.0] + [0.0] * 15)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": -1.0, "forces": np.zeros((len(atoms), 3))}

    logdir = tmp_path / "model"
    logdir.mkdir()
    (logdir / "energy-convention.json").write_text("{}")
    structure = tmp_path / "structure.extxyz"
    write(structure, Atoms("Si", cell=[5, 5, 5], pbc=True))
    monkeypatch.setattr(factory, "build_hydragnn_calculator", lambda cfg: RoutedCalculator())
    result = relaxation.relax_with_mlip(
        RelaxStructureInput(structure_path=str(structure), logdir=str(logdir), maxiter=1)
    )
    assert result.top_branch == 0
    if active_branches == 1:
        assert result.top_branch_weight is None
        assert "branch-weight UQ disabled" in result.notes
    else:
        assert result.top_branch_weight == pytest.approx(1)
        assert result.notes is None


@pytest.mark.parametrize("metadata_name", ["routing.json", "newhead.json", None])
def test_unmarked_relaxation_rejects_fitted_offsets_before_loading(
    tmp_path, monkeypatch, metadata_name
):
    import matsim_agents.backends.mlip.relaxation as relaxation

    logdir = tmp_path / "model"
    logdir.mkdir()
    if metadata_name is not None:
        (logdir / metadata_name).write_text(json.dumps({"reference_energies": {"Si": -1}}))
    inference = ModuleType("inference_fused")
    loaded = []

    def load(*args):
        loaded.append(args)
        raise RuntimeError("foundation loader reached")

    inference.load_fused_stack = load
    monkeypatch.setitem(sys.modules, "inference_fused", inference)
    request = RelaxStructureInput(
        structure_path=str(tmp_path / "structure.extxyz"),
        output_dir=str(tmp_path / "output"),
        logdir=str(logdir),
        hydragnn_branch_mlp_checkpoint=str(tmp_path / "mlp.pt"),
    )

    if metadata_name is None:
        with pytest.raises(RuntimeError, match="foundation loader reached"):
            relaxation.relax_with_mlip(request)
        assert len(loaded) == 1
    else:
        with pytest.raises(ValueError, match="lacks verified energy convention"):
            relaxation.relax_with_mlip(request)
        assert not loaded


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

    monkeypatch.setattr(relaxation, "atoms_to_graph", graph)
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


@pytest.mark.parametrize("mlp_precision", ["fp32", "bf16"])
def test_fused_xpu_builder_moves_and_exposes_mlp(tmp_path, monkeypatch, mlp_precision):
    import torch

    from matsim_agents.backends.mlip.relaxation import _load_fused_stack

    (tmp_path / "config.json").write_text("{}")
    device = torch.device("xpu:0")
    model = SimpleNamespace()
    moves = []
    mlp = SimpleNamespace(to=lambda **kwargs: moves.append(kwargs) or mlp)
    config = {"NeuralNetwork": {"Architecture": {}}}
    inference = ModuleType("inference_fused")
    inference.load_fused_stack = lambda *args: (
        model,
        mlp,
        config,
        device,
        nullcontext(),
        torch.float32,
        16,
        torch.device("cpu"),
        nullcontext(),
        False,
        "fp32",
        mlp_precision,
    )
    inference.run_fused_inference = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "inference_fused", inference)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    contexts = []
    monkeypatch.setattr(
        torch, "autocast", lambda *args, **kwargs: contexts.append((args, kwargs)) or nullcontext()
    )
    stack = _load_fused_stack(tmp_path, None, "mlp.pt", "fp32", mlp_precision, "xpu")
    assert stack[7] == device
    assert stack[9] is (mlp_precision == "fp32")
    assert bool(contexts) is (mlp_precision == "bf16")
    if contexts:
        assert contexts == [(("xpu",), {"dtype": torch.bfloat16})]
    result = build_hydragnn_calculator(
        HydraGNNConfig(logdir=tmp_path, mlp_checkpoint=tmp_path / "mlp.pt", mlp_device="xpu")
    )
    assert result.model is model and result.mlp is mlp
    assert moves == [{"device": device}, {"device": device}]


def test_fused_xpu_loader_rejects_unavailable_xpu_and_cpu_gnn(monkeypatch):
    import torch

    from matsim_agents.backends.mlip.relaxation import _load_fused_stack

    inference = ModuleType("inference_fused")
    calls = []
    inference.load_fused_stack = lambda *args: (
        calls.append(args)
        or (
            None,
            None,
            {},
            torch.device("cpu"),
            None,
            None,
            16,
            torch.device("cpu"),
            None,
            False,
            "fp32",
            "fp32",
        )
    )
    monkeypatch.setitem(sys.modules, "inference_fused", inference)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        _load_fused_stack("model", None, "mlp.pt", None, None, "xpu")
    assert not calls
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="but XPU was requested"):
        _load_fused_stack("model", None, "mlp.pt", None, None, "xpu")
    for device in ("cpu", "cuda"):
        stack = _load_fused_stack("model", None, "mlp.pt", None, None, device)
        assert stack[3].type == "cpu"
        assert calls[-1][-1] == device
