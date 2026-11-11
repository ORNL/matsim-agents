import pytest

torch = pytest.importorskip("torch")

from matsim_agents.backends.mlip.hydragnn_checkpoint import (  # noqa: E402
    load_hydragnn_state_dict,
    restore_hydragnn_checkpoint,
)


class LazyConditioning(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.graph_conditioner = None
        self.graph_concat_projector = None
        self.graph_pool_projector = None

    def _ensure_graph_conditioner(self, attrs, device):
        self.graph_conditioner = torch.nn.Sequential(
            torch.nn.Linear(attrs, 3),
            torch.nn.SiLU(),
            torch.nn.Linear(3, 4),
        ).to(device=device)

    def _ensure_graph_concat_projector(self, attrs, channels, device, dtype):
        self.graph_concat_projector = torch.nn.Linear(channels + attrs, channels).to(
            device=device, dtype=dtype
        )
        self.graph_concat_projector_in_dim = channels + attrs

    def _ensure_graph_pool_projector(self, attrs, channels, device, dtype):
        self.graph_pool_projector = torch.nn.Sequential(
            torch.nn.Linear(channels + attrs, channels),
            torch.nn.SiLU(),
            torch.nn.Linear(channels, channels),
        ).to(device=device, dtype=dtype)
        self.graph_pool_projector_in_dim = channels + attrs


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("projector", ["concat", "pool"])
def test_restores_lazy_conditioning_exactly(dtype, wrapped, projector):
    trained = LazyConditioning().to(dtype=dtype)
    getattr(trained, f"_ensure_graph_{projector}_projector")(2, 2, "cpu", dtype)
    rebuilt = LazyConditioning().to(dtype=dtype)
    if wrapped:
        trained = torch.nn.ModuleDict({"model": trained})
        rebuilt = torch.nn.ModuleDict({"model": rebuilt})
    state = trained.state_dict()
    load_hydragnn_state_dict(rebuilt, state)
    assert state.keys() == rebuilt.state_dict().keys()
    for key, expected in state.items():
        torch.testing.assert_close(rebuilt.state_dict()[key], expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_hydragnn_state_dict(rebuilt, {})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("wrapped", [False, True])
def test_restores_lazy_film_conditioner_exactly(dtype, wrapped):
    trained = LazyConditioning().to(dtype=dtype)
    trained._ensure_graph_conditioner(2, "cpu")
    trained.graph_conditioner.to(dtype=dtype)
    rebuilt = LazyConditioning().to(dtype=dtype)
    if wrapped:
        trained = torch.nn.ModuleDict({"model": trained})
        rebuilt = torch.nn.ModuleDict({"model": rebuilt})
    state = trained.state_dict()
    load_hydragnn_state_dict(rebuilt, state)
    assert state.keys() == rebuilt.state_dict().keys()
    for key, expected in state.items():
        torch.testing.assert_close(rebuilt.state_dict()[key], expected, rtol=0, atol=0)


def test_rejects_invalid_projector_shape():
    with pytest.raises(ValueError, match="projector shape"):
        load_hydragnn_state_dict(
            LazyConditioning(), {"graph_concat_projector.weight": torch.ones(2, 2)}
        )


def test_inference_restores_ddp_checkpoint_and_freezes_lazy_parameters(tmp_path):
    trained = LazyConditioning()
    trained._ensure_graph_concat_projector(2, 2, "cpu", torch.float32)
    path = tmp_path / "model.pk"
    torch.save(
        {"model_state_dict": {f"module.{k}": v for k, v in trained.state_dict().items()}}, path
    )
    rebuilt = LazyConditioning()
    restore_hydragnn_checkpoint(rebuilt, path, device="cpu")
    assert not rebuilt.training
    assert all(not parameter.requires_grad for parameter in rebuilt.parameters())
    for key, expected in trained.state_dict().items():
        torch.testing.assert_close(rebuilt.state_dict()[key], expected, rtol=0, atol=0)


def test_inference_restores_nested_module_model_checkpoint_prefix(tmp_path):
    trained = LazyConditioning().double()
    trained._ensure_graph_concat_projector(2, 2, "cpu", torch.float64)
    path = tmp_path / "model.pk"
    state = {f"module.model.{key}": value for key, value in trained.state_dict().items()}
    state["module.model.branch_weight_mlp.weight"] = torch.ones(3, 3, dtype=torch.float64)
    torch.save({"model_state_dict": state}, path)

    rebuilt = LazyConditioning().double()
    restore_hydragnn_checkpoint(rebuilt, path, device="cpu")

    for key, expected in trained.state_dict().items():
        torch.testing.assert_close(rebuilt.state_dict()[key], expected, rtol=0, atol=0)
    report = rebuilt._hydragnn_checkpoint_load_report
    assert report["model_tensor_count"] == len(trained.state_dict())
    assert report["matched_tensor_count"] == len(trained.state_dict())
    assert report["missing_tensor_count"] == 0
    assert report["unused_checkpoint_tensor_count"] == 1
    assert report["wrapper_prefixes_removed"] == 2


def test_inference_rejects_checkpoint_with_no_matching_model_keys(tmp_path):
    path = tmp_path / "model.pk"
    torch.save(
        {"model_state_dict": {"module.model.unrelated.weight": torch.ones(2, 2)}},
        path,
    )
    with pytest.raises(RuntimeError, match="no keys matching the constructed model"):
        restore_hydragnn_checkpoint(LazyConditioning(), path, device="cpu")


@pytest.mark.parametrize("ddp", [False, True])
def test_fused_loader_strictly_restores_model_prefix_and_lazy_weights(tmp_path, monkeypatch, ddp):
    import sys
    from types import ModuleType

    from benchmarks.e4_mlip.pilot import validate_checkpoint
    from matsim_agents.backends.mlip.relaxation import _load_fused_stack

    trained = torch.nn.ModuleDict({"model": LazyConditioning().double()})
    trained["model"]._ensure_graph_concat_projector(2, 2, "cpu", torch.float64)
    state = trained.state_dict()
    path = tmp_path / "model.pk"
    torch.save(
        {"model_state_dict": {f"module.{k}" if ddp else k: v for k, v in state.items()}},
        path,
    )
    before = path.read_bytes()
    rebuilt = torch.nn.ModuleDict({"model": LazyConditioning().double()})
    upstream = ModuleType("inference_fused")
    upstream.load_fused_stack = lambda *args: (rebuilt, None, {}, torch.device("cpu"))
    monkeypatch.setitem(sys.modules, "inference_fused", upstream)
    stack = _load_fused_stack(tmp_path, path, "unused", "fp64", "fp64", "cpu")
    assert stack[0] is rebuilt
    assert validate_checkpoint(rebuilt) == len(state)
    assert all(k.startswith("model.") for k in rebuilt._hydragnn_checkpoint_state_keys)
    for key, expected in state.items():
        torch.testing.assert_close(rebuilt.state_dict()[key], expected, rtol=0, atol=0)
    assert path.read_bytes() == before


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape", "double-ddp"])
def test_fused_loader_rejects_invalid_state_before_inference(tmp_path, monkeypatch, problem):
    import sys
    from types import ModuleType

    from matsim_agents.backends.mlip.relaxation import _load_fused_stack

    model = torch.nn.ModuleDict({"model": torch.nn.Linear(2, 2)})
    state = {f"module.{k}": v.clone() for k, v in model.state_dict().items()}
    if problem == "missing":
        state.pop("module.model.bias")
    elif problem == "unexpected":
        state["module.model.extra"] = torch.zeros(1)
    elif problem == "shape":
        state["module.model.weight"] = torch.zeros(3, 2)
    else:
        state = {f"module.{k}": v for k, v in state.items()}
    path = tmp_path / "model.pk"
    torch.save({"model_state_dict": state}, path)
    upstream = ModuleType("inference_fused")
    upstream.load_fused_stack = lambda *args: (model, None, {}, torch.device("cpu"))
    monkeypatch.setitem(sys.modules, "inference_fused", upstream)
    with pytest.raises(RuntimeError):
        _load_fused_stack(tmp_path, path, "unused", "fp32", "fp32", "cpu")
    assert not hasattr(model, "_hydragnn_checkpoint_state_keys")
