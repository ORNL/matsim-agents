import pytest

torch = pytest.importorskip("torch")

from matsim_agents.active_learning.hydragnn_checkpoint import (  # noqa: E402
    load_hydragnn_state_dict,
    restore_hydragnn_checkpoint,
)


class LazyConditioning(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.graph_concat_projector = None
        self.graph_pool_projector = None

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
