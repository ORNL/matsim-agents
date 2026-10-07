from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_training_loss_uses_requested_autocast_and_preserves_force_gradients(routed, bf16):
    import torch

    from matsim_agents.active_learning.finetune_hydragnn import _batch_loss
    from matsim_agents.active_learning.finetune_hydragnn_newhead import _batch_loss_single_head

    model = torch.nn.Linear(3, 1)
    seen = []
    batch = SimpleNamespace(
        num_graphs=1,
        pos=torch.randn(3, 3),
        batch=torch.zeros(3, dtype=torch.long),
        energy=torch.tensor([-3.0]),
        forces=torch.zeros(3, 3),
        chemical_composition=torch.zeros(1, 118),
    )

    def forward(graph):
        seen.append(torch.is_autocast_enabled("cpu"))
        output = model(graph.pos.square())
        assert output.dtype == (torch.bfloat16 if bf16 else torch.float32)
        return output.sum().reshape(1, 1)

    dtype = torch.bfloat16 if bf16 else None
    if routed:
        loss = _batch_loss(
            forward,
            lambda _: torch.zeros(1, 16),
            batch,
            torch.tensor([7]),
            1,
            1,
            torch.float32,
            autocast_dtype=dtype,
        )
    else:
        loss = _batch_loss_single_head(forward, batch, 1, 1, autocast_dtype=dtype)
    loss.backward()
    assert seen == [bf16]
    assert torch.isfinite(loss)
    assert torch.isfinite(model.weight.grad).all()
    assert model.weight.grad.abs().sum() > 0
    assert not torch.is_autocast_enabled("cpu")
