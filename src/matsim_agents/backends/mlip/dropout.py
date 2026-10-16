"""Test-time (MC) dropout helpers for MLIP torch models."""

from __future__ import annotations


def enable_dropout(model) -> None:
    """Set every Dropout layer in a torch ``nn.Module`` to train() while keeping BN in eval."""
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def inject_inference_dropout(
    model,
    p: float = 0.1,
    target_layers: str = "linear",
    max_layers: int | None = None,
) -> int:
    """Insert test-time dropout into a model that was trained without it.

    For backends such as UMA there are no native ``nn.Dropout`` layers, so
    MC-Dropout would measure zero variance. This attaches an ``nn.Dropout``
    child to each target layer and registers a forward hook that passes the
    layer's output through it. The injected modules start in ``eval()`` mode
    (identity), so deterministic prediction/relaxation is unaffected; they only
    become stochastic when :func:`score_mc_dropout` toggles dropout layers into
    ``train()`` mode.

    Idempotent: layers already carrying an injected dropout are skipped.

    Returns the number of layers instrumented.

    Note: this is a *heuristic* uncertainty (uncalibrated), not a Bayesian
    posterior — the model was not trained with dropout in these positions.
    """
    import torch
    import torch.nn as nn

    if target_layers == "linear":
        types: tuple[type, ...] = (nn.Linear,)
    elif target_layers == "all":
        types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    else:
        raise ValueError(f"target_layers must be 'linear' or 'all', got {target_layers!r}")

    # Snapshot targets first: we mutate ._modules during the loop, which would
    # otherwise invalidate the model.modules() generator.
    targets = [
        m
        for m in model.modules()
        if isinstance(m, types) and not getattr(m, "_mc_dropout_injected", False)
    ]

    def _make_hook(drop: nn.Module):
        def _hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                return drop(output)
            return output

        return _hook

    n = 0
    for module in targets:
        drop = nn.Dropout(p=float(p))
        drop.eval()  # dormant until score_mc_dropout flips it to train()
        module.add_module("_mc_injected_dropout", drop)
        module.register_forward_hook(_make_hook(drop))
        module._mc_dropout_injected = True  # type: ignore[attr-defined]
        n += 1
        if max_layers is not None and n >= max_layers:
            break
    return n
