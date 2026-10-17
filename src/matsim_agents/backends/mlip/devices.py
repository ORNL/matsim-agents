"""Accelerator selection shared by HydraGNN inference and fine-tuning."""

from __future__ import annotations

ACCELERATOR_TYPES = frozenset({"cuda", "xpu"})


def default_torch_device() -> str:
    """Return ``"cuda"``, then ``"xpu"`` (Intel GPUs), else ``"cpu"``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    return "cpu"
