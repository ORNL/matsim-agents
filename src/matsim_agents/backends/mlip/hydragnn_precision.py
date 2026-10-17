"""Use the model's resolved autocast dtype without changing master parameters."""

from contextlib import nullcontext


def hydragnn_autocast(device, autocast_dtype):
    if autocast_dtype is None:
        return nullcontext()
    import torch

    return torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype)
