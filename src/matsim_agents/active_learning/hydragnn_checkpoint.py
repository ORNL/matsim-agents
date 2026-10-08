"""Restore HydraGNN conditioning modules created lazily by the first forward."""

from collections.abc import Mapping


def restore_hydragnn_checkpoint(model, checkpoint_path, *, device):
    import torch

    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model_state_dict", state)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    load_hydragnn_state_dict(model, state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_hydragnn_state_dict(model, state_dict: Mapping, *, strict=True):
    for prefix, module in model.named_modules():
        prefix = f"{prefix}." if prefix else ""
        for name, weight_name in (
            ("graph_concat_projector", "weight"),
            ("graph_pool_projector", "0.weight"),
        ):
            key = f"{prefix}{name}.{weight_name}"
            if key not in state_dict:
                continue
            ensure = getattr(module, f"_ensure_{name}", None)
            if ensure is None:
                continue  # Strict loading reports unsupported checkpoint modules.
            weight = state_dict[key]
            if weight.ndim != 2 or weight.shape[1] <= weight.shape[0]:
                raise ValueError(f"Invalid HydraGNN conditioning projector shape: {key}")
            parameter = next(module.parameters())
            channels, inputs = weight.shape
            ensure(inputs - channels, channels, parameter.device, dtype=parameter.dtype)
    return model.load_state_dict(state_dict, strict=strict)
