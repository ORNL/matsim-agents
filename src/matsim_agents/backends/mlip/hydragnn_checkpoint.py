"""Restore HydraGNN conditioning modules created lazily by the first forward."""

from collections.abc import Mapping


def _normalize_state_dict_prefix(model, state_dict: Mapping):
    """Strip at most one recognized wrapper prefix, choosing the best model-key match."""
    target_keys = set(model.state_dict())
    candidates = [(0, dict(state_dict))]
    for prefix in ("module.", "model."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            candidates.append((1, {key[len(prefix) :]: value for key, value in state_dict.items()}))
    depth, normalized = max(
        candidates,
        key=lambda item: (len(target_keys.intersection(item[1])), -item[0]),
    )
    matched = len(target_keys.intersection(normalized))
    if not matched:
        examples = list(normalized)[:3]
        raise RuntimeError(
            "HydraGNN checkpoint has no keys matching the constructed model; "
            f"checkpoint keys begin {examples}, model keys begin {list(target_keys)[:3]}"
        )
    return normalized, matched, depth


def restore_hydragnn_checkpoint(model, checkpoint_path, *, device):
    import torch

    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model_state_dict", state)
    state_dict, _, prefix_depth = _normalize_state_dict_prefix(model, state_dict)
    load_hydragnn_state_dict(model, state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_tensor_count = len(model.state_dict())
    model._hydragnn_checkpoint_load_report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_tensor_count": len(state_dict),
        "model_tensor_count": model_tensor_count,
        # Strict loading succeeded, so every model tensor (including lazy modules) was restored.
        "matched_tensor_count": model_tensor_count,
        "missing_tensor_count": 0,
        "unused_checkpoint_tensor_count": 0,
        "wrapper_prefixes_removed": prefix_depth,
    }
    return tuple(state_dict)


def load_hydragnn_state_dict(model, state_dict: Mapping, *, strict=True):
    for prefix, module in model.named_modules():
        prefix = f"{prefix}." if prefix else ""
        conditioner_key = f"{prefix}graph_conditioner.0.weight"
        if conditioner_key in state_dict:
            ensure = getattr(module, "_ensure_graph_conditioner", None)
            if ensure is not None:
                weight = state_dict[conditioner_key]
                if weight.ndim != 2 or weight.shape[1] < 1:
                    raise ValueError(f"Invalid HydraGNN graph conditioner shape: {conditioner_key}")
                parameter = next(module.parameters())
                ensure(weight.shape[1], parameter.device)
                module.graph_conditioner.to(device=parameter.device, dtype=parameter.dtype)
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
