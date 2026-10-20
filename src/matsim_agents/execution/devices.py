"""Device masks for child processes and in-process device selection.

Placement device indices are positions among the devices this process can
see. Without a device mask that is every device of the node; under a mask (for
example a Slurm step, which renumbers its GPUs from zero) index ``i`` means
the ``i``-th visible device.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping, Sequence

from matsim_agents.execution.presets import DEVICE_MASK_VARIABLES


def _visible(variable: str) -> list[str] | None:
    """Devices a mask variable leaves visible; ``None`` when nothing is masked.

    An empty ``CUDA_VISIBLE_DEVICES`` or ``ROCR_VISIBLE_DEVICES`` hides every
    device, so it yields an empty list. An empty ``ZE_AFFINITY_MASK`` applies no
    mask, like an unset one.
    """

    inherited = os.environ.get(variable)
    if inherited is None:
        return None
    inherited = inherited.strip()
    if not inherited:
        return None if variable == "ZE_AFFINITY_MASK" else []
    return [item.strip() for item in inherited.split(",") if item.strip()]


def _effective(variable: str) -> list[str] | None:
    """Devices this process can use through *variable*'s vendor masks.

    On ROCm, ``HIP_VISIBLE_DEVICES`` selects among the devices that
    ``ROCR_VISIBLE_DEVICES`` leaves visible (or among all devices without it).
    """

    if variable != "ROCR_VISIBLE_DEVICES":
        return _visible(variable)
    rocr = _visible("ROCR_VISIBLE_DEVICES")
    hip = _visible("HIP_VISIBLE_DEVICES")
    if hip is None:
        return rocr
    if rocr is None:
        return hip
    try:
        positions = [int(item) for item in hip]
    except ValueError:
        raise ValueError(
            f"HIP_VISIBLE_DEVICES={','.join(hip)} must list indices into "
            f"ROCR_VISIBLE_DEVICES={','.join(rocr)}"
        ) from None
    if any(position >= len(rocr) for position in positions):
        raise ValueError(
            f"HIP_VISIBLE_DEVICES={','.join(hip)} selects devices outside "
            f"ROCR_VISIBLE_DEVICES={','.join(rocr)}"
        )
    return [rocr[position] for position in positions]


def apply_mask_environment(env: MutableMapping[str, str], mask: Mapping[str, str]) -> None:
    """Install a translated device mask, dropping masks it replaces.

    A child's ``ROCR_VISIBLE_DEVICES`` already names its exact devices, so an
    inherited ``HIP_VISIBLE_DEVICES`` would wrongly re-index them.
    """

    env.update(mask)
    if "ROCR_VISIBLE_DEVICES" in mask:
        env.pop("HIP_VISIBLE_DEVICES", None)


def child_mask_environment(env: Mapping[str, str], *, owner: str) -> dict[str, str]:
    """Translate the device mask in *env* through the mask this process already has."""

    result = dict(env)
    for variable in set(DEVICE_MASK_VARIABLES.values()) & set(env):
        visible = _effective(variable)
        if visible is None:
            continue
        requested = [int(item) for item in env[variable].split(",") if item]
        if any(index >= len(visible) for index in requested):
            raise ValueError(
                f"{owner} uses devices {requested}, but this process sees only "
                f"{variable}={','.join(visible) or '(none)'}; give the driver every device "
                f"of {owner}"
            )
        result[variable] = ",".join(visible[index] for index in requested)
    return result


def visible_ordinals(vendor: str, devices: Sequence[int], *, owner: str) -> tuple[int, ...]:
    """Check that *devices* are visible to this process and return their ordinals."""

    variable = DEVICE_MASK_VARIABLES.get(vendor)
    visible = _effective(variable) if variable else None
    if visible is not None and any(index >= len(visible) for index in devices):
        raise ValueError(
            f"{owner} uses devices {list(devices)}, but this process sees only "
            f"{variable}={','.join(visible) or '(none)'}"
        )
    return tuple(devices)


__all__ = ["apply_mask_environment", "child_mask_environment", "visible_ordinals"]
