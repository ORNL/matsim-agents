"""Device masks for child processes and in-process device selection.

Placement device indices are positions among the devices this process can
see. Without a device mask that is every device of the node; under a mask (for
example a Slurm step, which renumbers its GPUs from zero) index ``i`` means
the ``i``-th visible device.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from matsim_agents.execution.presets import DEVICE_MASK_VARIABLES


def _visible(variable: str) -> list[str] | None:
    inherited = os.environ.get(variable, "").strip()
    if not inherited:
        return None
    return [item.strip() for item in inherited.split(",") if item.strip()]


def child_mask_environment(env: Mapping[str, str], *, owner: str) -> dict[str, str]:
    """Translate the device mask in *env* through the mask this process already has."""

    result = dict(env)
    for variable in set(DEVICE_MASK_VARIABLES.values()) & set(env):
        visible = _visible(variable)
        if visible is None:
            continue
        requested = [int(item) for item in env[variable].split(",") if item]
        if any(index >= len(visible) for index in requested):
            raise ValueError(
                f"{owner} uses devices {requested}, but this process sees only "
                f"{variable}={','.join(visible)}; give the driver every device of {owner}"
            )
        result[variable] = ",".join(visible[index] for index in requested)
    return result


def visible_ordinals(vendor: str, devices: Sequence[int], *, owner: str) -> tuple[int, ...]:
    """Check that *devices* are visible to this process and return their ordinals."""

    variable = DEVICE_MASK_VARIABLES.get(vendor)
    visible = _visible(variable) if variable else None
    if visible is not None and any(index >= len(visible) for index in devices):
        raise ValueError(
            f"{owner} uses devices {list(devices)}, but this process sees only "
            f"{variable}={','.join(visible)}"
        )
    return tuple(devices)


__all__ = ["child_mask_environment", "visible_ordinals"]
