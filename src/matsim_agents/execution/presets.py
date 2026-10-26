"""Built-in node geometries and launch flavors for supported facilities.

Presets describe hardware only. They never carry accounts, queues, or site
paths; scenario layouts that use them live in ``deployments/<facility>/resources``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Vendor = Literal["cuda", "rocm", "xpu", "cpu"]
LauncherFlavor = Literal["srun", "mpiexec", "local"]

DEVICE_MASK_VARIABLES: dict[str, str] = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "rocm": "ROCR_VISIBLE_DEVICES",
    "xpu": "ZE_AFFINITY_MASK",
}


class NodeGeometry(BaseModel):
    """Hardware shape shared by every node of a resource plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vendor: Vendor
    devices_per_node: int = Field(ge=0)
    cores_per_node: int = Field(ge=1)
    device_memory_gb: float | None = Field(default=None, gt=0)
    device_unit: str = Field(
        default="gpu", min_length=1, description="What one device is: a gpu, gcd, or tile."
    )
    devices_per_gpu: int = Field(
        default=1, ge=1, description="Devices that make up one physical GPU package."
    )

    @property
    def device_kind(self) -> str:
        return f"{self.vendor}-{self.device_unit}"

    def device_mask_environment(self, devices: tuple[int, ...]) -> dict[str, str]:
        """Return the variables that restrict a process to *devices*."""

        variable = DEVICE_MASK_VARIABLES.get(self.vendor)
        if variable is None:
            return {}
        env = {variable: ",".join(str(index) for index in devices)}
        if self.vendor == "xpu":
            # ZE_AFFINITY_MASK indexes PVC tiles only in FLAT mode.
            env["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
        return env


class FacilityPreset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    launcher: LauncherFlavor
    geometry: NodeGeometry


PRESETS: dict[str, FacilityPreset] = {
    preset.name: preset
    for preset in (
        FacilityPreset(
            name="perlmutter-gpu",
            description="Perlmutter GPU node: 4x A100 40 GB, 64 cores",
            launcher="srun",
            geometry=NodeGeometry(
                vendor="cuda", devices_per_node=4, cores_per_node=64, device_memory_gb=40
            ),
        ),
        FacilityPreset(
            name="perlmutter-gpu-80gb",
            description="Perlmutter GPU node requested with -C 'gpu&hbm80g': 4x A100 80 GB",
            launcher="srun",
            geometry=NodeGeometry(
                vendor="cuda", devices_per_node=4, cores_per_node=64, device_memory_gb=80
            ),
        ),
        FacilityPreset(
            name="frontier",
            description="Frontier node: 8 MI250X GCDs (64 GB each), 56 usable cores",
            launcher="srun",
            geometry=NodeGeometry(
                vendor="rocm",
                devices_per_node=8,
                cores_per_node=56,
                device_memory_gb=64,
                device_unit="gcd",
                devices_per_gpu=2,
            ),
        ),
        FacilityPreset(
            name="aurora",
            description="Aurora node in FLAT mode: 12 PVC tiles (64 GB each), 104 cores",
            launcher="mpiexec",
            geometry=NodeGeometry(
                vendor="xpu",
                devices_per_node=12,
                cores_per_node=104,
                device_memory_gb=64,
                device_unit="tile",
                devices_per_gpu=2,
            ),
        ),
    )
}


def get_preset(name: str) -> FacilityPreset:
    try:
        return PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown resource preset {name!r}; known presets: {known}") from None


__all__ = [
    "DEVICE_MASK_VARIABLES",
    "PRESETS",
    "FacilityPreset",
    "LauncherFlavor",
    "NodeGeometry",
    "Vendor",
    "get_preset",
]
