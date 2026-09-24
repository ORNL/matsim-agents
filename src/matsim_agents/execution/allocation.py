"""Scheduler-neutral discovery and partitioning of an existing allocation."""

from __future__ import annotations

import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Allocation:
    scheduler: str
    nodes: tuple[str, ...]
    gpus_per_node: int | None = None

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def groups(self, nodes_per_job: int) -> list[tuple[str, ...]]:
        if nodes_per_job < 1:
            raise ValueError("nodes_per_job must be positive")
        return [
            self.nodes[start : start + nodes_per_job]
            for start in range(0, self.node_count - nodes_per_job + 1, nodes_per_job)
        ]


def _expand_slurm_nodelist(value: str) -> tuple[str, ...]:
    """Use scontrol when available; accept a plain comma list in tests/fallbacks."""

    import shutil
    import subprocess

    if shutil.which("scontrol"):
        result = subprocess.run(
            ["scontrol", "show", "hostnames", value],
            check=False,
            capture_output=True,
            text=True,
        )
        nodes = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if result.returncode == 0 and nodes:
            return nodes
    if "[" not in value:
        return tuple(node.strip() for node in value.split(",") if node.strip())
    return ()


def _int_from_environment(*names: str) -> int | None:
    for name in names:
        raw = os.environ.get(name)
        if not raw:
            continue
        # Slurm may report values such as ``gpu:a100:4``; the allocation
        # count is the final integer, not the digits embedded in a GPU model.
        matches = re.findall(r"\d+", raw)
        if matches:
            return int(matches[-1])
    return None


def discover_allocation() -> Allocation:
    """Discover Slurm, PBS/PALS, or a safe single-node local allocation."""

    override_gpus = _int_from_environment("MATSIM_GPUS_PER_NODE")
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_NUM_NODES"):
        nodes = _expand_slurm_nodelist(os.environ.get("SLURM_JOB_NODELIST", ""))
        count = _int_from_environment("SLURM_JOB_NUM_NODES") or len(nodes) or 1
        if not nodes:
            nodes = tuple(f"slurm-node-{index}" for index in range(count))
        return Allocation(
            scheduler="slurm",
            nodes=nodes,
            gpus_per_node=override_gpus
            or _int_from_environment("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"),
        )

    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and Path(nodefile).is_file():
        # PBS nodefiles may repeat a host per CPU slot; DFT placement needs
        # unique nodes while preserving scheduler order.
        nodes = tuple(
            dict.fromkeys(
                line.strip() for line in Path(nodefile).read_text().splitlines() if line.strip()
            )
        )
        return Allocation(scheduler="pbs", nodes=nodes, gpus_per_node=override_gpus)

    return Allocation(
        scheduler="local",
        nodes=(socket.gethostname(),),
        gpus_per_node=override_gpus,
    )


def validate_dft_resources(
    allocation: Allocation,
    *,
    nodes_per_job: int,
    ranks_per_node: int,
) -> None:
    if nodes_per_job > allocation.node_count:
        raise ValueError(
            f"DFT job requests {nodes_per_job} node(s), but allocation has {allocation.node_count}"
        )
    if allocation.gpus_per_node is not None and ranks_per_node != allocation.gpus_per_node:
        raise ValueError(
            f"ranks_per_node={ranks_per_node} does not use exactly the allocated "
            f"{allocation.gpus_per_node} GPU(s) per node"
        )


__all__ = ["Allocation", "discover_allocation", "validate_dft_resources"]
