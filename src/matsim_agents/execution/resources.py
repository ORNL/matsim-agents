"""Portable descriptions of resources requested by a scientific task."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Scheduler-neutral resources required by one execution step."""

    nodes: int = 1
    tasks_per_node: int = 1
    gpus_per_node: int = 0
    walltime: str | None = None
    queue: str | None = None
    account: str | None = None
