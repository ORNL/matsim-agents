"""HPC-neutral resource, launch, and provenance interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from matsim_agents.execution.provenance import RunStore, append_jsonl_record
from matsim_agents.execution.resources import ResourceRequest


@runtime_checkable
class ExecutionPlatform(Protocol):
    """Scheduler-neutral contract for an HPC deployment target.

    Each concrete platform (Perlmutter/SLURM, Frontier/PBS, Aurora/PBS, …)
    provides a :meth:`submit` that dispatches a command and returns an
    opaque job identifier.  The platform is also the source of truth for
    what resources are actually available to the current allocation.
    """

    name: str

    def submit(
        self,
        command: Sequence[str],
        *,
        resources: ResourceRequest,
        workdir: str,
        env: dict[str, str] | None = None,
    ) -> str:
        """Submit *command* and return an opaque job identifier."""
        ...

    def available_resources(self) -> ResourceRequest:
        """Return the resources visible to the current allocation."""
        ...


__all__ = ["ExecutionPlatform", "ResourceRequest", "RunStore", "append_jsonl_record"]
