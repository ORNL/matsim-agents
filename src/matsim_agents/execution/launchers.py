"""Interfaces for launching deterministic scientific computations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from matsim_agents.execution.resources import ResourceRequest


class Launcher(Protocol):
    """Scheduler-independent launcher contract."""

    def launch(
        self,
        command: Sequence[str],
        *,
        resources: ResourceRequest,
        workdir: Path,
    ) -> int:
        """Launch *command* and return an implementation-defined job identifier."""
