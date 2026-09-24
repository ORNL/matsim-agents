"""Collision-resistant, restart-friendly storage for scientific runs."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from matsim_agents.execution.contracts import ProvenanceRecord

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_STANDARD_DIRS = ("structures", "calculations", "datasets", "models")


def safe_component(value: str, *, fallback: str = "run") -> str:
    """Return a path-safe component; path separators can never escape a run."""

    cleaned = _SAFE_NAME.sub("-", value.strip()).strip("-._")
    return (cleaned or fallback)[:96]


def make_run_id(now: datetime | None = None) -> str:
    """Return ``UTC timestamp + random suffix`` suitable for concurrent jobs."""

    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{stamp.strftime('%Y-%m-%dT%H-%M-%SZ')}_{uuid.uuid4().hex[:8]}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


class ScientificRunDirectory:
    """Own the canonical files and event stream for one scientific run."""

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        workflow: str,
        request: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        provenance: ProvenanceRecord,
        run_id: str | None = None,
    ) -> ScientificRunDirectory:
        run_id = safe_component(run_id or make_run_id())
        path = Path(root) / run_id
        path.mkdir(parents=True, exist_ok=False)
        run = cls(path, run_id)
        for name in _STANDARD_DIRS:
            (path / name).mkdir()
        run.write_json("request.json", dict(request))
        run.write_json("resolved_config.json", dict(resolved_config))
        run.write_json("provenance.json", provenance)
        run.append_event("run_created", {"workflow": workflow})
        return run

    @classmethod
    def open(cls, path: str | Path) -> ScientificRunDirectory:
        path = Path(path)
        if not path.is_dir() or not (path / "provenance.json").is_file():
            raise ValueError(f"not a matsim-agents scientific run directory: {path}")
        return cls(path, path.name)

    def write_json(self, name: str, payload: Any) -> Path:
        """Atomically write JSON inside the run directory."""

        destination = self.path / safe_component(name)
        encoded = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=self.path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def append_event(self, event: str, payload: Mapping[str, Any]) -> Path:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": _jsonable(dict(payload)),
        }
        destination = self.path / "events.jsonl"
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return destination


__all__ = ["ScientificRunDirectory", "make_run_id", "safe_component"]
