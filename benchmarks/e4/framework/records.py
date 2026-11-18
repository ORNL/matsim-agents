"""Append-only attempt records that double as the benchmark checkpoint.

Every attempt is appended to ``attempts.jsonl`` and flushed to stable storage
before the next attempt of that task starts. A task is complete once one of its
records is terminal, so a resumed run never repeats completed work and keeps
retry budgets and attempt numbers from the interrupted run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TERMINAL_FIELD = "terminal"


@dataclass
class TaskHistory:
    """Attempts already recorded for one task."""

    attempts: int = 0
    terminal: dict[str, Any] | None = None
    retries_by_failure: dict[str, int] = field(default_factory=dict)


class AttemptRecorder:
    """Thread-safe, fsync-backed attempt log; also measures its own overhead."""

    def __init__(self, path: str | Path, *, clock_origin: float, segment: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock_origin = clock_origin
        self.segment = segment
        self.repaired_bytes = _drop_torn_tail(self.path)
        self._lock = threading.Lock()
        self.overhead_seconds = 0.0
        self.records_written = 0

    def offset(self, monotonic: float | None = None) -> float:
        return round((time.monotonic() if monotonic is None else monotonic) - self.clock_origin, 6)

    def append(self, record: Mapping[str, Any]) -> None:
        started = time.perf_counter()
        line = (
            json.dumps({**record, "segment": self.segment}, sort_keys=True, allow_nan=True) + "\n"
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            self.records_written += 1
            self.overhead_seconds += time.perf_counter() - started


def _drop_torn_tail(path: Path) -> int:
    """Truncate a final line left incomplete by a crash so appends stay parseable."""

    if not path.is_file():
        return 0
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    keep = data.rfind(b"\n") + 1
    with path.open("r+b") as stream:
        stream.truncate(keep)
        stream.flush()
        os.fsync(stream.fileno())
    return len(data) - keep


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read all complete records; a torn final line from a crash is ignored."""

    path = Path(path)
    if not path.is_file():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return records


def task_histories(records: Iterable[Mapping[str, Any]]) -> dict[str, TaskHistory]:
    histories: dict[str, TaskHistory] = {}
    for record in records:
        history = histories.setdefault(record["task_id"], TaskHistory())
        history.attempts = max(history.attempts, int(record["attempt"]))
        if record.get("retry_scheduled") and record.get("failure_class"):
            failure = record["failure_class"]
            history.retries_by_failure[failure] = history.retries_by_failure.get(failure, 0) + 1
        if record.get(TERMINAL_FIELD):
            history.terminal = dict(record)
    return histories


__all__ = ["AttemptRecorder", "TaskHistory", "read_records", "task_histories"]
