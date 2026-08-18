"""Small, dependency-free helpers for auditable workflow handoffs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunStore(Protocol):
    """Minimal contract for a persistent run/provenance store.

    A store accepts structured records (one per DFT job, AL iteration, or
    agent action) and lets callers iterate over them for auditing, plotting,
    or resumption.  The backing medium — JSONL file, SQLite, remote object
    store — is implementation-specific.
    """

    def append(self, record: Mapping[str, Any]) -> None:
        """Persist one record."""
        ...

    def iter_records(self) -> Iterable[Mapping[str, Any]]:
        """Yield all previously persisted records in insertion order."""
        ...


def append_jsonl_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one stable JSON record and return the resolved output path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return output


class JsonlRunStore:
    """Concrete :class:`RunStore` backed by a newline-delimited JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append(self, record: Mapping[str, Any]) -> None:
        append_jsonl_record(self._path, record)

    def iter_records(self) -> Iterable[Mapping[str, Any]]:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


__all__ = ["RunStore", "JsonlRunStore", "append_jsonl_record"]
