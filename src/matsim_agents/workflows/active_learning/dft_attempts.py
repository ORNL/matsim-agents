"""Persistent, uniquely reserved directories for active-learning DFT attempts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from matsim_agents.core.fileio import atomic_write_text

ATTEMPTS_DIR = "dft_attempts"


def create_attempt(root: Path, *, iteration: int, kind: str, identity: str) -> Path:
    """Reserve a unique directory before starting DFT; never reuse an existing directory."""
    parent = root.resolve() / ATTEMPTS_DIR
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        directory = parent / uuid4().hex
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        break
    write_attempt(
        directory,
        {
            "attempt_id": directory.name,
            "iteration": iteration,
            "kind": kind,
            "identity": identity,
            "status": "prepared",
        },
    )
    return directory


def write_attempt(directory: Path, record: dict[str, Any]) -> None:
    atomic_write_text(directory / "attempt.json", json.dumps(record, indent=2, sort_keys=True))


def update_attempt(directory: Path, **fields: Any) -> None:
    record = json.loads((directory / "attempt.json").read_text(encoding="utf-8"))
    record.update(fields)
    write_attempt(directory, record)
