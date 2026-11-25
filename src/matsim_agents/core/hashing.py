"""Content hashing and canonical JSON used for provenance and cache keys."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, streamed in 1 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any, *, default: Any = None) -> str:
    """Deterministic compact JSON: sorted keys, no whitespace, ASCII only."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=default
    )


def sha256_json(value: Any, *, default: Any = None) -> str:
    """Hex SHA-256 of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(value, default=default).encode()).hexdigest()


__all__ = ["canonical_json", "sha256_file", "sha256_json"]
