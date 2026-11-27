"""Crash-safe file writes: atomic replace and durable line appends."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# os.umask can only be read by setting it; mkstemp's owner-only mode would hide shared runs.
_UMASK = os.umask(0o022)
os.umask(_UMASK)
FILE_MODE = 0o666 & ~_UMASK


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write ``text`` so readers see either the old file or the complete new one."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        os.fchmod(fd, FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def append_line(path: str | Path, line: str) -> Path:
    """Append one newline-terminated line and fsync it before returning."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(line if line.endswith("\n") else line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return target


__all__ = ["FILE_MODE", "append_line", "atomic_write_text"]
