"""Done-work records that let an interrupted run skip what it already finished."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from matsim_agents.core.fileio import append_line

log = logging.getLogger(__name__)


class KeyedResultLog:
    """Append-only JSONL of ``{"key", "result"}`` entries; the key says what the result is for.

    A caller puts everything that would invalidate a result (inputs, method settings) into the
    key, so a changed configuration simply misses and the work reruns.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        encode: Callable[[Any], str] = json.dumps,
        decode: Callable[[str], Any] = json.loads,
    ) -> None:
        self.path = Path(path)
        self._encode = encode
        self._decode = decode

    def load(self, revive: Callable[[Any], Any] | None = None) -> dict[str, Any]:
        """Return ``key -> result``; a later entry replaces an earlier one with the same key."""
        results: dict[str, Any] = {}
        if not self.path.is_file():
            return results
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                entry = self._decode(line)
                result = entry["result"]
                results[entry["key"]] = revive(result) if revive else result
            except Exception as exc:  # noqa: BLE001
                # A write cut short by the interruption; that work simply reruns.
                log.warning("Ignoring unreadable entry in %s: %s", self.path, exc)
        return results

    def save(self, key: str, result: Any) -> None:
        """Durably append one finished result."""
        line = self._encode({"key": key, "result": result})
        append_line(self.path, line.replace("\n", " "))


__all__ = ["KeyedResultLog"]
