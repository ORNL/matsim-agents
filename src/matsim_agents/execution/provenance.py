"""Small, dependency-free helpers for auditable workflow handoffs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def append_jsonl_record(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one stable JSON record and return the resolved output path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return output
