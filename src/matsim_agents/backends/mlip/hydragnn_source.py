"""Tie a HydraGNN checkpoint to the HydraGNN source commit it loads with.

A logdir carries ``hydragnn-source.json`` next to its checkpoint. The imported
``hydragnn`` must be a clean git checkout at exactly that commit; whatever else
happens to be on ``PYTHONPATH`` is refused, never silently used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

RECORD_NAME = "hydragnn-source.json"
REGISTRY_PATH = Path(__file__).with_name("hydragnn_sources.json")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def imported_hydragnn_root() -> Path:
    """Directory containing the ``hydragnn`` package that this process would import."""
    spec = importlib.util.find_spec("hydragnn")
    if spec is None or not spec.origin:
        raise ModuleNotFoundError("hydragnn is not importable")
    return Path(spec.origin).resolve().parent.parent


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def _registry_record(logdir: Path) -> dict | None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    for checkpoint in sorted(logdir.glob("*.pk")):
        record = registry.get(_sha256(checkpoint))
        if record is not None:
            return record
    return None


def read_hydragnn_source_record(logdir: str | Path) -> dict:
    """The logdir's own record, else the registry entry for one of its checkpoints."""
    logdir = Path(logdir)
    path = logdir / RECORD_NAME
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
    else:
        record = _registry_record(logdir)
        if record is None:
            raise FileNotFoundError(
                f"{path} is missing and no checkpoint in {logdir} is in {REGISTRY_PATH.name}: "
                'record the HydraGNN commit it loads with {"commit": "<40-hex sha>", '
                '"checkout": "<directory name>"} before using it'
            )
    if not _COMMIT.fullmatch(str(record.get("commit", ""))):
        raise ValueError(f"{logdir}: 'commit' must be a full 40-character lowercase git sha")
    return record


def verify_hydragnn_source(logdir: str | Path) -> dict:
    """Refuse unless the imported HydraGNN is the clean checkout the logdir records."""
    record = read_hydragnn_source_record(logdir)
    expected = record["commit"]
    checkout = record.get("checkout", "the recorded checkout")
    root = imported_hydragnn_root()
    hint = (
        f"Use {checkout} (HYDRAGNN_ROOT={os.environ.get('HYDRAGNN_ROOT', '<unset>')}); "
        "source deployments/common/hydragnn-env.sh to put it first on PYTHONPATH."
    )
    try:
        actual = _git(root, "rev-parse", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError(
            f"{Path(logdir) / RECORD_NAME} requires HydraGNN {expected}, but the imported "
            f"hydragnn at {root} is not a readable git checkout ({exc}). {hint}"
        ) from exc
    if actual != expected:
        raise ValueError(
            f"{Path(logdir) / RECORD_NAME} requires HydraGNN {expected}, but the imported "
            f"hydragnn at {root} is {actual}. {hint}"
        )
    if dirty:
        raise ValueError(
            f"HydraGNN at {root} has uncommitted changes to tracked files, so it is not "
            f"commit {expected}. {hint}"
        )
    _require_example_scripts_from(root, hint)
    return record


def _require_example_scripts_from(root: Path, hint: str) -> None:
    """The example scripts are loaded by name, so a logdir's sibling copy could shadow them."""
    try:
        spec = importlib.util.find_spec("inference_random_structures")
    except (ImportError, ValueError):
        return
    if spec is None or not spec.origin:
        return
    origin = Path(spec.origin).resolve()
    if not origin.is_relative_to(root):
        raise ValueError(
            f"inference_random_structures is imported from {origin}, outside the verified "
            f"HydraGNN checkout {root}. {hint}"
        )


def write_hydragnn_source_record(logdir: str | Path, record: dict) -> Path:
    path = Path(logdir) / RECORD_NAME
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
