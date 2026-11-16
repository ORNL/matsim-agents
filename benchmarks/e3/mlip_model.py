"""Frozen E3-MLIP model identity and checkpoint verification."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

E3_MLIP_MODEL = "medium-omat-0"
E3_MLIP_CHECKPOINT_RELATIVE_PATH = Path("mace/maceomat0mediummodel")
E3_MLIP_CHECKPOINT_SHA256 = "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a"


def _verified_checkpoint() -> tuple[Path, str]:
    artifact_dir = os.environ.get("MACE_ARTIFACT_DIR")
    if not artifact_dir:
        raise RuntimeError("MACE_ARTIFACT_DIR must identify the frozen E3-MLIP model cache")
    checkpoint_path = Path(artifact_dir) / E3_MLIP_CHECKPOINT_RELATIVE_PATH
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"frozen E3-MLIP checkpoint is missing: {checkpoint_path}")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != E3_MLIP_CHECKPOINT_SHA256:
        raise ValueError(
            "frozen E3-MLIP checkpoint hash mismatch: "
            f"expected {E3_MLIP_CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    return checkpoint_path.resolve(), actual_sha256


def _verified_checkpoint_sha256() -> str:
    return _verified_checkpoint()[1]
