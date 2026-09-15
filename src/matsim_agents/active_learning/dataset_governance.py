"""Validation and immutable manifests for DFT-labelled active-learning data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DatasetValidationSummary(BaseModel):
    accepted: int = 0
    rejected: int = 0
    duplicate: int = 0
    rejection_reasons: list[str] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    dataset_id: str
    created_at_utc: str
    path: str
    sha256: str
    dft_backend: str
    energy_reference: str
    parent_dataset_id: str | None = None
    split_role: str = "training_pool"
    validation: DatasetValidationSummary


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_labelled_frames(frames: Iterable[Any]) -> tuple[list[Any], DatasetValidationSummary]:
    """Reject malformed/non-finite labels and exact duplicate geometries."""

    import numpy as np

    accepted: list[Any] = []
    summary = DatasetValidationSummary()
    seen: set[str] = set()
    for index, frame in enumerate(frames):
        try:
            atoms = getattr(frame, "atoms", frame)
            energy = float(getattr(frame, "energy_eV", atoms.info.get("energy")))
            raw_forces = getattr(frame, "forces_eV_per_A", atoms.arrays.get("forces"))
            forces = np.asarray(raw_forces, dtype=float)
            positions = np.asarray(atoms.positions, dtype=float)
            if forces.shape != positions.shape:
                raise ValueError(
                    f"forces shape {forces.shape} != positions shape {positions.shape}"
                )
            if (
                not np.isfinite(energy)
                or not np.isfinite(forces).all()
                or not np.isfinite(positions).all()
            ):
                raise ValueError("energy, forces, and positions must be finite")
            key_data = {
                "numbers": atoms.numbers.tolist(),
                "positions": np.round(positions, decimals=8).tolist(),
                "cell": np.round(np.asarray(atoms.cell), decimals=8).tolist(),
                "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
            }
            key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            if key in seen:
                summary.duplicate += 1
                continue
            seen.add(key)
            accepted.append(frame)
        except Exception as exc:  # noqa: BLE001 - record malformed external labels
            summary.rejected += 1
            summary.rejection_reasons.append(f"frame {index}: {exc}")
    summary.accepted = len(accepted)
    return accepted, summary


def write_dataset_manifest(
    dataset_path: str | Path,
    *,
    dft_backend: str,
    energy_reference: str,
    validation: DatasetValidationSummary,
    parent_dataset_id: str | None = None,
) -> Path:
    path = Path(dataset_path)
    digest = sha256_file(path)
    manifest = DatasetManifest(
        dataset_id=digest[:16],
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        path=str(path.resolve()),
        sha256=digest,
        dft_backend=dft_backend,
        energy_reference=energy_reference,
        parent_dataset_id=parent_dataset_id,
        validation=validation,
    )
    destination = path.with_suffix(path.suffix + ".manifest.json")
    destination.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return destination


__all__ = [
    "DatasetManifest",
    "DatasetValidationSummary",
    "sha256_file",
    "validate_labelled_frames",
    "write_dataset_manifest",
]
