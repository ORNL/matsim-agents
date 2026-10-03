"""Validation and immutable manifests for DFT-labelled active-learning data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
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
    method_signature: str | None = None
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


def structure_identity(atoms: Any) -> str:
    """Hash chemistry and geometry independently of atom ordering and translation."""
    import numpy as np

    numbers = np.asarray(atoms.numbers, dtype=int)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    scaled = np.asarray(atoms.get_scaled_positions(wrap=False))
    scaled[:, pbc] = np.mod(scaled[:, pbc], 1.0)
    metric = np.asarray(atoms.cell) @ np.asarray(atoms.cell).T
    origins = scaled if len(scaled) else np.zeros((1, 3))
    representations: list[str] = []
    for origin in origins:
        shifted = scaled - origin
        shifted[:, pbc] = np.mod(shifted[:, pbc], 1.0)
        sites = sorted(
            (int(number), *(round(float(value), 8) for value in position))
            for number, position in zip(numbers, shifted, strict=True)
        )
        payload = {
            "sites": sites,
            "metric": np.round(metric, decimals=8).tolist(),
            "pbc": pbc.tolist(),
        }
        representations.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256(min(representations).encode("utf-8")).hexdigest()


def validate_labelled_frames(
    frames: Iterable[Any],
    *,
    existing_frames: Iterable[Any] = (),
    expected_atomic_numbers: set[int] | None = None,
) -> tuple[list[Any], DatasetValidationSummary]:
    """Reject malformed/non-finite labels and exact duplicate geometries."""

    import numpy as np

    accepted: list[Any] = []
    summary = DatasetValidationSummary()
    seen = {
        structure_identity(getattr(existing, "atoms", existing)) for existing in existing_frames
    }
    for index, frame in enumerate(frames):
        try:
            atoms = getattr(frame, "atoms", frame)
            energy = float(getattr(frame, "energy_eV", atoms.info.get("energy")))
            raw_forces = getattr(frame, "forces_eV_per_A", atoms.arrays.get("forces"))
            forces = np.asarray(raw_forces, dtype=float)
            positions = np.asarray(atoms.positions, dtype=float)
            numbers = np.asarray(atoms.numbers, dtype=int)
            if np.any(numbers <= 0):
                raise ValueError("atomic numbers must be positive")
            if expected_atomic_numbers is not None and not set(numbers).issubset(
                expected_atomic_numbers
            ):
                raise ValueError("frame contains an atomic number outside the campaign element set")
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
            key = structure_identity(atoms)
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
    method_signature: str | None = None,
) -> Path:
    path = Path(dataset_path)
    digest = sha256_file(path)
    manifest = DatasetManifest(
        dataset_id=digest[:16],
        created_at_utc=datetime.now(UTC).isoformat(),
        path=str(path.resolve()),
        sha256=digest,
        dft_backend=dft_backend,
        method_signature=method_signature,
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
    "structure_identity",
    "validate_labelled_frames",
    "write_dataset_manifest",
]
