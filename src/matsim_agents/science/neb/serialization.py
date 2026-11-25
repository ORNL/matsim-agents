"""Serialize NEB pathways: JSON metadata plus ASE-readable extended-XYZ geometries.

Layout of a pathway directory::

    pathway.json    metadata only: transition record, assessment, file hashes, fingerprints
    initial.extxyz  initial endpoint
    final.extxyz    final endpoint
    images.extxyz   all images, one frame each

Coordinates live only in the geometry files, never in the JSON. ASE's own extxyz writer keeps
8 decimals, which would break geometry fingerprints, so frames are written here with shortest
round-trip float text (``repr``); ``ase.io.read`` then returns bit-identical coordinates. The
files stay ordinary extxyz and can be read with ``ase.io.read(path, ":")``.

``pathway_id`` is a hash of the canonical metadata, so identical inputs give identical IDs.
``read_pathway`` verifies file hashes and geometry fingerprints and re-runs the validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from pydantic import BaseModel, ConfigDict

from matsim_agents.core.hashing import sha256_file as _sha256_file
from matsim_agents.science.defects.point_defects import geometry_fingerprint
from matsim_agents.science.neb.contracts import NebTransition
from matsim_agents.science.neb.images import DEFAULT_MAX_STEP, DEFAULT_MIN_DISTANCE
from matsim_agents.science.neb.validation import PathwayAssessment, assess_pathway

SCHEMA = "matsim-neb-pathway/1"
METADATA_FILE = "pathway.json"
_GEOMETRY_FILES = ("initial.extxyz", "final.extxyz", "images.extxyz")


class LoadedPathway(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    pathway_id: str
    transition: NebTransition
    initial: Atoms
    final: Atoms
    images: tuple[Atoms, ...]
    assessment: PathwayAssessment
    parameters: dict


def _frame(atoms: Atoms) -> str:
    if "atom_id" not in atoms.arrays:
        raise ValueError("structure has no atom_id array; stable identities are required")
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError("structure has non-finite coordinates or cell")
    lattice = " ".join(repr(float(x)) for x in atoms.cell.array.reshape(-1))
    pbc = " ".join("T" if p else "F" for p in atoms.pbc)
    lines = [
        str(len(atoms)),
        f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:atom_id:I:1 pbc="{pbc}"',
    ]
    for symbol, position, atom_id in zip(
        atoms.get_chemical_symbols(), atoms.positions, atoms.get_array("atom_id"), strict=True
    ):
        coords = " ".join(repr(float(x)) for x in position)
        lines.append(f"{symbol} {coords} {int(atom_id)}")
    return "\n".join(lines) + "\n"


def _write_frames(path: Path, frames: Sequence[Atoms]) -> None:
    path.write_text("".join(_frame(a) for a in frames), encoding="utf-8")


def _canonical(data: dict) -> str:
    return json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _read_frames(path: Path) -> list[Atoms]:
    frames = read(path, index=":", format="extxyz")
    return frames if isinstance(frames, list) else [frames]


def _assessment_parameters(min_distance: float, max_step: float) -> dict:
    return {"min_distance": float(min_distance), "max_step": float(max_step)}


def write_pathway(
    directory: str | os.PathLike,
    transition: NebTransition,
    initial: Atoms,
    final: Atoms,
    images: Sequence[Atoms],
    *,
    parameters: dict | None = None,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    max_step: float = DEFAULT_MAX_STEP,
) -> str:
    """Write a pathway directory atomically and return its ``pathway_id``.

    The validation outcome is computed here and stored; a rejected pathway is still written so
    the reason survives. Refuses to overwrite an existing pathway.
    """
    target = Path(directory)
    if (target / METADATA_FILE).exists():
        raise FileExistsError(f"{target} already contains a pathway; refusing to overwrite")
    images = list(images)
    assessment = assess_pathway(
        transition, initial, final, images, min_distance=min_distance, max_step=max_step
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pathway-", dir=target.parent))
    try:
        _write_frames(staging / "initial.extxyz", [initial])
        _write_frames(staging / "final.extxyz", [final])
        _write_frames(staging / "images.extxyz", images)
        body = {
            "schema": SCHEMA,
            "transition": transition.model_dump(mode="json"),
            "assessment": assessment.model_dump(mode="json"),
            "assessment_parameters": _assessment_parameters(min_distance, max_step),
            "parameters": parameters or {},
            "files": {
                "initial.extxyz": {
                    "sha256": _sha256_file(staging / "initial.extxyz"),
                    "fingerprints": [geometry_fingerprint(initial)],
                },
                "final.extxyz": {
                    "sha256": _sha256_file(staging / "final.extxyz"),
                    "fingerprints": [geometry_fingerprint(final)],
                },
                "images.extxyz": {
                    "sha256": _sha256_file(staging / "images.extxyz"),
                    "fingerprints": [geometry_fingerprint(a) for a in images],
                },
            },
        }
        pathway_id = hashlib.sha256(_canonical(body).encode()).hexdigest()[:16]
        (staging / METADATA_FILE).write_text(
            _canonical({**body, "pathway_id": pathway_id}), encoding="utf-8"
        )
        target.mkdir(parents=True, exist_ok=True)
        for name in (*_GEOMETRY_FILES, METADATA_FILE):  # metadata last: it marks completion
            os.replace(staging / name, target / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return pathway_id


def read_pathway(directory: str | os.PathLike) -> LoadedPathway:
    """Load and verify a pathway directory; raise ``ValueError`` on any inconsistency."""
    source = Path(directory)
    try:
        meta = json.loads((source / METADATA_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{source} has no {METADATA_FILE}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{METADATA_FILE} is not valid JSON: {exc}") from exc
    if meta.get("schema") != SCHEMA:
        raise ValueError(f"unsupported pathway schema {meta.get('schema')!r}")
    claimed = meta.get("pathway_id")
    body = {k: v for k, v in meta.items() if k != "pathway_id"}
    if hashlib.sha256(_canonical(body).encode()).hexdigest()[:16] != claimed:
        raise ValueError("pathway metadata does not match its pathway_id")

    frames: dict[str, list[Atoms]] = {}
    for name in _GEOMETRY_FILES:
        entry = meta["files"][name]
        path = source / name
        if not path.is_file():
            raise ValueError(f"missing geometry file {name}")
        if _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{name} does not match its recorded hash")
        loaded = _read_frames(path)
        if [geometry_fingerprint(a) for a in loaded] != entry["fingerprints"]:
            raise ValueError(f"{name} geometry does not match its recorded fingerprints")
        frames[name] = loaded
    if len(frames["initial.extxyz"]) != 1 or len(frames["final.extxyz"]) != 1:
        raise ValueError("endpoint files must hold exactly one frame")

    transition = NebTransition.model_validate(meta["transition"])
    initial, final = frames["initial.extxyz"][0], frames["final.extxyz"][0]
    images = tuple(frames["images.extxyz"])
    settings = meta["assessment_parameters"]
    assessment = assess_pathway(
        transition,
        initial,
        final,
        list(images),
        min_distance=settings["min_distance"],
        max_step=settings["max_step"],
    )
    if assessment.model_dump(mode="json") != meta["assessment"]:
        raise ValueError("stored validation outcome differs from a fresh validation")
    return LoadedPathway(
        pathway_id=claimed,
        transition=transition,
        initial=initial,
        final=final,
        images=images,
        assessment=assessment,
        parameters=meta["parameters"],
    )


__all__ = ["LoadedPathway", "METADATA_FILE", "SCHEMA", "read_pathway", "write_pathway"]
