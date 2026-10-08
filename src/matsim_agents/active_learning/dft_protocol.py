"""Scientific DFT identities shared by label and elemental-reference workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from matsim_agents.active_learning.config import DFTConfig
from matsim_agents.active_learning.vasp_io import resolve_potcar_paths
from matsim_agents.backends.dft.qe_relax import resolve_pseudopotentials


def path_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"name": path.name, "sha256": digest.hexdigest()}
    if path.is_dir():
        digest = hashlib.sha256()
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            digest.update(hashlib.sha256(item.read_bytes()).digest())
        return {"name": path.name, "sha256": digest.hexdigest()}
    return {"name": path.name, "missing": True}


def scientific_dft_payload(cfg: DFTConfig, elements: set[str]) -> dict[str, Any]:
    if cfg.backend == "vasp":
        assert cfg.vasp is not None
        block = cfg.vasp
        return {
            "backend": "vasp",
            "executable": path_identity(block.vasp_bin),
            "incar_template": path_identity(block.incar_template),
            "kpoints_template": path_identity(block.kpoints_template),
            "potcars": {
                element: path_identity(path)
                for element, path in zip(
                    sorted(elements),
                    resolve_potcar_paths(sorted(elements), block.potcar_dir),
                    strict=True,
                )
            },
            "extra_incar": block.extra_incar,
        }
    assert cfg.qe is not None
    block = cfg.qe
    pseudopotentials = block.pseudopotentials or resolve_pseudopotentials(
        sorted(elements), str(block.pseudo_dir)
    )
    return {
        "backend": "qe",
        "executable": path_identity(block.pw_bin),
        "pseudopotential_files": {
            element: path_identity(block.pseudo_dir / filename)
            for element, filename in sorted(pseudopotentials.items())
            if element in elements
        },
        "pw_template": path_identity(block.pw_template),
        "ecutwfc_ry": block.ecutwfc_ry,
        "ecutrho_ry": block.ecutrho_ry,
        "kpts": block.kpts,
        "koffset": block.koffset,
        "occupations": block.occupations,
        "smearing": block.smearing,
        "degauss_ry": block.degauss_ry,
        "pseudopotentials": block.pseudopotentials,
        "extra_control": block.extra_control,
        "extra_system": block.extra_system,
        "extra_electrons": block.extra_electrons,
    }


def dft_method_signature(cfg: DFTConfig, elements: set[str]) -> str:
    payload = json.dumps(
        scientific_dft_payload(cfg, elements), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{cfg.backend}-{hashlib.sha256(payload).hexdigest()[:16]}"
