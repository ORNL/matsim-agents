"""Qualified E3 reference evidence for LiFePO4 and BCC, recovered from E2/E4 archives.

Every candidate source is audited against the frozen E3 protocol (= E4 production
protocol). Only ionic frames whose method signature matches exactly and whose SCF
converged below NELM are admitted. Admitted frames keep their group identity so
correlated frames from one starting structure never straddle validation/test roles.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ase.io import read, write

ARCHIVE = Path("/lus/flare/projects/HydraGNN/mlupopa/AgenticAI4Materials")
AB = ARCHIVE / "e4-vasp-ab-pilot-inputs/Nb22Ta22V21Hf21Zr21Ti21/case-1"
NELM = 170

# Sources inspected and rejected, with the concrete incompatibility (verified 2026-11-08).
EXCLUDED_SOURCES = [
    {
        "material": "bcc_refractory",
        "source": str(AB / "alcf/repeat-0[12]/attempts/attempt-01"),
        "records": 261,
        "reason": "PREC=Low stage; the frozen reference stage is PREC=Normal",
    },
    {
        "material": "bcc_refractory",
        "source": str(AB / "custom/repeat-0[12]"),
        "records": 280,
        "reason": "custom VASP executable; frozen protocol pins ALCF VASP 6.6.1",
    },
    {
        "material": "bcc_refractory",
        "source": str(AB / "alcf/repeat-03"),
        "records": 4,
        "reason": "restart-qualification segments, PREC=Low only",
    },
    {
        "material": "bcc_refractory",
        "source": str(ARCHIVE / "e2-dft-v1-20261104/cases/bcc_hea_NbTaVHfZrTi_equiatomic_128_*"),
        "records": 4,
        "reason": "E2 signature ENCUT 345 eV, SIGMA 0.10, EDIFFG -0.02 (not E4 350/0.20/-0.01)",
    },
    {
        "material": "bcc_refractory",
        "source": "e4-first-batch-01 (running jobs 9052129, 9052147, 9052148)",
        "records": 0,
        "reason": "incomplete; only PREC=Low frames so far; also excluded to protect E4",
    },
    {
        "material": "lifepo4",
        "source": str(
            ARCHIVE / "matsim-agents-dft-reference/dft_reference/vasp/calculations/"
            "critical_minerals_LiFePO4_olivine_{ideal,Li_vacancy,FeLi_antisite}"
        ),
        "records": 3,
        "reason": (
            "same INCAR/PAW labels but VASP 6.6.0 (not ALCF 6.6.1) and unit-cell k-mesh "
            "4x7x8, not the 4x8x8 density of the frozen 2x4x8 supercell mesh"
        ),
    },
    {
        "material": "lifepo4",
        "source": "e4-first-batch-01 LiFePO4 cases",
        "records": 0,
        "reason": "no LiFePO4 calculation has executed (qsub rejected by per-user limit)",
    },
]


def _incar(path: Path) -> dict[str, str]:
    tags = {}
    for line in path.read_text().splitlines():
        line = re.split(r"[#!]", line, maxsplit=1)[0]
        if "=" in line:
            key, value = line.split("=", 1)
            tags[key.strip().upper()] = value.strip()
    return tags


def _scf_iterations(oszicar: Path) -> list[int]:
    steps, current = [], 0
    for line in oszicar.read_text().splitlines():
        if line.startswith(("DAV:", "RMM:")):
            current += 1
        elif re.match(r"^\s*\d+\s+F=", line):
            steps.append(current)
            current = 0
    return steps


def bcc_reference_frames(expected: dict[str, str]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Normal-stage ALCF frames from the E4 A/B pilot, checked tag by tag."""
    frames, audit = [], []
    for repeat in ("repeat-01", "repeat-02"):
        directory = AB / "alcf" / repeat / "attempts" / "attempt-02"
        tags = _incar(directory / "INCAR")
        mismatched = {k: tags.get(k) for k, v in expected.items() if tags.get(k) != v}
        outcar = (directory / "OUTCAR").read_text()
        version = re.search(r"vasp\.(\d+\.\d+\.\d+)", outcar).group(1)
        scf = _scf_iterations(directory / "OSZICAR")
        steps = read(directory / "OUTCAR", index=":", format="vasp-out")
        ok = not mismatched and version == "6.6.1" and len(steps) == len(scf)
        audit.append(
            {
                "source": str(directory),
                "vasp_version": version,
                "incar_mismatches": mismatched,
                "ionic_frames": len(steps),
                "max_scf_iterations": max(scf),
                "scf_unconverged_frames": sum(n >= NELM for n in scf),
                "admitted": ok,
            }
        )
        if not ok:
            continue
        for index, (atoms, n_scf) in enumerate(zip(steps, scf, strict=True)):
            if n_scf >= NELM:
                continue
            atoms.info.update(
                frame_id=f"bcc-ab-alcf-{repeat}-normal-{index:03d}",
                e3_group="ab-pilot:Nb22Ta22V21Hf21Zr21Ti21",
                trajectory_id=repeat,
                energy_eV=float(atoms.get_potential_energy(force_consistent=True)),
            )
            frames.append(atoms)
    return frames, audit


def write_references(frames: list[Any], path: Path) -> str:
    write(path, frames, format="extxyz")
    return hashlib.sha256(path.read_bytes()).hexdigest()
