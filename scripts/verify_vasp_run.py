#!/usr/bin/env python3
"""Qualify one VASP phase directory from raw OUTCAR/OSZICAR/INCAR, never from a summary file.

Usage: verify_vasp_run.py DIR --phase relax|static [--max-force F] [--max-stress S]
                          [--expect TAG=VALUE ...] [--expect-paw Nb_pv,Ta_pv,O] [-o OUT.json]
Exit status 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

SCF_LINE = re.compile(r"^(DAV|RMM|CG|SDA|DIA|EDD)\s*:")


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def scf_steps_per_ionic(oszicar: str) -> list[int]:
    counts, n = [], 0
    for line in oszicar.splitlines():
        if SCF_LINE.match(line.strip()):
            n += 1
        elif " F= " in line:
            counts.append(n)
            n = 0
    return counts


def last_stress_kbar(outcar: str) -> np.ndarray | None:
    rows = re.findall(r"^\s+in kB((?:\s+-?\d+\.\d+){6})", outcar, re.M)
    if not rows:
        return None
    xx, yy, zz, xy, yz, zx = map(float, rows[-1].split())
    return np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])


def last_max_force(outcar_path: Path) -> float | None:
    from ase.io import read

    try:
        return float(np.linalg.norm(read(outcar_path, index=-1, format="vasp-out").get_forces(), axis=1).max())
    except Exception:
        return None


def incar_tags(path: Path) -> dict[str, str]:
    tags = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#")[0].split("!")[0]
        for part in line.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                tags[k.strip().upper()] = v.strip()
    return tags


def same_value(actual: str | None, expected: str) -> bool:
    if actual is None:
        return False
    try:
        return abs(float(actual.replace("E", "e")) - float(expected.replace("E", "e"))) < 1e-9
    except ValueError:
        norm = lambda s: s.strip().strip(".").upper()
        return norm(actual) == norm(expected)


def verify(d: Path, phase: str, max_force: float | None, max_stress: float | None,
           expect: dict[str, str], expect_paw: list[str] | None) -> dict:
    outcar_p, oszicar_p, incar_p = d / "OUTCAR", d / "OSZICAR", d / "INCAR"
    checks: dict[str, dict] = {}

    def check(name: str, ok: bool, **detail) -> None:
        checks[name] = {"passed": bool(ok), **detail}

    if not outcar_p.is_file() or not oszicar_p.is_file() or not incar_p.is_file():
        check("files_present", False, missing=[p.name for p in (outcar_p, oszicar_p, incar_p) if not p.is_file()])
        return {"directory": str(d), "phase": phase, "qualified": False, "checks": checks}
    outcar = outcar_p.read_text(errors="replace")
    oszicar = oszicar_p.read_text(errors="replace")
    tags = incar_tags(incar_p)

    check("normal_termination", "General timing and accounting" in outcar)
    nelm_m = re.search(r"NELM\s*=\s*(\d+)", outcar)
    nelm = int(nelm_m.group(1)) if nelm_m else int(tags.get("NELM", 60))
    scf = scf_steps_per_ionic(oszicar)
    unconverged = [i + 1 for i, n in enumerate(scf) if n >= nelm]
    check("scf_converged_every_ionic_step", bool(scf) and not unconverged,
          ionic_steps=len(scf), nelm=nelm, max_scf_steps=max(scf) if scf else None,
          unconverged_ionic_steps=unconverged[:20])
    if phase == "relax":
        check("ionic_accuracy_marker", "reached required accuracy" in outcar)
    force = last_max_force(outcar_p)
    if max_force is not None:
        check("max_force", force is not None and force <= max_force, value_eV_per_A=force, criterion=max_force)
    stress = last_stress_kbar(outcar)
    principal = None if stress is None else float(np.abs(np.linalg.eigvalsh(stress)).max())
    if max_stress is not None:
        check("max_abs_principal_stress", principal is not None and principal <= max_stress,
              value_kbar=principal, criterion=max_stress)
    for key, want in expect.items():
        check(f"incar_{key}", same_value(tags.get(key), want), expected=want, actual=tags.get(key))
    titels = re.findall(r"TITEL\s*=\s*\S+\s+(\S+)", outcar)
    paw = list(dict.fromkeys(titels))
    if expect_paw is not None:
        check("paw_datasets", paw == expect_paw, expected=expect_paw, actual=paw)

    energies = re.findall(r"free  energy   TOTEN\s*=\s*(-?\d+\.\d+)", outcar)
    sigma0 = re.findall(r"energy\(sigma->0\)\s*=\s*(-?\d+\.\d+)", outcar)
    nions = int(re.search(r"NIONS\s*=\s*(\d+)", outcar).group(1))
    mags = re.findall(r"mag=\s*(-?\d+\.\d+)", oszicar)
    nkpts = re.search(r"Found\s+(\d+)\s+irreducible k-points", outcar)
    version = outcar.splitlines()[0].strip() if outcar else None
    return {
        "directory": str(d), "phase": phase,
        "qualified": all(c["passed"] for c in checks.values()),
        "checks": checks,
        "values": {
            "nions": nions,
            "final_free_energy_eV": float(energies[-1]) if energies else None,
            "final_energy_sigma0_eV": float(sigma0[-1]) if sigma0 else None,
            "energy_sigma0_per_atom_eV": float(sigma0[-1]) / nions if sigma0 else None,
            "max_force_eV_per_A": force,
            "stress_kbar": None if stress is None else stress.tolist(),
            "max_abs_principal_stress_kbar": principal,
            "final_total_magnetization_muB": float(mags[-1]) if mags else None,
            "irreducible_kpoints": int(nkpts.group(1)) if nkpts else None,
        },
        "method": {"vasp_header": version, "paw_titels": paw,
                   "incar": {k: tags[k] for k in sorted(tags)}},
        "provenance_sha256": {n: sha256(d / n) for n in ("INCAR", "KPOINTS", "POSCAR", "POTCAR", "CONTCAR", "OUTCAR", "vasprun.xml")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--phase", choices=("relax", "static"), required=True)
    ap.add_argument("--max-force", type=float)
    ap.add_argument("--max-stress", type=float)
    ap.add_argument("--expect", action="append", default=[], metavar="TAG=VALUE")
    ap.add_argument("--expect-paw", help="comma-separated PAW labels in POTCAR order")
    ap.add_argument("-o", "--output", type=Path)
    a = ap.parse_args()
    expect = dict(e.split("=", 1) for e in a.expect)
    result = verify(a.directory.resolve(), a.phase, a.max_force, a.max_stress,
                    {k.upper(): v for k, v in expect.items()},
                    a.expect_paw.split(",") if a.expect_paw else None)
    text = json.dumps(result, indent=2) + "\n"
    if a.output:
        a.output.write_text(text)
    failed = [k for k, c in result["checks"].items() if not c["passed"]]
    print(f"{a.directory}: {'QUALIFIED' if result['qualified'] else 'NOT QUALIFIED'}"
          + (f" (failed: {', '.join(failed)})" if failed else ""), file=sys.stderr)
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
