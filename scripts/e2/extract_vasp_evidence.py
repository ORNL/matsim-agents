"""Extract verifiable numbers from archived VASP calculations for E2 evidence packets."""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read

ARCHIVE = Path(sys.argv[1])
NAMES = sys.argv[2:]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def incar(path: Path) -> dict[str, str]:
    tags = {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.split("!")[0].split("#")[0]
        if "=" in line:
            key, value = line.split("=", 1)
            tags[key.strip().upper()] = value.strip()
    return tags


report = {}
for name in NAMES:
    calc = ARCHIVE / "dft_reference/vasp/calculations" / name
    outcar = calc / "OUTCAR"
    text = outcar.read_text(errors="replace") if outcar.is_file() else ""
    entry: dict = {"directory": str(calc.relative_to(ARCHIVE))}
    entry["files"] = sorted(p.name for p in calc.iterdir() if p.is_file())[:40]
    if text:
        entry["outcar_sha256"] = sha(outcar)
        entry["ionic_accuracy_reached"] = "reached required accuracy" in text
        entry["scf_ediff_reached_marker"] = "aborting loop because EDIFF is reached" in text
        entry["scf_unconverged_marker"] = "EDIFF was not reached (unconverged)" in text
        version = re.search(r"vasp\.(\d+\.\d+\.\d+)", text)
        entry["vasp_version"] = version.group(1) if version else None
        entry["potcars"] = sorted(set(re.findall(r"POTCAR:\s+(PAW_\S+ \S+ \S+)", text)))
        entry["ionic_steps"] = text.count("TOTAL-FORCE")
        try:
            frames = read(outcar, index=":")
            last = frames[-1]
            forces = last.get_forces()
            entry["final_energy_eV"] = float(last.get_potential_energy())
            entry["final_max_force_eV_A"] = float(np.linalg.norm(forces, axis=1).max())
            entry["natoms"] = len(last)
            entry["formula"] = last.get_chemical_formula()
            try:
                stress = last.get_stress(voigt=False) * 1602.1766  # eV/A^3 -> kbar
                entry["final_max_abs_principal_stress_kbar"] = float(
                    np.abs(np.linalg.eigvalsh((stress + stress.T) / 2)).max()
                )
            except Exception:
                entry["final_max_abs_principal_stress_kbar"] = None
            magmoms = re.findall(r"number of electron\s+\S+\s+magnetization\s+(\S+)", text)
            entry["final_total_magnetization"] = float(magmoms[-1]) if magmoms else None
        except Exception as exc:  # incomplete OUTCAR
            entry["parse_error"] = f"{type(exc).__name__}: {exc}"
    if (calc / "INCAR").is_file():
        tags = incar(calc / "INCAR")
        entry["incar"] = {
            k: tags[k]
            for k in (
                "ENCUT", "PREC", "EDIFF", "EDIFFG", "ISPIN", "MAGMOM", "LDAU", "LDAUU",
                "ISMEAR", "SIGMA", "IBRION", "ISIF", "NSW", "KSPACING", "IVDW", "GGA",
            )
            if k in tags
        }
    report[name] = entry
print(json.dumps(report, indent=1))
