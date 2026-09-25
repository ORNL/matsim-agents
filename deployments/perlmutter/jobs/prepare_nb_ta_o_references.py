#!/usr/bin/env python3
"""Create transparent bootstrap structures for Nb-Ta-O hull references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ase.build import bulk, molecule
from ase.io import write


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend", choices=("qe", "vasp"), default="qe")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    structures = {
        "Nb": bulk("Nb", "bcc", a=3.300),
        "Ta": bulk("Ta", "bcc", a=3.306),
        "O2": molecule("O2"),
    }
    structures["O2"].set_cell([15.0, 15.0, 15.0])
    structures["O2"].center()
    structures["O2"].set_pbc(True)

    manifest: dict[str, dict[str, object]] = {}
    for formula, atoms in structures.items():
        path = (args.output_dir / f"{formula}.extxyz").resolve()
        write(path, atoms)
        manifest[formula] = {"path": str(path), "relax_cell": formula != "O2"}
    if args.backend == "qe":
        manifest["O2"]["settings"] = {
            "kpts": [1, 1, 1],
            "koffset": [0, 0, 0],
            "extra_system": {"nspin": 2, "starting_magnetization(1)": 1.0},
        }
    else:
        manifest["O2"]["settings"] = {
            "kspacing": 10.0,
            "kgamma": True,
            "ispin": 2,
            "extra_incar": {"MAGMOM": "2*1.0"},
        }
    manifest_path = args.output_dir / "reference_structures.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())