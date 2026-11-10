"""Inventory local structure files without importing a model or changing source archives."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read

from benchmarks.e4_mlip.run import geometry_hash, sha256

log = logging.getLogger(__name__)
BCC = {"Nb", "Ta", "V", "Hf", "Zr", "Ti"}
FCC = {"Cr", "Mn", "Fe", "Co", "Ni"}


def family_for(composition: dict[str, int]) -> str | None:
    species = set(composition)
    if len(species) >= 2 and species <= BCC:
        return "bcc_refractory"
    if species == FCC:
        return "fcc_hea"
    if species == {"Li", "Fe", "P", "O"}:
        return "lifepo4"
    return None


def inventory(roots: list[Path]) -> dict:
    records, errors = [], []
    seen = set()
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        candidates = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and (
                path.name in {"POSCAR", "CONTCAR"}
                or path.suffix.lower() in {".poscar", ".contcar", ".vasp", ".xyz", ".extxyz"}
            )
        )
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                atoms = read(path, format="vasp" if "CAR" in path.name else None)
                composition = dict(Counter(atoms.get_chemical_symbols()))
                family = family_for(composition)
                if family is None:
                    continue
                valid_geometry = (
                    bool(atoms.pbc.all())
                    and atoms.get_volume() > 0
                    and len(atoms) > 0
                    and bool(np.isfinite(atoms.positions).all())
                    and bool(np.isfinite(atoms.cell.array).all())
                )
                expected = 112 if family == "lifepo4" else 128
                records.append(
                    {
                        "path": str(resolved),
                        "root": str(root.resolve()),
                        "family": family,
                        "composition": composition,
                        "atom_count": len(atoms),
                        "sha256": sha256(path),
                        "geometry_sha256": geometry_hash(atoms),
                        "initial_status": (
                            "intermediate" if "CONTCAR" in path.name.upper() else "unknown"
                        ),
                        "size_and_periodicity_suitable": valid_geometry and len(atoms) == expected,
                        "qualification": (
                            "chemistry filter only; lattice and checkpoint coverage unverified"
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - record each unreadable archive explicitly
                log.warning("Cannot inventory %s: %s", path, exc)
                errors.append({"path": str(path), "error": str(exc)})
    summaries = []
    for root in roots:
        for family in ("bcc_refractory", "fcc_hea", "lifepo4"):
            selected = [
                row
                for row in records
                if row["root"] == str(root.resolve()) and row["family"] == family
            ]
            summaries.append(
                {
                    "root": str(root.resolve()),
                    "family": family,
                    "files": len(selected),
                    "unique_geometry_hashes": len({row["geometry_sha256"] for row in selected}),
                    "atom_counts": dict(Counter(str(row["atom_count"]) for row in selected)),
                    "size_suitable_unique": len(
                        {
                            row["geometry_sha256"]
                            for row in selected
                            if row["size_and_periodicity_suitable"]
                        }
                    ),
                }
            )
    return {
        "scanned_files": len(seen),
        "summaries": summaries,
        "records": records,
        "errors": errors,
        "note": (
            "Hashes preserve site ordering; not a symmetry-canonical duplicate proof. "
            "CONTCAR does not prove convergence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = inventory(args.root)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summaries": report["summaries"], "errors": len(report["errors"])}, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
