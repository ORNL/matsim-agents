#!/usr/bin/env python3
"""Convert standardized prediction artifacts into a Codabench submission."""

from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path


def _validate_formation_energies(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"structure_id", "formation_energy_eV_per_atom"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is not a formation-energy file; missing {sorted(missing)}. "
                "Raw model total energies cannot be submitted as formation energies."
            )


def _zip_files(source: Path, output: Path, *, prefix: str = "") -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        return
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"{prefix}{path.relative_to(source).as_posix()}")


def package_submission(
    predictions: Path,
    output: Path,
    *,
    formation_energies: Path | None = None,
    task4_energies: Path | None = None,
) -> list[Path]:
    """Create task files from a prediction directory and return created paths."""
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    energy_path = formation_energies or predictions / "formation_energies.csv"
    if energy_path.is_file():
        _validate_formation_energies(energy_path)
        for name in ("task1.csv", "task5.csv"):
            target = output / name
            shutil.copy2(energy_path, target)
            created.append(target)

    for source_name, target_name in (("forces", "task2.zip"), ("relaxed", "task3.zip")):
        source = predictions / source_name
        if source.is_dir() and any(source.iterdir()):
            target = output / target_name
            _zip_files(source, target)
            created.append(target)

    task4_relaxed = predictions / "task4_relaxed"
    if task4_relaxed.is_dir() and any(task4_relaxed.iterdir()):
        target = output / "task4.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(task4_relaxed.rglob("*.xyz")):
                archive.write(path, f"task4_relaxed/{path.name}")
            energy4 = task4_energies or predictions / "task4_energies.csv"
            if energy4.is_file():
                _validate_formation_energies(energy4)
                archive.write(energy4, "task4_energies.csv")
        created.append(target)

    if not created:
        raise ValueError(f"No packageable predictions found beneath {predictions}")
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--formation-energies", type=Path)
    parser.add_argument("--task4-energies", type=Path)
    args = parser.parse_args()
    created = package_submission(
        args.predictions,
        args.output,
        formation_energies=args.formation_energies,
        task4_energies=args.task4_energies,
    )
    print("Created:")
    for path in created:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
