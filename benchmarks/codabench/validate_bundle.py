#!/usr/bin/env python3
"""Validate a Codabench competition source tree or assembled upload tree."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

REQUIRED_STATIC = (
    "competition.yaml",
    "logo.png",
    "pages/overview.md",
    "pages/evaluation.md",
    "pages/data.md",
    "pages/terms.md",
    "scoring_program/score.py",
    "scoring_program/metadata",
    "starting_kit/README.md",
    "starting_kit/MODEL_INTERFACE.md",
)
REQUIRED_PUBLIC = ("structures_metadata.csv", "structures")
REQUIRED_REFERENCE = (
    "formation_energies.csv",
    "forces",
    "relaxed",
    "structures_metadata.csv",
    "public_ids.txt",
    "private_ids.txt",
    "elemental_energies.json",
)
LEADERBOARD_KEYS = {
    "public_overall_score",
    "public_Task1_energy_MAE_eV_per_atom",
    "public_Task2_forces_MAE_eV_per_A",
    "public_Task3_relaxation_RMSD_A",
    "public_Task5_phase_spearman_rho",
}


def _has_files(path: Path) -> bool:
    return path.is_dir() and any(candidate.is_file() for candidate in path.rglob("*"))


def validate_bundle(root: Path, *, release: bool) -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_STATIC:
        if not (root / rel).exists():
            errors.append(f"missing static asset: {rel}")

    manifest_path = root / "competition.yaml"
    if manifest_path.is_file():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        keys = {
            column["key"]
            for board in manifest.get("leaderboards", [])
            for column in board.get("columns", [])
        }
        if missing := LEADERBOARD_KEYS - keys:
            errors.append(f"leaderboard is missing keys: {sorted(missing)}")
        for task in manifest.get("tasks", []):
            for field in ("input_data", "reference_data", "scoring_program"):
                rel = task.get(field)
                if not rel or not (root / rel).exists():
                    errors.append(f"task {task.get('index')} has missing {field}: {rel!r}")

    if release:
        for rel in REQUIRED_PUBLIC:
            if not (root / "public_data" / rel).exists():
                errors.append(f"missing public release asset: public_data/{rel}")
        for rel in REQUIRED_REFERENCE:
            if not (root / "reference_data" / rel).exists():
                errors.append(f"missing protected reference asset: reference_data/{rel}")

        for rel in ("public_data/structures", "reference_data/forces", "reference_data/relaxed"):
            if (path := root / rel).exists() and not _has_files(path):
                errors.append(f"release asset directory is empty: {rel}")

        metadata = root / "public_data" / "structures_metadata.csv"
        if metadata.is_file():
            with metadata.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                rel = row.get("file_path", "")
                if not rel or not (root / "public_data" / "structures" / rel).is_file():
                    errors.append(
                        f"missing public structure for {row.get('structure_id')}: {rel!r}"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).parent)
    parser.add_argument("--release", action="store_true", help="Require all upload data assets")
    args = parser.parse_args()
    errors = validate_bundle(args.root.resolve(), release=args.release)
    if errors:
        print("Codabench validation failed:", file=sys.stderr)
        print("\n".join(f"  - {error}" for error in errors), file=sys.stderr)
        return 1
    print("Codabench validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
