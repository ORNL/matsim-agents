#!/usr/bin/env python3
"""Compare the invariant metadata and optional numerical summaries of runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def compare_runs(paths: list[Path]) -> list[str]:
    records = [
        json.loads((path / "environment.json").read_text(encoding="utf-8")) for path in paths
    ]
    errors: list[str] = []
    hashes = {record["structure_sha256"] for record in records}
    commits = {record["git_commit"] for record in records}
    if len(hashes) != 1:
        errors.append("runs used different structure bytes")
    if len(commits) != 1:
        errors.append("runs used different git commits")
    manifest = json.loads((HERE / "manifest.yaml").read_text(encoding="utf-8"))
    tolerance = manifest["acceptance"]["cross_facility"]
    summaries = []
    for path in paths:
        summary = path / "scientific_summary.json"
        if not summary.is_file():
            errors.append(f"{path} is missing scientific_summary.json")
            continue
        summaries.append(json.loads(summary.read_text(encoding="utf-8")))
    qualifications = {summary.get("qualification") for summary in summaries}
    if len(qualifications) > 1:
        errors.append("cannot compare contract and compute qualification results")
    if summaries:
        metrics = (
            ("energy_eV_per_atom", "energy_eV_per_atom_atol"),
            ("max_force_eV_per_A", "max_force_eV_per_A_atol"),
        )
        for key, tol_key in metrics:
            values = [float(summary[key]) for summary in summaries if key in summary]
            if values and max(values) - min(values) > float(tolerance[tol_key]):
                errors.append(f"{key} range exceeds {tolerance[tol_key]}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("provide at least two run directories")
    errors = compare_runs(args.runs)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print("PASS: runs satisfy cross-facility invariants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
