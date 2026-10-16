#!/usr/bin/env python3
"""Collect paper benchmark outputs into one machine-readable dataset.

Scans one or more run roots for:
- active-learning-uq-*/outputs/uq_summary.csv
- qe-warmstart-*/**/comparison.json
- vasp-warmstart-*/**/comparison.json (if present)

Writes:
- consolidated CSV (long format)
- consolidated JSON (same records)

Example:
  python scripts/diagnostics/collect_paper_results.py \
    --runs-root /global/cfs/projectdirs/m5216/mlupopa/runs \
    --runs-root /global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs \
    --output-csv research/paper/manuscript/results/paper_results_master.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, Iterable, List


DEFAULT_RUN_ROOTS = [
    "/global/cfs/projectdirs/m5216/mlupopa/runs",
    "/global/cfs/projectdirs/amsc001/cm2us/mlupopa/runs",
    "/global/cfs/cdirs/amsc001/cm2us/mlupopa/runs",
]


def _existing_roots(candidates: Iterable[str]) -> List[str]:
    return [os.path.abspath(p) for p in candidates if os.path.isdir(p)]


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def _collect_uq(root: str) -> List[Dict[str, object]]:
    records = []
    pattern = os.path.join(root, "active-learning-uq-*", "outputs", "uq_summary.csv")
    for uq_csv in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(os.path.dirname(uq_csv))
        run_name = os.path.basename(run_dir)
        job_id = run_name.rsplit("-", 1)[-1]
        for row in _read_csv_rows(uq_csv):
            records.append(
                {
                    "workflow": "uq",
                    "run_root": root,
                    "run_dir": run_dir,
                    "job_id": job_id,
                    "source_file": uq_csv,
                    "structure": row.get("structure"),
                    "optimized_structure": row.get("optimized_structure"),
                    "final_energy_eV": _safe_float(row.get("final_energy_eV")),
                    "final_max_force_eV_per_A": _safe_float(row.get("final_max_force_eV_per_A")),
                    "num_steps": int(row["num_steps"]) if row.get("num_steps") else None,
                    "converged": row.get("converged"),
                    "mean_top_weight": _safe_float(row.get("mean_top_weight")),
                    "mean_entropy": _safe_float(row.get("mean_entropy")),
                    "is_unreliable": row.get("is_unreliable"),
                    "reason": row.get("reason"),
                }
            )
    return records


def _collect_warmstart(root: str) -> List[Dict[str, object]]:
    records = []
    run_patterns = [
        os.path.join(root, "qe-warmstart-*"),
        os.path.join(root, "vasp-warmstart-*"),
    ]
    for run_pattern in run_patterns:
        for run_dir in sorted(glob.glob(run_pattern)):
            run_name = os.path.basename(run_dir)
            job_id = run_name.rsplit("-", 1)[-1]
            comparison_files = glob.glob(os.path.join(run_dir, "**", "comparison.json"), recursive=True)
            for comp_json in sorted(comparison_files):
                with open(comp_json, "r") as fh:
                    data = json.load(fh)
                cold = data.get("cold") or {}
                warm = data.get("warm") or {}
                hydragnn = data.get("hydragnn") or {}
                structure_path = data.get("structure_path")
                fixture = os.path.splitext(os.path.basename(structure_path or ""))[0]

                is_vasp = "n_ionic_steps" in cold or "scf_iterations_per_step" in cold and "bfgs_steps" not in cold
                workflow = "warmstart_vasp" if is_vasp else "warmstart_qe"

                records.append(
                    {
                        "workflow": workflow,
                        "run_root": root,
                        "run_dir": run_dir,
                        "job_id": job_id,
                        "source_file": comp_json,
                        "fixture": fixture,
                        "structure": structure_path,
                        "cold_converged": cold.get("converged"),
                        "warm_converged": warm.get("converged") if warm else None,
                        "cold_steps": cold.get("bfgs_steps", cold.get("n_ionic_steps")),
                        "warm_steps": warm.get("bfgs_steps", warm.get("n_ionic_steps")) if warm else None,
                        "cold_scf_total": cold.get("scf_iterations_total"),
                        "warm_scf_total": warm.get("scf_iterations_total") if warm else None,
                        "cold_wall_time_sec": cold.get("wall_time_sec"),
                        "warm_wall_time_sec": warm.get("wall_time_sec") if warm else None,
                        "delta_steps": data.get("delta_bfgs_steps", data.get("delta_ionic_steps")),
                        "delta_scf_iterations": data.get("delta_scf_iterations"),
                        "delta_energy_ev": data.get("delta_energy_ev"),
                        "speedup": data.get("speedup_bfgs", data.get("speedup_ionic")),
                        "warm_helped": data.get("warm_helped"),
                        "hydragnn_num_steps": hydragnn.get("num_steps") if hydragnn else None,
                        "hydragnn_final_energy_eV": hydragnn.get("final_energy_eV") if hydragnn else None,
                        "hydragnn_final_max_force_eV_per_A": hydragnn.get("final_max_force_eV_per_A") if hydragnn else None,
                    }
                )
    return records


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()}) if rows else ["workflow"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def _dedupe_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Drop duplicates caused by mirrored filesystem roots (cdirs/projectdirs)."""
    seen = set()
    unique = []
    for row in rows:
        source = row.get("source_file") or ""
        source_real = os.path.realpath(str(source)) if source else ""
        key = (
            row.get("workflow"),
            row.get("job_id"),
            row.get("case"),
            row.get("fixture"),
            row.get("structure"),
            source_real,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect manuscript benchmark outputs.")
    parser.add_argument(
        "--runs-root",
        action="append",
        default=[],
        help="Run root to scan (can be provided multiple times).",
    )
    parser.add_argument(
        "--output-csv",
        default="research/paper/manuscript/results/paper_results_master.csv",
        help="Path to consolidated CSV.",
    )
    parser.add_argument(
        "--output-json",
        default="research/paper/manuscript/results/paper_results_master.json",
        help="Path to consolidated JSON.",
    )
    args = parser.parse_args()

    roots = _existing_roots(args.runs_root or DEFAULT_RUN_ROOTS)
    if not roots:
        print("No existing run roots found. Provide at least one --runs-root.")
        return 2

    all_rows: List[Dict[str, object]] = []
    counts = {"uq": 0, "warmstart": 0}

    for root in roots:
        uq_rows = _collect_uq(root)
        ws_rows = _collect_warmstart(root)

        counts["uq"] += len(uq_rows)
        counts["warmstart"] += len(ws_rows)

        all_rows.extend(uq_rows)
        all_rows.extend(ws_rows)

    all_rows = _dedupe_rows(all_rows)

    counts = {
        "uq": sum(1 for r in all_rows if r.get("workflow") == "uq"),
        "warmstart": sum(1 for r in all_rows if str(r.get("workflow", "")).startswith("warmstart")),
    }

    _write_csv(args.output_csv, all_rows)
    _write_json(
        args.output_json,
        {
            "run_roots": roots,
            "counts": counts,
            "n_records": len(all_rows),
            "records": all_rows,
        },
    )

    print("Paper results collection complete.")
    print("Run roots:")
    for root in roots:
        print("  - {}".format(root))
    print("Records:")
    print("  - uq: {}".format(counts["uq"]))
    print("  - warmstart: {}".format(counts["warmstart"]))
    print("  - total: {}".format(len(all_rows)))
    print("CSV:  {}".format(os.path.abspath(args.output_csv)))
    print("JSON: {}".format(os.path.abspath(args.output_json)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
