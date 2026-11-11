#!/usr/bin/env python3
"""Analyze E4 raw JSONL measurements; never synthesize missing observations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.e4.consistency import compare_consistency_records  # noqa: E402
from benchmarks.e4.planning import analyze_scaling  # noqa: E402


def _write_class_plots(rows: list[dict], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted({row["material_class"] for row in rows})
    written = []
    metrics = (
        ("median_makespan_seconds", "Strong makespan (s)", "strong"),
        ("speedup", "Strong speedup", "strong"),
        ("efficiency", "Strong efficiency", "strong"),
        ("median_valid_task_throughput_per_second", "Valid tasks/s/node", "weak"),
        ("weak_efficiency", "Weak efficiency", "weak"),
        ("allocated_node_hours", "Allocated node-hours", "all"),
    )
    for material_class in classes:
        class_rows = [row for row in rows if row["material_class"] == material_class]
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for axis, (key, title, experiment) in zip(axes.flat, metrics, strict=True):
            for kind in ("strong", "weak") if experiment == "all" else (experiment,):
                points = sorted(
                    (row for row in class_rows if row["experiment"] == kind),
                    key=lambda row: row["nodes"],
                )
                if not points:
                    continue
                x = [row["nodes"] for row in points]
                y = [row.get(key) for row in points]
                axis.plot(x, y, marker="o", label=kind)
            axis.set_title(title)
            axis.set_xlabel("Aurora nodes")
            axis.set_xscale("log", base=2)
            axis.grid(True, alpha=0.25)
            if axis.lines:
                axis.legend()
        figure.suptitle(f"E4 workflow scaling: {material_class}")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", material_class)
        target = output_dir / f"{safe_name}.png"
        figure.savefig(target, dpi=160)
        plt.close(figure)
        written.append(str(target))
    return written


def _manifest_status(manifest_path: Path, rows: list[dict]) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    actual = {(row["material_class"], row["experiment"], row["nodes"]): row for row in rows}
    status = []
    for entry in manifest["material_classes"]:
        material_class = entry["material_class"]
        for experiment, key in (("strong", "strong_scaling"), ("weak", "weak_scaling")):
            for tier in entry[key]["tiers"]:
                measured = actual.get((material_class, experiment, tier["nodes"]))
                if measured is None:
                    execution_status = "planned_unexecuted"
                elif measured["fully_valid_repetitions"] == measured["repetitions"]:
                    execution_status = "measured_complete"
                else:
                    execution_status = "measured_incomplete_or_failed"
                status.append(
                    {
                        "material_class": material_class,
                        "experiment": experiment,
                        "nodes": tier["nodes"],
                        "plan_status": tier["status"],
                        "queue_eligible": tier.get("queue_eligible"),
                        "execution_status": execution_status,
                        "measured_repetitions": measured["repetitions"] if measured else 0,
                        "completed_tasks": measured["valid_tasks"] if measured else 0,
                        "failed_tasks": measured["failed_tasks"] if measured else 0,
                    }
                )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--kind", choices=("scaling", "consistency"), default="scaling")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    records = [
        json.loads(line) for line in args.measurements.read_text().splitlines() if line.strip()
    ]
    rows = (
        analyze_scaling(records)
        if args.kind == "scaling"
        else [compare_consistency_records(records)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("no strong/weak scaling records found")
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.kind == "scaling":
        summary = {
            "actual_measurement_count": len(records),
            "aggregate_count": len(rows),
            "aggregates": rows,
            "planned_vs_measured": (
                _manifest_status(args.workload_manifest, rows) if args.workload_manifest else None
            ),
            "synthetic_values_generated": False,
        }
        args.output.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if not args.no_plots:
            plot_dir = args.output.parent / f"{args.output.stem}-plots"
            _write_class_plots(rows, plot_dir)
    print(f"wrote {len(rows)} aggregates from {len(records)} raw runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
