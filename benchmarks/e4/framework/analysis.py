"""Tables and figures from E4 framework measurements; never synthesizes observations.

Scaling aggregates reuse ``benchmarks.e4.planning.analyze_scaling``; this module adds
per-run rows (resumed segments combined) and the E4.3 mixed-vs-homogeneous table.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from benchmarks.e4.planning import analyze_scaling

RUN_COLUMNS = (
    "run_id", "facility", "experiment", "policy", "material_class", "nodes", "repetition",
    "segments", "timing_complete", "planned_tasks", "attempted_tasks", "executed_tasks", "converged_tasks",
    "valid_tasks", "failed_tasks", "retried_tasks", "total_attempts", "failed_attempts",
    "interrupted_attempts", "application_makespan_seconds", "allocation_seconds",
    "completed_tasks_per_second", "qualified_results_per_second",
    "qualified_results_per_node_second", "framework_initialization_seconds",
    "framework_finalization_seconds", "placement_seconds", "checkpoint_provenance_seconds",
    "median_queue_wait_seconds", "median_launch_overhead_seconds",
    "p95_launch_overhead_seconds", "recovery_seconds", "node_hours", "logical_device_hours",
    "physical_gpu_hours", "busy_device_hours", "idle_device_hours", "idle_dft_node_hours",
    "utilization_device_weighted", "utilization_dft_node_weighted", "failures_by_class",
)  # fmt: skip
_SUMMED = (
    "application_makespan_seconds", "allocation_seconds", "framework_initialization_seconds",
    "framework_finalization_seconds", "placement_seconds", "checkpoint_provenance_seconds",
    "recovery_seconds", "node_hours", "logical_device_hours", "physical_gpu_hours",
    "busy_device_hours", "idle_device_hours", "idle_dft_node_hours",
)  # fmt: skip


def load_measurements(inputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for raw in map(Path, inputs):
        files.extend([raw] if raw.is_file() else sorted(raw.rglob("measurements.jsonl")))
    records = [
        json.loads(line)
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in records if r.get("benchmark") == "e4-framework"]


def combine_segments(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per run: cumulative counts from the last segment, times summed.

    A segment killed before it wrote its measurement leaves a gap in the segment
    numbers; such runs keep their counts but get no summed timing, so they cannot
    report an inflated throughput.
    """

    by_run: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_run.setdefault(record["run_id"], []).append(record)
    rows = []
    for segments in by_run.values():
        segments.sort(key=lambda r: r.get("segment", 1))
        row = dict(segments[-1], segments=len(segments))
        numbers = [int(s.get("segment", 1)) for s in segments]
        row["timing_complete"] = numbers == list(range(1, len(numbers) + 1))
        for field in _SUMMED:
            values = [s[field] for s in segments if s.get(field) is not None]
            row[field] = sum(values) if values and row["timing_complete"] else None
        row["task_execution_seconds"] = [
            x for s in segments for x in s.get("task_execution_seconds", [])
        ]
        span = row["application_makespan_seconds"]
        row["completed_tasks_per_second"] = row["executed_tasks"] / span if span else None
        row["qualified_results_per_second"] = row["valid_tasks"] / span if span else None
        row["qualified_results_per_node_second"] = (
            row["valid_tasks"] / span / row["nodes"] if span else None
        )
        rows.append(row)
    return sorted(rows, key=lambda r: r["run_id"])


def heterogeneous_comparison(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """E4.3: each mixed fixed-workload run against the measured single-class runs.

    The baseline is the sum of single-class makespans on the same node count,
    i.e. running the classes one after another.
    """

    fixed = [r for r in rows if r["experiment"] == "fixed"]
    output = []
    for mixed in fixed:
        classes = mixed.get("task_classes") or []
        if len(classes) < 2:
            continue
        key = (mixed["facility"], mixed["nodes"], mixed["repetition"])
        single = {
            r["task_classes"][0]: r
            for r in fixed
            if len(r.get("task_classes") or []) == 1
            and (r["facility"], r["nodes"], r["repetition"]) == key
            and r["task_classes"][0] in classes
        }
        serial = (
            sum(r["application_makespan_seconds"] for r in single.values())
            if set(single) == set(classes)
            else None
        )
        output.append(
            {
                "run_id": mixed["run_id"],
                "facility": mixed["facility"],
                "nodes": mixed["nodes"],
                "repetition": mixed["repetition"],
                "policy": mixed.get("policy"),
                "task_classes": "+".join(classes),
                "makespan_seconds": mixed["application_makespan_seconds"],
                "homogeneous_runs_found": "+".join(sorted(single)),
                "homogeneous_serial_makespan_seconds": serial,
                "makespan_ratio_vs_homogeneous_serial": (
                    mixed["application_makespan_seconds"] / serial if serial else None
                ),
                **{
                    field: mixed.get(field)
                    for field in (
                        "utilization_device_weighted",
                        "utilization_dft_node_weighted",
                        "idle_device_hours",
                        "idle_dft_node_hours",
                        "median_queue_wait_seconds",
                        "failed_attempts",
                        "retried_tasks",
                        "recovery_seconds",
                        "valid_tasks",
                        "planned_tasks",
                    )
                },
            }
        )
    return output


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns or rows[0]), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, dict | list) else v
                    for k, v in row.items()
                }
            )


def _plots(scaling: Sequence[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("weak", "median_valid_task_throughput_per_second", "Qualified results/s"),
        ("weak", "weak_efficiency", "Weak efficiency"),
        ("weak", "allocated_node_hours", "Node-hours"),
        ("strong", "median_makespan_seconds", "Makespan (s)"),
        ("strong", "speedup", "Strong speedup"),
        ("strong", "efficiency", "Strong efficiency"),
    )
    written = []
    for label in sorted({r["material_class"] for r in scaling}):
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for axis, (experiment, metric, title) in zip(axes.flat, panels, strict=True):
            points = [
                r
                for r in scaling
                if r["material_class"] == label
                and r["experiment"] == experiment
                and r.get(metric) is not None
            ]
            if points:
                for site in sorted({p.get("facility", "unknown") for p in points}):
                    series = [p for p in points if p.get("facility", "unknown") == site]
                    axis.plot(
                        [p["nodes"] for p in series], [p[metric] for p in series], "o-", label=site
                    )
                axis.set_xscale("log", base=2)
                axis.legend()
            else:
                axis.text(0.5, 0.5, "no measured points", ha="center", transform=axis.transAxes)
            axis.set(title=f"{experiment}: {title}", xlabel="nodes")
        figure.suptitle(label)
        path = output / f"scaling-{label}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(str(path))
    return written


def analyze_runs(
    inputs: Iterable[str | Path], output: str | Path, *, plots: bool = True
) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = load_measurements(inputs)
    if not records:
        raise SystemExit("no e4-framework measurements found")
    rows = combine_segments(records)
    scaling = analyze_scaling(
        [r for r in rows if r["experiment"] in {"weak", "strong"} and r["timing_complete"]]
    )
    hetero = heterogeneous_comparison(rows)
    outputs: dict[str, Any] = {"runs": str(output / "runs.csv")}
    _write_csv(output / "runs.csv", rows, RUN_COLUMNS)
    for name, table in (("scaling", scaling), ("heterogeneous", hetero)):
        if table:
            _write_csv(output / f"{name}.csv", table)
            outputs[name] = str(output / f"{name}.csv")
    if plots and scaling:
        outputs["figures"] = _plots(scaling, output)
    report = {
        "raw_measurements": len(records),
        "runs": len(rows),
        "resumed_runs": sum(1 for r in rows if r["segments"] > 1),
        "incomplete_runs": [r["run_id"] for r in rows if r["valid_tasks"] != r["planned_tasks"]],
        "missing_segment_timing_runs": [r["run_id"] for r in rows if not r["timing_complete"]],
        "synthetic_values_generated": False,
        "outputs": outputs,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


__all__ = ["analyze_runs", "combine_segments", "heterogeneous_comparison", "load_measurements"]
