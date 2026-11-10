"""Consolidate measured E4-B records only; validate consistency before plotting."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.geometry import find_mic


def consistency(reference: dict, candidate: dict, tolerances: dict) -> list[str]:
    failures = []
    original = {row["structure_id"]: row for row in reference["results"]}
    current = {row["structure_id"]: row for row in candidate["results"]}
    if original.keys() != current.keys():
        return ["Different structure IDs"]
    for identifier, a in original.items():
        b = current[identifier]
        if a["status"] != "ok" or b["status"] != "ok":
            failures.append(f"{identifier}: failed task")
            continue
        if a["converged"] != b["converged"]:
            failures.append(f"{identifier}: different convergence status")
        if abs(a["energy_eV"] - b["energy_eV"]) > tolerances["energy_eV"]:
            failures.append(f"{identifier}: energy mismatch")
        af, bf = np.asarray(a["forces_eV_A"]), np.asarray(b["forces_eV_A"])
        if (
            af.shape != bf.shape
            or not np.isfinite(bf).all()
            or not np.allclose(af, bf, atol=tolerances["force_eV_A"], rtol=0)
        ):
            failures.append(f"{identifier}: force mismatch")
        cell = np.asarray(a["cell_A"])
        if not np.allclose(cell, b["cell_A"], atol=1e-10, rtol=0):
            failures.append(f"{identifier}: cell mismatch")
        positions_a, positions_b = np.asarray(a["positions_A"]), np.asarray(b["positions_A"])
        if positions_a.shape != positions_b.shape or not np.isfinite(positions_b).all():
            failures.append(f"{identifier}: invalid final geometry")
        else:
            _, distances = find_mic(positions_b - positions_a, cell, pbc=True)
            if distances.max() > tolerances["position_A"]:
                failures.append(f"{identifier}: geometry mismatch")
    return failures


def consolidate(records: list[dict], tolerances: dict) -> list[dict]:
    baselines = {}
    keys = set()
    for record in records:
        key = (record["backend"], record["mode"], record["devices"], record["repetition"])
        if key in keys:
            raise ValueError(f"Duplicate scaling record: {key}")
        keys.add(key)
        if (
            not record["measurement_valid"]
            or len(record["results"]) != record["attempted"]
            or record["attempted"] != 120
            or len({row["structure_id"] for row in record["results"]}) != 120
        ):
            raise ValueError(f"Invalid scaling measurement: {key}")
        if not np.isfinite(record["steady_seconds"]) or record["steady_seconds"] <= 0:
            raise ValueError(f"Invalid duration: {key}")
        for row in record["results"]:
            if row["status"] != "ok" or not row["synchronized"]:
                raise ValueError(f"Failed or unsynchronized task: {key}")
            values = [row["energy_eV"], row["max_force_eV_A"]]
            if not np.isfinite(values).all() or not np.isfinite(row["forces_eV_A"]).all():
                raise ValueError(f"Nonfinite outputs: {key}")
        if record["devices"] == 1:
            baselines[(record["backend"], record["mode"], record["repetition"])] = record
    rows = []
    for record in records:
        baseline_key = (record["backend"], record["mode"], record["repetition"])
        if baseline_key not in baselines:
            raise ValueError(f"Missing one-device baseline: {baseline_key}")
        baseline = baselines[baseline_key]
        for field in (
            "manifest_sha256",
            "model_metadata_sha256",
            "precision",
            "fmax",
            "max_steps",
            "optimizer",
            "relax_cell",
        ):
            if baseline[field] != record[field]:
                raise ValueError(f"Changed fixed-workload setting: {field}")
        if failures := consistency(baseline, record, tolerances):
            raise ValueError(f"Numerical consistency failure: {failures}")
        speedup = baseline["steady_seconds"] / record["steady_seconds"]
        results = record["results"]
        completed = sum(row["converged"] is True for row in results)
        valid = len(results)
        rows.append(
            {
                "backend": record["backend"],
                "mode": record["mode"],
                "devices": record["devices"],
                "repetition": record["repetition"],
                "attempted": record["attempted"],
                "valid": valid,
                "failed": 0,
                "converged": completed,
                "convergence_rate": completed / valid if record["mode"] == "relaxation" else None,
                "throughput": (completed if record["mode"] == "relaxation" else valid)
                / record["steady_seconds"],
                "speedup": speedup,
                "efficiency": speedup / record["devices"],
                "steady_seconds": record["steady_seconds"],
                "total_seconds": record["total_seconds"],
                "cold_seconds": record["cold_phase_seconds"],
                "peak_device_memory_bytes": max(row["peak_allocated_bytes"] for row in results),
                "mean_steps": float(np.mean([row["steps"] for row in results])),
                "step_count_differences": sum(
                    {r["structure_id"]: r["steps"] for r in baseline["results"]}[
                        row["structure_id"]
                    ]
                    != row["steps"]
                    for row in results
                ),
                "mean_force_evaluations": float(
                    np.mean([row["force_evaluations"] for row in results])
                ),
            }
        )
    return rows


def plot(rows: list[dict], output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["backend"], row["mode"], row["devices"])].append(row)
    for metric, modes, name in (
        ("speedup", ["inference"], "E4-B1-inference-speedup"),
        ("throughput", ["relaxation"], "E4-B2-relaxation-throughput"),
        ("efficiency", ["inference", "relaxation"], "E4-B3-parallel-efficiency"),
    ):
        fig, ax = plt.subplots()
        for backend, mode in sorted({(r["backend"], r["mode"]) for r in rows}):
            if mode not in modes:
                continue
            devices = sorted(d for b, m, d in grouped if (b, m) == (backend, mode))
            values = [[r[metric] for r in grouped[(backend, mode, d)]] for d in devices]
            ax.errorbar(
                devices,
                [np.mean(v) for v in values],
                yerr=[np.std(v, ddof=1) if len(v) > 1 else 0 for v in values],
                marker="o",
                label=f"{backend} {mode}",
            )
        if metric == "speedup":
            ax.plot([1, 12], [1, 12], "--", label="ideal linear")
        elif metric == "efficiency":
            ax.axhline(1, linestyle="--", label="ideal")
        ax.set_xlabel("Logical XPU tiles (one Aurora node)")
        ax.set_ylabel(metric)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{name}.png")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="+")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    rows = consolidate(
        [json.loads(path.read_text()) for path in args.records], metadata["proposed_tolerances"]
    )
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "scaling.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "records": len(rows),
                "repetitions": {
                    f"{backend}/{mode}/{devices}": sum(
                        (row["backend"], row["mode"], row["devices"]) == (backend, mode, devices)
                        for row in rows
                    )
                    for backend, mode, devices in {
                        (row["backend"], row["mode"], row["devices"]) for row in rows
                    }
                },
                "variability": (
                    "Figures use sample standard deviation; missing repetitions are not fabricated"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    plot(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
