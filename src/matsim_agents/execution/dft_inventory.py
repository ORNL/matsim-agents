"""Read-only inventory of retained DFT workflow results and measured costs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

_QE_TIMER = re.compile(r"PWSCF\s*:\s*(.*?)\s+CPU\s+(.*?)\s+WALL")
_DURATION_PART = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([hms])")
_VASP_WALL = re.compile(r"Elapsed time\s*\(sec\):\s*([0-9.]+)")


def _duration_seconds(value: str) -> float | None:
    parts = _DURATION_PART.findall(value)
    if not parts:
        return None
    factors = {"h": 3600.0, "m": 60.0, "s": 1.0}
    return sum(float(amount) * factors[unit] for amount, unit in parts)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _parse_application_timers(work_dir: Path, backend: str) -> dict[str, float | None]:
    if backend == "qe":
        output = work_dir / "pw.out"
        try:
            text = output.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {"application_cpu_seconds": None, "application_wall_seconds": None}
        matches = list(_QE_TIMER.finditer(text))
        if not matches:
            return {"application_cpu_seconds": None, "application_wall_seconds": None}
        match = matches[-1]
        return {
            "application_cpu_seconds": _duration_seconds(match.group(1)),
            "application_wall_seconds": _duration_seconds(match.group(2)),
        }

    if backend == "vasp":
        output = work_dir / "OUTCAR"
        try:
            text = output.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {"application_cpu_seconds": None, "application_wall_seconds": None}
        matches = _VASP_WALL.findall(text)
        return {
            "application_cpu_seconds": None,
            "application_wall_seconds": float(matches[-1]) if matches else None,
        }

    return {"application_cpu_seconds": None, "application_wall_seconds": None}


def _load_accounting(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"run_directory", "elapsed_seconds", "nodes", "gpus"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "accounting CSV requires columns: run_directory, elapsed_seconds, nodes, gpus"
            )
        for row in reader:
            run_directory = str(Path(row["run_directory"]).resolve())
            records[run_directory] = row
    return records


def inventory_dft_runs(
    root: str | Path, accounting_csv: str | Path | None = None
) -> dict[str, Any]:
    """Inventory ``results.json`` records under root without invoking a scheduler."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"inventory root is not a directory: {root_path}")
    accounting = _load_accounting(Path(accounting_csv).expanduser() if accounting_csv else None)
    runs: list[dict[str, Any]] = []
    for result_path in sorted(root_path.rglob("results.json")):
        result = _read_json(result_path)
        if result is None:
            continue
        stages = result.get("stages")
        if not isinstance(stages, list):
            continue
        dft_stages = [
            stage
            for stage in stages
            if isinstance(stage, dict) and stage.get("backend") in {"qe", "vasp"}
        ]
        if not dft_stages:
            continue

        run_directory = Path(str(result.get("run_directory") or result_path.parent)).expanduser()
        if not run_directory.is_absolute():
            run_directory = result_path.parent / run_directory
        run_directory = run_directory.resolve()
        stage_records = []
        for stage in dft_stages:
            backend = str(stage["backend"])
            artifacts = stage.get("artifacts") if isinstance(stage.get("artifacts"), dict) else {}
            work_dir = Path(str(artifacts.get("work_dir") or ""))
            if not work_dir.is_absolute():
                work_dir = (
                    run_directory / work_dir
                    if str(work_dir)
                    else run_directory / "calculations" / backend
                )
            timers = _parse_application_timers(work_dir, backend)
            stage_records.append(
                {
                    "backend": backend,
                    "converged": stage.get("converged"),
                    "failure_reason": stage.get("failure_reason"),
                    "steps": stage.get("steps"),
                    **timers,
                    "timer_source": "native_output" if any(timers.values()) else None,
                }
            )

        directory_key = str(run_directory)
        allocation_row = accounting.get(directory_key)
        allocation = None
        if allocation_row is not None:
            elapsed = float(allocation_row["elapsed_seconds"])
            nodes = int(allocation_row["nodes"])
            gpus = int(allocation_row["gpus"])
            allocation = {
                "elapsed_seconds": elapsed,
                "nodes": nodes,
                "gpus": gpus,
                "node_hours": elapsed * nodes / 3600.0,
                "gpu_hours": elapsed * gpus / 3600.0,
                "source": "provided_accounting_csv",
            }
        runs.append(
            {
                "run_directory": directory_key,
                "run_id": result.get("run_id"),
                "status": result.get("status"),
                "dft_stages": stage_records,
                "allocation": allocation,
            }
        )

    return {
        "root": str(root_path),
        "run_count": len(runs),
        "allocation_record_count": sum(run["allocation"] is not None for run in runs),
        "runs": runs,
        "accounting_note": (
            "Application timers come from native QE/VASP outputs. Allocation metrics are included "
            "only for exact run_directory matches in the supplied accounting CSV; no scheduler "
            "commands are run and missing allocation data are not estimated."
        ),
    }


__all__ = ["inventory_dft_runs"]
