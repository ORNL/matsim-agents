"""E4.2 cross-platform portability evidence from existing run artefacts.

Three levels per facility, each backed by a real run directory:

1. ``deployment`` -- a ``benchmarks/portability/run.py --suite smoke --execute``
   output: the installed package imports and the facility DFT step launcher
   validates. Checked with ``benchmarks.portability.validate.validate_run``.
2. ``stage`` -- an E4 framework run (``cli.py run``) containing real ``dft`` and
   ``mlip-relax`` tasks. Passes only if every real class has qualified tasks and
   the provenance below is complete.
3. ``workflow`` -- a ``benchmarks/portability/run.py --qualification compute``
   output: the production relaxation workflow end-to-end with real MLIP and QE
   backends. Contract-qualification runs (deterministic adapters) never count.

Installation alone never counts: level 1 requires an executed smoke run. The
numerical comparison groups qualified level-2 results by (task class, input
structure hash) and checks the spread across facilities against the
``cross_facility`` tolerances of ``benchmarks/portability/manifest.yaml``;
bitwise identity is not required. Level-3 runs are compared with
``benchmarks.portability.compare.compare_runs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.e4.framework.records import read_records  # noqa: E402

REAL_KINDS = ("dft", "mlip-relax")
METRICS = (
    ("energy_eV_per_atom", "energy_eV_per_atom_atol"),
    ("max_force_eV_per_A", "max_force_eV_per_A_atol"),
)


def tolerances() -> dict[str, float]:
    manifest = json.loads(
        (REPO / "benchmarks/portability/manifest.yaml").read_text(encoding="utf-8")
    )
    return {key: float(value) for key, value in manifest["acceptance"]["cross_facility"].items()}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deployment_level(path: Path) -> dict[str, Any]:
    from benchmarks.portability.validate import validate_run

    errors = list(validate_run(path))
    result = _load(path / "result.json") if (path / "result.json").is_file() else {}
    smoke = (result.get("execution") or {}).get("smoke")
    if smoke is None:
        errors.append("no executed smoke suite (installation alone is not portability)")
    elif smoke.get("return_code", 0) != 0:
        errors.append("smoke suite failed")
    return {"level": "deployment", "run": str(path), "passed": not errors, "errors": errors}


def workflow_level(path: Path) -> dict[str, Any]:
    from benchmarks.portability.validate import validate_run

    errors = list(validate_run(path))
    result = _load(path / "result.json") if (path / "result.json").is_file() else {}
    if result.get("qualification") != "compute":
        errors.append(
            "not a compute-qualification run; contract adapters are not workflow evidence"
        )
    if result.get("status") != "passed":
        errors.append(f"workflow status is {result.get('status')!r}")
    return {"level": "workflow", "run": str(path), "passed": not errors, "errors": errors}


def stage_provenance(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Provenance of one framework run and the list of missing items."""

    errors: list[str] = []
    files = {
        name: run_dir / name for name in ("identity.json", "manifest.json", "resolved_config.json")
    }
    for name, path in files.items():
        if not path.is_file():
            errors.append(f"missing {name}")
    environments = sorted(run_dir.glob("environment-*.json"))
    if not environments:
        errors.append("missing environment-*.json")
    if errors:
        return {}, errors
    identity = _load(files["identity.json"])
    manifest = _load(files["manifest.json"])
    config = _load(files["resolved_config.json"])
    envs = [_load(path) for path in environments]
    facility = identity["facility"]
    for env in envs:
        if not env.get("git_commit"):
            errors.append(f"{env.get('recorded_utc')}: no git commit")
        if facility != "local" and not env.get("scheduler"):
            errors.append(f"{env.get('recorded_utc')}: no scheduler job ID")
    kinds = {task["kind"] for task in manifest["tasks"]}
    dft = (config.get("dft_backends") or {}).get(facility)
    if "dft" in kinds:
        block = (dft or {}).get((dft or {}).get("backend", ""), {})
        if not block:
            errors.append("no DFT backend settings for this facility")
        elif dft["backend"] == "qe" and not block.get("pseudopotentials"):
            errors.append("QE pseudopotentials not pinned")
        elif dft["backend"] == "vasp" and not (
            block.get("potcar_dir") and block.get("incar_template")
        ):
            errors.append("VASP POTCAR directory or INCAR template not pinned")
    for task in manifest["tasks"]:
        if task["kind"] in REAL_KINDS and not task.get("structure_sha256"):
            errors.append(f"{task['task_id']}: no input structure hash")
    provenance = {
        "run_id": identity["run_id"],
        "facility": facility,
        "git_commits": sorted({env.get("git_commit") or "" for env in envs}),
        "git_dirty": any(env.get("git_dirty") for env in envs),
        "software": envs[-1].get("software"),
        "modules": envs[-1].get("modules"),
        "job_ids": [env.get("scheduler") for env in envs],
        "config_sha256": identity["config_sha256"],
        "workload_hash": identity["workload_hash"],
        "resource_plan_sha256": identity["resource_plan_sha256"],
        "recovery_policy": identity["recovery_policy"],
        "dft_settings": dft,
        "checkpoint": "attempts.jsonl",
    }
    return provenance, errors


def stage_results(run_dir: Path) -> list[dict[str, Any]]:
    """Terminal, qualified real-task attempts with their numerical results."""

    manifest = _load(run_dir / "manifest.json")
    structures = {task["task_id"]: task.get("structure_sha256") for task in manifest["tasks"]}
    rows = []
    for record in read_records(run_dir / "attempts.jsonl"):
        if record.get("record") != "attempt" or record.get("kind") not in REAL_KINDS:
            continue
        outcome = record.get("outcome") or {}
        if not record.get("terminal") or not outcome.get("qualified"):
            continue
        rows.append(
            {
                "task_id": record["task_id"],
                "task_class": record["task_class"],
                "kind": record["kind"],
                "structure_sha256": structures.get(record["task_id"]),
                "energy_eV_per_atom": outcome.get("energy_eV_per_atom"),
                "max_force_eV_per_A": outcome.get("max_force_eV_per_A"),
                "input_sha256": outcome.get("input_sha256"),
                "output_sha256": outcome.get("output_sha256"),
                "settings": record.get("settings"),
            }
        )
    return rows


def stage_level(run_dir: Path) -> dict[str, Any]:
    provenance, errors = stage_provenance(run_dir)
    if errors:
        return {"level": "stage", "run": str(run_dir), "passed": False, "errors": errors}
    manifest = _load(run_dir / "manifest.json")
    real_classes = sorted({t["task_class"] for t in manifest["tasks"] if t["kind"] in REAL_KINDS})
    if not real_classes:
        errors.append("run has no real dft/mlip-relax tasks")
    results = stage_results(run_dir)
    counts = {name: sum(row["task_class"] == name for row in results) for name in real_classes}
    for name, count in counts.items():
        if count == 0:
            errors.append(f"class {name}: no qualified results")
    for row in results:
        if not row["output_sha256"]:
            errors.append(f"{row['task_id']}: qualified without an output hash")
        if row["kind"] == "dft" and not row["input_sha256"]:
            errors.append(f"{row['task_id']}: DFT result without input hashes")
    return {
        "level": "stage",
        "run": str(run_dir),
        "passed": not errors,
        "errors": errors,
        "qualified_by_class": counts,
        "provenance": provenance,
        "results": results,
    }


def numerical_comparison(stages: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Cross-facility spread of per-structure medians for each qualified class."""

    tol = tolerances()
    medians: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for facility, stage in stages.items():
        groups: dict[tuple[str, str], dict[str, list[float]]] = {}
        for row in stage.get("results", []):
            key = (row["task_class"], row["structure_sha256"] or "")
            for metric, _ in METRICS:
                if row.get(metric) is not None:
                    groups.setdefault(key, {}).setdefault(metric, []).append(float(row[metric]))
        for key, values in groups.items():
            medians.setdefault(key, {})[facility] = {
                metric: statistics.median(items) for metric, items in values.items()
            }
    rows = []
    for (task_class, structure), by_facility in sorted(medians.items()):
        if len(by_facility) < 2:
            continue
        for metric, tol_key in METRICS:
            values = {fac: item[metric] for fac, item in by_facility.items() if metric in item}
            if len(values) < 2:
                continue
            spread = max(values.values()) - min(values.values())
            rows.append(
                {
                    "task_class": task_class,
                    "structure_sha256": structure,
                    "metric": metric,
                    "facilities": ",".join(sorted(values)),
                    "values": json.dumps(dict(sorted(values.items()))),
                    "spread": spread,
                    "tolerance": tol[tol_key],
                    "within_tolerance": spread <= tol[tol_key],
                }
            )
    return rows


def evaluate(levels: Mapping[str, Mapping[str, Path]], output: str | Path) -> dict[str, Any]:
    """``levels[level][facility] = run_dir``; writes portability.json and CSVs."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checks = {"deployment": deployment_level, "stage": stage_level, "workflow": workflow_level}
    facilities = sorted({fac for runs in levels.values() for fac in runs})
    matrix: dict[str, dict[str, Any]] = {fac: {} for fac in facilities}
    for level, runs in levels.items():
        for facility, path in runs.items():
            matrix[facility][level] = checks[level](Path(path))
    stages = {fac: item["stage"] for fac, item in matrix.items() if "stage" in item}
    numeric = numerical_comparison(stages)
    workflow_runs = [Path(levels["workflow"][fac]) for fac in sorted(levels.get("workflow", {}))]
    workflow_errors: list[str] = []
    if len(workflow_runs) >= 2:
        from benchmarks.portability.compare import compare_runs

        workflow_errors = compare_runs(workflow_runs)
    summary_rows = [
        {
            "facility": fac,
            **{
                level: ("pass" if matrix[fac][level]["passed"] else "fail")
                if level in matrix[fac]
                else "not-run"
                for level in checks
            },
        }
        for fac in facilities
    ]
    report = {
        "benchmark": "e4.2-portability",
        "levels": matrix,
        "summary": summary_rows,
        "numerical_comparison": numeric,
        "numerical_comparison_passed": all(row["within_tolerance"] for row in numeric),
        "workflow_comparison_errors": workflow_errors,
        "tolerances": tolerances(),
    }
    (output / "portability.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "portability-levels.csv", summary_rows, ["facility", *checks])
    if numeric:
        _write_csv(output / "portability-numerics.csv", numeric, list(numeric[0]))
    return report


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _pairs(values: Sequence[str] | None) -> dict[str, Path]:
    pairs = {}
    for value in values or []:
        facility, _, path = value.partition("=")
        if not path:
            raise SystemExit(f"expected FACILITY=DIR, got {value!r}")
        pairs[facility] = Path(path)
    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deployment", nargs="*", metavar="FACILITY=DIR")
    parser.add_argument("--stage", nargs="*", metavar="FACILITY=DIR")
    parser.add_argument("--workflow", nargs="*", metavar="FACILITY=DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    levels = {
        level: _pairs(getattr(args, level))
        for level in ("deployment", "stage", "workflow")
        if getattr(args, level)
    }
    if not levels:
        parser.error("provide at least one level")
    report = evaluate(levels, args.output)
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "numerical_comparison_passed": report["numerical_comparison_passed"],
                "workflow_comparison_errors": report["workflow_comparison_errors"],
            },
            indent=2,
        )
    )
    failed = any(row[level] == "fail" for row in report["summary"] for level in levels)
    if failed or report["workflow_comparison_errors"]:
        return 1
    return 0 if report["numerical_comparison_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
