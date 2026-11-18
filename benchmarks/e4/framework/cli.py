#!/usr/bin/env python3
"""Command line for E4 orchestration measurements.

plan     print the frozen workload (task counts, hash) for each node point; no execution
run      execute one measurement inside the current allocation (or locally)
resume   continue an interrupted run directory without repeating finished tasks
smoke    local weak+strong sweep, then the scaling analysis (CI-sized)
analyze  tables and figures from measurements.jsonl files
jobs     print the batch submissions and resource estimates of a sweep (never submits)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.e4.framework.workload import build_workload, load_config  # noqa: E402

FACILITIES = ("local", "aurora", "frontier", "perlmutter")


def _classes(value: str | None) -> list[str] | None:
    return None if not value else [item for item in value.split(",") if item]


def cmd_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    points = args.nodes or config.get("node_points", {}).get(args.experiment, [1])
    rows = []
    for nodes in points:
        workload = build_workload(
            config,
            experiment=args.experiment,
            nodes=nodes,
            root=ROOT,
            classes=_classes(args.classes),
        )
        counts: dict[str, int] = {}
        for task in workload.tasks:
            counts[task.task_class] = counts.get(task.task_class, 0) + 1
        rows.append(
            {
                "nodes": nodes,
                "planned_tasks": workload.planned_tasks,
                "tasks_per_node": workload.planned_tasks / nodes,
                "by_class": counts,
                "workload_hash": workload.workload_hash,
            }
        )
    print(
        json.dumps(
            {"workload_id": config["workload_id"], "experiment": args.experiment, "points": rows},
            indent=2,
        )
    )
    return 0


def _run(args: argparse.Namespace, *, resume: bool) -> int:
    from benchmarks.e4.framework.runner import BenchmarkRun

    try:
        return _execute(BenchmarkRun, args, resume=resume)
    except FileExistsError:
        if not args.auto_resume:
            raise
        return _execute(BenchmarkRun, args, resume=True)


def _execute(BenchmarkRun, args: argparse.Namespace, *, resume: bool) -> int:  # noqa: N803
    run = BenchmarkRun(
        load_config(args.config),
        facility=args.facility,
        experiment=args.experiment,
        repetition=args.repetition,
        output_root=args.output_root,
        nodes=args.nodes,
        policy=args.policy,
        classes=_classes(args.classes),
        run_id=args.run_id,
        resume=resume,
        local_device_slots=args.local_device_slots,
    )
    summary = run.execute()
    print(
        json.dumps(
            {
                "run_dir": str(run.run_dir),
                "counts": summary["counts"],
                "throughput": summary["throughput"],
            },
            indent=2,
        )
    )
    counts = summary["counts"]
    return 0 if counts["unfinished_tasks"] == 0 and counts["failed_tasks"] == 0 else 2


def cmd_smoke(args: argparse.Namespace) -> int:
    from benchmarks.e4.framework.analysis import analyze_runs
    from benchmarks.e4.framework.runner import BenchmarkRun

    config = load_config(args.config)
    output = Path(args.output_root)
    for experiment in ("weak", "strong"):
        for nodes in config["node_points"][experiment]:
            run = BenchmarkRun(
                config,
                facility="local",
                experiment=experiment,
                repetition=0,
                output_root=output / "runs",
                nodes=nodes,
                local_device_slots=args.local_device_slots,
            )
            counts = run.execute()["counts"]
            print(
                f"{run.run_id}: qualified {counts['qualified_tasks']}/{counts['planned_tasks']}, "
                f"attempts {counts['total_attempts']}, failed {counts['failed_tasks']}"
            )
    report = analyze_runs([output / "runs"], output / "analysis", plots=not args.no_plots)
    print(json.dumps(report["outputs"], indent=2))
    return 0


JOB_SCRIPTS = {
    "aurora": ("qsub", "deployments/aurora/jobs/job-e4-framework-aurora.pbs"),
    "frontier": ("sbatch", "deployments/frontier/jobs/job-e4-framework-frontier.sh"),
    "perlmutter": ("sbatch", "deployments/perlmutter/jobs/job-e4-framework-perlmutter.sh"),
}
STARTUP_SECONDS = 300.0


def _walltime(seconds: float) -> str:
    minutes = max(10, int(-(-seconds // 600)) * 10)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def estimate_point(
    config: dict,
    experiment: str,
    nodes: int,
    classes: list[str] | None,
    facility: str | None = None,
) -> dict:
    """Planning estimate of one run: ideal makespan from per-task seconds, plus margins.

    Controlled tasks use their frozen durations; real classes use ``estimate_seconds``
    from the config. Node-group tasks share ``nodes - mlip pool`` nodes; device tasks
    share the device slots of the mlip pool. Retries are included through the
    configured fault fraction. This is an estimate for budgeting, not a measurement.
    """

    from matsim_agents.execution.presets import get_preset

    workload = build_workload(
        config, experiment=experiment, nodes=nodes, root=ROOT, classes=classes
    )
    specs = {spec["name"]: spec for spec in config["task_classes"]}
    node_seconds = device_seconds = 0.0
    for task in workload.tasks:
        seconds = task.seconds or float(specs[task.task_class].get("estimate_seconds", 0.0))
        seconds *= 1 + task.fail_attempts
        if task.pool == "dft":
            node_seconds += seconds
        else:
            device_seconds += seconds
    resources = config["resources"].get(facility) or next(iter(config["resources"].values()))
    mlip = resources["pools"].get("mlip")
    mlip_nodes = int(mlip["nodes"]) if mlip else 0
    dft_nodes = max(nodes - mlip_nodes, 1)
    slots = mlip_nodes * get_preset(resources["preset"]).geometry.devices_per_node if mlip else 1
    makespan = max(node_seconds / dft_nodes, device_seconds / max(slots, 1))
    wall = STARTUP_SECONDS + 1.5 * makespan
    return {
        "nodes": nodes,
        "planned_tasks": workload.planned_tasks,
        "ideal_makespan_seconds": round(makespan, 1),
        "walltime": _walltime(wall),
        "node_hours_estimate": round(nodes * wall / 3600, 2),
    }


def cmd_jobs(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    submit, script = JOB_SCRIPTS[args.facility]
    points = args.nodes or config.get("node_points", {}).get(args.experiment, [1])
    rows, total = [], 0.0
    for repetition in range(args.repetitions or int(config.get("repetitions", 1))):
        for nodes in points:
            row = estimate_point(
                config, args.experiment, nodes, _classes(args.classes), args.facility
            )
            env = [
                f"E4_CONFIG={args.config}",
                f"E4_EXPERIMENT={args.experiment}",
                f"E4_REPETITION={repetition}",
                f"E4_OUTPUT_ROOT={args.output_root}",
                f"E4_POLICY={args.policy}",
            ]
            if args.classes:
                env.append(f"E4_CLASSES={args.classes}")
            if submit == "qsub":
                command = (
                    f"qsub -l select={nodes} -l walltime={row['walltime']} "
                    f"-v {','.join(env)} {script}"
                )
            else:
                command = (
                    f"sbatch -N {nodes} -t {row['walltime']} --export=ALL,{','.join(env)} {script}"
                )
            rows.append({**row, "repetition": repetition, "command": command})
            total += row["node_hours_estimate"]
    print(
        json.dumps(
            {
                "facility": args.facility,
                "experiment": args.experiment,
                "runs": rows,
                "total_node_hours_estimate": round(total, 1),
            },
            indent=2,
        )
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from benchmarks.e4.framework.analysis import analyze_runs

    report = analyze_runs(args.inputs, args.output, plots=not args.no_plots)
    print(json.dumps(report["outputs"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("config", type=Path)
    plan.add_argument("--experiment", choices=("weak", "strong", "fixed"), default="weak")
    plan.add_argument("--nodes", type=int, nargs="*")
    plan.add_argument("--classes")
    plan.set_defaults(func=cmd_plan)

    for name, resume in (("run", False), ("resume", True)):
        command = sub.add_parser(name)
        command.add_argument("config", type=Path)
        command.add_argument("--facility", choices=FACILITIES, required=True)
        command.add_argument("--experiment", choices=("weak", "strong", "fixed"), required=True)
        command.add_argument("--repetition", type=int, default=0)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument(
            "--nodes", type=int, help="required for local; checked against the allocation otherwise"
        )
        command.add_argument("--policy", choices=("concurrent", "staged"), default="concurrent")
        command.add_argument("--classes", help="comma-separated task classes (homogeneous runs)")
        command.add_argument("--run-id")
        command.add_argument(
            "--local-device-slots",
            type=int,
            help="local smoke only: CPU worker slots for device pools",
        )
        command.add_argument(
            "--auto-resume", action="store_true", help="resume if the run directory exists"
        )
        command.set_defaults(func=lambda a, resume=resume: _run(a, resume=resume))

    smoke = sub.add_parser("smoke")
    smoke.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path(__file__).parent / "configs" / "smoke-local.json",
    )
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--local-device-slots", type=int, default=2)
    smoke.add_argument("--no-plots", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    analyze = sub.add_parser("analyze")
    analyze.add_argument(
        "inputs", type=Path, nargs="+", help="run directories, roots, or measurements.jsonl files"
    )
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--no-plots", action="store_true")
    analyze.set_defaults(func=cmd_analyze)

    jobs = sub.add_parser("jobs")
    jobs.add_argument("config", type=Path)
    jobs.add_argument("--facility", choices=sorted(JOB_SCRIPTS), required=True)
    jobs.add_argument("--experiment", choices=("weak", "strong", "fixed"), required=True)
    jobs.add_argument("--nodes", type=int, nargs="*")
    jobs.add_argument("--repetitions", type=int)
    jobs.add_argument("--policy", choices=("concurrent", "staged"), default="concurrent")
    jobs.add_argument("--classes")
    jobs.add_argument("--output-root", default="${RUNS_ROOT}/e4-framework")
    jobs.set_defaults(func=cmd_jobs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
