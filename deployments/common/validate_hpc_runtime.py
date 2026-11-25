#!/usr/bin/env python3
"""Exercise the scheduler- and launcher-facing code on a real allocation or login node.

``allocation`` and ``qe`` run inside a Slurm allocation; ``slurm-jobs`` runs where ``sbatch``
and ``sacct`` work. Each check records pass/fail and the process exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

RESULTS: list[dict[str, str]] = []
FACILITY = "frontier"


def check(name: str) -> Callable[[Callable[..., str]], Callable[..., str]]:
    def wrap(fn: Callable[..., str]) -> Callable[..., str]:
        def run(*args: object) -> str:
            try:
                detail = fn(*args)
                RESULTS.append({"check": name, "status": "pass", "detail": detail})
            except Exception as exc:  # a failed check must not stop the others
                RESULTS.append({"check": name, "status": "fail", "detail": f"{exc!r}"})
                print(f"FAIL {name}: {exc!r}", flush=True)
                return "fail"
            print(f"PASS {name}: {detail}", flush=True)
            return "pass"

        run.__name__ = fn.__name__
        return run

    return wrap


def _sleepers() -> list[str]:
    out = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-f", "^sleep 600$"], capture_output=True, text=True
    ).stdout.split()
    return sorted(out)


def _job_steps(job_id: str) -> list[str]:
    out = subprocess.run(
        ["squeue", "-s", "-h", "-j", job_id, "-o", "%i"], capture_output=True, text=True
    ).stdout.split()
    return [step for step in out if not step.endswith((".batch", ".extern"))]


def _facility():
    from matsim_agents.execution.presets import get_preset

    return get_preset(FACILITY)


@check("allocation discovery")
def allocation() -> str:
    from matsim_agents.execution.allocation import discover_allocation

    preset = _facility()
    expected = "slurm" if preset.launcher == "srun" else "pbs"
    found = discover_allocation()
    assert found.scheduler == expected, f"expected {expected}, found {found}"
    nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    assert found.node_count == nodes, found
    return f"{found.scheduler} nodes={list(found.nodes)} gpus_per_node={found.gpus_per_node}"


def probe_mask(vendor: str) -> int:
    from matsim_agents.execution.devices import child_mask_environment, visible_ordinals
    from matsim_agents.execution.presets import DEVICE_MASK_VARIABLES

    variable = DEVICE_MASK_VARIABLES[vendor]
    ok = visible_ordinals(vendor, [0, 1], owner="probe")
    try:
        visible_ordinals(vendor, [2], owner="probe")
        bad = None
    except ValueError as exc:
        bad = str(exc)
    child = child_mask_environment({variable: "1"}, owner="probe")[variable]
    print(
        json.dumps({"mask": os.environ.get(variable), "ok": list(ok), "bad": bad, "child": child})
    )
    return 0


@check("device mask inside a launcher step")
def step_masks() -> str:
    preset = _facility()
    vendor = preset.geometry.vendor
    probe = [sys.executable, str(Path(__file__).resolve()), "probe-mask", "--vendor", vendor]
    env = dict(os.environ)
    if preset.launcher == "srun":
        argv = ["srun", "-N1", "-n1", "--gpus-per-task=2", "--gpu-bind=none", *probe]
    else:
        # No scheduler step mask under PBS, so impose a two-device one.
        argv = ["mpiexec", "-n", "1", *probe]
        env.update(preset.geometry.device_mask_environment((2, 3)))
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=env)
    assert proc.returncode == 0, proc.stderr[-400:]
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    mask = (info["mask"] or "").split(",")
    assert len(mask) == 2, f"expected a two-device mask, got {info['mask']!r}"
    assert info["ok"] == [0, 1] and info["bad"], info
    assert info["child"] == mask[1], info
    return f"mask {info['mask']}; child of index 1 -> {info['child']}; index 2 rejected"


def _silicon():
    from ase import Atoms

    return Atoms(
        "Si2",
        cell=[[0, 2.75, 2.75], [2.75, 0, 2.75], [2.75, 2.75, 0]],
        scaled_positions=[[0, 0, 0], [0.26, 0.25, 0.25]],
        pbc=True,
    )


@check("QE relaxation through run_pw and the real launcher")
def qe(work: Path) -> str:
    from matsim_agents.backends.dft.qe_relax import (
        QESettings,
        find_default_launcher,
        run_pw,
        write_pw_input,
    )

    launcher = find_default_launcher()
    assert launcher, "MATSIM_QE_LAUNCHER is unset or does not resolve"
    pseudo = os.environ["MATSIM_QE_PSEUDO_DIR"]
    settings = QESettings(
        pseudo_dir=pseudo,
        calculation="relax",
        ecutwfc_ry=20.0,
        ecutrho_ry=160.0,
        occupations="fixed",
        kpts=(2, 2, 2),
        pseudopotentials={"Si": "Si_r.upf"},
    )
    work.mkdir(parents=True, exist_ok=True)
    deck = write_pw_input(_silicon(), settings, str(work / "pw.in"), outdir=str(work / "tmp"))
    geometry = _facility().geometry
    if geometry.vendor == "rocm":
        # An inherited HIP mask must be dropped for the placed ROCR mask.
        os.environ["HIP_VISIBLE_DEVICES"] = "1"
    mask = geometry.device_mask_environment(tuple(range(geometry.devices_per_node)))
    result = run_pw(deck, str(work), shlex.split(launcher), timeout_sec=900, env=mask)
    assert result.return_code == 0, f"return code {result.return_code}; see {result.stdout_path}"
    assert result.job_done and result.converged, (
        f"job_done={result.job_done} converged={result.converged}"
    )
    assert result.final_max_force_eV_per_A is not None and result.final_max_force_eV_per_A < 0.05
    return (
        f"E={result.final_energy_ev:.6f} eV, bfgs={result.bfgs_steps}, "
        f"max|F|={result.final_max_force_eV_per_A:.4f} eV/A, wall={result.wall_time_sec:.1f}s"
    )


@check("launcher timeout stops the launched step")
def timeout_cleanup(work: Path) -> str:
    from matsim_agents.backends.dft.process import TIMEOUT_RETURN_CODE, run_launcher

    work.mkdir(parents=True, exist_ok=True)
    slurm = _facility().launcher == "srun"
    job = os.environ.get("SLURM_JOB_ID", "")
    argv = (
        ["srun", "-N1", "-n1", "sleep", "600"] if slurm else ["mpiexec", "-n", "1", "sleep", "600"]
    )
    before = _job_steps(job) if slurm else _sleepers()
    started = time.monotonic()
    with open(work / "sleep.out", "w") as out:
        code = run_launcher(argv, cwd=str(work), stdout=out, timeout_sec=5, env=None, grace_sec=90)
    elapsed = time.monotonic() - started
    assert code == TIMEOUT_RETURN_CODE, code
    after = _job_steps(job) if slurm else _sleepers()
    assert after == before, f"left behind: {sorted(set(after) - set(before))}"
    return f"returned {code} after {elapsed:.0f}s; nothing left running"


@check("slurm_jobs submit, dedupe and sacct refresh")
def slurm_jobs(account: str, manifest: Path, qos: str, partition: str) -> str:
    from matsim_agents.execution.slurm_jobs import JobSpec, refresh_manifest, submit_job

    manifest.parent.mkdir(parents=True, exist_ok=True)
    spec = JobSpec(
        name="matsim-validate-hostname",
        command=("hostname",),
        time_limit="00:02:00",
        account=account,
        qos=qos,
        partition=partition,
        cwd=str(manifest.parent),
    )
    entry, submitted = submit_job(spec, manifest)
    again, resubmitted = submit_job(spec, manifest)
    assert again.job_id == entry.job_id and not resubmitted, "duplicate spec was resubmitted"
    states = {e.job_id: e.state for e in refresh_manifest(manifest)}
    return (
        f"job {entry.job_id} submitted={submitted}; duplicate suppressed; "
        f"sacct state {states.get(entry.job_id)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["allocation", "qe", "slurm-jobs", "refresh", "probe-mask"]
    )
    parser.add_argument("--facility", default="frontier", help="resource preset name")
    parser.add_argument("--vendor", default="rocm", help="probe-mask only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--account", help="scheduler account for slurm-jobs")
    parser.add_argument("--qos", default="debug")
    parser.add_argument("--partition", default="batch")
    args = parser.parse_args()
    global FACILITY
    FACILITY = args.facility
    if args.phase == "probe-mask":
        return probe_mask(args.vendor)
    assert args.output, "--output is required"
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "validation_jobs.json"
    if args.phase == "allocation":
        allocation()
        step_masks()
    elif args.phase == "qe":
        qe(args.output / "qe")
        timeout_cleanup(args.output / "timeout")
    elif args.phase == "slurm-jobs":
        assert args.account, "--account is required"
        slurm_jobs(args.account, manifest, args.qos, args.partition)
    else:
        from matsim_agents.execution.slurm_jobs import refresh_manifest

        for entry in refresh_manifest(manifest):
            RESULTS.append(
                {
                    "check": f"job {entry.job_id}",
                    "status": "pass" if entry.state == "COMPLETED" else "fail",
                    "detail": f"{entry.state} exit={entry.exit_code} elapsed={entry.elapsed}",
                }
            )
            print(RESULTS[-1], flush=True)
    (args.output / f"{args.phase}.json").write_text(json.dumps(RESULTS, indent=2))
    failed = [r for r in RESULTS if r["status"] != "pass"]
    for r in failed:
        print(f"FAIL {r['check']}: {r['detail']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
