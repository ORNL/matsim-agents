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


def _job_steps(job_id: str) -> list[str]:
    out = subprocess.run(
        ["squeue", "-s", "-h", "-j", job_id, "-o", "%i"], capture_output=True, text=True
    ).stdout.split()
    return [step for step in out if not step.endswith((".batch", ".extern"))]


@check("allocation discovery")
def allocation() -> str:
    from matsim_agents.execution.allocation import discover_allocation

    found = discover_allocation()
    assert found.scheduler == "slurm", found
    assert found.node_count == int(os.environ["SLURM_JOB_NUM_NODES"]), found
    return f"{found.scheduler} nodes={list(found.nodes)} gpus_per_node={found.gpus_per_node}"


@check("device mask inside a Slurm step")
def step_masks() -> str:
    code = (
        "import os, json;"
        "from matsim_agents.execution.devices import visible_ordinals, child_mask_environment;"
        "v=os.environ.get('ROCR_VISIBLE_DEVICES');"
        "ok=visible_ordinals('rocm',[0,1],owner='probe');"
        "bad=None\n"
        "try: visible_ordinals('rocm',[2],owner='probe')\n"
        "except ValueError as e: bad=str(e)\n"
        "child=child_mask_environment({'ROCR_VISIBLE_DEVICES':'1'},owner='probe');"
        "print(json.dumps({'mask':v,'ok':list(ok),'bad':bad,'child':child['ROCR_VISIBLE_DEVICES']}))"
    )
    proc = subprocess.run(
        [
            "srun",
            "-N1",
            "-n1",
            "--gpus-per-node=2",
            "--gpu-bind=closest",
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    mask = (info["mask"] or "").split(",")
    assert len(mask) == 2, f"expected a two-device step mask, got {info['mask']!r}"
    assert info["ok"] == [0, 1] and info["bad"], info
    assert info["child"] == mask[1], info
    return f"step mask {info['mask']}; child of index 1 -> {info['child']}; index 2 rejected"


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
    # An inherited HIP mask must be dropped for the placed ROCR mask.
    os.environ["HIP_VISIBLE_DEVICES"] = "1"
    mask = {"ROCR_VISIBLE_DEVICES": ",".join(str(i) for i in range(8))}
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


@check("launcher timeout stops the srun step")
def timeout_cleanup(work: Path) -> str:
    from matsim_agents.backends.dft.process import TIMEOUT_RETURN_CODE, run_launcher

    work.mkdir(parents=True, exist_ok=True)
    job = os.environ["SLURM_JOB_ID"]
    before = _job_steps(job)
    started = time.monotonic()
    with open(work / "sleep.out", "w") as out:
        code = run_launcher(
            ["srun", "-N1", "-n1", "sleep", "600"],
            cwd=str(work),
            stdout=out,
            timeout_sec=5,
            env=None,
            grace_sec=90,
        )
    elapsed = time.monotonic() - started
    assert code == TIMEOUT_RETURN_CODE, code
    after = _job_steps(job)
    assert after == before, f"steps left behind: {sorted(set(after) - set(before))}"
    return f"returned {code} after {elapsed:.0f}s; no step left in job {job}"


@check("slurm_jobs submit, dedupe and sacct refresh")
def slurm_jobs(account: str, manifest: Path) -> str:
    from matsim_agents.execution.slurm_jobs import JobSpec, refresh_manifest, submit_job

    manifest.parent.mkdir(parents=True, exist_ok=True)
    spec = JobSpec(
        name="matsim-validate-hostname",
        command=("hostname",),
        time_limit="00:02:00",
        account=account,
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
    parser.add_argument("phase", choices=["allocation", "qe", "slurm-jobs", "refresh"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--account", help="scheduler account for slurm-jobs")
    args = parser.parse_args()
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
        slurm_jobs(args.account, manifest)
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
