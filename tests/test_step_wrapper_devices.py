"""Slurm step wrappers keep the placement's exact GPUs (srun is stubbed)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "perlmutter-qe": ("deployments/perlmutter/launchers/_qe-step-perlmutter.sh", "CUDA"),
    "perlmutter-vasp": ("deployments/perlmutter/launchers/_vasp-step-perlmutter.sh", "CUDA"),
    "frontier-qe": ("deployments/frontier/launchers/_qe-step-frontier.sh", "ROCR"),
    "frontier-vasp": ("deployments/frontier/launchers/_vasp-step-frontier.sh", "ROCR"),
}


def _run(tmp_path, script, mask, devices_per_node, node_devices="4"):
    """Run the srun-building tail of a wrapper with stub srun/module tooling."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / "srun.args"
    (bin_dir / "srun").write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {record}\n')
    (bin_dir / "srun").chmod(0o755)
    text = (ROOT / script).read_text()
    tail = text[text.index("# Slurm picks which GPUs") :]
    prelude = (
        "set -euo pipefail\n"
        "NODE_ARGS=(); NNODES=1; TOTAL_RANKS=3; THREADS_PER_RANK=16\n"
        f"GPUS_PER_NODE={devices_per_node}; PW_BIN=pw.x; INPUT=pw.in; VASP_BIN=vasp_std\n"
    )
    stub = tmp_path / "wrapper-tail.sh"
    stub.write_text(prelude + tail)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "MATSIM_DEVICE_MASK": mask,
        "MATSIM_NODE_DEVICES": node_devices,
    }
    subprocess.run(["bash", str(stub)], env=env, check=True, capture_output=True, text=True)
    return record.read_text().splitlines()


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_subset_placement_takes_the_node_and_masks_inside_the_step(tmp_path, name):
    script, vendor = WRAPPERS[name]
    args = _run(tmp_path, script, "1,2,3", "3")
    assert "--gpus-per-node=4" in args and "--gpu-bind=none" in args
    # Other pools may hold the node's other GPUs, so the step overlaps them.
    assert "--overlap" in args and "--exclusive" not in args
    assert f"{vendor}_VISIBLE_DEVICES=1,2,3" in args
    assert args.index(f"{vendor}_VISIBLE_DEVICES=1,2,3") < len(args) - 1


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_whole_node_placement_keeps_slurm_binding(tmp_path, name):
    script, vendor = WRAPPERS[name]
    args = _run(tmp_path, script, "0,1,2,3", "4")
    assert "--gpus-per-node=4" in args and "--gpu-bind=closest" in args
    assert "--exclusive" in args and "--overlap" not in args
    assert not any(arg.startswith(f"{vendor}_VISIBLE_DEVICES") for arg in args)


AURORA = {
    "aurora-qe": "deployments/aurora/launchers/_qe-step-aurora.sh",
    "aurora-vasp": "deployments/aurora/launchers/_vasp-step-aurora.sh",
}


def _run_aurora(tmp_path, script, mask, ranks_per_node, threads=8):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / "mpiexec.args"
    (bin_dir / "mpiexec").write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$@" "ZE=${{ZE_AFFINITY_MASK:-unset}}" > {record}\n'
    )
    (bin_dir / "mpiexec").chmod(0o755)
    text = (ROOT / script).read_text()
    tail = text[text.index("# Each rank owns") :]
    stub = tmp_path / "wrapper-tail.sh"
    stub.write_text(
        "set -euo pipefail\n"
        f"HOSTS=x1; RANKS_PER_NODE={ranks_per_node}; TOTAL_RANKS={ranks_per_node}\n"
        f"THREADS_PER_RANK={threads}; PW_BIN=pw.x; INPUT=pw.in; VASP_BIN=vasp_std\n" + tail
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "MATSIM_DEVICE_MASK": mask,
        "MATSIM_DEVICES_PER_NODE": str(len(mask.split(","))),
        "MATSIM_NODE_DEVICES": "12",
        "ZE_AFFINITY_MASK": mask,
    }
    subprocess.run(["bash", str(stub)], env=env, check=True, capture_output=True, text=True)
    return record.read_text().splitlines()


@pytest.mark.parametrize("name", sorted(AURORA))
def test_aurora_steps_bind_rank_cores_and_placed_tiles(tmp_path, name):
    args = _run_aurora(tmp_path, AURORA[name], "4,5", ranks_per_node=4)
    assert "--depth=8" in args and "--cpu-bind=depth" in args
    assert "--gpu-bind=list:4:5:4:5" in args
    assert args[-1] == "ZE=unset"  # mpiexec sets each rank's mask from --gpu-bind


@pytest.mark.parametrize("name", sorted(AURORA))
def test_aurora_whole_node_steps_bind_nearest_tile(tmp_path, name):
    mask = ",".join(str(index) for index in range(12))
    args = _run_aurora(tmp_path, AURORA[name], mask, ranks_per_node=12, threads=1)
    assert "--gpu-bind=closest" in args and "--depth=1" in args


def _run_aurora_vllm(tmp_path, in_step):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / "calls"
    for tool in ("mpiexec", "python"):
        (bin_dir / tool).write_text(
            f'#!/bin/bash\nprintf "{tool} %s\\n" "$*" >> {record}\n'
            f'printf "{tool} PMI_RANK=%s\\n" "${{PMI_RANK:-unset}}" >> {record}\n'
        )
        (bin_dir / tool).chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "ZE_AFFINITY_MASK": "0,1",
        "PMI_RANK": "0",
        "PBS_JOBID": "1",
    }
    if in_step:
        env["MATSIM_SERVER_IN_STEP"] = "1"
    script = ROOT / "deployments/aurora/launchers/vllm-server-aurora.sh"
    subprocess.run(
        ["bash", str(script), "/models/m", "--port", "8000"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return record.read_text().splitlines()


def test_aurora_vllm_launcher_starts_its_own_mpiexec_step(tmp_path):
    calls = _run_aurora_vllm(tmp_path, in_step=False)
    assert calls[0].startswith("mpiexec --hosts ")
    assert "-n 1 --ppn 1 env -u PMI_RANK" in calls[0]


def test_aurora_vllm_launcher_runs_directly_inside_an_existing_step(tmp_path):
    calls = _run_aurora_vllm(tmp_path, in_step=True)
    assert not any(call.startswith("mpiexec") for call in calls)
    assert calls[0].startswith("python ") and "--model /models/m" in calls[0]
    assert calls[1] == "python PMI_RANK=unset"
