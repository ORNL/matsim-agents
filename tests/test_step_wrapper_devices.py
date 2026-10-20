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
    assert f"{vendor}_VISIBLE_DEVICES=1,2,3" in args
    assert args.index(f"{vendor}_VISIBLE_DEVICES=1,2,3") < len(args) - 1


@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_whole_node_placement_keeps_slurm_binding(tmp_path, name):
    script, vendor = WRAPPERS[name]
    args = _run(tmp_path, script, "0,1,2,3", "4")
    assert "--gpus-per-node=4" in args and "--gpu-bind=closest" in args
    assert not any(arg.startswith(f"{vendor}_VISIBLE_DEVICES") for arg in args)
