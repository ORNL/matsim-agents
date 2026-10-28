from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_qualification_launchers_share_device_runner() -> None:
    common = ROOT / "deployments/common/qualify_torch_pyg_device.py"
    assert common.is_file()
    expected = {
        "aurora": ("deployments/aurora/jobs/qualify_torch_pyg_xpu.py", '"xpu"'),
        "frontier": (
            "deployments/frontier/jobs/job-qualify-torch-pyg-frontier.sh",
            "--device cuda",
        ),
        "perlmutter": (
            "deployments/perlmutter/jobs/job-qualify-torch-pyg-perlmutter.sh",
            "--device cuda",
        ),
    }
    for relative_path, device in expected.values():
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert device in content
        if relative_path.endswith(".sh"):
            assert "deployments/common/qualify_torch_pyg_device.py" in content


def test_gpu_qualification_cpu_operations(tmp_path: Path) -> None:
    if any(
        importlib.util.find_spec(package) is None
        for package in ("torch", "torch_geometric", "torch_scatter")
    ):
        pytest.skip("Torch, PyG, and torch_scatter are required for the CPU qualification")

    output = tmp_path / "qualification.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "deployments/common/qualify_torch_pyg_device.py"),
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "PASSED"
    assert result["device"] == "cpu"
    assert {
        "pyg_scatter_sum_and_backward",
        "pyg_global_add_pool",
        "torch_scatter_add",
        "torch_scatter_sum",
        "torch_scatter_mean",
        "torch_scatter_reductions_backward",
    }.issubset(result["operations"])