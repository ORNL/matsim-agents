"""Exercise facility launcher argument translation without Slurm or HydraGNN."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("platform", ["frontier", "perlmutter", "aurora"])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize(
    "script_name", ["finetune_hydragnn", "finetune_hydragnn_newhead", "custom"]
)
def test_launcher_executes_expected_cli(tmp_path, platform, configured, script_name):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "deployments" / platform / "launchers" / f"_hydragnn-train-step-{platform}.sh"
    script = tmp_path / f"{script_name}.py"
    script.write_text(
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE_ARGS'], 'w') as output:\n"
        "    json.dump(sys.argv[1:], output)\n"
    )
    for scheduler in ("srun", "mpiexec"):
        stub = tmp_path / scheduler
        stub.write_text(
            '#!/bin/bash\nset -eu\nwhile [[ "$1" != python ]]; do shift; done\nexec "$@"\n'
        )
        stub.chmod(0o755)
    (tmp_path / "python").symlink_to(sys.executable)
    manifest = tmp_path / "elemental references.json"
    manifest.write_text("{}")
    branch_mlp = tmp_path / "branch mlp.pt"
    branch_mlp.touch()
    output = tmp_path / "output"
    capture = tmp_path / "args.json"
    args = [
        "bash",
        str(launcher),
        str(script),
        str(tmp_path / "dataset.extxyz"),
        str(output),
        str(tmp_path / "resume"),
        "2",
        "1",
        "1",
        str(manifest),
        "incumbent.pk",
        str(branch_mlp),
    ]
    if configured:
        args.extend(
            [
                str(tmp_path / "fine tuning repo"),
                "-2.5",
                "3.0",
                str(tmp_path / "HydraGNN source"),
            ]
        )
    proc = subprocess.run(
        args,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE_ARGS": str(capture),
            # Set by the training pool's ProcessLaunch.
            "MATSIM_ASSIGNED_NODES": "node0",
            "ZE_AFFINITY_MASK": "3",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if script_name == "custom":
        assert proc.returncode == 2
        assert "unsupported HydraGNN trainer" in proc.stderr
        assert not capture.exists()
        return
    assert proc.returncode == 0, proc.stderr
    actual = json.loads(capture.read_text())
    expected = [
        "--dataset",
        str(tmp_path / "dataset.extxyz"),
        "--output-dir",
        str(output),
        "--gfm-logdir",
        str(tmp_path / "resume"),
        "--epochs",
        "2",
        "--elemental-reference-manifest",
        str(manifest),
    ]
    if platform == "aurora":
        expected.extend(["--device", "xpu"])
    expected.extend(
        [
            "--gfm-checkpoint",
            "incumbent.pk",
            "--charge",
            "-2.5" if configured else "0.0",
            "--spin",
            "3.0" if configured else "0.0",
        ]
    )
    if script_name == "finetune_hydragnn_newhead" and configured:
        expected.extend(["--ft-repo", str(tmp_path / "fine tuning repo")])
    if configured:
        expected.extend(["--hydragnn-root", str(tmp_path / "HydraGNN source")])
    if script_name == "finetune_hydragnn":
        expected.extend(["--branch-mlp", str(branch_mlp)])
    assert actual == expected


@pytest.mark.parametrize(
    ("cuda", "xpu", "expected"),
    [(True, True, "cuda"), (False, True, "xpu"), (False, False, "cpu"), (False, None, "cpu")],
)
def test_default_torch_device_prefers_cuda_then_xpu(monkeypatch, cuda, xpu, expected):
    from types import SimpleNamespace

    import torch

    from matsim_agents.backends.mlip.devices import default_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    if xpu is None:
        monkeypatch.delattr(torch, "xpu", raising=False)
    else:
        monkeypatch.setattr(torch, "xpu", SimpleNamespace(is_available=lambda: xpu), raising=False)
    assert default_torch_device() == expected
