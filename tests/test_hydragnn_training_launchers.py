"""Exercise facility launcher argument translation without Slurm or HydraGNN."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("platform", ["frontier", "perlmutter"])
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
    srun = tmp_path / "srun"
    srun.write_text('#!/bin/bash\nset -eu\nwhile [[ "$1" != python ]]; do shift; done\nexec "$@"\n')
    srun.chmod(0o755)
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
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if script_name == "custom" and platform == "perlmutter":
        assert proc.returncode == 2
        assert "unsupported HydraGNN trainer" in proc.stderr
        assert not capture.exists()
        return
    assert proc.returncode == 0, proc.stderr
    actual = json.loads(capture.read_text())
    built_in = script_name != "custom"
    expected = [
        "--dataset",
        str(tmp_path / "dataset.extxyz"),
        "--output-dir" if built_in else "--logdir",
        str(output),
        "--gfm-logdir" if built_in else "--resume_from",
        str(tmp_path / "resume"),
        "--epochs",
        "2",
        "--elemental-reference-manifest",
        str(manifest),
    ]
    if built_in:
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
    if built_in and configured:
        expected.extend(["--hydragnn-root", str(tmp_path / "HydraGNN source")])
    if script_name == "finetune_hydragnn":
        expected.extend(["--branch-mlp", str(branch_mlp)])
    assert actual == expected
