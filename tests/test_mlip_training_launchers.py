"""Exercise the site fine-tuning launchers without Slurm, PBS, or ML packages."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = {
    "perlmutter": {"hydragnn", "uma", "mace"},
    "aurora": {"hydragnn", "uma", "mace"},
    "frontier": {"hydragnn", "uma", "mace"},
}
SCHEDULER = {"perlmutter": "srun", "frontier": "srun", "aurora": "mpiexec"}
FINE_TUNE = [
    "-m",
    "matsim_agents.backends.mlip.training.finetune_x",
    "--dataset",
    "labelled data.extxyz",
    "--output-dir",
    "out dir",
    "--lora",
]


def _launcher(platform: str) -> Path:
    return ROOT / "deployments" / platform / "launchers" / f"_mlip-train-step-{platform}.sh"


def _site(tmp_path: Path) -> dict[str, str]:
    """Stub scheduler, python, module system, and backend venvs; return the job env."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for scheduler in ("srun", "mpiexec"):
        stub = bin_dir / scheduler
        stub.write_text(
            "#!/bin/bash\n"
            f'printf "%s\\n" "$@" > "{tmp_path}/{scheduler}.args"\n'
            'while [[ "$1" != python && "$1" != env ]]; do shift; done\n'
            'exec "$@"\n'
        )
        stub.chmod(0o755)
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "json.dump({'argv': sys.argv[1:], 'venv': os.environ.get('TEST_VENV', 'driver'),\n"
        "           'pythonpath': os.environ.get('PYTHONPATH', ''),\n"
        "           'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "           'rocr': os.environ.get('ROCR_VISIBLE_DEVICES')},\n"
        "          open(os.environ['CAPTURE'], 'w'))\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE": str(tmp_path / "python.json"),
        "BASH_ENV": str(tmp_path / "bash-env"),
        "MATSIM_MODEL_ARTIFACTS_ROOT": str(tmp_path / "artifacts"),
        "ZE_AFFINITY_MASK": "3",
    }
    (tmp_path / "bash-env").write_text("module() { return 0; }\nml() { return 0; }\n")
    for name, variable in (("venv-uma", "MATSIM_FAIRCHEM_VENV"), ("venv-mace", "MATSIM_MACE_VENV")):
        venv = tmp_path / name
        (venv / "bin").mkdir(parents=True)
        (venv / "bin/activate").write_text(f"export TEST_VENV={name}\n")
        env[variable] = str(venv)
    env.pop("TEST_VENV", None)
    return env


def _run(platform: str, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_launcher(platform)), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("platform", sorted(SUPPORTED))
@pytest.mark.parametrize("backend", ["hydragnn", "uma", "mace", "orb"])
def test_launcher_runs_the_fine_tuner_unchanged_in_the_backend_environment(
    tmp_path, platform, backend
):
    env = _site(tmp_path)
    proc = _run(platform, [backend, "1", "1", *FINE_TUNE], env)
    capture = tmp_path / "python.json"
    if backend not in SUPPORTED[platform]:
        assert proc.returncode == 2
        assert "unsupported MLIP backend" in proc.stderr
        assert not capture.exists()
        return
    assert proc.returncode == 0, proc.stderr
    result = json.loads(capture.read_text())
    assert result["argv"] == FINE_TUNE
    assert result["venv"] == {"hydragnn": "driver", "uma": "venv-uma", "mace": "venv-mace"}[backend]
    assert result["pythonpath"].split(os.pathsep)[0] == str(ROOT / "src")


@pytest.mark.parametrize("platform", sorted(SUPPORTED))
def test_launcher_rejects_arguments_without_a_module(tmp_path, platform):
    proc = _run(platform, ["hydragnn", "1", "1", "train.py"], _site(tmp_path))
    assert proc.returncode == 2
    assert "Usage:" in proc.stderr


@pytest.mark.parametrize(
    ("platform", "mask_variable"), [("perlmutter", "cuda"), ("frontier", "rocr")]
)
def test_slurm_launchers_apply_the_training_placement(tmp_path, platform, mask_variable):
    env = {
        **_site(tmp_path),
        "MATSIM_ASSIGNED_NODES": "nid001",
        "MATSIM_DEVICE_MASK": "2",
        "MATSIM_NODE_DEVICES": "4",
    }
    proc = _run(platform, ["hydragnn", "1", "1", *FINE_TUNE], env)
    assert proc.returncode == 0, proc.stderr
    srun = (tmp_path / "srun.args").read_text().splitlines()
    assert srun[: srun.index("python")][:3] == ["--overlap", "--nodelist", "nid001"]
    assert srun[srun.index("-N") + 1] == "1"
    assert srun[srun.index("-n") + 1] == "1"
    assert "--gpus-per-node=4" in srun
    assert "--gpu-bind=none" in srun
    assert json.loads((tmp_path / "python.json").read_text())[mask_variable] == "2"


def test_slurm_launcher_uses_whole_steps_without_a_partial_mask(tmp_path):
    proc = _run("perlmutter", ["mace", "2", "4", *FINE_TUNE], _site(tmp_path))
    assert proc.returncode == 0, proc.stderr
    srun = (tmp_path / "srun.args").read_text().splitlines()
    assert "--overlap" not in srun
    assert srun[srun.index("-N") + 1] == "2"
    assert srun[srun.index("-n") + 1] == "8"
    assert "--gpus-per-node=4" in srun
    assert "--gpu-bind=closest" in srun


def test_aurora_launcher_places_ranks_on_the_assigned_hosts_and_tiles(tmp_path):
    env = {**_site(tmp_path), "MATSIM_ASSIGNED_NODES": "x1001"}
    proc = _run("aurora", ["uma", "2", "1", *FINE_TUNE], env)
    assert proc.returncode == 0, proc.stderr
    mpiexec = (tmp_path / "mpiexec.args").read_text().splitlines()
    assert mpiexec[: mpiexec.index("python")] == ["--hosts", "x1001", "-n", "2", "--ppn", "1"]

    env.pop("ZE_AFFINITY_MASK")
    proc = _run("aurora", ["uma", "1", "1", *FINE_TUNE], env)
    assert proc.returncode != 0
    assert "XPU tiles" in proc.stderr


def test_trainer_command_reaches_the_fine_tuner_through_the_launcher(tmp_path):
    from matsim_agents.backends.mlip.config import HydraGNNConfig
    from matsim_agents.workflows.active_learning.trainer import _fine_tune_argv

    mlip = HydraGNNConfig(
        logdir=tmp_path / "gfm logdir",
        checkpoint="incumbent.pk",
        charge=-2.5,
    )
    trainer = SimpleNamespace(
        epochs_per_iter=2, finetune_mode="lora", train_launcher=_launcher("perlmutter")
    )
    argv = _fine_tune_argv(
        trainer,
        mlip,
        SimpleNamespace(node_count=1, ranks_per_node=1, total_ranks=1),
        dataset=tmp_path / "dataset.extxyz",
        out_dir=tmp_path / "out",
        references=tmp_path / "references.json",
        seed=0,
    )
    assert argv[:5] == ["bash", str(_launcher("perlmutter")), "hydragnn", "1", "1"]
    proc = subprocess.run(argv, env=_site(tmp_path), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert json.loads((tmp_path / "python.json").read_text())["argv"] == argv[5:]


@pytest.mark.parametrize(
    ("cuda", "xpu", "expected"),
    [(True, True, "cuda"), (False, True, "xpu"), (False, False, "cpu"), (False, None, "cpu")],
)
def test_default_torch_device_prefers_cuda_then_xpu(monkeypatch, cuda, xpu, expected):
    import torch

    from matsim_agents.backends.mlip.devices import default_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    if xpu is None:
        monkeypatch.delattr(torch, "xpu", raising=False)
    else:
        monkeypatch.setattr(torch, "xpu", SimpleNamespace(is_available=lambda: xpu), raising=False)
    assert default_torch_device() == expected
