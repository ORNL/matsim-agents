"""Slurm COMPLETED is not QE success: launchers must surface a failing pw.x.

These tests run the real launcher script with a stub ``srun``, ``module`` and ``pw.x``.
They verify launcher control flow only; no Quantum ESPRESSO run takes place.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTIER = ROOT / "deployments" / "frontier"
KCONV = FRONTIER / "campaigns" / "qe-1024" / "qe_kconv_job.sbatch"
PINNED = [
    FRONTIER / "launchers" / "_qe-step-frontier.sh",
    FRONTIER / "launchers" / "run-pw-gpu-frontier.sh",
    FRONTIER / "jobs" / "job-qe-gpu-smoke-frontier.sh",
    FRONTIER / "campaigns" / "qe-1024" / "qe_1024_job.sbatch",
    KCONV,
]


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _run_kconv(tmp_path: Path, pw_body: str) -> subprocess.CompletedProcess:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _exe(bindir / "srun", '#!/bin/bash\nwhile [[ "$1" == -* ]]; do shift; done\nexec "$@"\n')
    pw = _exe(tmp_path / "pw.x", "#!/bin/bash\n" + pw_body)
    for mesh in (4, 6, 8):
        d = tmp_path / "kconv" / f"k{mesh}"
        d.mkdir(parents=True)
        (d / "pw.in").write_text("&control\n/\n")
    # `module` is a shell function on the real system; export a no-op one here.
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "PW_BIN": str(pw),
        "KCONV_DIR": str(tmp_path / "kconv"),
        "SLURM_JOB_NUM_NODES": "1",
        "BASH_FUNC_module%%": "() { :; }",
    }
    return subprocess.run(
        ["bash", str(KCONV)], env=env, capture_output=True, text=True, timeout=60, check=False
    )


def test_segfaulting_pw_makes_the_job_fail(tmp_path):
    result = _run_kconv(tmp_path, 'echo "     FFT dimensions"\nkill -SEGV $$\n')
    assert result.returncode != 0
    assert result.stderr.count("FAILED exit 139") == 3
    assert "OK exit" not in result.stdout


def test_exit_zero_without_a_converged_scf_is_still_a_failure(tmp_path):
    result = _run_kconv(tmp_path, 'echo "starting"\nexit 0\n')
    assert result.returncode != 0
    assert "FAILED exit 0" in result.stderr


def test_converged_pw_with_job_done_succeeds(tmp_path):
    body = 'echo "     convergence has been achieved in   9 iterations"\necho "     JOB DONE."\n'
    result = _run_kconv(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("OK exit 0") == 3


@pytest.mark.parametrize("script", PINNED, ids=lambda p: p.name)
def test_launcher_pins_the_build_toolchain(script):
    loads = [
        ln.strip() for ln in script.read_text().splitlines() if ln.strip().startswith("module load")
    ]
    assert "cpe/24.11" in loads[0]
    assert any("PrgEnv-cray" in ln for ln in loads[1:])
