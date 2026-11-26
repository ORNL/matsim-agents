"""LAMMPS adapter: discovery, input writing, process handling and log parsing with a fake binary."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest
from ase.build import bulk

from matsim_agents.backends.external.lammps import availability as av
from matsim_agents.backends.external.lammps.inputs import (
    DATA_NAME,
    INPUT_NAME,
    INPUT_RECORD,
    LammpsInputError,
    LammpsTask,
    Potential,
    potential_from_file,
    write_inputs,
)
from matsim_agents.backends.external.lammps.parser import LammpsParseError, parse_log
from matsim_agents.backends.external.lammps.runner import run_local, submit_slurm
from matsim_agents.execution.slurm_jobs import CommandResult

HELP = """
Large-scale Atomic/Molecular Massively Parallel Simulator - 29 Aug 2024 - Update 1

OS: Linux "Test" x86_64

Compiler: GNU C++ 14.2.0 with OpenMP 4.5
MPI v3.1: MPI VERSION    : CRAY MPICH version 8.1.32

Installed packages:

KSPACE MANYBODY MOLECULE
RIGID

List of individual style options included in this LAMMPS executable
"""

LOG = """LAMMPS (29 Aug 2024 - Update 1)
Step Temp PotEng KinEng TotEng Press Volume
         0   0   -472.5   0   -472.5   12.0   857.0
Loop time of 0.001 on 1 procs for 0 steps with 54 atoms
Step Temp PotEng KinEng TotEng Press Volume
         0   0   -472.5   0   -472.5   12.0   857.0
        10   0   -473.0   0   -473.0    1.0   857.0
Loop time of 0.002 on 1 procs for 10 steps with 54 atoms
MATSIM_LAMMPS_DONE
Total wall time: 0:00:00
"""


def _fake(tmp_path: Path, body: str, name: str = "lmp") -> Path:
    exe = tmp_path / "bin" / name
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text(f"#!{sys.executable}\nimport sys\n{body}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


def _env(exe: Path) -> dict[str, str]:
    return {"PATH": str(exe.parent)}


def test_not_installed_without_binary(tmp_path):
    result = av.check_availability(environ={"PATH": str(tmp_path)})
    assert result.status is av.Status.NOT_INSTALLED
    assert result.executable is None


def test_probe_parses_version_packages_and_build(tmp_path):
    exe = _fake(tmp_path, f"print({HELP!r})")
    result = av.check_availability(environ=_env(exe))
    assert result.status is av.Status.READY
    assert result.version == "29 Aug 2024 - Update 1"
    assert result.packages == ("KSPACE", "MANYBODY", "MOLECULE", "RIGID")
    assert result.has_package("manybody") and not result.has_package("ML-IAP")
    assert result.build["compiler"].startswith("GNU C++")
    assert len(result.executable_sha256) == 64
    assert av.check_availability(environ=_env(exe), invoke=False).status is av.Status.INSTALLED


def test_probe_without_banner_is_execution_failed(tmp_path):
    exe = _fake(tmp_path, "print('not lammps'); sys.exit(0)")
    assert av.check_availability(environ=_env(exe)).status is av.Status.EXECUTION_FAILED


def test_explicit_env_executable_wins(tmp_path):
    exe = _fake(tmp_path, f"print({HELP!r})", name="lmp_custom")
    env = {"PATH": "", av.ENV_EXECUTABLE: str(exe)}
    assert av.find_executable(environ=env) == exe.resolve()


def test_potential_is_required_and_files_must_exist(tmp_path):
    with pytest.raises(LammpsInputError):
        Potential(pair_style="eam/alloy", pair_coeff=())
    with pytest.raises(LammpsInputError):
        Potential(pair_style="lj/cut 2.5", pair_coeff=("* * 1 1\nrun 1000",))
    pot = potential_from_file("eam/alloy", tmp_path / "missing.eam.alloy", ["W"])
    with pytest.raises(LammpsInputError, match="not found"):
        write_inputs(bulk("W", "bcc", a=3.165), tmp_path / "run", pot, LammpsTask())


def test_task_validation():
    with pytest.raises(LammpsInputError):
        LammpsTask("nve")
    with pytest.raises(LammpsInputError):
        LammpsTask("nvt", steps=10)
    with pytest.raises(LammpsInputError, match="seed"):
        LammpsTask("nvt", steps=10, temperature_k=300.0)


def test_write_inputs_records_hashes_and_species_order(tmp_path):
    pot_file = tmp_path / "W.eam.alloy"
    pot_file.write_text("fake potential\n")
    atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat(2)
    script = write_inputs(
        atoms,
        tmp_path / "run",
        potential_from_file("eam/alloy", pot_file, ["W"]),
        LammpsTask("nvt", steps=10, temperature_k=300.0, seed=3),
    )
    text = script.read_text()
    assert script.name == INPUT_NAME
    assert f"read_data {DATA_NAME}" in text and "pair_style eam/alloy" in text
    assert "velocity all create 300.0" in text and "run 10" in text
    assert "MATSIM_LAMMPS_DONE" in text
    record = (tmp_path / "run" / INPUT_RECORD).read_text()
    assert '"species_order": [\n    "W"\n  ]' in record and str(pot_file.resolve()) in record
    second = write_inputs(
        atoms,
        tmp_path / "run2",
        potential_from_file("eam/alloy", pot_file, ["W"]),
        LammpsTask("nvt", steps=10, temperature_k=300.0, seed=3),
    )
    assert second.read_text() == text


def test_parse_log_blocks():
    blocks = parse_log(LOG)
    assert len(blocks) == 2
    assert blocks[-1].column("Step") == (0.0, 10.0)
    assert blocks[-1].last("PotEng") == -473.0
    with pytest.raises(LammpsParseError):
        blocks[-1].column("Missing")


def _staged(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    write_inputs(
        bulk("W", "bcc", a=3.165, cubic=True),
        run,
        Potential(pair_style="lj/cut 2.5", pair_coeff=("* * 1.0 1.0",)),
        LammpsTask(),
    )
    return run


def test_run_local_success_needs_both_markers(tmp_path):
    run = _staged(tmp_path)
    good = _fake(tmp_path, f"open(sys.argv[sys.argv.index('-log') + 1], 'w').write({LOG!r})")
    outcome = run_local(good, run, 30)
    assert outcome.process_succeeded and outcome.argv[-4:] == (
        "-in",
        INPUT_NAME,
        "-log",
        "log.lammps",
    )
    assert (run / "run_record.json").is_file()
    truncated = LOG.replace("MATSIM_LAMMPS_DONE\n", "")
    bad = _fake(
        tmp_path, f"open(sys.argv[sys.argv.index('-log') + 1], 'w').write({truncated!r})", "lmp2"
    )
    assert not run_local(bad, run, 30).process_succeeded


def test_run_local_failure_and_timeout(tmp_path):
    run = _staged(tmp_path)
    failing = _fake(tmp_path, "sys.stderr.write('ERROR: boom'); sys.exit(1)")
    outcome = run_local(failing, run, 30)
    assert outcome.returncode == 1 and not outcome.process_succeeded
    assert "boom" in (run / "stderr.txt").read_text()
    slow = _fake(tmp_path, "import time; time.sleep(30)", "slow")
    outcome = run_local(slow, run, 1)
    assert outcome.timed_out and outcome.returncode is None


def test_run_local_requires_staged_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_local(Path("/bin/true"), tmp_path, 5)


def test_cli_check_reports_status_and_exit_code(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    monkeypatch.delenv(av.ENV_EXECUTABLE, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    missing = CliRunner().invoke(app, ["external", "lammps", "check"])
    assert missing.exit_code == 1 and "NOT_INSTALLED" in missing.output
    exe = _fake(tmp_path, f"print({HELP!r})")
    ready = CliRunner().invoke(app, ["external", "lammps", "check", "--executable", str(exe)])
    assert ready.exit_code == 0 and "READY" in ready.output


def _real_eam() -> tuple[Path, Path] | None:
    exe = av.find_executable()
    root = os.environ.get(av.ENV_POTENTIALS)
    if exe is None or not root or not (Path(root) / "W_zhou.eam.alloy").is_file():
        return None
    return exe, Path(root) / "W_zhou.eam.alloy"


@pytest.mark.skipif(_real_eam() is None, reason="needs a LAMMPS binary and W_zhou.eam.alloy")
def test_real_lammps_matches_ase_eam(tmp_path):
    import numpy as np
    from ase.calculators.eam import EAM

    from matsim_agents.backends.external.lammps.inputs import FINAL_DUMP, LOG_NAME
    from matsim_agents.backends.external.lammps.parser import parse_log_file, read_final_atoms
    from matsim_agents.core.seeding import numpy_rng

    exe, pot = _real_eam()
    atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    atoms.positions += numpy_rng(7, "lammps-xcheck").normal(0, 0.05, atoms.positions.shape)
    write_inputs(atoms, tmp_path, potential_from_file("eam/alloy", pot, ["W"]), LammpsTask())
    assert run_local(
        exe, tmp_path, 120, env=dict(os.environ, OMP_NUM_THREADS="1")
    ).process_succeeded
    ref = atoms.copy()
    ref.calc = EAM(potential=str(pot), form="alloy")
    energy = parse_log_file(tmp_path / LOG_NAME)[-1].last("PotEng")
    forces = read_final_atoms(tmp_path / FINAL_DUMP).get_forces()
    assert abs(energy - ref.get_potential_energy()) / len(atoms) < 1e-6
    assert np.abs(forces - ref.get_forces()).max() < 1e-4


def test_submit_slurm_uses_job_manifest(tmp_path):
    run = _staged(tmp_path)
    calls = []

    def runner(argv):
        calls.append(list(argv))
        return CommandResult(0, "4242\n")

    outcome = submit_slurm(
        Path("/opt/lmp"),
        run,
        name="lmp-test",
        manifest=tmp_path / "m.json",
        account="abc123",
        runner=runner,
    )
    assert outcome.mode == "slurm" and outcome.job_id == "4242" and outcome.submitted
    assert outcome.argv[:2] == ("srun", "/opt/lmp")
    assert calls and os.path.basename(calls[0][0]) == "sbatch"
