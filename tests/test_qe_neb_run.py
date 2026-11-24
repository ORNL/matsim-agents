"""B2.3: neb.x execution with mock executables (no QE needed)."""

import hashlib
import json
import stat

import pytest

from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.backends.dft.qe_neb_run import build_neb_argv, run_neb

SPECIES = ("Mo", "W")


def _cfg(workdir, **over):
    base = {
        "n_images": 5,
        "species": SPECIES,
        "pseudo_dir": "/pseudos",
        "pseudopotentials": {"Mo": "Mo.upf", "W": "W.upf"},
        "ecutwfc_ry": 50.0,
        "ecutrho_ry": 400.0,
        "occupations": "fixed",
        "kpts": (1, 1, 1),
        "nspin": 1,
        "workdir": str(workdir),
    }
    base.update(over)
    return QENebConfig(**base)


def _mock(tmp_path, body, name="neb.x"):
    exe = tmp_path / name
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


def _input(workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "neb.in"
    path.write_text("BEGIN\nEND\n")
    return path


def test_argv_uses_inp_flag_and_never_stdin(tmp_path):
    cfg = _cfg(tmp_path, neb_executable="neb.x")
    argv = build_neb_argv(cfg, tmp_path / "neb.in", str(tmp_path))
    assert argv == ["neb.x", "-inp", "neb.in"]
    assert "<" not in argv


def test_argv_includes_launcher_parallel_flags_and_extras(tmp_path):
    cfg = _cfg(
        tmp_path,
        launcher=("srun", "-n", "8"),
        nimage=4,
        npool=2,
        extra_args=("-ndiag", "1"),
    )
    argv = build_neb_argv(cfg, tmp_path / "neb.in", str(tmp_path))
    assert argv == [
        "srun",
        "-n",
        "8",
        "neb.x",
        "-nimage",
        "4",
        "-npool",
        "2",
        "-ndiag",
        "1",
        "-inp",
        "neb.in",
    ]


def test_input_outside_workdir_is_passed_absolute(tmp_path):
    cfg = _cfg(tmp_path / "work")
    other = tmp_path / "elsewhere" / "neb.in"
    other.parent.mkdir()
    other.write_text("x")
    assert build_neb_argv(cfg, other, str(tmp_path / "work"))[-1] == str(other.resolve())


def test_successful_run_is_recorded_but_not_called_converged(tmp_path):
    work = tmp_path / "w"
    exe = _mock(tmp_path, 'echo "args: $@"\necho "to stderr" >&2\nexit 0\n')
    cfg = _cfg(work, neb_executable=exe)
    path = _input(work)
    record = run_neb(cfg, path, timeout_sec=30)
    assert record.status == "exited_zero" and record.exit_code == 0
    assert record.convergence_assessed is False
    assert record.argv == (exe, "-inp", "neb.in")
    log = (work / "neb.out").read_text()
    assert "args: -inp neb.in" in log and "to stderr" in log  # both streams captured
    assert record.stdout_sha256 == hashlib.sha256(log.encode()).hexdigest()
    assert record.input_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record.config_hash == cfg.config_hash()
    assert record.cwd == str(work.resolve()) and record.streams == "stderr merged into stdout"
    assert record.started_at <= record.finished_at
    saved = json.loads((work / "neb.out.record.json").read_text())
    assert saved["status"] == "exited_zero" and saved["exit_code"] == 0


def test_nonzero_exit_is_recorded_not_raised(tmp_path):
    work = tmp_path / "w"
    exe = _mock(tmp_path, "echo failing\nexit 3\n")
    record = run_neb(_cfg(work, neb_executable=exe), _input(work))
    assert record.status == "exited_nonzero" and record.exit_code == 3
    assert "failing" in (work / "neb.out").read_text()


def test_timeout_is_distinguished(tmp_path):
    work = tmp_path / "w"
    exe = _mock(tmp_path, "echo started\nsleep 30\n")
    record = run_neb(_cfg(work, neb_executable=exe), _input(work), timeout_sec=0.5)
    assert record.status == "timed_out" and record.exit_code == 124


def test_exit_124_without_timeout_is_just_nonzero(tmp_path):
    work = tmp_path / "w"
    exe = _mock(tmp_path, "exit 124\n")
    record = run_neb(_cfg(work, neb_executable=exe), _input(work))
    assert record.status == "exited_nonzero" and record.exit_code == 124


def test_missing_executable_gives_not_started_record(tmp_path):
    work = tmp_path / "w"
    cfg = _cfg(work, neb_executable=str(tmp_path / "does-not-exist"))
    record = run_neb(cfg, _input(work))
    assert record.status == "not_started" and record.exit_code is None
    assert "FileNotFoundError" in record.error
    assert (work / "neb.out.record.json").is_file()


def test_existing_output_is_never_overwritten(tmp_path):
    work = tmp_path / "w"
    exe = _mock(tmp_path, "echo one\n")
    cfg = _cfg(work, neb_executable=exe)
    path = _input(work)
    run_neb(cfg, path)
    before = (work / "neb.out").read_bytes()
    with pytest.raises(FileExistsError):
        run_neb(cfg, path)
    assert (work / "neb.out").read_bytes() == before
    again = run_neb(cfg, path, stdout_name="neb.stage2.out")
    assert again.status == "exited_zero" and (work / "neb.stage2.out").is_file()


def test_missing_input_and_bad_timeout(tmp_path):
    work = tmp_path / "w"
    cfg = _cfg(work, neb_executable=_mock(tmp_path, "exit 0\n"))
    with pytest.raises(FileNotFoundError):
        run_neb(cfg, work / "neb.in")
    path = _input(work)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="timeout"):
            run_neb(cfg, path, timeout_sec=bad)
    assert not (work / "neb.out").exists()


def test_environment_is_passed_through_the_launcher_hook(tmp_path):
    work = tmp_path / "w"
    seen = {}

    def fake(argv, *, cwd, stdout, timeout_sec, env):
        seen.update(argv=argv, cwd=cwd, env=env, timeout=timeout_sec)
        stdout.write("fake output\n")
        return 0

    cfg = _cfg(work)
    record = run_neb(cfg, _input(work), env={"OMP_NUM_THREADS": "2"}, timeout_sec=9, launcher=fake)
    assert seen["env"] == {"OMP_NUM_THREADS": "2"} and seen["timeout"] == 9
    assert seen["argv"] == ["neb.x", "-inp", "neb.in"] and record.status == "exited_zero"


def test_no_stray_files_after_run(tmp_path):
    work = tmp_path / "w"
    run_neb(_cfg(work, neb_executable=_mock(tmp_path, "exit 0\n")), _input(work))
    assert sorted(p.name for p in work.iterdir()) == ["neb.in", "neb.out", "neb.out.record.json"]
