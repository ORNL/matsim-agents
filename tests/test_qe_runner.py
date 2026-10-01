from __future__ import annotations

from types import SimpleNamespace

from matsim_agents.backends.dft import qe_relax


def test_run_pw_expands_launcher_placeholders(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "pw.in"
    input_path.touch()
    observed: dict[str, list[str]] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        return SimpleNamespace(returncode=0)

    result = SimpleNamespace(wall_time_sec=None)
    monkeypatch.setattr(qe_relax.subprocess, "run", fake_run)
    monkeypatch.setattr(qe_relax, "parse_pw_stdout", lambda *_args, **_kwargs: result)

    qe_relax.run_pw(
        str(input_path),
        str(tmp_path),
        ["bash", "wrapper.sh", "{work_dir}", "pw.x", "{input}", "1", "4", "16"],
    )

    assert observed["argv"] == [
        "bash",
        "wrapper.sh",
        str(tmp_path),
        "pw.x",
        str(input_path),
        "1",
        "4",
        "16",
    ]
