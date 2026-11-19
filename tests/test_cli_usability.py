"""Tests for repository-wide CLI usability guarantees."""

from __future__ import annotations

import json
import subprocess
import sys

from typer.testing import CliRunner

from matsim_agents.cli import app, system
from matsim_agents.cli._app import package_version


def test_cli_help_does_not_load_langgraph_or_print_warnings():
    probe = (
        "import sys; from matsim_agents.cli import app; "
        "from typer.testing import CliRunner; "
        "assert CliRunner().invoke(app, ['--help']).exit_code == 0; "
        "print(sorted(m for m in ('langgraph', 'langchain_core') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-W", "default", "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
    assert "Warning" not in result.stderr


def test_root_help_and_version_are_available_without_a_command():
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0, help_result.output
    assert "workflow list" in help_result.output
    assert "--debug" in help_result.output

    version_result = runner.invoke(app, ["--version"])
    assert version_result.exit_code == 0, version_result.output
    assert version_result.output.strip() == package_version()


def test_doctor_json_does_not_load_or_run_a_workflow(monkeypatch):
    class Capability:
        def __init__(self, kind, name, available, missing=()):
            self.kind = kind
            self.name = name
            self.available = available
            self.missing = missing

        def model_dump(self, mode="json"):
            return {
                "kind": self.kind,
                "name": self.name,
                "available": self.available,
                "missing": list(self.missing),
            }

    monkeypatch.setattr(
        system,
        "check_all",
        lambda: [Capability("mlip", "mace", True), Capability("llm", "openai", False, ("key",))],
    )
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["python_supported"] is True
    assert report["backends"]["mlip"] == {"available": 1, "configured": 1}
    assert report["required"] == {}


def test_doctor_required_backend_failure_is_explicit(monkeypatch):
    class Capability:
        kind = "llm"
        name = "openai"
        available = False
        missing = ("env OPENAI_API_KEY",)

        def model_dump(self, mode="json"):
            return {
                "kind": self.kind,
                "name": self.name,
                "available": False,
                "missing": list(self.missing),
            }

    monkeypatch.setattr(system, "check_all", lambda: [Capability()])
    result = CliRunner().invoke(app, ["doctor", "--require", "openai"])
    assert result.exit_code == 1


def test_unhandled_configuration_error_is_friendly(tmp_path):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("{}\n")
    result = CliRunner().invoke(app, ["relax", str(invalid)])
    assert result.exit_code == 2
    assert "Error:" in result.output
    assert "--debug" in result.output
    assert "Traceback" not in result.output

    debug_result = CliRunner().invoke(app, ["--debug", "relax", str(invalid)])
    assert debug_result.exit_code != 0
    assert debug_result.exception is not None
