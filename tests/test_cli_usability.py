"""Tests for repository-wide CLI usability guarantees."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
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


class _AnyConfig:
    @staticmethod
    def model_validate(raw):
        return raw


def test_validation_errors_list_one_line_per_field(tmp_path):
    invalid = tmp_path / "relax.yaml"
    invalid.write_text("mode: bogus\nmax_steps: -1\n")
    result = CliRunner().invoke(app, ["relax", str(invalid)])
    assert result.exit_code == 2
    assert "invalid ScientificRelaxationConfig (3 problem(s)):" in result.output
    assert "  mode: Input should be 'mlip', 'dft' or 'mlip-dft'" in result.output
    assert "  structure_path: Field required" in result.output
    assert "  max_steps: Input should be greater than or equal to 1" in result.output


def test_config_init_copies_templates_without_overwriting(tmp_path):
    from matsim_agents.cli.config_tools import CONFIG_KINDS, REPO

    runner = CliRunner()
    listed = runner.invoke(app, ["config", "list"])
    assert listed.exit_code == 0
    for kind, (_, example, _) in CONFIG_KINDS.items():
        assert kind in listed.output
        assert (REPO / example).is_file(), example
    target = tmp_path / "relax.yaml"
    created = runner.invoke(app, ["config", "init", "relax", str(target)])
    assert created.exit_code == 0, created.output
    assert "matsim-agents relax" in created.output
    assert target.read_text() == (REPO / CONFIG_KINDS["relax"][1]).read_text()
    again = runner.invoke(app, ["config", "init", "relax", str(target)])
    assert again.exit_code != 0
    assert "refusing to overwrite" in again.output
    unknown = runner.invoke(app, ["config", "init", "nope", str(tmp_path / "x.yaml")])
    assert unknown.exit_code == 2


def test_config_schema_exports_every_kind(tmp_path):
    from matsim_agents.cli.config_tools import CONFIG_KINDS

    runner = CliRunner()
    for kind in CONFIG_KINDS:
        result = runner.invoke(app, ["config", "schema", kind])
        assert result.exit_code == 0, (kind, result.output)
        assert "properties" in json.loads(result.output)
    written = tmp_path / "relax.schema.json"
    assert runner.invoke(app, ["config", "schema", "relax", "-o", str(written)]).exit_code == 0
    assert json.loads(written.read_text())["title"] == "ScientificRelaxationConfig"


def test_relax_prints_a_summary_and_json_on_request(monkeypatch, tmp_path):
    from matsim_agents.execution.contracts import EvidenceLevel, WorkflowStatus
    from matsim_agents.workflows import relaxation
    from matsim_agents.workflows.relaxation import (
        RelaxationMode,
        RelaxationStageResult,
        ScientificRelaxationResult,
    )

    result = ScientificRelaxationResult(
        run_id="r1",
        run_directory=str(tmp_path / "run"),
        mode=RelaxationMode.MLIP,
        status=WorkflowStatus.COMPLETE,
        stages=[
            RelaxationStageResult(
                stage="mlip",
                backend="mace",
                evidence_level=EvidenceLevel.MLIP_RELAXATION,
                input_structure_path="in.cif",
                energy_eV=-12.5,
                max_force_eV_per_A=0.01,
                steps=40,
                converged=True,
            )
        ],
        final_structure_path="out.cif",
    )
    monkeypatch.setattr(relaxation, "ScientificRelaxationConfig", _AnyConfig)
    monkeypatch.setattr(relaxation, "run_relaxation", lambda cfg: result)
    config = tmp_path / "relax.yaml"
    config.write_text("{}\n")
    runner = CliRunner()
    readable = runner.invoke(app, ["relax", str(config)])
    assert readable.exit_code == 0, readable.output
    assert "Relaxation complete (mlip)" in readable.output
    assert "mlip [mace]: converged, E=-12.5 eV" in readable.output
    assert "Final structure: out.cif" in readable.output
    as_json = runner.invoke(app, ["relax", str(config), "--json"])
    assert json.loads(as_json.output)["final_structure_path"] == "out.cif"


def test_llm_check_summary_reports_failed_stages(monkeypatch, tmp_path):
    from matsim_agents.execution.contracts import WorkflowStatus
    from matsim_agents.workflows import llm_check
    from matsim_agents.workflows.llm_check import LLMCheckResult

    result = LLMCheckResult(
        run_id="c1",
        run_directory=str(tmp_path / "run"),
        status=WorkflowStatus.FAILED,
        provider="vllm",
        model="qwen",
        stages={"server": True, "chat": False},
        failure_reason="chat timed out",
    )
    monkeypatch.setattr(llm_check, "LLMCheckConfig", _AnyConfig)
    monkeypatch.setattr(llm_check, "run_llm_check", lambda cfg: result)
    config = tmp_path / "check.yaml"
    config.write_text("{}\n")
    readable = CliRunner().invoke(app, ["llm-check", str(config)])
    assert readable.exit_code == 1
    assert "LLM check failed: vllm qwen" in readable.output
    assert "chat: failed" in readable.output
    assert "Failure: chat timed out" in readable.output


def test_debate_summary_points_to_the_transcript(monkeypatch, tmp_path):
    from matsim_agents.execution.contracts import WorkflowStatus
    from matsim_agents.workflows import debate
    from matsim_agents.workflows.debate import ScientificDebateResult

    result = ScientificDebateResult(
        run_id="d1",
        run_directory=str(tmp_path / "run"),
        status=WorkflowStatus.COMPLETE,
        hypothesis="NbTaO5 is stable",
        rounds_completed=2,
        turns=[],
        verdicts=[],
        synthesis="",
        transcript_path=str(tmp_path / "run" / "transcript.md"),
        contributions_path=str(tmp_path / "run" / "contributions.jsonl"),
    )
    monkeypatch.setattr(debate, "ScientificDebateConfig", _AnyConfig)
    monkeypatch.setattr(debate, "run_scientific_debate", lambda cfg: result)
    config = tmp_path / "debate.yaml"
    config.write_text("{}\n")
    readable = CliRunner().invoke(app, ["debate", str(config)])
    assert readable.exit_code == 0, readable.output
    assert "Debate complete: 2 round(s)" in readable.output
    assert "transcript.md" in readable.output


def test_every_registered_tool_module_has_a_main():
    import ast
    from importlib.util import find_spec

    from matsim_agents.cli.tools import TOOLS

    for name, (module, _) in TOOLS.items():
        spec = find_spec(module)
        assert spec is not None and spec.origin, (name, module)
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body
        ), (name, module)


@pytest.mark.parametrize("takes_argv", [True, False])
def test_tools_run_passes_arguments_and_exit_code(monkeypatch, takes_argv):
    import types

    from matsim_agents.cli import tools

    seen = {}
    if takes_argv:

        def main(argv=None):
            seen["args"], seen["prog"] = argv, sys.argv[0]
            return 3

    else:

        def main():
            seen["args"], seen["prog"] = sys.argv[1:], sys.argv[0]
            return 3

    fake = types.ModuleType("fake_tool")
    fake.main = main
    monkeypatch.setitem(tools.TOOLS, "fake", ("fake_tool", "test tool"))
    monkeypatch.setitem(sys.modules, "fake_tool", fake)
    result = CliRunner().invoke(app, ["tools", "run", "fake", "--dataset", "d.json", "--help"])
    assert result.exit_code == 3
    assert seen == {
        "args": ["--dataset", "d.json", "--help"],
        "prog": "matsim-agents tools run fake",
    }


def test_tools_list_and_unknown_tool():
    runner = CliRunner()
    listed = runner.invoke(app, ["tools", "list"])
    assert listed.exit_code == 0
    assert "finetune-mace" in listed.output
    unknown = runner.invoke(app, ["tools", "run", "nope"])
    assert unknown.exit_code == 2
    assert "unknown tool 'nope'" in unknown.output


def test_shell_completion_is_offered():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--install-completion" in result.output
    assert "--show-completion" in result.output
