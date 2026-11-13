import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from matsim_agents.cli import app
from matsim_agents.cli.workflow_catalog import ADAPTERS, WorkflowConfig, load_workflow
from matsim_agents.cli.workflows import _check, _command, _validate_native_cli
from matsim_agents.execution.contracts import EvidenceLevel, ProvenanceRecord
from matsim_agents.execution.locking import exclusive_lock
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.execution.workflow_record import WorkflowInvocation, read_invocation


@pytest.fixture
def workflow_yaml(tmp_path):
    path = tmp_path / "workflow.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "workflow": "objective",
                "arguments": ["Find a Cu structure"],
                "output_root": str(tmp_path / "records"),
            }
        )
    )
    return path


@pytest.fixture
def native_process(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if kwargs.get("stdout") is not None:
            kwargs["stdout"].write("Native workflow result: no scientific convergence asserted.\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("matsim_agents.cli.workflows.subprocess.run", run)

    def execute(command, output, emit, *, cwd, env):
        result = run(command, stdout=output)
        return result.returncode

    monkeypatch.setattr("matsim_agents.cli.workflows.run_native_command", execute)
    return calls


def test_check_parses_native_cli_without_running(workflow_yaml, native_process):
    result = CliRunner().invoke(app, ["workflow", "check", str(workflow_yaml)])
    assert result.exit_code == 0, result.output
    assert "Preflight passed" in result.output
    assert native_process == []
    assert not (workflow_yaml.parent / "records").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--tier", "misspelled"],
        ["--case", "not-a-case"],
        ["/nonexistent/suite"],
    ],
)
def test_known_answer_preflight_rejects_invalid_suite_requests(arguments, tmp_path, native_process):
    cfg = WorkflowConfig(workflow="known-answers", arguments=arguments)
    with pytest.raises((ValueError, OSError)):
        _check(cfg)
    assert native_process == []


@pytest.mark.parametrize(
    "workflow,arguments",
    [
        ("objective", ["Find Cu"]),
        ("plan", ["Find Cu"]),
        ("supervisor", ["Cu"]),
        ("chat", ["--llm-only"]),
        ("known-answers", []),
    ],
)
def test_advanced_argument_adapters_parse_without_invoking(workflow, arguments, native_process):
    adapter = ADAPTERS[workflow]
    _validate_native_cli([*adapter.command, *arguments])
    assert native_process == []


@pytest.mark.parametrize(
    "name", ["explore", "relax", "characterize", "debate", "llm-check", "al", "e1", "agent-eval"]
)
def test_configured_native_adapter_arguments_parse(name, tmp_path, native_process):
    path = tmp_path / "native.yaml"
    path.write_text("{}")
    adapter = ADAPTERS[name]
    arguments = [*adapter.command, str(path)]
    if name == "e1":
        arguments.extend(["--output", str(tmp_path / "e1")])
    _validate_native_cli(arguments)
    assert native_process == []


def test_run_status_summary_and_report(workflow_yaml, native_process):
    result = CliRunner().invoke(app, ["workflow", "run", str(workflow_yaml)])
    assert result.exit_code == 0, result.output
    directory = next((workflow_yaml.parent / "records").iterdir())
    record = read_invocation(directory)
    assert record.state == "completed"
    assert native_process[0][-2:] == ["run", "Find a Cu structure"]
    report = (directory / "report.md").read_text()
    assert "Invocation completion is not scientific convergence" in report
    assert "workflow-output.log" in report
    assert (directory / "workflow-summary.json").is_file()
    for action in ("status", "summary"):
        result = CliRunner().invoke(app, ["workflow", action, str(directory)])
        assert result.exit_code == 0, result.output
    assert len(native_process) == 1


def test_failed_native_command_is_persisted(workflow_yaml, monkeypatch):
    def fail(command, output, emit, *, cwd, env):
        output.write("native scientific backend failed\n")
        return 7

    monkeypatch.setattr("matsim_agents.cli.workflows.run_native_command", fail)
    result = CliRunner().invoke(app, ["workflow", "run", str(workflow_yaml)])
    assert result.exit_code == 1
    record = read_invocation(next((workflow_yaml.parent / "records").iterdir()))
    assert record.state == "failed" and record.return_code == 7
    assert "exited 7" in record.failure_reason


def test_nonresumable_workflows_are_never_replayed(workflow_yaml, native_process):
    CliRunner().invoke(app, ["workflow", "run", str(workflow_yaml)])
    directory = next((workflow_yaml.parent / "records").iterdir())
    result = CliRunner().invoke(app, ["workflow", "resume", str(directory)])
    assert result.exit_code == 1
    assert "no native resume contract" in result.output
    assert len(native_process) == 1


def test_e3_approval_flag_is_not_silently_applied_to_other_workflows(workflow_yaml, native_process):
    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "run",
            str(workflow_yaml),
            "--approve-execution",
        ],
    )
    assert result.exit_code == 1
    assert "only for E3 presets" in result.output
    assert native_process == []


@pytest.mark.parametrize(
    "change",
    [
        {"workflow": "unknown"},
        {"arguments": ["Find Cu", "--unknown-option"]},
        {"config": "missing.yaml"},
        {"unexpected": True},
    ],
)
def test_invalid_configuration_does_not_dispatch(workflow_yaml, change, native_process):
    raw = yaml.safe_load(workflow_yaml.read_text())
    workflow_yaml.write_text(yaml.safe_dump({**raw, **change}))
    result = CliRunner().invoke(app, ["workflow", "run", str(workflow_yaml)])
    assert result.exit_code != 0, result.output
    assert native_process == []


def test_presets_copy_annotations_and_rebase_paths(tmp_path, native_process):
    for preset in ("e3-mlip", "e3-dft", "explore", "debate"):
        path = tmp_path / f"{preset}.yaml"
        result = CliRunner().invoke(app, ["workflow", "init", preset, str(path)])
        assert result.exit_code == 0, result.output
        assert path.read_text().startswith("#")
        cfg = load_workflow(path)
        assert cfg.workflow == preset
        if cfg.settings is not None:
            assert cfg.settings.manifest.is_absolute()
        else:
            assert cfg.config.is_absolute()
        again = CliRunner().invoke(app, ["workflow", "init", preset, str(path)])
        assert again.exit_code == 1
        assert "overwrite" in again.output
    assert native_process == []


@pytest.fixture
def mlip_preset(tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text("{}")
    resource = tmp_path / "budget.json"
    resource.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cap_gpu_seconds": 2_764_800,
                "reservations": {"123": {"reserved_gpu_seconds": 460_800}},
            }
        )
    )
    return WorkflowConfig.model_validate(
        {
            "workflow": "e3-mlip",
            "settings": {
                "manifest": str(manifest),
                "resource_state": str(resource),
                "output_root": str(tmp_path / "science"),
                "material": "nb_ta_o",
            },
        }
    )


def test_e3_mlip_requires_approval_allocation_and_reservation(mlip_preset, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="approve-execution"):
        _command(mlip_preset, approved=False)
    with pytest.raises(ValueError, match="Slurm allocation"):
        _command(mlip_preset, approved=True)
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    with pytest.raises(ValueError, match="full 16-node/two-hour"):
        _command(mlip_preset, approved=True)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    command = _command(mlip_preset, approved=True)
    assert command[2:4] == ["benchmarks.e3.production", "run-campaign"]
    assert "--workers" in command and "64" in command
    # Command construction never executes the campaign.


def test_e3_mlip_check_uses_existing_protocol_validator(mlip_preset, native_process):
    assert _check(mlip_preset)
    assert native_process[0][2:4] == ["benchmarks.e3.production", "validate"]
    assert "run-replicate" not in native_process[0]


def test_native_e3_protocol_validation_never_executes_campaign(monkeypatch, capsys):
    from benchmarks.e3 import production
    from matsim_agents.cli.workflows import REPO

    def unexpected(*args, **kwargs):
        pytest.fail("protocol validation must not execute or load a model")

    monkeypatch.setattr(production, "run_replicate", unexpected)
    monkeypatch.setattr(production, "_verified_checkpoint_sha256", unexpected)
    assert (
        production.main(
            [
                "validate",
                "--manifest",
                str(REPO / "benchmarks/e3/manifests/e3-mlip-production.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["execution_performed"] is False


def test_e3_static_check_from_outside_the_checkout(tmp_path, mlip_preset):
    from matsim_agents.cli.workflows import REPO

    mlip_preset.settings.manifest = REPO / "benchmarks/e3/manifests/e3-mlip-production.json"
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(mlip_preset.model_dump(mode="json")))
    result = subprocess.run(
        [sys.executable, "-m", "matsim_agents.cli", "workflow", "check", str(path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Preflight passed" in result.stdout
    assert not (tmp_path / "science").exists()


@pytest.fixture
def dft_preset(tmp_path):
    path = tmp_path / "vasp.json"
    path.write_text(
        json.dumps(
            {
                "batch_id": "E3-round1-bcc-01",
                "cases": [{"calculation_id": "E3-R1-BCC-0001"}],
            }
        )
    )
    return WorkflowConfig.model_validate(
        {
            "workflow": "e3-dft",
            "settings": {
                "manifest": str(path),
                "calculation_ids": ["E3-R1-BCC-0001"],
            },
        }
    )


def test_e3_dft_check_and_run_default_to_native_validation(dft_preset, native_process):
    _check(dft_preset)
    assert "--dry-run" in native_process[0]
    assert "--submit" not in native_process[0]
    assert "--dry-run" in _command(dft_preset, approved=False)
    assert "--submit" in _command(dft_preset, approved=True)
    # Approved command is inspected only, never dispatched by this test.


def test_e3_dft_validation_failure_is_explicit(dft_preset, monkeypatch):
    monkeypatch.setattr(
        "matsim_agents.cli.workflows.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "PAW mismatch", ""),
    )
    with pytest.raises(ValueError, match="PAW mismatch"):
        _check(dft_preset)


def test_interruption_records_error_without_success(workflow_yaml, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("native interruption")

    monkeypatch.setattr("matsim_agents.cli.workflows.run_native_command", interrupt)
    result = CliRunner().invoke(app, ["workflow", "run", str(workflow_yaml)])
    assert result.exit_code != 0
    record = read_invocation(next((workflow_yaml.parent / "records").iterdir()))
    assert record.state == "interrupted"
    assert "native interruption" in record.failure_reason


def test_invocation_lock_prevents_concurrent_resume(tmp_path, native_process):
    from matsim_agents.cli.workflows import _execute

    run = ScientificRunDirectory.create(
        tmp_path,
        workflow="objective",
        request={},
        resolved_config={},
        provenance=ProvenanceRecord(workflow="objective", evidence_level=EvidenceLevel.HYPOTHESIS),
    )
    record = WorkflowInvocation(
        workflow="objective",
        command=["test-only"],
        working_directory=tmp_path,
        state="running",
        configuration_sha256={},
    )
    with (
        exclusive_lock(run.path / ".workflow.lock", timeout=0),
        pytest.raises(ValueError, match="another process"),
    ):
        _execute(run, record)
    assert native_process == []


def test_catalog_covers_all_native_scientific_entrypoints():
    expected = {
        ("explore", "run"),
        ("relax",),
        ("characterize",),
        ("debate",),
        ("llm-check",),
        ("al", "run"),
        ("run",),
        ("plan",),
        ("supervisor-run",),
        ("chat",),
        ("e1-development",),
        ("benchmark", "known-answers"),
        ("benchmark", "agent-eval"),
    }
    assert {adapter.command for adapter in ADAPTERS.values() if adapter.command} == expected


@pytest.mark.parametrize("workflow", ["chat", "explore"])
def test_relative_native_resume_path_is_resolved_in_callers_directory(
    tmp_path, monkeypatch, workflow
):
    from types import SimpleNamespace

    run = _saved_invocation(tmp_path, "explore" if workflow == "explore" else "characterize")
    if workflow == "chat":
        from matsim_agents.cli.workflows import _hashes
        from matsim_agents.execution.workflow_record import write_invocation

        cfg = WorkflowConfig(workflow="chat", output_root=tmp_path)
        run.write_json("resolved_config.json", cfg.model_dump(mode="json"))
        record = read_invocation(run.path)
        record.workflow = "chat"
        record.configuration_sha256 = _hashes(cfg)
        write_invocation(run, record)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    native = ScientificRunDirectory.create(
        elsewhere,
        workflow=workflow,
        request={},
        resolved_config={},
        provenance=ProvenanceRecord(workflow=workflow, evidence_level=EvidenceLevel.HYPOTHESIS),
    )
    observed = []
    monkeypatch.setattr(
        "matsim_agents.workflows.exploration_checkpoint._state",
        lambda directory: SimpleNamespace(config="same"),
    )
    monkeypatch.setattr(
        "matsim_agents.workflows.exploration_config.load_exploration_config",
        lambda path: "same",
    )

    def execute(command, output, emit, *, cwd, env):
        observed.append(command)
        return 0

    monkeypatch.setattr("matsim_agents.cli.workflows.run_native_command", execute)
    monkeypatch.chdir(elsewhere)
    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "resume",
            str(run.path),
            "--native-run",
            native.path.name,
        ],
    )
    assert result.exit_code == 0, result.output
    assert str(native.path) in observed[0]
    assert read_invocation(run.path).native_run_directory == str(native.path)


def test_record_rejects_ambient_relative_working_directory():
    with pytest.raises(ValueError, match="must be absolute"):
        WorkflowInvocation(
            workflow="characterize",
            command=["test"],
            working_directory=Path("."),
            state="running",
            configuration_sha256={},
        )


def _saved_invocation(tmp_path, workflow, *, state="interrupted", arguments=None):
    from matsim_agents.cli.workflows import _hashes
    from matsim_agents.execution.workflow_record import write_invocation

    native = tmp_path / "native.yaml"
    native.write_text("{}")
    cfg = WorkflowConfig(
        workflow=workflow,
        config=native,
        arguments=arguments or [],
        output_root=tmp_path / "invocations",
    )
    run = ScientificRunDirectory.create(
        cfg.output_root,
        workflow=workflow,
        request={},
        resolved_config=cfg.model_dump(mode="json"),
        provenance=ProvenanceRecord(workflow=workflow, evidence_level=EvidenceLevel.HYPOTHESIS),
    )
    write_invocation(
        run,
        WorkflowInvocation(
            workflow=workflow,
            command=["test"],
            working_directory=tmp_path,
            state=state,
            configuration_sha256=_hashes(cfg),
        ),
    )
    return run


@pytest.mark.parametrize("workflow", ["characterize", "e1"])
def test_native_journal_resume_delegates_without_enabling_retries(
    workflow, tmp_path, native_process, monkeypatch
):
    run = _saved_invocation(tmp_path, workflow)
    monkeypatch.setattr("matsim_agents.cli.workflows._check", lambda cfg: [])
    result = CliRunner().invoke(app, ["workflow", "resume", str(run.path)])
    assert result.exit_code == 0, result.output
    assert "--retry-errors" not in native_process[0]
    assert ADAPTERS[workflow].command[0] in native_process[0]
    assert read_invocation(run.path).state == "completed"


def test_completed_resumable_invocation_is_noop(tmp_path, native_process, monkeypatch):
    run = _saved_invocation(tmp_path, "characterize", state="completed")
    monkeypatch.setattr(
        "matsim_agents.cli.workflows._check", lambda cfg: pytest.fail("must not check")
    )
    result = CliRunner().invoke(app, ["workflow", "resume", str(run.path)])
    assert result.exit_code == 0, result.output
    assert "already completed" in result.output
    assert native_process == []


def test_resume_retains_original_native_working_directory(tmp_path, monkeypatch):
    run = _saved_invocation(tmp_path, "characterize")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    observed = []

    def execute(command, output, emit, *, cwd, env):
        observed.append(cwd)
        return 0

    monkeypatch.setattr("matsim_agents.cli.workflows.run_native_command", execute)
    result = CliRunner().invoke(app, ["workflow", "resume", str(run.path)])
    assert result.exit_code == 0, result.output
    assert observed == [tmp_path]


def test_changed_wrapper_settings_refuse_resume(tmp_path, native_process):
    run = _saved_invocation(tmp_path, "characterize")
    saved = json.loads((run.path / "resolved_config.json").read_text())
    saved["arguments"] = ["--different-policy"]
    run.write_json("resolved_config.json", saved)
    result = CliRunner().invoke(app, ["workflow", "resume", str(run.path)])
    assert result.exit_code == 1
    assert "configuration or campaign manifest changed" in " ".join(result.output.split())
    assert native_process == []


def test_changed_native_input_refuses_resume(tmp_path, native_process):
    run = _saved_invocation(tmp_path, "characterize")
    (tmp_path / "native.yaml").write_text("changed: true")
    result = CliRunner().invoke(app, ["workflow", "resume", str(run.path)])
    assert result.exit_code == 1
    assert "changed" in result.output
    assert native_process == []


def test_al_resume_requires_native_loop_setting(tmp_path, native_process, monkeypatch):
    from types import SimpleNamespace

    run = _saved_invocation(tmp_path, "al")
    monkeypatch.setattr("matsim_agents.cli.workflows._check", lambda cfg: [])
    monkeypatch.setattr(
        "matsim_agents.active_learning.ALConfig.from_yaml",
        lambda path: SimpleNamespace(loop=SimpleNamespace(resume=False)),
    )
    result = CliRunner().invoke(app, ["workflow", "resume", str(run.path)])
    assert result.exit_code == 1
    assert "loop.resume" in result.output
    assert native_process == []


def test_exploration_resume_requires_matching_native_configuration(
    tmp_path, native_process, monkeypatch
):
    from types import SimpleNamespace

    run = _saved_invocation(tmp_path, "explore")
    native = ScientificRunDirectory.create(
        tmp_path / "science",
        workflow="explore",
        request={},
        resolved_config={},
        provenance=ProvenanceRecord(workflow="explore", evidence_level=EvidenceLevel.HYPOTHESIS),
    )
    monkeypatch.setattr("matsim_agents.cli.workflows._check", lambda cfg: [])
    monkeypatch.setattr(
        "matsim_agents.workflows.exploration_checkpoint._state",
        lambda directory: SimpleNamespace(config="another request"),
    )
    monkeypatch.setattr(
        "matsim_agents.workflows.exploration_config.load_exploration_config",
        lambda path: "original request",
    )
    result = CliRunner().invoke(
        app,
        [
            "workflow",
            "resume",
            str(run.path),
            "--native-run",
            str(native.path),
        ],
    )
    assert result.exit_code == 1
    assert "different scientific configuration" in " ".join(result.output.split())
    assert native_process == []
