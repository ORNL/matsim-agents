"""Consistent check/plan/run/status/resume entry points over native workflows."""

from __future__ import annotations

import hashlib
import json
import shlex
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import typer
import yaml
from typer.core import TyperGroup
from typer.main import get_command

from matsim_agents.cli._app import app, console
from matsim_agents.cli.workflow_catalog import (
    ADAPTERS,
    WorkflowConfig,
    load_workflow,
    native_arguments,
)
from matsim_agents.core.contracts import EvidenceLevel, ProvenanceRecord
from matsim_agents.core.hashing import sha256_file as file_sha256
from matsim_agents.execution.locking import LockTimeoutError, exclusive_lock
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.execution.workflow_record import (
    WorkflowInvocation,
    read_invocation,
    saved_workflow_config,
    write_invocation,
)
from matsim_agents.execution.workflow_runner import run_native_command

workflow_app = typer.Typer(
    help=(
        "Check/run/status/summary/resume for native scientific workflows. "
        "When to use: a named, pre-defined workflow with a config file, not a free-text objective."
    )
)
app.add_typer(workflow_app, name="workflow")


def _validate_native_cli(arguments: list[str]) -> dict[str, Any]:
    """Parse all native arguments without invoking any command callback."""
    command = get_command(app)
    remaining = list(arguments)
    parent = None
    name = "matsim-agents"
    with ExitStack() as contexts:
        while isinstance(command, TyperGroup):
            context = contexts.enter_context(
                command.make_context(name, [], parent=parent, resilient_parsing=True)
            )
            if not remaining:
                raise ValueError("A native workflow command is required")
            name, child, remaining = command.resolve_command(context, remaining)
            if child is None:
                raise ValueError(f"Unknown native command: {name}")
            parent, command = context, child
        if "--help" in remaining:
            raise ValueError("Use the native --help command directly, not a workflow invocation")
        context = contexts.enter_context(command.make_context(name, remaining, parent=parent))
        if context.args:
            raise ValueError(f"Unrecognized native arguments: {' '.join(context.args)}")
        return context.params


def _check(cfg: WorkflowConfig) -> list[str]:
    notes = [
        "Static configuration preflight only: no scientific command, "
        "LLM, model load, or job submission.",
        "Device/ABI, authentication, and live scheduler readiness are not qualified by this check.",
    ]
    if cfg.config is not None and not cfg.config.is_file():
        raise ValueError(f"Native configuration does not exist: {cfg.config}")
    parameters = _validate_native_cli(native_arguments(cfg))
    if cfg.workflow in {"known-answers", "agent-eval"}:
        if cfg.workflow == "known-answers":
            from matsim_agents.workflows.evaluation.known_answers import load_suite

            if parameters["tier"] not in {"replay", "live"}:
                raise ValueError("Known-answer tier must be replay or live")
        else:
            from matsim_agents.workflows.evaluation.agent_eval import load_suite

        suite = load_suite(parameters["suite"])
        if set(parameters["case"] or []) - {case.case_id for case in suite.cases}:
            raise ValueError("Requested benchmark case IDs are not in the selected suite")
    if cfg.config is not None:
        if cfg.workflow == "explore":
            from matsim_agents.workflows.exploration.api import (
                check_exploration,
                load_exploration_config,
            )

            result = check_exploration(load_exploration_config(cfg.config))
            if not result.ready:
                raise ValueError("\n".join(result.errors))
        elif cfg.workflow == "characterize":
            from matsim_agents.workflows.characterization import load_contract

            load_contract(cfg.config)
        elif cfg.workflow == "al":
            from matsim_agents.workflows.active_learning import ALConfig

            ALConfig.from_yaml(cfg.config)
        else:
            from matsim_agents.cli.workflow_commands import _load_workflow_config

            if cfg.workflow in {"debate", "llm-check"}:
                raw = _load_workflow_config(cfg.config)
            else:
                raw = yaml.safe_load(cfg.config.read_text())
            if cfg.workflow == "relax":
                from matsim_agents.workflows.relaxation import ScientificRelaxationConfig

                ScientificRelaxationConfig.model_validate(raw)
            elif cfg.workflow == "debate":
                from matsim_agents.workflows.debate import ScientificDebateConfig

                ScientificDebateConfig.model_validate(raw)
            elif cfg.workflow == "llm-check":
                from matsim_agents.workflows.llm_check import LLMCheckConfig

                LLMCheckConfig.model_validate(raw)
            elif cfg.workflow == "agent-eval":
                from matsim_agents.workflows.evaluation.agent_eval import AgentEvalConfig

                AgentEvalConfig.model_validate(raw)
    return notes


def _command(cfg: WorkflowConfig) -> list[str]:
    return [sys.executable, "-m", "matsim_agents.cli", *native_arguments(cfg)]


def _hashes(cfg: WorkflowConfig) -> dict[str, str]:
    paths = [cfg.config] if cfg.config is not None else []
    return {
        "workflow_config": hashlib.sha256(cfg.model_dump_json().encode()).hexdigest(),
        **{str(path): file_sha256(path) for path in paths},
    }


def _execute(run: ScientificRunDirectory, record: WorkflowInvocation) -> None:
    """Preserve native output and status; never reinterpret completion as convergence."""
    try:
        with exclusive_lock(run.path / ".workflow.lock", timeout=0):
            state_path = run.path / "workflow-state.json"
            if state_path.is_file() and read_invocation(run.path).state == "completed":
                console.print("Invocation already completed; no command executed.")
                return
            _execute_locked(run, record)
    except LockTimeoutError as error:
        raise ValueError("Workflow invocation is already running in another process") from error


def _execute_locked(run: ScientificRunDirectory, record: WorkflowInvocation) -> None:
    write_invocation(run, record)
    console.print(f"Workflow run directory: {run.path}", markup=False)
    try:
        with (run.path / "workflow-output.log").open("a") as output:
            return_code = run_native_command(
                record.command,
                output,
                lambda text: console.print(text, end="", markup=False, highlight=False),
                cwd=record.working_directory,
            )
        record.return_code = return_code
        record.state = "completed" if return_code == 0 else "failed"
        if return_code != 0:
            record.failure_reason = (
                f"Native command exited {return_code}; inspect workflow-output.log"
            )
    except BaseException as error:
        record.state = (
            "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        )
        record.failure_reason = f"{type(error).__name__}: {error}"
        try:
            write_invocation(run, record)
        except OSError as persistence_error:
            error.add_note(f"Could not persist interruption report: {persistence_error}")
        raise
    write_invocation(run, record)
    console.print(f"Report: {run.path / 'report.md'}", markup=False)
    if record.state == "failed":
        raise typer.Exit(1)
    console.print(
        f"Next: `matsim-agents workflow summary {run.path}` for the report, or "
        f"`matsim-agents workflow status {run.path}` later.",
        markup=False,
        soft_wrap=True,
    )


def _load(config: Path) -> WorkflowConfig:
    """Load a workflow file; an unreadable or invalid file is a usage error (exit 2)."""
    try:
        return load_workflow(config)
    except (ValueError, OSError, yaml.YAMLError) as error:
        console.print(f"Configuration error in {config}: {error}", markup=False)
        raise typer.Exit(2) from error


@workflow_app.command("list")
def list_workflows() -> None:
    """List native workflows and whether they have a supported restart operation."""
    for name, adapter in ADAPTERS.items():
        console.print(
            f"{name}: {adapter.description}; resume="
            f"{'unsupported' if adapter.resume == 'unsupported' else 'native'}",
            markup=False,
        )


@workflow_app.command("init")
def init(
    workflow: str = typer.Argument(..., help="A workflow name from `workflow list`."),
    output: Path = typer.Argument(..., help="Destination YAML; never overwrite an existing file"),
    config: Path | None = typer.Option(None, help="Native configuration file for the workflow."),
) -> None:
    """Write a workflow file for a registered workflow."""
    try:
        if workflow not in ADAPTERS:
            raise ValueError(f"Unknown workflow {workflow!r}; choose {', '.join(ADAPTERS)}")
        if output.exists():
            raise ValueError(f"Refusing to overwrite {output}")
        data: dict[str, Any] = {"workflow": workflow, "output_root": "./workflow-records"}
        if ADAPTERS[workflow].config_argument:
            if config is None:
                raise ValueError(f"{workflow} requires --config path/to/native-config")
            data["config"] = str(config.expanduser().resolve())
        WorkflowConfig.model_validate(data)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
        console.print(f"Created {output}; next: workflow check {output}", markup=False)
        console.print("No model loaded, workflow started, or job submitted.")
    except (ValueError, OSError) as error:
        console.print(f"Cannot create preset: {error}", markup=False)
        raise typer.Exit(1) from error


@workflow_app.command("check")
def check(config: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Validate a workflow file and its native configuration without running anything."""
    cfg = _load(config)
    try:
        notes = _check(cfg)
    except (ValueError, OSError, yaml.YAMLError) as error:
        console.print(f"Preflight failed: {error}", markup=False)
        raise typer.Exit(1) from error
    console.print("Preflight passed")
    for note in notes:
        console.print(note, markup=False)
    console.print(
        f"Next: `matsim-agents workflow plan {config}` to see the exact command, "
        f"then `matsim-agents workflow run {config}`.",
        markup=False,
        soft_wrap=True,
    )


def _plan(cfg: WorkflowConfig) -> dict[str, Any]:
    notes = _check(cfg)
    command = _command(cfg)
    return {
        "workflow": cfg.workflow,
        "output_root": str(cfg.output_root),
        "command": command,
        "command_display": shlex.join(command),
        "notes": notes,
        "would_execute": True,
    }


def _print_plan(plan: dict[str, Any]) -> None:
    console.print(f"Workflow: {plan['workflow']}", markup=False)
    console.print(f"Output root: {plan['output_root']}", markup=False)
    console.print("Command:", markup=False)
    console.print(f"  {plan['command_display']}", markup=False, soft_wrap=True)
    for note in plan["notes"]:
        console.print(f"Note: {note}", markup=False)
    console.print("No run directory, model load, scheduler job, or scientific command created.")


@workflow_app.command("plan")
def plan(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    as_json: bool = typer.Option(False, "--json", help="Print the plan as JSON."),
) -> None:
    """Validate a workflow and show the exact command without executing it."""
    cfg = _load(config)
    try:
        result = _plan(cfg)
    except (ValueError, OSError, yaml.YAMLError) as error:
        console.print(f"Cannot plan workflow: {error}", markup=False)
        raise typer.Exit(1) from error
    if as_json:
        console.print_json(json.dumps(result))
    else:
        _print_plan(result)


@workflow_app.command("run")
def run(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate and print the command without executing it."
    ),
) -> None:
    """Validate and run a workflow, recording the invocation in a new run directory."""
    cfg = _load(config)
    try:
        if dry_run:
            _print_plan(_plan(cfg))
            return
        _check(cfg)
        command = _command(cfg)
        directory = ScientificRunDirectory.create(
            cfg.output_root,
            workflow=cfg.workflow,
            request={"workflow_config": str(config.resolve())},
            resolved_config=cfg.model_dump(mode="json"),
            provenance=ProvenanceRecord(
                workflow=cfg.workflow, evidence_level=EvidenceLevel.HYPOTHESIS
            ),
        )
        record = WorkflowInvocation(
            workflow=cfg.workflow,
            command=command,
            working_directory=Path.cwd(),
            state="running",
            configuration_sha256=_hashes(cfg),
        )
        _execute(directory, record)
    except (ValueError, OSError, yaml.YAMLError) as error:
        console.print(f"Workflow error: {error}", markup=False)
        raise typer.Exit(1) from error


@workflow_app.command("status")
def status(directory: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    try:
        record = read_invocation(directory)
    except (ValueError, OSError) as error:
        console.print(f"Cannot read status: {error}", markup=False)
        raise typer.Exit(1) from error
    console.print(f"{record.workflow}: {record.state}; exit={record.return_code}", markup=False)
    console.print("Last saved invocation status; not scheduler status or a liveness probe.")


@workflow_app.command("summary")
def summary(directory: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    try:
        read_invocation(directory)
        report = (directory / "report.md").read_text()
    except (ValueError, OSError) as error:
        console.print(f"Cannot read report: {error}", markup=False)
        raise typer.Exit(1) from error
    console.print(report, markup=False)


@workflow_app.command("resume")
def resume(
    directory: Path = typer.Argument(..., exists=True, file_okay=False),
    native_run: Path | None = typer.Option(
        None, help="Native run/session directory for exploration or chat, shown in native output."
    ),
) -> None:
    try:
        if native_run is not None:
            native_run = native_run.expanduser().resolve()
        record = read_invocation(directory)
        cfg = WorkflowConfig.model_validate(saved_workflow_config(directory))
        adapter = ADAPTERS[cfg.workflow]
        if adapter.resume == "unsupported":
            raise ValueError(
                f"{cfg.workflow} has no native resume contract. Start a fresh workflow run; "
                "completed scientific work will not be silently replayed."
            )
        if _hashes(cfg) != record.configuration_sha256:
            raise ValueError("Resume refused: native configuration changed")
        if record.state == "completed":
            console.print("Invocation already completed; no command executed.")
            return
        if adapter.resume == "directory":
            if native_run is None:
                raise ValueError(
                    "Explore resume requires --native-run with its scientific run directory"
                )
            from matsim_agents.workflows.exploration.checkpoint import _state
            from matsim_agents.workflows.exploration.config import load_exploration_config

            native_state = _state(ScientificRunDirectory.open(native_run))
            if native_state.config != load_exploration_config(cfg.config):
                raise ValueError(
                    "Native exploration directory has a different scientific configuration"
                )
            command = [
                sys.executable,
                "-m",
                "matsim_agents.cli",
                "explore",
                "resume",
                str(native_run),
            ]
        elif adapter.resume == "chat":
            if native_run is None:
                raise ValueError(
                    "Chat resume requires --native-run with its saved session directory"
                )
            if "--resume" in cfg.arguments:
                raise ValueError("Use --native-run for resume, not a stored --resume argument")
            command = [*_command(cfg), "--resume", str(native_run)]
        elif cfg.workflow == "al":
            from matsim_agents.workflows.active_learning import ALConfig

            if not ALConfig.from_yaml(cfg.config).loop.resume:
                raise ValueError("Native AL configuration must enable loop.resume")
            command = _command(cfg)
        else:
            command = _command(cfg)
        record.command = command
        record.state = "running"
        record.failure_reason = None
        record.return_code = None
        record.native_run_directory = str(native_run.resolve()) if native_run else None
        _execute(ScientificRunDirectory.open(directory), record)
    except (ValueError, OSError, yaml.YAMLError) as error:
        console.print(f"Cannot resume workflow: {error}", markup=False)
        raise typer.Exit(1) from error
