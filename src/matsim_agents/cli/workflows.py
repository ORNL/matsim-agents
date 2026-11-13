"""Consistent entry points over native advanced workflows and E3 campaign runners."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
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
    E3DFTSettings,
    E3MLIPSettings,
    WorkflowConfig,
    load_workflow,
    native_arguments,
)
from matsim_agents.execution.contracts import EvidenceLevel, ProvenanceRecord
from matsim_agents.execution.ledger import file_sha256
from matsim_agents.execution.locking import LockTimeoutError, exclusive_lock
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.execution.workflow_record import (
    WorkflowInvocation,
    read_invocation,
    saved_workflow_config,
    write_invocation,
)
from matsim_agents.execution.workflow_runner import run_native_command

workflow_app = typer.Typer(help="Check/run/status/summary/resume for native scientific workflows.")
app.add_typer(workflow_app, name="workflow")
REPO = Path(__file__).resolve().parents[3]
SUBMIT_BATCH = REPO / "deployments/aurora/jobs/e4-first-batch/submit_batch.py"


def _native_environment() -> dict[str, str]:
    return {
        "PYTHONPATH": os.pathsep.join(
            [str(REPO / "src"), str(REPO), os.environ.get("PYTHONPATH", "")]
        )
    }


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
    if isinstance(cfg.settings, E3MLIPSettings):
        if not (REPO / "benchmarks/e3/production.py").is_file():
            raise ValueError("E3-MLIP requires a source checkout containing the benchmark runner")
        checked = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.e3.production",
                "validate",
                "--manifest",
                str(cfg.settings.manifest),
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
            env={**os.environ, **_native_environment()},
        )
        if checked.returncode != 0:
            detail = checked.stdout + checked.stderr
            raise ValueError(f"E3-MLIP protocol preflight failed:\n{detail}")
        notes.append("E3-MLIP protocol validated; run requires Slurm and approval.")
        return notes
    if isinstance(cfg.settings, E3DFTSettings):
        manifest = json.loads(cfg.settings.manifest.read_text())
        if manifest.get("batch_id") != "E3-round1-bcc-01":
            raise ValueError("E3-DFT preset requires the frozen E3-round1-bcc-01 batch manifest")
        if not SUBMIT_BATCH.is_file():
            raise ValueError("E3-DFT requires a source checkout containing the frozen batch runner")
        cases = manifest.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("E3-DFT manifest has no cases")
        ids = [case["calculation_id"] for case in cases]
        if len(set(cfg.settings.calculation_ids)) != len(cfg.settings.calculation_ids):
            raise ValueError("Duplicate calculation IDs in E3-DFT settings")
        if set(cfg.settings.calculation_ids) - set(ids):
            raise ValueError("Requested calculation IDs are not in the frozen batch")
        command = [
            sys.executable,
            str(SUBMIT_BATCH),
            "--manifest",
            str(cfg.settings.manifest),
            "--dry-run",
        ]
        if cfg.settings.calculation_ids:
            command.extend(["--only", *cfg.settings.calculation_ids])
        checked = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env={**os.environ, **_native_environment()},
        )
        if checked.returncode != 0:
            raise ValueError("Frozen E3/VASP preflight failed:\n" + checked.stdout + checked.stderr)
        notes.append(
            "Native PAW/executable/input/collision/submission-ledger dry-run checks passed."
        )
        return notes
    parameters = _validate_native_cli(native_arguments(cfg))
    if cfg.workflow in {"known-answers", "agent-eval"}:
        if cfg.workflow == "known-answers":
            from matsim_agents.campaign.known_answers import load_suite

            if parameters["tier"] not in {"replay", "live"}:
                raise ValueError("Known-answer tier must be replay or live")
        else:
            from matsim_agents.campaign.agent_eval import load_suite

        suite = load_suite(parameters["suite"])
        if set(parameters["case"] or []) - {case.case_id for case in suite.cases}:
            raise ValueError("Requested benchmark case IDs are not in the selected suite")
    if cfg.config is not None:
        if cfg.workflow == "explore":
            from matsim_agents.workflows.exploration import (
                check_exploration,
                load_exploration_config,
            )

            result = check_exploration(load_exploration_config(cfg.config))
            if not result.ready:
                raise ValueError("\n".join(result.errors))
        elif cfg.workflow == "characterize":
            from matsim_agents.campaign.characterization import load_contract

            load_contract(cfg.config)
        elif cfg.workflow == "al":
            from matsim_agents.active_learning import ALConfig

            ALConfig.from_yaml(cfg.config)
        else:
            from matsim_agents.cli.workflow_commands import _load_workflow_config

            if cfg.workflow in {"debate", "llm-check"}:
                raw = _load_workflow_config(cfg.config)
            elif cfg.workflow == "e1":
                raw = json.loads(cfg.config.read_text())
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
            elif cfg.workflow == "e1":
                from matsim_agents.workflows.e1 import E1DevelopmentManifest

                E1DevelopmentManifest.model_validate(raw)
            elif cfg.workflow == "agent-eval":
                from matsim_agents.campaign.agent_eval import AgentEvalConfig

                AgentEvalConfig.model_validate(raw)
    return notes


def _command(cfg: WorkflowConfig, *, approved: bool) -> list[str]:
    if approved and cfg.workflow not in {"e3-mlip", "e3-dft"}:
        raise ValueError(
            "--approve-execution is only for E3 presets; use native approval settings otherwise"
        )
    if isinstance(cfg.settings, E3MLIPSettings):
        if not approved:
            raise ValueError("E3-MLIP execution requires --approve-execution")
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("E3-MLIP production must run inside an existing Slurm allocation")
        settings = cfg.settings
        budget = json.loads(settings.resource_state.read_text())
        cap = 2_764_800
        if budget.get("schema_version") != 1 or budget.get("cap_gpu_seconds") != cap:
            raise ValueError("E3-MLIP requires the 768-GPU-hour benchmark resource ledger")
        job_id = os.environ["SLURM_JOB_ID"]
        reservation_id = os.environ.get("MATSIM_E3_MLIP_RESERVATION_ID", job_id)
        reservation = budget.get("reservations", {}).get(reservation_id)
        if (
            not isinstance(reservation, dict)
            or reservation.get("reserved_gpu_seconds") != 460_800
            or reservation.get("job_id", job_id) != job_id
        ):
            raise ValueError("E3-MLIP requires this job's full 16-node/two-hour reservation.")
        return [
            sys.executable,
            "-m",
            "benchmarks.e3.production",
            "run-campaign",
            "--manifest",
            str(settings.manifest),
            "--output-root",
            str(settings.output_root),
            "--resource-state",
            str(settings.resource_state),
            "--material",
            settings.material,
            "--workers",
            str(settings.workers),
            "--worker-python",
            str(settings.worker_python or (REPO / ".venv-mace/bin/python")),
            "--hard-optimization-seconds",
            str(settings.hard_optimization_seconds),
        ]
    if isinstance(cfg.settings, E3DFTSettings):
        command = [
            sys.executable,
            str(SUBMIT_BATCH),
            "--manifest",
            str(cfg.settings.manifest),
            "--submit" if approved else "--dry-run",
        ]
        if cfg.settings.calculation_ids:
            command.extend(["--only", *cfg.settings.calculation_ids])
        return command
    return [sys.executable, "-m", "matsim_agents.cli", *native_arguments(cfg)]


def _hashes(cfg: WorkflowConfig) -> dict[str, str]:
    paths = [cfg.config] if cfg.config is not None else []
    if cfg.settings is not None:
        paths.append(cfg.settings.manifest)
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
                env=_native_environment(),
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
    preset: str = typer.Argument(..., help="e3-mlip, e3-dft, explore, or debate"),
    output: Path = typer.Argument(..., help="Destination YAML; never overwrite an existing file"),
) -> None:
    """Copy an annotated preset, rebasing its native paths to the destination."""
    presets = {
        "e3-mlip": "e3-mlip.yaml",
        "e3-dft": "e3-dft.yaml",
        "explore": "exploration-workflow.yaml",
        "debate": "debate-workflow.yaml",
    }
    try:
        if preset not in presets:
            raise ValueError(f"Unknown preset {preset!r}; choose {', '.join(presets)}")
        if output.exists():
            raise ValueError(f"Refusing to overwrite {output}")
        source = REPO / "examples/workflows" / presets[preset]
        cfg = load_workflow(source)
        text = source.read_text()
        raw = yaml.safe_load(text)
        # Keep annotations while replacing only the preset's path values.
        for field in ("config",):
            if field in raw:
                text = text.replace(
                    f"{field}: {raw[field]}", f"{field}: {json.dumps(str(getattr(cfg, field)))}"
                )
        if cfg.settings is not None:
            for field in ("manifest", "output_root", "resource_state", "worker_python"):
                resolved = getattr(cfg.settings, field, None)
                if resolved is not None:
                    original = raw["settings"].get(field)
                    if original is None:
                        continue
                    text = text.replace(
                        f"{field}: {original}",
                        f"{field}: {json.dumps(str(resolved))}",
                    )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(text)
        console.print(
            f"Created {output}; edit settings, then workflow check {output}", markup=False
        )
        console.print("No model loaded, campaign started, or job submitted.")
    except (ValueError, OSError) as error:
        console.print(f"Cannot create preset: {error}", markup=False)
        raise typer.Exit(1) from error


@workflow_app.command("check")
def check(config: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    try:
        cfg = load_workflow(config)
        notes = _check(cfg)
    except (ValueError, OSError, yaml.YAMLError) as error:
        console.print(f"Preflight failed: {error}", markup=False)
        raise typer.Exit(1) from error
    console.print("Preflight passed")
    for note in notes:
        console.print(note, markup=False)


@workflow_app.command("run")
def run(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    approve_execution: bool = typer.Option(
        False, help="Explicitly approve E3-MLIP execution or E3/VASP scheduler submission."
    ),
) -> None:
    try:
        cfg = load_workflow(config)
        _check(cfg)
        command = _command(cfg, approved=approve_execution)
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
            execution_approved=approve_execution,
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
    approve_execution: bool = typer.Option(
        False, help="Explicit approval for E3-MLIP continuation."
    ),
) -> None:
    try:
        if native_run is not None:
            native_run = native_run.expanduser().resolve()
        record = read_invocation(directory)
        cfg = WorkflowConfig.model_validate(saved_workflow_config(directory))
        if approve_execution and cfg.workflow != "e3-mlip":
            raise ValueError("--approve-execution on resume is only for E3-MLIP")
        adapter = ADAPTERS[cfg.workflow]
        if adapter.resume == "unsupported":
            raise ValueError(
                f"{cfg.workflow} has no native resume contract. Start a fresh workflow run; "
                "completed scientific work will not be silently replayed."
            )
        if _hashes(cfg) != record.configuration_sha256:
            raise ValueError("Resume refused: native configuration or campaign manifest changed")
        if record.state == "completed":
            console.print("Invocation already completed; no command executed.")
            return
        if adapter.resume == "directory":
            if native_run is None:
                raise ValueError(
                    "Explore resume requires --native-run with its scientific run directory"
                )
            from matsim_agents.workflows.exploration_checkpoint import _state
            from matsim_agents.workflows.exploration_config import load_exploration_config

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
            command = [*_command(cfg, approved=False), "--resume", str(native_run)]
        elif cfg.workflow == "al":
            from matsim_agents.active_learning import ALConfig

            if not ALConfig.from_yaml(cfg.config).loop.resume:
                raise ValueError("Native AL configuration must enable loop.resume")
            command = _command(cfg, approved=False)
        else:
            command = _command(cfg, approved=approve_execution)
        record.command = command
        record.state = "running"
        record.failure_reason = None
        record.return_code = None
        record.native_run_directory = str(native_run.resolve()) if native_run else None
        record.execution_approved = approve_execution
        _execute(ScientificRunDirectory.open(directory), record)
    except (ValueError, OSError, yaml.YAMLError) as error:
        console.print(f"Cannot resume workflow: {error}", markup=False)
        raise typer.Exit(1) from error
