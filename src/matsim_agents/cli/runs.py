"""Config-file commands: ``validate``, ``plan``, ``run``, ``resume``, ``status``, ``evidence``.

Every command is a thin call into :mod:`matsim_agents.workflows.runner`; the same entry points
serve scripts and agents.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console, error_console
from matsim_agents.cli.options import opt_mlip_config
from matsim_agents.workflows import run_records, runner

BLOCKED_EXIT = 3

_RUNS_DIR = typer.Option("runs", "--runs-dir", help="Directory that receives the run record.")
_FILE = typer.Argument(..., exists=True, dir_okay=False, help="A maintained config YAML file.")


def _options(
    mlip_config: Path | None, allow_template: bool, dispatch: bool, approved_by: str | None
) -> runner.RunOptions:
    return runner.RunOptions(
        mlip_config=mlip_config,
        allow_template=allow_template,
        dispatch=dispatch,
        approved_by=approved_by,
    )


def _report_preflight(kind: str, pre: runner.Preflight) -> None:
    console.print(f"Kind: {kind}", markup=False)
    console.print_json(json.dumps(pre.summary, default=str))
    for blocker in pre.blockers:
        error_console.print(f"Blocked: {blocker}", markup=False, highlight=False)


@app.command()
def validate(
    config: Path = _FILE,
    mlip_config: Path | None = opt_mlip_config(),
    dispatch: bool = typer.Option(False, help="Check as if execution were requested."),
):
    """Check a config file without running anything.

    When to use: before `run`, to learn whether the config is valid and unblocked.

    Exit 0: valid and unblocked. Exit 3: valid but a backend, approval or input is missing.
    Exit 2: invalid.
    """
    kind, pre = runner.validate(config, _options(mlip_config, False, dispatch, None))
    _report_preflight(kind, pre)
    if pre.blockers:
        raise typer.Exit(BLOCKED_EXIT)
    console.print("Valid: nothing blocks `run`.", markup=False)


@app.command()
def plan(
    config: Path = _FILE,
    mlip_config: Path | None = opt_mlip_config(),
    dispatch: bool = typer.Option(False, help="Plan the submitting variant."),
):
    """Show what `run` would do for a config file, then stop.

    When to use: to preview a run, including which backends or approvals it needs.
    """
    kind, pre = runner.validate(config, _options(mlip_config, False, dispatch, None))
    _report_preflight(kind, pre)
    state = "would be blocked" if pre.blockers else "would execute"
    console.print(f"Plan: `run {config}` {state}. Nothing was executed.", markup=False)
    if pre.blockers:
        raise typer.Exit(BLOCKED_EXIT)


@app.command()
def run(
    config: Path = _FILE,
    runs_dir: Path = _RUNS_DIR,
    mlip_config: Path | None = opt_mlip_config(),
    allow_template: bool = typer.Option(
        False, "--allow-template", help="Run a repository example config as given."
    ),
    dispatch: bool = typer.Option(
        False, "--dispatch", help="Submit or execute external work instead of a dry run."
    ),
    approved_by: str | None = typer.Option(None, "--approved-by", help="Approver identity."),
):
    """Execute a config file and record the run.

    When to use: you have a maintained config YAML (application, relaxation, AL, NEB, ...).

    Exit 0: complete or partial. Exit 3: blocked, nothing executed. Exit 1: failed.
    """
    directory, record = runner.run(
        config,
        runs_dir=runs_dir,
        options=_options(mlip_config, allow_template, dispatch, approved_by),
    )
    _print_record(directory, record)
    raise typer.Exit(runner.exit_code(record))


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run ID or run directory."),
    runs_dir: Path = typer.Option(None, "--runs-dir", help="Directory to search."),
):
    """Continue an interrupted run, or retry a blocked one.

    When to use: a run was interrupted or blocked and the cause is now fixed.
    """
    directory, record = runner.resume(run_id, runs_dir=runs_dir)
    _print_record(directory, record)
    raise typer.Exit(runner.exit_code(record))


@app.command()
def status(
    run_id: str | None = typer.Argument(None, help="Run ID or directory; omit to list runs."),
    runs_dir: Path = typer.Option(None, "--runs-dir", help="Directory to search."),
):
    """Show one run's state, or list recorded runs."""
    roots = [runs_dir] if runs_dir else None
    if run_id is None:
        for directory, record in run_records.list_runs(roots or run_records.DEFAULT_RUN_ROOTS):
            console.print(
                f"{record.run_id}  {record.state:<11} {record.kind:<12} {directory}", markup=False
            )
        return
    _print_record(*_find(run_id, roots))


@app.command()
def evidence(
    run_id: str = typer.Argument(..., help="Run ID or run directory."),
    runs_dir: Path = typer.Option(None, "--runs-dir", help="Directory to search."),
):
    """Print a run's record: state, summary, blockers and the native run directory."""
    _, record = _find(run_id, [runs_dir] if runs_dir else None)
    console.print_json(record.model_dump_json())


def _find(run_id: str, roots):
    directory = run_records.find_run(run_id, roots)
    return directory, run_records.load_record(directory)


def _print_record(directory: Path, record: run_records.RunRecord) -> None:
    console.print(f"Run {record.run_id}: {record.state}", markup=False)
    console.print(f"Record: {directory / run_records.RECORD_FILE}", markup=False, soft_wrap=True)
    if record.native_run_directory:
        console.print(f"Native run: {record.native_run_directory}", markup=False, soft_wrap=True)
    for blocker in record.blockers:
        error_console.print(f"Blocked: {blocker}", markup=False, highlight=False)
    if record.failure_reason:
        error_console.print(f"Failed: {record.failure_reason}", markup=False, highlight=False)
