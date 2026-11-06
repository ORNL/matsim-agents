"""Git-backed coordination commands for work distributed across HPC facilities."""

from __future__ import annotations

import getpass
import json
import socket
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console
from matsim_agents.execution.task_board import TaskBoard, TaskBoardError

coordination_app = typer.Typer(
    add_completion=False,
    help="Coordinate HPC tasks through the current branch on a shared Git remote.",
)
app.add_typer(coordination_app, name="coordination")


def _board(repo: Path, remote: str) -> TaskBoard:
    try:
        return TaskBoard(repo, remote=remote)
    except TaskBoardError as exc:
        console.print(f"[red]coordination error[/red]: {exc}")
        raise typer.Exit(code=1) from None


def _run(operation):
    try:
        result = operation()
    except (TaskBoardError, ValueError) as exc:
        console.print(f"[red]coordination error[/red]: {exc}")
        raise typer.Exit(code=1) from None
    if result is not None:
        console.print_json(json.dumps(result, indent=2))


@coordination_app.command("init")
def coordination_init(
    repo: Path = typer.Option(
        Path("."), "--repo", help="Source checkout containing the Git remote."
    ),
    remote: str = typer.Option("origin", "--remote", help="Configured Git remote name."),
):
    """Create the task-board directory on the current branch if needed."""

    branch = _board(repo, remote)
    _run(lambda: branch.initialize())
    console.print(f"[green]ready[/green]: {remote}/{branch.branch}")


@coordination_app.command("add")
def coordination_add(
    task_id: str = typer.Argument(..., help="Stable unique task identifier."),
    title: str = typer.Option(..., "--title", help="Short scientific task title."),
    facility: str = typer.Option(..., "--facility", help="Target HPC facility."),
    command: str = typer.Option(
        ..., "--command", help="Documented command; never executed by this tool."
    ),
    artifact_uri: str | None = typer.Option(
        None, "--artifact-uri", help="Shared path or approved artifact URI for outputs."
    ),
    details: str | None = typer.Option(
        None, "--details", help="Inputs, acceptance criteria, and limits."
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Source checkout containing the Git remote."
    ),
    remote: str = typer.Option("origin", "--remote", help="Configured Git remote name."),
):
    """Publish a task specification without dispatching a job."""

    _run(
        lambda: _board(repo, remote).add_task(
            task_id,
            title=title,
            facility=facility,
            command=command,
            artifact_uri=artifact_uri,
            details=details,
        )
    )


@coordination_app.command("list")
def coordination_list(
    status: str | None = typer.Option(
        None, "--status", help="Filter by pending/running/completed/failed."
    ),
    repo: Path = typer.Option(
        Path("."), "--repo", help="Source checkout containing the Git remote."
    ),
    remote: str = typer.Option("origin", "--remote", help="Configured Git remote name."),
):
    """Fetch and display tasks from the shared remote branch."""

    _run(lambda: _board(repo, remote).list_tasks(status=status))


def _worker(value: str | None) -> str:
    return value or f"{getpass.getuser()}@{socket.gethostname()}"


@coordination_app.command("claim")
def coordination_claim(
    task_id: str = typer.Argument(...),
    worker: str | None = typer.Option(
        None, "--worker", help="Worker identity; defaults to user@host."
    ),
    lease_minutes: int = typer.Option(60, "--lease-minutes", min=1, help="Claim lease duration."),
    repo: Path = typer.Option(Path("."), "--repo"),
    remote: str = typer.Option("origin", "--remote"),
):
    """Claim a pending task or reclaim it after its lease expires."""

    _run(
        lambda: _board(repo, remote).claim(
            task_id, worker=_worker(worker), lease_minutes=lease_minutes
        )
    )


@coordination_app.command("heartbeat")
def coordination_heartbeat(
    task_id: str = typer.Argument(...),
    worker: str | None = typer.Option(
        None, "--worker", help="Must match the claim's worker identity."
    ),
    lease_minutes: int = typer.Option(60, "--lease-minutes", min=1),
    repo: Path = typer.Option(Path("."), "--repo"),
    remote: str = typer.Option("origin", "--remote"),
):
    """Extend the current worker's task lease."""

    _run(
        lambda: _board(repo, remote).heartbeat(
            task_id, worker=_worker(worker), lease_minutes=lease_minutes
        )
    )


def _finish(
    task_id: str,
    worker: str | None,
    note: str | None,
    success: bool,
    repo: Path,
    remote: str,
):
    _run(
        lambda: _board(repo, remote).finish(
            task_id, worker=_worker(worker), success=success, note=note
        )
    )


@coordination_app.command("complete")
def coordination_complete(
    task_id: str = typer.Argument(...),
    worker: str | None = typer.Option(None, "--worker"),
    note: str | None = typer.Option(
        None, "--note", help="Short result summary or artifact location."
    ),
    repo: Path = typer.Option(Path("."), "--repo"),
    remote: str = typer.Option("origin", "--remote"),
):
    """Publish successful completion for the currently owned task."""

    _finish(task_id, worker, note, True, repo, remote)


@coordination_app.command("fail")
def coordination_fail(
    task_id: str = typer.Argument(...),
    worker: str | None = typer.Option(None, "--worker"),
    note: str | None = typer.Option(
        None, "--note", help="Failure summary or diagnostic artifact location."
    ),
    repo: Path = typer.Option(Path("."), "--repo"),
    remote: str = typer.Option("origin", "--remote"),
):
    """Publish failure for the currently owned task."""

    _finish(task_id, worker, note, False, repo, remote)
