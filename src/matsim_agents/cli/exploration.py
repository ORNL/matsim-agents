"""Beginner-facing, config-driven phase search commands."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from matsim_agents.cli._app import app, console
from matsim_agents.workflows.exploration import (
    ExplorationConfig,
    ExplorationSummary,
    check_exploration,
    load_exploration_config,
    resume_exploration,
    run_exploration,
    summarize_exploration,
)

explore_app = typer.Typer(help="Check, run, and inspect a bounded MLIP-only phase search.")
app.add_typer(explore_app, name="explore")


def _progress(message: str) -> None:
    console.print(message, markup=False, soft_wrap=True)


def _load_config(path: Path) -> ExplorationConfig:
    try:
        return load_exploration_config(path)
    except (ValueError, OSError, yaml.YAMLError) as error:
        console.print(f"Configuration error: {error}", style="red", markup=False)
        raise typer.Exit(2) from error


def _print_summary(summary: ExplorationSummary) -> None:
    console.print(f"Run directory: {summary.run_directory}", markup=False, soft_wrap=True)
    console.print(
        f"Execution: {summary.status}; scientific outcome: {summary.outcome}", markup=False
    )
    for key in ("generated", "attempted", "converged", "nonconverged", "backend_errors"):
        console.print(f"  {key}: {summary.counts.get(key, 0)}")
    for key in ("selected", "duplicate", "deferred"):
        if key in summary.counts:
            console.print(f"  {key}: {summary.counts[key]}")
    for reason, count in summary.termination_reasons.items():
        console.print(f"  termination {reason}: {count}", markup=False)
    if summary.calculator_loads is not None:
        console.print(f"Calculator loads: {summary.calculator_loads}")
    console.print(f"Rounds: {summary.rounds_completed}/{summary.rounds_requested}")
    for error in summary.errors:
        console.print(f"Error: {error}", style="red", markup=False)
    for note in summary.notes:
        console.print(f"Note: {note}", markup=False)
    console.print("Completed execution does not imply convergence or thermodynamic stability.")
    console.print("This command does not launch LLM, DFT, or active-learning work.")


@explore_app.command("check")
def check(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, help="Exploration YAML."),
) -> None:
    """Validate configuration and dependencies without running calculations."""
    result = check_exploration(_load_config(config))
    console.print(f"Python: {result.python}", markup=False, soft_wrap=True)
    console.print(f"Preflight: {'passed' if result.ready else 'failed'}")
    if result.checkpoint_sha256:
        console.print(f"Checkpoint SHA-256: {result.checkpoint_sha256}")
    for note in result.notes:
        console.print(f"Note: {note}", markup=False)
    for error in result.errors:
        console.print(f"Error: {error}", style="red", markup=False)
    if not result.ready:
        raise typer.Exit(1)


@explore_app.command("run")
def run(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, help="Exploration YAML."),
) -> None:
    """Generate, select, and relax checkpointed rounds; never submit a scheduler job."""
    cfg = _load_config(config)
    try:
        summary = run_exploration(cfg, on_progress=_progress)
    except (ValueError, OSError) as error:
        console.print(f"Exploration error: {error}", style="red", markup=False)
        raise typer.Exit(1) from error
    _print_summary(summary)
    if summary.status == "failed":
        raise typer.Exit(1)


@explore_app.command("summary")
def summary(
    run_directory: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Inspect saved evidence without loading a model or repeating work."""
    try:
        result = summarize_exploration(run_directory)
    except (ValueError, OSError) as error:
        console.print(f"Cannot read exploration evidence: {error}", style="red", markup=False)
        raise typer.Exit(1) from error
    _print_summary(result)
    if result.status == "failed":
        raise typer.Exit(1)


@explore_app.command("resume")
def resume(
    run_directory: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Continue a checkpointed run; do not retry committed failures."""
    try:
        result = resume_exploration(run_directory, on_progress=_progress)
    except (ValueError, OSError) as error:
        console.print(f"Cannot resume exploration: {error}", style="red", markup=False)
        raise typer.Exit(1) from error
    _print_summary(result)
    if result.status == "failed":
        raise typer.Exit(1)
