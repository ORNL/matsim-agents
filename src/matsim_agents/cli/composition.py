"""Composition mode: ``supervisor-run`` for one target composition."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel

from matsim_agents.cli._app import app, console
from matsim_agents.cli.options import (
    opt_active_learning_config,
    opt_active_learning_dry_run,
    opt_al_handoff_audit_path,
    opt_checkpoint,
    opt_hydragnn_branch_mlp_checkpoint,
    opt_hydragnn_inference_head,
    opt_logdir,
    opt_mlip_backend,
    opt_mlp_device,
    opt_mlp_precision,
    opt_precision,
    opt_trigger_al_handoff,
    opt_uma_model_name,
    opt_uma_task,
    opt_uq_min_relaxations_for_handoff,
    opt_uq_min_unreliable_fraction,
    opt_uq_top_weight_threshold,
    path_or_none,
)


@app.command("supervisor-run")
def supervisor_run(
    composition: str = typer.Argument(..., help="Target composition, e.g. Li2MnO3."),
    mlip_backend: str = opt_mlip_backend(),
    logdir: Path | None = opt_logdir(),
    hydragnn_branch_mlp_checkpoint: Path | None = opt_hydragnn_branch_mlp_checkpoint(),
    hydragnn_inference_head: str | None = opt_hydragnn_inference_head(),
    uma_model_name: str = opt_uma_model_name(),
    uma_task: str = opt_uma_task(),
    output_dir: Path = typer.Option(Path("./outputs"), help="Root directory for artifacts."),
    checkpoint: str | None = opt_checkpoint("HydraGNN checkpoint filename or path."),
    mlp_device: str = opt_mlp_device(),
    precision: str | None = opt_precision("HydraGNN precision override."),
    mlp_precision: str | None = opt_mlp_precision("MLP precision override."),
    optimizer: str = typer.Option(
        "FIRE", "--ase-structure-optimizer", help="ASE structure optimizer for relaxations."
    ),
    maxiter: int = typer.Option(200, help="Max relaxation steps per phase."),
    maxstep: float = typer.Option(1e-2, help="Optimizer max step."),
    fmax: float = typer.Option(0.02, help="Stop relaxation when max force < fmax (eV/Å)."),
    relative_increase_threshold: float = typer.Option(
        0.05,
        "--relative-increase-threshold",
        help="Abort+rollback when |F|max grows by more than this fraction in one step.",
    ),
    n_random: int = typer.Option(
        50,
        "--n-random",
        help="Number of supplementary pyXtal random structures per composition.",
    ),
    random_seed: int = typer.Option(0, "--random-seed", help="Seed for pyXtal RNG."),
    trigger_active_learning_on_high_uq: bool = opt_trigger_al_handoff(
        True,
        "Automatically hand off to active learning when UQ is high.",
    ),
    active_learning_config: Path | None = opt_active_learning_config(
        "Path to a base active-learning YAML config."
    ),
    active_learning_dry_run: bool = opt_active_learning_dry_run(
        "Plan/report AL handoff only (dry-run) or execute AL loop."
    ),
    uq_top_weight_threshold: float = opt_uq_top_weight_threshold(),
    uq_min_unreliable_fraction: float = opt_uq_min_unreliable_fraction(),
    uq_min_relaxations_for_handoff: int = opt_uq_min_relaxations_for_handoff(
        3,
        "Minimum number of relaxations before handoff UQ evaluation.",
    ),
    al_handoff_audit_path: Path | None = opt_al_handoff_audit_path(
        "Optional JSONL path for UQ decision and handoff audit records."
    ),
):
    """Run LangGraph supervisor: discovery exploration -> UQ gate -> optional AL handoff."""
    from matsim_agents.orchestration.composition_graph import SupervisorConfig, run_supervisor

    cfg = SupervisorConfig(
        composition=composition,
        mlip_backend=mlip_backend,
        logdir=path_or_none(logdir),
        hydragnn_branch_mlp_checkpoint=path_or_none(hydragnn_branch_mlp_checkpoint),
        hydragnn_inference_head=hydragnn_inference_head,
        uma_model_name=uma_model_name,
        uma_task=uma_task,
        output_dir=str(output_dir),
        checkpoint=checkpoint,
        mlp_device=mlp_device,
        precision=precision,
        mlp_precision=mlp_precision,
        optimizer=optimizer,
        maxiter=maxiter,
        maxstep=maxstep,
        fmax=fmax,
        relative_increase_threshold=relative_increase_threshold,
        n_random=n_random,
        random_seed=random_seed,
        trigger_active_learning_on_high_uq=trigger_active_learning_on_high_uq,
        active_learning_config=path_or_none(active_learning_config),
        active_learning_dry_run=active_learning_dry_run,
        uq_top_weight_threshold=uq_top_weight_threshold,
        uq_min_unreliable_fraction=uq_min_unreliable_fraction,
        uq_min_relaxations_for_handoff=uq_min_relaxations_for_handoff,
        al_handoff_audit_path=path_or_none(al_handoff_audit_path),
    )

    final = run_supervisor(cfg)
    console.print(Panel.fit("matsim-agents supervisor summary", style="bold cyan"))
    console.print(final.get("summary") or "(no summary)")
