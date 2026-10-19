"""Interactive mode: ``chat`` hypothesis-generation session."""

from __future__ import annotations

from pathlib import Path

import typer

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


@app.command()
def chat(
    llm_only: bool = typer.Option(
        False,
        "--llm-only/--with-mlip",
        help="Run conversational LLMs only, without MLIP validation or atomistic tools.",
    ),
    mlip_backend: str = opt_mlip_backend(),
    logdir: Path | None = opt_logdir(),
    hydragnn_branch_mlp_checkpoint: Path | None = opt_hydragnn_branch_mlp_checkpoint(),
    hydragnn_inference_head: str | None = opt_hydragnn_inference_head(),
    uma_model_name: str = opt_uma_model_name(),
    uma_task: str = opt_uma_task(),
    output_dir: Path = typer.Option(
        Path("./outputs"), help="Root directory for discovery artifacts."
    ),
    checkpoint: str | None = opt_checkpoint("HydraGNN checkpoint filename or path."),
    mlp_device: str = opt_mlp_device(),
    resources: Path | None = typer.Option(
        None,
        "--resources",
        help=(
            "Resource plan (a file with a resources: block). Its mlip pool relaxes "
            "candidates concurrently, one worker per device; see docs/resource-dispatch.md."
        ),
    ),
    precision: str | None = opt_precision("HydraGNN precision override."),
    mlp_precision: str | None = opt_mlp_precision("MLP precision override."),
    optimizer: str = typer.Option(
        "FIRE", "--ase-structure-optimizer", help="ASE structure optimizer for relaxations."
    ),
    maxiter: int = typer.Option(200, help="Max relaxation steps per phase."),
    fmax: float = typer.Option(0.02, help="Stop relaxation when max force < fmax (eV/Å)."),
    relative_increase_threshold: float = typer.Option(
        0.05,
        "--relative-increase-threshold",
        help="Abort+rollback if |F|max grows by more than this fraction in one step. "
        "Increase (e.g. 10.0) to let FIRE cross small barriers from symmetric starts.",
    ),
    n_random: int = typer.Option(
        50,
        "--n-random",
        help="Number of supplementary pyXtal random structures per composition "
        "(in addition to every applicable AFLOW prototype decoration). "
        "Set to 0 to disable. Silently degrades to 0 if pyxtal is not installed.",
    ),
    random_seed: int = typer.Option(0, "--random-seed", help="Seed for the pyXtal RNG."),
    llm_provider: str = typer.Option("ollama", "--llm-provider", case_sensitive=False),
    llm_model: str = typer.Option("qwen2.5:14b", "--llm-model"),
    llm_base_url: str | None = typer.Option(None, "--llm-base-url"),
    llm_peer_review: bool = typer.Option(
        False,
        "--llm-peer-review/--no-llm-peer-review",
        help="Enable proposer/critic multi-LLM rounds to challenge and refine hypotheses.",
    ),
    peer_review_rounds: int = typer.Option(
        1,
        "--peer-review-rounds",
        help="Number of proposer/critic revision rounds when --llm-peer-review is enabled.",
    ),
    critic_llm_provider: str | None = typer.Option(
        None,
        "--critic-llm-provider",
        help="Critic provider (defaults to --llm-provider).",
    ),
    critic_llm_model: str | None = typer.Option(
        None,
        "--critic-llm-model",
        help="Critic model id (provider-specific).",
    ),
    critic_llm_base_url: str | None = typer.Option(
        None,
        "--critic-llm-base-url",
        help="Optional critic endpoint override.",
    ),
    critic_panel_models: str | None = typer.Option(
        None,
        "--critic-panel-models",
        help="Comma-separated critic model ids for panel mode.",
    ),
    critic_panel_providers: str | None = typer.Option(
        None,
        "--critic-panel-providers",
        help="Optional comma-separated providers aligned with --critic-panel-models.",
    ),
    critic_panel_base_urls: str | None = typer.Option(
        None,
        "--critic-panel-base-urls",
        help="Optional comma-separated base URLs aligned with --critic-panel-models.",
    ),
    critic_cross_critique: bool = typer.Option(
        False,
        "--critic-cross-critique/--no-critic-cross-critique",
        help="Enable critic-to-critic challenge before proposer synthesis.",
    ),
    auto_confirm: bool = typer.Option(
        False,
        "--auto-confirm/--ask",
        help="If set, skip the y/N prompt and explore every detected composition.",
    ),
    trigger_active_learning_on_high_uq: bool = opt_trigger_al_handoff(
        True,
        "Automatically hand off discovery to active learning when UQ is high.",
    ),
    active_learning_config: Path | None = opt_active_learning_config(
        "Path to a base AL YAML config. On handoff, discovery overrides "
        "seed_source to compositions=[detected_formula]."
    ),
    active_learning_dry_run: bool = opt_active_learning_dry_run(
        "Plan/report AL handoff only (dry-run) or execute AL loop."
    ),
    uq_top_weight_threshold: float = opt_uq_top_weight_threshold(),
    uq_min_unreliable_fraction: float = opt_uq_min_unreliable_fraction(),
    uq_min_relaxations_for_handoff: int = opt_uq_min_relaxations_for_handoff(
        3,
        "Minimum number of relaxations required before handoff UQ evaluation.",
    ),
    al_handoff_audit_path: Path | None = opt_al_handoff_audit_path(
        "Optional JSONL artifact path for discovery->AL handoff audit records "
        "(UQ metrics, trigger rationale, action)."
    ),
):
    """Interactive hypothesis-generation chat that triggers atomistic exploration.

    Defaults to Qwen 2.5 (14B) served via a local Ollama daemon.
    """
    from matsim_agents.orchestration.interactive_chat import DiscoveryChatConfig, run_chat

    panel_models = [s.strip() for s in (critic_panel_models or "").split(",") if s.strip()]
    panel_providers = [s.strip() for s in (critic_panel_providers or "").split(",") if s.strip()]
    panel_base_urls = [s.strip() for s in (critic_panel_base_urls or "").split(",") if s.strip()]

    cfg = DiscoveryChatConfig(
        llm_only=llm_only,
        mlip_backend=mlip_backend,
        logdir=path_or_none(logdir),
        hydragnn_branch_mlp_checkpoint=path_or_none(hydragnn_branch_mlp_checkpoint),
        hydragnn_inference_head=hydragnn_inference_head,
        uma_model_name=uma_model_name,
        uma_task=uma_task,
        output_dir=str(output_dir),
        checkpoint=checkpoint,
        mlp_device=mlp_device,
        resources=path_or_none(resources),
        precision=precision,
        mlp_precision=mlp_precision,
        optimizer=optimizer,
        maxiter=maxiter,
        fmax=fmax,
        relative_increase_threshold=relative_increase_threshold,
        n_random=n_random,
        random_seed=random_seed,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        enable_hypothesis_debate=llm_peer_review,
        debate_rounds=peer_review_rounds,
        critic_llm_provider=critic_llm_provider,
        critic_llm_model=critic_llm_model,
        critic_llm_base_url=critic_llm_base_url,
        critic_panel_models=panel_models,
        critic_panel_providers=panel_providers,
        critic_panel_base_urls=panel_base_urls,
        critic_cross_critique=critic_cross_critique,
        auto_confirm=auto_confirm,
        trigger_active_learning_on_high_uq=trigger_active_learning_on_high_uq,
        active_learning_config=path_or_none(active_learning_config),
        active_learning_dry_run=active_learning_dry_run,
        uq_top_weight_threshold=uq_top_weight_threshold,
        uq_min_unreliable_fraction=uq_min_unreliable_fraction,
        uq_min_relaxations_for_handoff=uq_min_relaxations_for_handoff,
        al_handoff_audit_path=path_or_none(al_handoff_audit_path),
    )
    session = run_chat(cfg)
    console.print(
        f"\n[bold]Session finished.[/bold] {len(session.explorations)} composition(s) explored."
    )
