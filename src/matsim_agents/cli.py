"""Command-line interface for matsim-agents."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from matsim_agents.graph import build_graph
from matsim_agents.state import MatSimState

app = typer.Typer(add_completion=False, help="Multi-agent AI for atomistic materials simulation.")
console = Console()


@app.command()
def run(
    objective: str = typer.Argument(..., help="Free-form objective for the agent system."),
    mlp_backend: str = typer.Option(
        "hydragnn",
        "--mlp-backend",
        help="Relaxation backend: hydragnn | uma.",
        case_sensitive=False,
    ),
    logdir: Path | None = typer.Option(
        None,
        help="HydraGNN logdir with config.json and checkpoint (required for --mlp-backend hydragnn).",
    ),
    mlp_checkpoint: Path | None = typer.Option(
        None,
        help="Path to BranchWeightMLP checkpoint (.pt), required for --mlp-backend hydragnn.",
    ),
    uma_model_name: str = typer.Option(
        "uma-s-1p1",
        "--uma-model-name",
        help="UMA model name/checkpoint (used when --mlp-backend uma).",
    ),
    uma_task: str = typer.Option(
        "omat",
        "--uma-task",
        help="UMA task head: omat | omol (used when --mlp-backend uma).",
    ),
    checkpoint: str | None = typer.Option(
        None, help="HydraGNN checkpoint filename or absolute path."
    ),
    output_dir: Path | None = typer.Option(None, help="Where to write artifacts."),
    mlp_device: str = typer.Option("cuda", help="Device for the auxiliary MLP (cuda|cpu)."),
    precision: str | None = typer.Option(
        None, help="HydraGNN precision override (fp32|fp64|bf16)."
    ),
    mlp_precision: str | None = typer.Option(None, help="MLP precision override (fp32|fp64|bf16)."),
    max_iterations: int = typer.Option(5, help="Maximum executor iterations."),
    llm_provider: str = typer.Option(
        "ollama",
        "--llm-provider",
        help="LLM backend: ollama | vllm | openai | anthropic.",
        case_sensitive=False,
    ),
    llm_model: str | None = typer.Option(
        None, "--llm-model", help="Model identifier (provider-specific)."
    ),
    llm_base_url: str | None = typer.Option(
        None,
        "--llm-base-url",
        help="Override server URL (Ollama or vLLM /v1 endpoint).",
    ),
    trigger_active_learning_on_high_uq: bool = typer.Option(
        False,
        "--trigger-al-handoff/--no-trigger-al-handoff",
        help="After run relaxations, evaluate UQ and optionally hand off to active learning.",
    ),
    active_learning_config: Path | None = typer.Option(
        None,
        "--al-config",
        help="Path to a base AL YAML config used when UQ-triggered handoff fires.",
    ),
    active_learning_dry_run: bool = typer.Option(
        True,
        "--al-dry-run/--al-run",
        help="Plan/report run->AL handoff only, or execute AL loop.",
    ),
    uq_top_weight_threshold: float = typer.Option(
        0.6,
        "--uq-top-weight-threshold",
        help="Trigger handoff when mean top branch weight is below this value.",
    ),
    uq_min_unreliable_fraction: float = typer.Option(
        0.25,
        "--uq-min-unreliable-fraction",
        help="Trigger handoff when this fraction of relaxations is low-confidence.",
    ),
    uq_min_relaxations_for_handoff: int = typer.Option(
        1,
        "--uq-min-relaxations-for-handoff",
        help="Minimum relaxations required before evaluating run-path handoff.",
    ),
    al_handoff_audit_path: Path | None = typer.Option(
        None,
        "--al-handoff-audit-path",
        help="Optional JSONL path for UQ decision and run->AL handoff audit records.",
    ),
):
    """Run the planner -> executor -> analyst graph for a given objective."""
    graph = build_graph()
    state = MatSimState(
        objective=objective,
        max_iterations=max_iterations,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
    )
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "mlp_backend": mlp_backend,
            "logdir": str(logdir) if logdir else None,
            "mlp_checkpoint": str(mlp_checkpoint) if mlp_checkpoint else None,
            "uma_model_name": uma_model_name,
            "uma_task": uma_task,
            "checkpoint": checkpoint,
            "output_dir": str(output_dir) if output_dir else None,
            "mlp_device": mlp_device,
            "precision": precision,
            "mlp_precision": mlp_precision,
            "trigger_al_handoff": trigger_active_learning_on_high_uq,
            "active_learning_config": str(active_learning_config) if active_learning_config else None,
            "active_learning_dry_run": active_learning_dry_run,
            "uq_top_weight_threshold": uq_top_weight_threshold,
            "uq_min_unreliable_fraction": uq_min_unreliable_fraction,
            "uq_min_relaxations_for_handoff": uq_min_relaxations_for_handoff,
            "al_handoff_audit_path": str(al_handoff_audit_path) if al_handoff_audit_path else None,
        }
    }

    final = graph.invoke(state, config=config)

    console.print(Panel.fit("matsim-agents run summary", style="bold cyan"))
    console.print(f"Objective: {final['objective']}")
    console.print(f"Tasks executed: {len(final['results'])}")
    for r in final["results"]:
        console.print(f"  • {r.optimized_structure_path}  E={r.final_energy_eV:.4f} eV")
    console.print(Panel(final.get("analysis") or "(no analysis)", title="Analyst"))


@app.command()
def plan(objective: str):
    """Run the planner only and print the proposed task list as JSON."""
    from matsim_agents.agents.planner import planner_node

    state = MatSimState(objective=objective)
    out = planner_node(state)
    console.print_json(json.dumps([t.model_dump() for t in out["plan"]]))


@app.command()
def chat(
    mlp_backend: str = typer.Option(
        "hydragnn",
        "--mlp-backend",
        help="Relaxation backend: hydragnn | uma.",
        case_sensitive=False,
    ),
    logdir: Path | None = typer.Option(
        None,
        help="HydraGNN logdir with config.json and checkpoint (required for --mlp-backend hydragnn).",
    ),
    mlp_checkpoint: Path | None = typer.Option(
        None,
        help="Path to BranchWeightMLP checkpoint (.pt), required for --mlp-backend hydragnn.",
    ),
    uma_model_name: str = typer.Option(
        "uma-s-1p1",
        "--uma-model-name",
        help="UMA model name/checkpoint (used when --mlp-backend uma).",
    ),
    uma_task: str = typer.Option(
        "omat",
        "--uma-task",
        help="UMA task head: omat | omol (used when --mlp-backend uma).",
    ),
    output_dir: Path = typer.Option(
        Path("./outputs"), help="Root directory for discovery artifacts."
    ),
    checkpoint: str | None = typer.Option(None, help="HydraGNN checkpoint filename or path."),
    mlp_device: str = typer.Option("cuda", help="Device for the auxiliary MLP (cuda|cpu)."),
    precision: str | None = typer.Option(None, help="HydraGNN precision override."),
    mlp_precision: str | None = typer.Option(None, help="MLP precision override."),
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
    auto_confirm: bool = typer.Option(
        False,
        "--auto-confirm/--ask",
        help="If set, skip the y/N prompt and explore every detected composition.",
    ),
    trigger_active_learning_on_high_uq: bool = typer.Option(
        True,
        "--trigger-al-handoff/--no-trigger-al-handoff",
        help="Automatically hand off discovery to active learning when UQ is high.",
    ),
    active_learning_config: Path | None = typer.Option(
        None,
        "--al-config",
        help=(
            "Path to a base AL YAML config. On handoff, discovery overrides "
            "seed_source to compositions=[detected_formula]."
        ),
    ),
    active_learning_dry_run: bool = typer.Option(
        True,
        "--al-dry-run/--al-run",
        help="Plan/report AL handoff only (dry-run) or execute AL loop.",
    ),
    uq_top_weight_threshold: float = typer.Option(
        0.6,
        "--uq-top-weight-threshold",
        help="Trigger handoff when mean top branch weight is below this value.",
    ),
    uq_min_unreliable_fraction: float = typer.Option(
        0.25,
        "--uq-min-unreliable-fraction",
        help="Trigger handoff when this fraction of relaxations is low-confidence.",
    ),
    uq_min_relaxations_for_handoff: int = typer.Option(
        3,
        "--uq-min-relaxations-for-handoff",
        help="Minimum number of relaxations required before handoff UQ evaluation.",
    ),
    al_handoff_audit_path: Path | None = typer.Option(
        None,
        "--al-handoff-audit-path",
        help=(
            "Optional JSONL artifact path for discovery->AL handoff audit records "
            "(UQ metrics, trigger rationale, action)."
        ),
    ),
):
    """Interactive hypothesis-generation chat that triggers atomistic exploration.

    Defaults to Qwen 2.5 (14B) served via a local Ollama daemon.
    """
    from matsim_agents.chat import DiscoveryChatConfig, run_chat

    cfg = DiscoveryChatConfig(
        mlp_backend=mlp_backend,
        logdir=str(logdir) if logdir else None,
        mlp_checkpoint=str(mlp_checkpoint) if mlp_checkpoint else None,
        uma_model_name=uma_model_name,
        uma_task=uma_task,
        output_dir=str(output_dir),
        checkpoint=checkpoint,
        mlp_device=mlp_device,
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
        auto_confirm=auto_confirm,
        trigger_active_learning_on_high_uq=trigger_active_learning_on_high_uq,
        active_learning_config=(str(active_learning_config) if active_learning_config else None),
        active_learning_dry_run=active_learning_dry_run,
        uq_top_weight_threshold=uq_top_weight_threshold,
        uq_min_unreliable_fraction=uq_min_unreliable_fraction,
        uq_min_relaxations_for_handoff=uq_min_relaxations_for_handoff,
        al_handoff_audit_path=(str(al_handoff_audit_path) if al_handoff_audit_path else None),
    )
    session = run_chat(cfg)
    console.print(
        f"\n[bold]Session finished.[/bold] {len(session.explorations)} composition(s) explored."
    )


@app.command("supervisor-run")
def supervisor_run(
    composition: str = typer.Argument(..., help="Target composition, e.g. Li2MnO3."),
    mlp_backend: str = typer.Option(
        "hydragnn",
        "--mlp-backend",
        help="Relaxation backend: hydragnn | uma.",
        case_sensitive=False,
    ),
    logdir: Path | None = typer.Option(
        None,
        help="HydraGNN logdir with config.json and checkpoint (required for --mlp-backend hydragnn).",
    ),
    mlp_checkpoint: Path | None = typer.Option(
        None,
        help="Path to BranchWeightMLP checkpoint (.pt), required for --mlp-backend hydragnn.",
    ),
    uma_model_name: str = typer.Option(
        "uma-s-1p1",
        "--uma-model-name",
        help="UMA model name/checkpoint (used when --mlp-backend uma).",
    ),
    uma_task: str = typer.Option(
        "omat",
        "--uma-task",
        help="UMA task head: omat | omol (used when --mlp-backend uma).",
    ),
    output_dir: Path = typer.Option(Path("./outputs"), help="Root directory for artifacts."),
    checkpoint: str | None = typer.Option(None, help="HydraGNN checkpoint filename or path."),
    mlp_device: str = typer.Option("cuda", help="Device for the auxiliary MLP (cuda|cpu)."),
    precision: str | None = typer.Option(None, help="HydraGNN precision override."),
    mlp_precision: str | None = typer.Option(None, help="MLP precision override."),
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
    trigger_active_learning_on_high_uq: bool = typer.Option(
        True,
        "--trigger-al-handoff/--no-trigger-al-handoff",
        help="Automatically hand off to active learning when UQ is high.",
    ),
    active_learning_config: Path | None = typer.Option(
        None,
        "--al-config",
        help="Path to a base active-learning YAML config.",
    ),
    active_learning_dry_run: bool = typer.Option(
        True,
        "--al-dry-run/--al-run",
        help="Plan/report AL handoff only (dry-run) or execute AL loop.",
    ),
    uq_top_weight_threshold: float = typer.Option(
        0.6,
        "--uq-top-weight-threshold",
        help="Trigger handoff when mean top branch weight is below this value.",
    ),
    uq_min_unreliable_fraction: float = typer.Option(
        0.25,
        "--uq-min-unreliable-fraction",
        help="Trigger handoff when this fraction of relaxations is low-confidence.",
    ),
    uq_min_relaxations_for_handoff: int = typer.Option(
        3,
        "--uq-min-relaxations-for-handoff",
        help="Minimum number of relaxations before handoff UQ evaluation.",
    ),
    al_handoff_audit_path: Path | None = typer.Option(
        None,
        "--al-handoff-audit-path",
        help="Optional JSONL path for UQ decision and handoff audit records.",
    ),
):
    """Run LangGraph supervisor: discovery exploration -> UQ gate -> optional AL handoff."""
    from matsim_agents.supervisor import SupervisorConfig, run_supervisor

    cfg = SupervisorConfig(
        composition=composition,
        mlp_backend=mlp_backend,
        logdir=str(logdir) if logdir else None,
        mlp_checkpoint=str(mlp_checkpoint) if mlp_checkpoint else None,
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
        active_learning_config=(str(active_learning_config) if active_learning_config else None),
        active_learning_dry_run=active_learning_dry_run,
        uq_top_weight_threshold=uq_top_weight_threshold,
        uq_min_unreliable_fraction=uq_min_unreliable_fraction,
        uq_min_relaxations_for_handoff=uq_min_relaxations_for_handoff,
        al_handoff_audit_path=(str(al_handoff_audit_path) if al_handoff_audit_path else None),
    )

    final = run_supervisor(cfg)
    console.print(Panel.fit("matsim-agents supervisor summary", style="bold cyan"))
    console.print(final.get("summary") or "(no summary)")


# --------------------------------------------------------------------------- #
# Active-learning subcommand group: `matsim-agents al ...`                    #
# --------------------------------------------------------------------------- #

al_app = typer.Typer(
    add_completion=False,
    help="HydraGNN <-> DFT (VASP or Quantum ESPRESSO) active-learning loop.",
)
app.add_typer(al_app, name="al")


@al_app.command("run")
def al_run(
    config: Path = typer.Argument(..., help="Path to AL YAML config (see ALConfig schema)."),
    log_level: str = typer.Option("INFO", help="Python logging level."),
):
    """Run the active-learning loop end-to-end."""
    import logging

    from matsim_agents.active_learning import ALConfig
    from matsim_agents.active_learning.loop import run_active_learning

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = ALConfig.from_yaml(config)
    run_active_learning(cfg)


@al_app.command("validate-config")
def al_validate_config(config: Path = typer.Argument(...)):
    """Validate an AL YAML config without running anything."""
    from matsim_agents.active_learning import ALConfig

    cfg = ALConfig.from_yaml(config)
    console.print_json(cfg.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
