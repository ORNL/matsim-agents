"""Command-line interface for matsim-agents."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from matsim_agents.orchestration.objective_graph import build_graph
from matsim_agents.orchestration.state import MatSimState

app = typer.Typer(add_completion=False, help="Multi-agent AI for atomistic materials simulation.")
console = Console()


@app.command("debate")
def debate_workflow(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="JSON or YAML ScientificDebateConfig file.",
    ),
) -> None:
    """Run a persisted multi-model debate over one scientific hypothesis."""

    import yaml

    from matsim_agents.workflows.debate import ScientificDebateConfig, run_scientific_debate

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg = ScientificDebateConfig.model_validate(raw)
    console.print_json(run_scientific_debate(cfg).model_dump_json())


@app.command("llm-check")
def llm_check_workflow(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="JSON or YAML LLMCheckConfig file.",
    ),
) -> None:
    """Qualify LLM readiness independently of scientific workflows."""

    import yaml

    from matsim_agents.workflows.llm_check import LLMCheckConfig, run_llm_check

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg = LLMCheckConfig.model_validate(raw)
    result = run_llm_check(cfg)
    console.print_json(result.model_dump_json())
    if result.status == "failed":
        raise typer.Exit(code=1)


@app.command("relax")
def relax_workflow(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="JSON or YAML ScientificRelaxationConfig file.",
    ),
) -> None:
    """Run a provenance-tracked MLIP, DFT, or MLIP-to-DFT relaxation."""

    import yaml

    from matsim_agents.workflows.relaxation import ScientificRelaxationConfig, run_relaxation

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg = ScientificRelaxationConfig.model_validate(raw)
    result = run_relaxation(cfg)
    console.print_json(result.model_dump_json())
    if result.status == "failed":
        raise typer.Exit(code=1)


def _path_or_none(value: Path | None) -> str | None:
    return str(value) if value else None


def _opt_mlip_backend():
    return typer.Option(
        "hydragnn",
        "--mlip-backend",
        help="Relaxation backend: hydragnn | uma.",
        case_sensitive=False,
    )


def _opt_logdir():
    return typer.Option(
        None,
        help="HydraGNN logdir with config.json and checkpoint (required for --mlip-backend hydragnn).",
    )


def _opt_hydragnn_branch_mlp_checkpoint():
    return typer.Option(
        None,
        help="Path to BranchWeightMLP checkpoint (.pt), required for --mlip-backend hydragnn.",
    )


def _opt_uma_model_name():
    return typer.Option(
        "uma-s-1p1",
        "--uma-model-name",
        help="UMA model name/checkpoint (used when --mlip-backend uma).",
    )


def _opt_uma_task():
    return typer.Option(
        "omat",
        "--uma-task",
        help="UMA task head: omat | omol (used when --mlip-backend uma).",
    )


def _opt_checkpoint(help_text: str):
    return typer.Option(None, help=help_text)


def _opt_mlp_device():
    return typer.Option("cuda", help="Device for the auxiliary MLP (cuda|cpu).")


def _opt_precision(help_text: str):
    return typer.Option(None, help=help_text)


def _opt_mlp_precision(help_text: str):
    return typer.Option(None, help=help_text)


def _opt_trigger_al_handoff(default: bool, help_text: str):
    return typer.Option(
        default,
        "--trigger-al-handoff/--no-trigger-al-handoff",
        help=help_text,
    )


def _opt_active_learning_config(help_text: str):
    return typer.Option(None, "--al-config", help=help_text)


def _opt_active_learning_dry_run(help_text: str):
    return typer.Option(True, "--al-dry-run/--al-run", help=help_text)


def _opt_uq_top_weight_threshold():
    return typer.Option(
        0.6,
        "--uq-top-weight-threshold",
        help="Trigger handoff when mean top branch weight is below this value.",
    )


def _opt_uq_min_unreliable_fraction():
    return typer.Option(
        0.25,
        "--uq-min-unreliable-fraction",
        help="Trigger handoff when this fraction of relaxations is low-confidence.",
    )


def _opt_uq_min_relaxations_for_handoff(default: int, help_text: str):
    return typer.Option(
        default,
        "--uq-min-relaxations-for-handoff",
        help=help_text,
    )


def _opt_al_handoff_audit_path(help_text: str):
    return typer.Option(None, "--al-handoff-audit-path", help=help_text)


def _build_run_config(
    *,
    thread_id: str,
    mlip_backend: str,
    logdir: Path | None,
    hydragnn_branch_mlp_checkpoint: Path | None,
    uma_model_name: str,
    uma_task: str,
    checkpoint: str | None,
    output_dir: Path | None,
    mlp_device: str,
    precision: str | None,
    mlp_precision: str | None,
    trigger_al_handoff: bool,
    active_learning_config: Path | None,
    active_learning_dry_run: bool,
    uq_top_weight_threshold: float,
    uq_min_unreliable_fraction: float,
    uq_min_relaxations_for_handoff: int,
    al_handoff_audit_path: Path | None,
) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "mlip_backend": mlip_backend,
            "logdir": _path_or_none(logdir),
            "hydragnn_branch_mlp_checkpoint": _path_or_none(hydragnn_branch_mlp_checkpoint),
            "uma_model_name": uma_model_name,
            "uma_task": uma_task,
            "checkpoint": checkpoint,
            "output_dir": _path_or_none(output_dir),
            "mlp_device": mlp_device,
            "precision": precision,
            "mlp_precision": mlp_precision,
            "trigger_al_handoff": trigger_al_handoff,
            "active_learning_config": _path_or_none(active_learning_config),
            "active_learning_dry_run": active_learning_dry_run,
            "uq_top_weight_threshold": uq_top_weight_threshold,
            "uq_min_unreliable_fraction": uq_min_unreliable_fraction,
            "uq_min_relaxations_for_handoff": uq_min_relaxations_for_handoff,
            "al_handoff_audit_path": _path_or_none(al_handoff_audit_path),
        }
    }


@app.command()
def run(
    objective: str = typer.Argument(..., help="Free-form objective for the agent system."),
    mlip_backend: str = _opt_mlip_backend(),
    logdir: Path | None = _opt_logdir(),
    hydragnn_branch_mlp_checkpoint: Path | None = _opt_hydragnn_branch_mlp_checkpoint(),
    uma_model_name: str = _opt_uma_model_name(),
    uma_task: str = _opt_uma_task(),
    checkpoint: str | None = _opt_checkpoint("HydraGNN checkpoint filename or absolute path."),
    output_dir: Path | None = typer.Option(None, help="Where to write artifacts."),
    mlp_device: str = _opt_mlp_device(),
    precision: str | None = _opt_precision("HydraGNN precision override (fp32|fp64|bf16)."),
    mlp_precision: str | None = _opt_mlp_precision("MLP precision override (fp32|fp64|bf16)."),
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
    trigger_active_learning_on_high_uq: bool = _opt_trigger_al_handoff(
        False,
        "After run relaxations, evaluate UQ and optionally hand off to active learning.",
    ),
    active_learning_config: Path | None = _opt_active_learning_config(
        "Path to a base AL YAML config used when UQ-triggered handoff fires."
    ),
    active_learning_dry_run: bool = _opt_active_learning_dry_run(
        "Plan/report run->AL handoff only, or execute AL loop."
    ),
    uq_top_weight_threshold: float = _opt_uq_top_weight_threshold(),
    uq_min_unreliable_fraction: float = _opt_uq_min_unreliable_fraction(),
    uq_min_relaxations_for_handoff: int = _opt_uq_min_relaxations_for_handoff(
        1,
        "Minimum relaxations required before evaluating run-path handoff.",
    ),
    al_handoff_audit_path: Path | None = _opt_al_handoff_audit_path(
        "Optional JSONL path for UQ decision and run->AL handoff audit records."
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
    config = _build_run_config(
        thread_id=str(uuid.uuid4()),
        mlip_backend=mlip_backend,
        logdir=logdir,
        hydragnn_branch_mlp_checkpoint=hydragnn_branch_mlp_checkpoint,
        uma_model_name=uma_model_name,
        uma_task=uma_task,
        checkpoint=checkpoint,
        output_dir=output_dir,
        mlp_device=mlp_device,
        precision=precision,
        mlp_precision=mlp_precision,
        trigger_al_handoff=trigger_active_learning_on_high_uq,
        active_learning_config=active_learning_config,
        active_learning_dry_run=active_learning_dry_run,
        uq_top_weight_threshold=uq_top_weight_threshold,
        uq_min_unreliable_fraction=uq_min_unreliable_fraction,
        uq_min_relaxations_for_handoff=uq_min_relaxations_for_handoff,
        al_handoff_audit_path=al_handoff_audit_path,
    )

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
    mlip_backend: str = _opt_mlip_backend(),
    logdir: Path | None = _opt_logdir(),
    hydragnn_branch_mlp_checkpoint: Path | None = _opt_hydragnn_branch_mlp_checkpoint(),
    uma_model_name: str = _opt_uma_model_name(),
    uma_task: str = _opt_uma_task(),
    output_dir: Path = typer.Option(
        Path("./outputs"), help="Root directory for discovery artifacts."
    ),
    checkpoint: str | None = _opt_checkpoint("HydraGNN checkpoint filename or path."),
    mlp_device: str = _opt_mlp_device(),
    precision: str | None = _opt_precision("HydraGNN precision override."),
    mlp_precision: str | None = _opt_mlp_precision("MLP precision override."),
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
    trigger_active_learning_on_high_uq: bool = _opt_trigger_al_handoff(
        True,
        "Automatically hand off discovery to active learning when UQ is high.",
    ),
    active_learning_config: Path | None = _opt_active_learning_config(
        "Path to a base AL YAML config. On handoff, discovery overrides "
        "seed_source to compositions=[detected_formula]."
    ),
    active_learning_dry_run: bool = _opt_active_learning_dry_run(
        "Plan/report AL handoff only (dry-run) or execute AL loop."
    ),
    uq_top_weight_threshold: float = _opt_uq_top_weight_threshold(),
    uq_min_unreliable_fraction: float = _opt_uq_min_unreliable_fraction(),
    uq_min_relaxations_for_handoff: int = _opt_uq_min_relaxations_for_handoff(
        3,
        "Minimum number of relaxations required before handoff UQ evaluation.",
    ),
    al_handoff_audit_path: Path | None = _opt_al_handoff_audit_path(
        "Optional JSONL artifact path for discovery->AL handoff audit records "
        "(UQ metrics, trigger rationale, action)."
    ),
):
    """Interactive hypothesis-generation chat that triggers atomistic exploration.

    Defaults to Qwen 2.5 (14B) served via a local Ollama daemon.
    """
    from matsim_agents.chat import DiscoveryChatConfig, run_chat

    panel_models = [s.strip() for s in (critic_panel_models or "").split(",") if s.strip()]
    panel_providers = [s.strip() for s in (critic_panel_providers or "").split(",") if s.strip()]
    panel_base_urls = [s.strip() for s in (critic_panel_base_urls or "").split(",") if s.strip()]

    cfg = DiscoveryChatConfig(
        mlip_backend=mlip_backend,
        logdir=_path_or_none(logdir),
        hydragnn_branch_mlp_checkpoint=_path_or_none(hydragnn_branch_mlp_checkpoint),
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
        critic_panel_models=panel_models,
        critic_panel_providers=panel_providers,
        critic_panel_base_urls=panel_base_urls,
        critic_cross_critique=critic_cross_critique,
        auto_confirm=auto_confirm,
        trigger_active_learning_on_high_uq=trigger_active_learning_on_high_uq,
        active_learning_config=_path_or_none(active_learning_config),
        active_learning_dry_run=active_learning_dry_run,
        uq_top_weight_threshold=uq_top_weight_threshold,
        uq_min_unreliable_fraction=uq_min_unreliable_fraction,
        uq_min_relaxations_for_handoff=uq_min_relaxations_for_handoff,
        al_handoff_audit_path=_path_or_none(al_handoff_audit_path),
    )
    session = run_chat(cfg)
    console.print(
        f"\n[bold]Session finished.[/bold] {len(session.explorations)} composition(s) explored."
    )


@app.command("supervisor-run")
def supervisor_run(
    composition: str = typer.Argument(..., help="Target composition, e.g. Li2MnO3."),
    mlip_backend: str = _opt_mlip_backend(),
    logdir: Path | None = _opt_logdir(),
    hydragnn_branch_mlp_checkpoint: Path | None = _opt_hydragnn_branch_mlp_checkpoint(),
    uma_model_name: str = _opt_uma_model_name(),
    uma_task: str = _opt_uma_task(),
    output_dir: Path = typer.Option(Path("./outputs"), help="Root directory for artifacts."),
    checkpoint: str | None = _opt_checkpoint("HydraGNN checkpoint filename or path."),
    mlp_device: str = _opt_mlp_device(),
    precision: str | None = _opt_precision("HydraGNN precision override."),
    mlp_precision: str | None = _opt_mlp_precision("MLP precision override."),
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
    trigger_active_learning_on_high_uq: bool = _opt_trigger_al_handoff(
        True,
        "Automatically hand off to active learning when UQ is high.",
    ),
    active_learning_config: Path | None = _opt_active_learning_config(
        "Path to a base active-learning YAML config."
    ),
    active_learning_dry_run: bool = _opt_active_learning_dry_run(
        "Plan/report AL handoff only (dry-run) or execute AL loop."
    ),
    uq_top_weight_threshold: float = _opt_uq_top_weight_threshold(),
    uq_min_unreliable_fraction: float = _opt_uq_min_unreliable_fraction(),
    uq_min_relaxations_for_handoff: int = _opt_uq_min_relaxations_for_handoff(
        3,
        "Minimum number of relaxations before handoff UQ evaluation.",
    ),
    al_handoff_audit_path: Path | None = _opt_al_handoff_audit_path(
        "Optional JSONL path for UQ decision and handoff audit records."
    ),
):
    """Run LangGraph supervisor: discovery exploration -> UQ gate -> optional AL handoff."""
    from matsim_agents.orchestration.composition_graph import SupervisorConfig, run_supervisor

    cfg = SupervisorConfig(
        composition=composition,
        mlip_backend=mlip_backend,
        logdir=_path_or_none(logdir),
        hydragnn_branch_mlp_checkpoint=_path_or_none(hydragnn_branch_mlp_checkpoint),
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
        active_learning_config=_path_or_none(active_learning_config),
        active_learning_dry_run=active_learning_dry_run,
        uq_top_weight_threshold=uq_top_weight_threshold,
        uq_min_unreliable_fraction=uq_min_unreliable_fraction,
        uq_min_relaxations_for_handoff=uq_min_relaxations_for_handoff,
        al_handoff_audit_path=_path_or_none(al_handoff_audit_path),
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
