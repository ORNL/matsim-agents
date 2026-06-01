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
    logdir: Path = typer.Option(..., help="HydraGNN logdir with config.json and checkpoint."),
    mlp_checkpoint: Path = typer.Option(..., help="Path to BranchWeightMLP checkpoint (.pt)."),
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
            "logdir": str(logdir),
            "mlp_checkpoint": str(mlp_checkpoint),
            "checkpoint": checkpoint,
            "output_dir": str(output_dir) if output_dir else None,
            "mlp_device": mlp_device,
            "precision": precision,
            "mlp_precision": mlp_precision,
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
    logdir: Path = typer.Option(..., help="HydraGNN logdir with config.json and checkpoint."),
    mlp_checkpoint: Path = typer.Option(..., help="Path to BranchWeightMLP checkpoint (.pt)."),
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
    auto_confirm: bool = typer.Option(
        False,
        "--auto-confirm/--ask",
        help="If set, skip the y/N prompt and explore every detected composition.",
    ),
):
    """Interactive hypothesis-generation chat that triggers atomistic exploration.

    Defaults to Qwen 2.5 (14B) served via a local Ollama daemon.
    """
    from matsim_agents.chat import DiscoveryChatConfig, run_chat

    cfg = DiscoveryChatConfig(
        logdir=str(logdir),
        mlp_checkpoint=str(mlp_checkpoint),
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
        auto_confirm=auto_confirm,
    )
    session = run_chat(cfg)
    console.print(
        f"\n[bold]Session finished.[/bold] {len(session.explorations)} composition(s) explored."
    )


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
