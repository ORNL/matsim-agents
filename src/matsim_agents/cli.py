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
    min_atoms: int = typer.Option(32, help="Auto-tile each prototype to at least this many atoms."),
    supercell: str | None = typer.Option(
        None, help="Explicit supercell, e.g. '2x2x2'. Overrides --min-atoms."
    ),
    include_2d: bool = typer.Option(
        False,
        "--include-2d/--no-include-2d",
        help="Also enumerate 2-D prototypes (graphene, h-BN, MX2).",
    ),
    num_layers: int = typer.Option(1, help="Layers stacked for every 2-D prototype."),
    vacuum: float = typer.Option(15.0, help="Vacuum gap (Å) along z for 2-D prototypes."),
    interlayer: float | None = typer.Option(
        None, help="Override interlayer separation (Å) for 2-D prototypes."
    ),
    n_orderings: int = typer.Option(
        1,
        help="Sample up to N symmetrically-distinct site decorations per prototype (multi-species only).",
    ),
    lattice_scales: str | None = typer.Option(
        None, help="Comma-separated isotropic cell-scale factors, e.g. '0.96,1.0,1.04'."
    ),
    ordering_seed: int = typer.Option(
        0, help="RNG seed for the ordering sampler (reproducibility)."
    ),
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

    sc_tuple: tuple[int, int, int] | None = None
    if supercell:
        try:
            parts = [int(x) for x in supercell.lower().replace(",", "x").split("x") if x]
            if len(parts) != 3:
                raise ValueError
            sc_tuple = (parts[0], parts[1], parts[2])
        except ValueError as exc:
            raise typer.BadParameter("--supercell must be 'NxNxN' (e.g. '2x2x2').") from exc

    scales_tuple: tuple[float, ...] | None = None
    if lattice_scales:
        try:
            scales_tuple = tuple(float(x) for x in lattice_scales.split(",") if x.strip())
            if not scales_tuple:
                scales_tuple = None
        except ValueError as exc:
            raise typer.BadParameter(
                "--lattice-scales must be a comma-separated list of floats, e.g. '0.96,1.0,1.04'."
            ) from exc

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
        supercell=sc_tuple,
        min_atoms=min_atoms,
        include_2d=include_2d,
        num_layers=num_layers,
        vacuum=vacuum,
        interlayer=interlayer,
        n_orderings=n_orderings,
        lattice_scales=scales_tuple,
        ordering_seed=ordering_seed,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        auto_confirm=auto_confirm,
    )
    session = run_chat(cfg)
    console.print(
        f"\n[bold]Session finished.[/bold] {len(session.explorations)} composition(s) explored."
    )


if __name__ == "__main__":
    app()
