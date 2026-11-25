"""Package tools behind one entry point: ``matsim-agents tools list|run``."""

from __future__ import annotations

import inspect
import subprocess
import sys
from importlib import import_module

import typer

from matsim_agents.cli._app import CLIState, app, console, error_console, fail

# name -> (module, one-line purpose); descriptions are static so ``list`` loads nothing heavy.
TOOLS: dict[str, tuple[str, str]] = {
    "finetune-eval": (
        "matsim_agents.workflows.active_learning.finetune_eval",
        "Fine-tune and evaluate one MLIP backend on one dataset",
    ),
    "evaluate": (
        "matsim_agents.workflows.active_learning.evaluate",
        "Evaluate an MLIP against a held-out DFT test set",
    ),
    "eval-report": (
        "matsim_agents.workflows.active_learning.eval_report",
        "Per-group error reports and comparisons of MLIP evaluations",
    ),
    "elemental-dft": (
        "matsim_agents.workflows.active_learning.elemental_dft",
        "Approved, cached DFT references for elemental phases",
    ),
    "splits": (
        "matsim_agents.data.splits",
        "Objective-aware train/validation/test splits of DFT datasets",
    ),
    "formation-labels": (
        "matsim_agents.data.formation_training",
        "Prepare formation-energy labels without modifying raw DFT data",
    ),
    "finetune-mace": (
        "matsim_agents.backends.mlip.training.finetune_mace",
        "Fine-tune a MACE foundation model on an AL dataset",
    ),
    "finetune-uma": (
        "matsim_agents.backends.mlip.training.finetune_uma",
        "Fine-tune a UMA (FairChem) foundation model on an AL dataset",
    ),
    "finetune-hydragnn": (
        "matsim_agents.backends.mlip.training.finetune_hydragnn",
        "Fine-tune the multi-branch HydraGNN foundation model",
    ),
    "finetune-hydragnn-newhead": (
        "matsim_agents.backends.mlip.training.finetune_hydragnn_newhead",
        "Fine-tune HydraGNN with one new head instead of the branch heads",
    ),
    "branch-routing": (
        "matsim_agents.backends.mlip.training.branch_routing",
        "Map AL datasets to HydraGNN branch heads",
    ),
}

tools_app = typer.Typer(help="List and run the package's training, dataset and benchmark tools.")
app.add_typer(tools_app, name="tools")


@tools_app.command("list")
def list_tools() -> None:
    """List available tools."""
    width = max(map(len, TOOLS))
    for name, (_, purpose) in TOOLS.items():
        console.print(f"{name:{width}}  {purpose}", markup=False)
    console.print(
        "Run one with `matsim-agents tools run NAME [ARGS...]`; add --help for its options."
    )


@tools_app.command(
    "run",
    add_help_option=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_tool(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Tool name from `matsim-agents tools list`."),
) -> None:
    """Run a tool; every argument after NAME (including --help) is passed to it."""
    if name not in TOOLS:
        fail(f"unknown tool {name!r}; run `matsim-agents tools list`", code=2)
    module = import_module(TOOLS[name][0])
    arguments = list(ctx.args)
    # argparse derives the usage line's program name from sys.argv[0].
    saved, sys.argv = sys.argv, [f"matsim-agents tools run {name}", *arguments]
    try:
        code = (
            module.main(arguments) if inspect.signature(module.main).parameters else module.main()
        )
    except (subprocess.CalledProcessError, RuntimeError) as error:
        # The tool ran and failed, so this is exit 1 rather than a usage error.
        if ctx.find_root().ensure_object(CLIState).debug:
            raise
        error_console.print(
            f"Error: tool {name!r} failed: {error}", style="red", markup=False, highlight=False
        )
        error_console.print(
            "Hint: rerun with --debug before the command for a traceback.", markup=False
        )
        raise typer.Exit(1) from None
    except SystemExit as error:
        if isinstance(error.code, int):
            raise
        if ctx.find_root().ensure_object(CLIState).debug:
            raise
        error_console.print(f"Configuration error: {error.code}", style="red", markup=False)
        raise typer.Exit(2) from None
    finally:
        sys.argv = saved
    raise typer.Exit(code or 0)
