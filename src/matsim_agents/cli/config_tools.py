"""Configuration helpers: ``matsim-agents config init`` and ``config schema``."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console, fail

REPO = Path(__file__).resolve().parents[3]

# kind -> (pydantic model, annotated example in a source checkout, command that consumes it)
CONFIG_KINDS: dict[str, tuple[str, str, str]] = {
    "relax": (
        "matsim_agents.workflows.relaxation:ScientificRelaxationConfig",
        "examples/relaxation/scientific_relaxation.example.yaml",
        "matsim-agents relax {path}",
    ),
    "characterize": (
        "matsim_agents.workflows.characterization:CharacterizationContract",
        "examples/characterization/relaxation.yaml",
        "matsim-agents characterize {path}",
    ),
    "debate": (
        "matsim_agents.workflows.debate:ScientificDebateConfig",
        "examples/debate/equal-independent.yaml",
        "matsim-agents debate {path}",
    ),
    "llm-check": (
        "matsim_agents.workflows.llm_check:LLMCheckConfig",
        "examples/llm_check/llm_check.example.yaml",
        "matsim-agents llm-check {path}",
    ),
    "al": (
        "matsim_agents.workflows.active_learning.config:ALConfig",
        "examples/active_learning/al_config.example.yaml",
        "matsim-agents al validate-config {path}",
    ),
    "explore": (
        "matsim_agents.workflows.exploration.config:ExplorationConfig",
        "examples/workflows/explore-nb2o5.yaml",
        "matsim-agents explore check {path}",
    ),
    "workflow": (
        "matsim_agents.cli.workflow_catalog:WorkflowConfig",
        "examples/workflows/exploration-workflow.yaml",
        "matsim-agents workflow check {path}",
    ),
}
KIND_HELP = ", ".join(CONFIG_KINDS)

config_app = typer.Typer(help="Create starter configuration files and export their JSON Schemas.")
app.add_typer(config_app, name="config")


def _kind(kind: str) -> tuple[str, str, str]:
    if kind not in CONFIG_KINDS:
        fail(f"unknown configuration kind {kind!r}; choose one of: {KIND_HELP}", code=2)
    return CONFIG_KINDS[kind]


@config_app.command("list")
def list_kinds() -> None:
    """List configuration kinds, their templates, and the command that reads them."""
    for kind, (_, example, command) in CONFIG_KINDS.items():
        console.print(f"{kind:13} {example}  ->  {command.format(path='FILE')}", markup=False)


@config_app.command("init")
def init(
    kind: str = typer.Argument(..., help=f"Configuration kind: {KIND_HELP}."),
    output: Path = typer.Argument(..., help="Destination file; never overwritten."),
) -> None:
    """Copy an annotated starter configuration to OUTPUT."""
    _, example, command = _kind(kind)
    source = REPO / example
    if not source.is_file():
        fail(f"template {example} is only available in a source checkout of matsim-agents")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as stream:
            stream.write(source.read_bytes())
    except FileExistsError:
        fail(f"refusing to overwrite {output}")
    console.print(f"Created {output} from {example}", markup=False)
    console.print(
        f"Relative paths in it were written for {source.parent}; check them before running.",
        markup=False,
    )
    console.print(f"Next: edit it, then run `{command.format(path=output)}`", markup=False)
    if kind == "al":
        console.print(
            "For a real campaign, prefer `matsim-agents al init-campaign`: it also proposes "
            "the elemental reference phases for your elements.",
            markup=False,
        )


@config_app.command("schema")
def schema(
    kind: str = typer.Argument(..., help=f"Configuration kind: {KIND_HELP}."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the schema here instead of stdout."
    ),
) -> None:
    """Print the JSON Schema for a configuration kind (for editor validation/completion)."""
    module, name = _kind(kind)[0].split(":")
    document = json.dumps(getattr(import_module(module), name).model_json_schema(), indent=2)
    if output is None:
        typer.echo(document)
    else:
        output.write_text(document + "\n", encoding="utf-8")
        console.print(f"Wrote {output}", markup=False)
