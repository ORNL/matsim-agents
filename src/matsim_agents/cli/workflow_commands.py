"""Config-file driven workflow commands: ``debate``, ``llm-check``, ``relax``."""

from __future__ import annotations

from pathlib import Path

import typer

from matsim_agents.cli._app import app, console


def _load_workflow_config(path: Path) -> dict:
    """Read a workflow config; its resources: block gets ${VAR} expansion."""
    import yaml

    from matsim_agents.execution.placement import ResourcePlan

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "resources" in raw:
        plan = ResourcePlan.from_mapping(raw, source=path)
        raw = {key: value for key, value in raw.items() if key != "vars"}
        raw["resources"] = plan
    return raw


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

    from matsim_agents.workflows.debate import ScientificDebateConfig, run_scientific_debate

    cfg = ScientificDebateConfig.model_validate(_load_workflow_config(config))
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

    from matsim_agents.workflows.llm_check import LLMCheckConfig, run_llm_check

    cfg = LLMCheckConfig.model_validate(_load_workflow_config(config))
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
