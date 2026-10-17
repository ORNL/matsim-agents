"""Config-file driven workflow commands: ``debate``, ``llm-check``, ``relax``."""

from __future__ import annotations

from pathlib import Path

import typer

from matsim_agents.cli._app import app, console


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
