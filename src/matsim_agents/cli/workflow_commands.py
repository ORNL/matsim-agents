"""Config-file driven workflow commands: ``debate``, ``llm-check``, ``relax``, ``characterize``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from matsim_agents.cli._app import app, console

JSON_OPTION = typer.Option(False, "--json", help="Print the full result as JSON.")


def _show(label: str, value: Any) -> None:
    if value not in (None, "", [], {}):
        console.print(f"{label}: {value}", markup=False, soft_wrap=True)


def _number(value: float | None, unit: str) -> str:
    return "n/a" if value is None else f"{value:.6g} {unit}"


def _load_workflow_config(path: Path) -> dict:
    """Read a workflow config; its resources: block gets ${VAR} expansion."""
    import yaml

    from matsim_agents.execution.placement import ResourcePlan

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Saved resolved configs carry "resources": null once their servers are resolved.
    if isinstance(raw, dict) and raw.get("resources") is not None:
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
    as_json: bool = JSON_OPTION,
) -> None:
    """Run a persisted multi-model debate over one scientific hypothesis."""

    from matsim_agents.workflows.debate import ScientificDebateConfig, run_scientific_debate

    cfg = ScientificDebateConfig.model_validate(_load_workflow_config(config))
    result = run_scientific_debate(cfg)
    if as_json:
        # The texts are in the run's transcript.md and contributions.jsonl.
        console.print_json(result.model_dump_json(exclude={"turns", "verdicts", "synthesis"}))
        return
    console.print(f"Debate {result.status}: {result.rounds_completed} round(s)", markup=False)
    _show("Hypothesis", result.hypothesis)
    _show("Verdicts", len(result.verdicts))
    _show("Structured hypotheses", len(result.scientific_hypotheses))
    _show("Incomplete contributions", ", ".join(result.incomplete_contributions))
    _show("Transcript", result.transcript_path)
    _show("Run directory", result.run_directory)


@app.command("llm-check")
def llm_check_workflow(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="JSON or YAML LLMCheckConfig file.",
    ),
    as_json: bool = JSON_OPTION,
) -> None:
    """Qualify LLM readiness independently of scientific workflows."""

    from matsim_agents.workflows.llm_check import LLMCheckConfig, run_llm_check

    cfg = LLMCheckConfig.model_validate(_load_workflow_config(config))
    result = run_llm_check(cfg)
    if as_json:
        console.print_json(result.model_dump_json())
    else:
        console.print(f"LLM check {result.status}: {result.provider} {result.model}", markup=False)
        for stage, passed in result.stages.items():
            console.print(f"  {stage}: {'passed' if passed else 'failed'}", markup=False)
        _show("Failure", result.failure_reason)
        _show("Incomplete replies", ", ".join(result.incomplete_replies))
        _show("Run directory", result.run_directory)
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
    as_json: bool = JSON_OPTION,
) -> None:
    """Run a provenance-tracked MLIP, DFT, or MLIP-to-DFT relaxation."""

    import yaml

    from matsim_agents.backends.mlip.config import resolve_mlip_paths
    from matsim_agents.workflows.relaxation import ScientificRelaxationConfig, run_relaxation

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if isinstance(raw.get("mlip"), dict):
        resolve_mlip_paths(raw["mlip"], config.resolve().parent)
    cfg = ScientificRelaxationConfig.model_validate(raw)
    result = run_relaxation(cfg)
    if as_json:
        console.print_json(result.model_dump_json())
    else:
        console.print(f"Relaxation {result.status} ({result.mode})", markup=False)
        for stage in result.stages:
            console.print(
                f"  {stage.stage} [{stage.backend}]: "
                f"{'converged' if stage.converged else 'not converged'}, "
                f"E={_number(stage.energy_eV, 'eV')}, "
                f"max force={_number(stage.max_force_eV_per_A, 'eV/A')}, steps={stage.steps}",
                markup=False,
                soft_wrap=True,
            )
            _show("    failure", stage.failure_reason)
        _show("Final structure", result.final_structure_path)
        _show("Failure", result.failure_reason)
        _show("Run directory", result.run_directory)
    if result.status == "failed":
        raise typer.Exit(code=1)


@app.command("characterize")
def characterize_workflow(
    contract: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="YAML or JSON CharacterizationContract file.",
    ),
    as_json: bool = JSON_OPTION,
) -> None:
    """Answer a characterization contract (relaxation, energy ranking, or hull stability).

    Running the same contract again resumes with its outstanding calculations.
    """

    from matsim_agents.campaign.characterization import load_contract, run_characterization

    report = run_characterization(load_contract(contract))
    if as_json:
        console.print_json(report.model_dump_json())
    else:
        console.print(
            f"Characterization {report.contract_id} ({report.recipe}): {report.status}",
            markup=False,
        )
        _show("Conclusion", report.conclusion)
        for value in report.observables:
            where = f" [{value.structure}]" if value.structure else ""
            console.print(
                f"  {value.name}{where} = {value.value} {value.unit} ({value.evidence_level})",
                markup=False,
                soft_wrap=True,
            )
        for item in report.missing:
            _show("Missing", item)
        for item in report.limitations:
            _show("Limitation", item)
        _show("Recommended next test", report.recommended_next_test)
    if report.status in {"invalid_protocol", "failed_infrastructure"}:
        raise typer.Exit(code=1)
