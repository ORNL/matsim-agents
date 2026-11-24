"""NEB subcommand group: ``matsim-agents neb ...`` (dry run unless told otherwise)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

neb_app = typer.Typer(
    add_completion=False,
    help="Prepare, validate and inspect Quantum ESPRESSO NEB calculations for defect migration.",
)
app.add_typer(neb_app, name="neb")

CONFIG_ARGUMENT = typer.Argument(..., exists=True, dir_okay=False, help="NEB YAML config.")
DIRECTORY_ARGUMENT = typer.Argument(..., exists=True, file_okay=False, help="NEB workflow dir.")


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error[/red]: {message}")
    return typer.Exit(code=1)


def _load(config: Path):
    import yaml

    from matsim_agents.backends.dft.qe_neb import QENebConfig
    from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
    from matsim_agents.discovery.neb_endpoints import generate_vacancy_hop, vacancy_neighbors
    from matsim_agents.discovery.radiation_defects import generate_vacancy

    try:
        data = yaml.safe_load(config.read_text())
        if not isinstance(data, dict):
            raise ValueError("the config must be a mapping")
        unknown = set(data) - {"parent", "defect", "lattice_parameter", "qe", "images", "method_id"}
        if unknown:
            raise ValueError(f"unknown keys {sorted(unknown)}")
        parent = generate_alloy(AlloySpec(**data["parent"]))
        lattice = float(data["lattice_parameter"])
        defect = data["defect"]
        kind = defect["mechanism"]
        initial, defect_config = generate_vacancy(parent, "parent", int(defect["site"]))
        if kind == "vacancy_hop":
            neighbors = vacancy_neighbors(initial, defect_config, lattice)
            if not neighbors:
                raise ValueError("the vacancy has no nearest neighbours")
            mover = int(defect.get("mover", neighbors[0]))
            pair = generate_vacancy_hop(initial, defect_config, parent, mover, lattice)
        elif kind == "frenkel_recombination":
            raise ValueError(
                "frenkel_recombination needs an interstitial start; build the pair in Python"
            )
        else:
            raise ValueError(f"unsupported mechanism {kind!r}")
        qe = QENebConfig(**data["qe"])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise _fail(f"{config}: {exc}") from None
    return data, pair, qe


@neb_app.command("prepare")
def neb_prepare(
    config: Path = CONFIG_ARGUMENT,
    out: Path = typer.Option(..., "--out", help="New workflow directory."),
    dispatch: bool = typer.Option(
        False, "--dispatch", help="Actually run neb.x. Default is a dry run."
    ),
    allow_unrelaxed_endpoints: bool = typer.Option(
        False, "--allow-unrelaxed-endpoints", help="Permit dispatch with generated endpoints."
    ),
    ledger: Path | None = typer.Option(None, "--ledger", help="Evidence ledger to append to."),
):
    """Generate endpoints and images, write QE input and plan; run only with --dispatch."""

    from matsim_agents.execution.ledger import EvidenceLedger
    from matsim_agents.workflows.neb import NebWorkflowError, run_neb_workflow

    data, pair, qe = _load(config)
    images = data.get("images", {})
    try:
        outcome = run_neb_workflow(
            pair,
            qe,
            out,
            dispatch=dispatch,
            allow_unrelaxed_endpoints=allow_unrelaxed_endpoints,
            ledger=EvidenceLedger(ledger) if ledger else None,
            method_id=data.get("method_id"),
            **{k: float(v) for k, v in images.items() if k in ("min_distance", "max_step")},
        )
    except NebWorkflowError as exc:
        raise _fail(str(exc)) from None
    plan = outcome.plan
    if plan.status == "rejected":
        for reason in plan.reasons:
            console.print(f"[red]rejected[/red]: {reason}")
        raise typer.Exit(code=1)
    mode = "dry run" if plan.dry_run else "dispatched"
    console.print(f"[green]{mode}[/green]: pathway {plan.pathway_id} ({plan.pathway_state})")
    console.print(f"input: {plan.input_path}")
    if outcome.result is not None:
        console.print(f"result: {outcome.result.state}")


@neb_app.command("validate")
def neb_validate(directory: Path = DIRECTORY_ARGUMENT):
    """Re-read a workflow's pathway with hash and validation checks."""

    from matsim_agents.discovery.neb_serialization import read_pathway

    try:
        pathway = read_pathway(directory / "pathway")
    except ValueError as exc:
        raise _fail(str(exc)) from None
    console.print(f"pathway {pathway.pathway_id}: {pathway.assessment.state}")
    for check in pathway.assessment.failed:
        console.print(f"[red]failed[/red]: {check.name}: {check.detail}")
    if pathway.assessment.state == "rejected":
        raise typer.Exit(code=1)


@neb_app.command("inspect")
def neb_inspect(
    directory: Path = DIRECTORY_ARGUMENT,
    ledger: Path | None = typer.Option(None, "--ledger", help="Evidence ledger to append to."),
    as_json: bool = typer.Option(False, "--json", help="Print the result as JSON."),
):
    """Parse the latest run stage and write neb-result.json."""

    from matsim_agents.execution.ledger import EvidenceLedger
    from matsim_agents.workflows.neb import NebWorkflowError, ingest_neb_results

    try:
        outcome = ingest_neb_results(directory, ledger=EvidenceLedger(ledger) if ledger else None)
    except NebWorkflowError as exc:
        raise _fail(str(exc)) from None
    if as_json:
        console.print_json(json.dumps(outcome.result.model_dump(mode="json")))
        return
    result = outcome.result
    console.print(f"state: {result.state}")
    console.print(f"forward barrier (eV): {result.forward_barrier_eV}")
    console.print(f"reverse barrier (eV): {result.reverse_barrier_eV}")
    for blocker in outcome.qualification_blockers:
        console.print(f"not qualified: {blocker}")


@neb_app.command("resume")
def neb_resume(
    directory: Path = DIRECTORY_ARGUMENT,
    kind: str = typer.Option("continue", "--kind", help="continue or ci."),
    nstep_path: int = typer.Option(..., "--nstep-path", help="Total path steps; above done."),
    run: bool = typer.Option(False, "--run", help="Run the prepared stage. Default prepares only."),
):
    """Explicitly prepare (and optionally run) a continue or climbing-image stage."""

    from matsim_agents.backends.dft.qe_neb import QENebConfig
    from matsim_agents.backends.dft.qe_neb_stages import (
        StageError,
        derive_stage_config,
        load_history,
        prepare_stage,
        run_stage,
    )
    from matsim_agents.discovery.neb_serialization import read_pathway

    if kind not in ("continue", "ci"):
        raise _fail(f"--kind must be continue or ci, not {kind!r}")
    try:
        workdir = directory / "qe"
        first = load_history(workdir).stages
        if not first:
            raise StageError("no stage exists; run `neb prepare` first")
        base = QENebConfig.model_validate(first[0].config)
        config = derive_stage_config(base, kind, nstep_path=nstep_path)
        entry = prepare_stage(config, read_pathway(directory / "pathway").images, kind=kind)
        console.print(f"prepared stage {entry.index} ({kind}): {entry.input_path}")
        if run:
            done = run_stage(workdir)
            console.print(f"stage {done.index}: {done.outcome}")
    except (StageError, ValueError) as exc:
        raise _fail(str(exc)) from None
