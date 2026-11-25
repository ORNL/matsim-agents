"""Hierarchical-transfer campaign commands: ``matsim-agents campaign ...``.

Every command is read-only or dry-run except ``approve``, which submits a bounded batch only
when the config has ``execution.dry_run: false`` and ``--approved-by`` matches the approver
written in the config. Nothing here claims real-alloy transferability.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

campaign_app = typer.Typer(
    add_completion=False,
    help=(
        "Hierarchical W-Ta-Cr-V-Hf-Mo transfer campaign: inventory, coverage, transferability, "
        "preview, cost, bounded dispatch, monitor, review. Dry run by default."
    ),
)
app.add_typer(campaign_app, name="campaign")

CONFIG_ARGUMENT = typer.Argument(..., exists=True, dir_okay=False, help="Campaign YAML config.")
WORKSPACE_OPTION = typer.Option(..., "--workspace", "-w", help="Campaign state directory.")
JSON_OPTION = typer.Option(False, "--json", help="Print JSON instead of a table.")


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error[/red]: {message}")
    return typer.Exit(code=1)


def _open(config: Path, workspace: Path):
    from pydantic import ValidationError

    from matsim_agents.campaign.hierarchy_campaign import CampaignWorkspace
    from matsim_agents.campaign.hierarchy_config import load_campaign_config

    try:
        cfg = load_campaign_config(config)
        return CampaignWorkspace(workspace, cfg)
    except (ValidationError, TypeError, ValueError, OSError) as exc:
        raise _fail(f"{config}: {exc}") from None


def _proposals(ws):
    from matsim_agents.campaign.hierarchy_campaign import load_proposals

    path = ws.config.data.candidates
    if path is None:
        raise _fail("data.candidates is not set; there is nothing to plan")
    try:
        return load_proposals(path)
    except (ValueError, OSError) as exc:
        raise _fail(f"{path}: {exc}") from None


def _emit(payload, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, indent=2, default=str))


@campaign_app.command("check")
def check(config: Path = CONFIG_ARGUMENT) -> None:
    """Validate the config; unknown keys and a non-1024-site target are errors."""
    from matsim_agents.campaign.hierarchy_config import load_campaign_config

    try:
        cfg = load_campaign_config(config)
    except Exception as exc:  # noqa: BLE001 - report any validation problem uniformly
        raise _fail(f"{config}: {exc}") from None
    mode = "dry run" if cfg.execution.dry_run else "LIVE (approved)"
    console.print(f"ok: {cfg.campaign.campaign_id} ({cfg.target.n_sites} sites, {mode})")


@campaign_app.command("inventory")
def inventory_cmd(
    config: Path = CONFIG_ARGUMENT,
    workspace: Path = WORKSPACE_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Ingest configured datasets once; no new DFT is run."""
    from matsim_agents.campaign.hierarchy_campaign import inventory

    rows = inventory(_open(config, workspace))
    if as_json:
        return _emit(rows, True)
    for row in rows:
        compat = row["compatibility"]["compatible"]
        console.print(
            f"{row['name']}: {row['n_entries']} entries "
            f"({row['n_new_calculations']} new, {row['n_duplicate_calculations']} duplicate), "
            f"subspaces={len(row['covered_subspaces'])}, off-target={row['off_target_elements']}, "
            f"compatible={compat}"
        )
    if not rows:
        console.print("no datasets configured")


@campaign_app.command("coverage")
def coverage_cmd(
    config: Path = CONFIG_ARGUMENT,
    workspace: Path = WORKSPACE_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Interaction coverage of the reused data and the explicit list of missing interactions."""
    from matsim_agents.campaign.hierarchy_campaign import coverage

    cov = coverage(_open(config, workspace))
    report = cov["report"]
    if as_json:
        return _emit({"basis": cov["basis"], **report.model_dump()}, True)
    console.print(
        f"coverage {report.fraction:.3f} ({report.n_covered}/{report.n_required}), "
        f"basis={','.join(cov['basis'])}"
    )
    console.print(
        f"missing: {len(report.missing)}, higher-order: {len(report.missing_higher_order)}"
    )
    for key in report.missing_higher_order[:20]:
        console.print(f"  missing {key}")


@campaign_app.command("transferability")
def transferability_cmd(
    config: Path = CONFIG_ARGUMENT,
    workspace: Path = WORKSPACE_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Held-out transfer verdicts; a property without data stays NOT_YET_ESTABLISHED."""
    from matsim_agents.campaign.hierarchy_campaign import transferability

    ws = _open(config, workspace)
    try:
        runs = transferability(ws)
    except (ValueError, OSError) as exc:
        raise _fail(str(exc)) from None
    if as_json:
        return _emit([r.model_dump() for r in runs], True)
    for run in runs:
        for v in run.verdicts:
            console.print(
                f"{run.property:>26} {run.experiment:<12} n_new={v.n_new:<4} "
                f"{v.registry_status}: {v.reason}"
            )


@campaign_app.command("preview")
def preview(
    config: Path = CONFIG_ARGUMENT,
    workspace: Path = WORKSPACE_OPTION,
    acquisition: str | None = typer.Option(
        None, "--acquisition", help="policy | deterministic_baseline (default: from config)."
    ),
    as_json: bool = JSON_OPTION,
) -> None:
    """Show the calculations that would be dispatched, with their gate status. Submits nothing."""
    from matsim_agents.campaign.hierarchy_campaign import plan

    ws = _open(config, workspace)
    if acquisition not in (None, "policy", "deterministic_baseline"):
        raise _fail("--acquisition must be policy or deterministic_baseline")
    result = plan(ws, _proposals(ws), acquisition=acquisition)
    if as_json:
        return _emit(result.model_dump(), True)
    console.print(
        f"acquisition={result.acquisition}: {len(result.selected)} selected, "
        f"{result.expected_node_hours:.1f} node-hours expected, "
        f"{result.ready_node_hours:.1f} READY"
    )
    for c in result.calculations:
        if c.selected:
            console.print(
                f"  {c.candidate_id} [{c.category}] {c.predicted_node_hours:.1f} nh "
                f"{c.status}: {'; '.join(c.status_reasons)}"
            )


@campaign_app.command("cost")
def cost(
    config: Path = CONFIG_ARGUMENT,
    workspace: Path = WORKSPACE_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Expected cost of the selected batch against the remaining allocation."""
    from matsim_agents.campaign.hierarchy_campaign import plan

    ws = _open(config, workspace)
    result = plan(ws, _proposals(ws))
    summary = ws.ledger.summary()
    payload = {
        "expected_node_hours": result.expected_node_hours,
        "ready_node_hours": result.ready_node_hours,
        "remaining_allocation": summary.remaining,
        "by_category": summary.by_category,
        "fits_remaining_allocation": result.expected_node_hours <= summary.remaining + 1e-9,
    }
    if as_json:
        return _emit(payload, True)
    console.print(
        f"expected {result.expected_node_hours:.1f} nh, READY {result.ready_node_hours:.1f} nh, "
        f"remaining {summary.remaining:.1f} nh, fits={payload['fits_remaining_allocation']}"
    )


@campaign_app.command("approve")
def approve(
    config: Path = CONFIG_ARGUMENT,
    workspace: Path = WORKSPACE_OPTION,
    approved_by: str = typer.Option(..., "--approved-by", help="Must match execution.approved_by."),
    as_json: bool = JSON_OPTION,
) -> None:
    """Submit one bounded batch of READY calculations; refuses under dry_run or a wrong approver."""
    from matsim_agents.campaign.hierarchy_campaign import dispatch, sbatch_submitter

    ws = _open(config, workspace)
    outcome = dispatch(ws, _proposals(ws), sbatch_submitter(ws.config), approved_by=approved_by)
    if as_json:
        return _emit(outcome.model_dump(), True)
    console.print(f"submitted={outcome.submitted} duplicates={outcome.skipped_duplicates}")
    for calc, why in outcome.blocked.items():
        console.print(f"  blocked {calc}: {why}")
    if outcome.needs_manual_check:
        console.print(f"[yellow]needs manual check[/yellow]: {outcome.needs_manual_check}")
    if outcome.stopped_because:
        console.print(f"stopped: {outcome.stopped_because}")
        if not outcome.submitted:
            raise typer.Exit(code=2)


@campaign_app.command("monitor")
def monitor_cmd(
    config: Path = CONFIG_ARGUMENT,
    workspace: Path = WORKSPACE_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Poll submitted jobs once and charge their actual node-hours."""
    from matsim_agents.campaign.hierarchy_campaign import monitor

    ws = _open(config, workspace)
    try:
        states = monitor(ws)
    except Exception as exc:  # noqa: BLE001 - scheduler errors must not corrupt state
        raise _fail(f"scheduler query failed: {exc}") from None
    if as_json:
        return _emit(states, True)
    for calc, state in states.items():
        console.print(f"{calc}: {state}")
    if not states:
        console.print("no submissions yet")


@campaign_app.command("review")
def review_cmd(
    config: Path = CONFIG_ARGUMENT,
    workspace: Path = WORKSPACE_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Report table, stage gates and the next decision."""
    from matsim_agents.campaign.hierarchy_campaign import review

    ws = _open(config, workspace)
    proposals = None
    if ws.config.data.candidates is not None:
        proposals = _proposals(ws)
    report, decision = review(ws, proposals)
    if as_json:
        return _emit({"report": report.model_dump(), "plan": decision.model_dump()}, True)
    for key, value in report.table():
        console.print(f"{key:<30} {value}")
    for note in report.notes:
        console.print(f"note: {note}")
