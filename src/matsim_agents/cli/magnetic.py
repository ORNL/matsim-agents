"""Collinear magnetic configuration commands: ``matsim-agents magnetic ...``.

These commands prepare and inspect starting states. They never run DFT.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

magnetic_app = typer.Typer(
    add_completion=False,
    help=(
        "Prepare and inspect collinear magnetic starting states. "
        "When to use: before a spin-polarized DFT comparison. Never runs DFT."
    ),
)
app.add_typer(magnetic_app, name="magnetic")


@magnetic_app.command("policies")
def policies(as_json: bool = typer.Option(False, "--json", help="Print JSON.")) -> None:
    """List material-family policies and their status (never reported as qualified)."""
    from matsim_agents.discovery.magnetism.policies import policy_readiness

    report = policy_readiness()
    if as_json:
        console.print_json(json.dumps(report))
        return
    for name, entry in report.items():
        console.print(f"{name}: trials={entry['trials']} qualified=False", markup=False)
    console.print("Policies choose starting states only; they make no ground-state claim.")


@magnetic_app.command("generate")
def generate(
    structure: Path = typer.Argument(..., exists=True, dir_okay=False, help="Structure file."),
    output: Path = typer.Option(..., "--output", "-o", help="Directory for configuration JSON."),
    policy: str | None = typer.Option(None, "--policy", help="Policy name; default: auto-select."),
    lattice_a: float | None = typer.Option(None, "--lattice-a", help="Cubic a (Angstrom)."),
    structure_id: str = typer.Option("structure", "--structure-id"),
) -> None:
    """Write the policy's trial configurations for a structure (one JSON per state)."""
    import numpy as np
    from ase.io import read

    from matsim_agents.discovery.magnetism.policies import (
        POLICIES,
        configurations_for_policy,
        select_policy,
    )

    try:
        atoms = read(str(structure))
    except Exception as error:  # noqa: BLE001 - any reader failure is a user input error
        console.print(f"Cannot read structure: {error}", style="red", markup=False)
        raise typer.Exit(2) from error
    if "atom_id" not in atoms.arrays:
        atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    chosen = next((p for p in POLICIES if p.name == policy), None) if policy else None
    if policy and chosen is None:
        console.print(f"Unknown policy {policy!r}", style="red", markup=False)
        raise typer.Exit(2)
    chosen = chosen or select_policy(atoms)
    if chosen is None:
        console.print("No policy matches this composition; create states explicitly.")
        raise typer.Exit(1)
    configs = configurations_for_policy(atoms, structure_id, chosen, lattice_a=lattice_a)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for config in configs:
        path = output / f"{config.config_id}.json"
        path.write_text(config.to_json())
        manifest.append(
            {
                "config_id": config.config_id,
                "mode": config.mode,
                "fingerprint": config.fingerprint(),
            }
        )
    (output / "manifest.json").write_text(
        json.dumps(
            {"policy": chosen.name, "statement": chosen.statement, "states": manifest}, indent=2
        )
    )
    console.print(f"Policy {chosen.name}: wrote {len(configs)} configuration(s) to {output}")
    console.print("Starting states only; DFT has not been run and nothing is qualified.")


@magnetic_app.command("show")
def show(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, help="Configuration JSON."),
) -> None:
    """Print the mode, assumptions and fingerprint of a saved configuration."""
    from matsim_agents.discovery.magnetism.config import MagneticConfiguration

    try:
        loaded = MagneticConfiguration.from_json(config.read_text())
    except ValueError as error:
        console.print(f"Invalid configuration: {error}", style="red", markup=False)
        raise typer.Exit(2) from error
    console.print(f"Mode: {loaded.mode}; atoms: {len(loaded.atom_ids)}", markup=False)
    console.print(f"Spin polarized: {loaded.spin_polarized}; qualification: {loaded.qualification}")
    console.print(f"Fingerprint: {loaded.fingerprint()}", markup=False)
    for assumption in loaded.assumptions:
        console.print(f"Assumption: {assumption}", markup=False, soft_wrap=True)
