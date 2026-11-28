"""Active-learning subcommand group: ``matsim-agents active-learning ...`` (alias ``al``)."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console, fail
from matsim_agents.cli.options import load_mlip_option, opt_mlip_config

al_app = typer.Typer(
    add_completion=False,
    help="MLIP <-> DFT (VASP or Quantum ESPRESSO) active-learning loop.",
)
app.add_typer(al_app, name="active-learning")
app.add_typer(al_app, name="al", hidden=True)


TEMPLATE_ENV = "MATSIM_ALLOW_TEMPLATE_CONFIG"


def _guard_template(config: Path, allow_template: bool) -> None:
    """Refuse to run a repository example unless the caller opts in explicitly."""
    from matsim_agents.cli.config_tools import REPO
    from matsim_agents.workflows.active_learning.campaign_init import is_repo_template

    if not is_repo_template(config, REPO):
        return
    message = (
        f"{config} is an example shipped with matsim-agents. Examples illustrate the "
        "schema; their seeds, DFT settings and elemental references belong to the "
        "example, not to your campaign. Create your own with "
        "`matsim-agents al init-campaign`."
    )
    if not (allow_template or os.environ.get(TEMPLATE_ENV) == "1"):
        fail(f"{message} To reproduce the example itself, pass --allow-template.", code=2)
    console.print(f"[yellow]Warning:[/yellow] {message}")


opt_allow_template = typer.Option(
    False,
    "--allow-template",
    help=f"Run a repository example as is (also set by {TEMPLATE_ENV}=1).",
)


@al_app.command("run")
def al_run(
    config: Path = typer.Argument(..., help="Path to AL YAML config (see ALConfig schema)."),
    mlip_config: Path | None = opt_mlip_config(),
    log_level: str = typer.Option("INFO", help="Python logging level."),
    allow_template: bool = opt_allow_template,
):
    """Run the active-learning loop end-to-end."""
    import logging

    from matsim_agents.workflows.active_learning import ALConfig
    from matsim_agents.workflows.active_learning.loop import run_active_learning

    _guard_template(config, allow_template)
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    cfg = ALConfig.from_yaml(config, mlip=load_mlip_option(mlip_config))
    run_active_learning(cfg)


@al_app.command("validate-config")
def al_validate_config(
    config: Path = typer.Argument(...),
    mlip_config: Path | None = opt_mlip_config(),
):
    """Validate an AL YAML config without running anything."""
    from matsim_agents.cli.config_tools import REPO
    from matsim_agents.workflows.active_learning import ALConfig
    from matsim_agents.workflows.active_learning.campaign_init import (
        check_reference_coverage,
        elements_of,
        is_repo_template,
    )

    cfg = ALConfig.from_yaml(config, mlip=load_mlip_option(mlip_config))
    strategies = cfg.sampling.strategies if cfg.sampling is not None else []
    seed_sources = [
        strategy.parameters.seed_source for strategy in strategies if strategy.method == "md"
    ]
    if cfg.md is not None:
        seed_sources.insert(0, cfg.md.seed_source)
    seed_paths = [path for source in seed_sources for path in source.paths if path.is_file()]
    compositions = [formula for source in seed_sources for formula in source.compositions]
    neb_paths = [
        path
        for strategy in strategies
        if strategy.method == "neb"
        for path in (strategy.parameters.initial_path, strategy.parameters.final_path)
        if path.is_file()
    ]
    try:
        check_reference_coverage(
            cfg.trainer,
            elements_of(seed_paths + neb_paths, compositions),
        )
    except ValueError as error:
        fail(str(error))
    console.print_json(cfg.model_dump_json(indent=2))
    if is_repo_template(config, REPO):
        console.print(
            "[yellow]Note:[/yellow] this is a repository example. Create your campaign's "
            "own configuration with `matsim-agents al init-campaign`."
        )


@al_app.command("init-campaign")
def al_init_campaign(
    campaign_dir: Path = typer.Argument(
        ..., help="New campaign directory; files are never overwritten."
    ),
    mlip_config: Path = typer.Option(
        ...,
        "--mlip-config",
        help="YAML with an mlip: block (e.g. examples/mlip/uma-s-1p1-omat.yaml).",
    ),
    dft_backend: str = typer.Option(..., "--dft-backend", help="qe or vasp."),
    seeds: list[Path] = typer.Option([], "--seed", help="Seed structure file (repeatable)."),
    compositions: list[str] = typer.Option(
        [], "--composition", help="Reduced formula to seed from prototypes (repeatable)."
    ),
    extra_elements: list[str] = typer.Option(
        [], "--element", help="Extra element to cover, e.g. a later dopant (repeatable)."
    ),
    stress_tolerance: float = typer.Option(
        0.01, help="Bulk reference relaxation stress tolerance, eV/A^3."
    ),
    reference_kpts: tuple[int, int, int] | None = typer.Option(
        None, help="k-point mesh for the primitive bulk reference cells (QE)."
    ),
    reference_kspacing: float | None = typer.Option(
        None,
        help="Max k-point spacing (2*pi/A) for each bulk reference cell (QE); "
        "alternative to --reference-kpts.",
    ),
    kspacing_from_mesh: tuple[float, int] | None = typer.Option(
        None,
        help="Match the spacing of a mesh on a cubic supercell: SIDE_A KPTS "
        "(e.g. 25.28 3 for 3x3x3 on an 8x8x8 bcc cell).",
    ),
    force_tolerance: float | None = typer.Option(
        None, help="Reference relaxation max-force tolerance, eV/A."
    ),
    magnetic_elements: list[str] = typer.Option(
        [],
        "--magnetic-element",
        help="Element whose NM, FM and AFM reference phases are all declared (repeatable); "
        "replaces the default Fe, Co, Ni, Cr.",
    ),
    dft_settings: Path | None = typer.Option(
        None,
        help="YAML mapping merged into the generated dft.<backend> block "
        "(e.g. pseudopotentials, degauss_ry, extra_electrons).",
    ),
    ecutwfc_ry: float | None = typer.Option(None, help="QE wavefunction cutoff, Ry."),
    ecutrho_ry: float | None = typer.Option(None, help="QE charge-density cutoff, Ry."),
    kpts: tuple[int, int, int] | None = typer.Option(None, help="QE k-points for labelled cells."),
    preset: str | None = typer.Option(None, help="Facility resource preset (e.g. frontier)."),
):
    """Create your campaign's AL config and its elemental reference phase plan."""
    from matsim_agents.workflows.active_learning.campaign_init import write_campaign

    if dft_backend not in {"qe", "vasp"}:
        fail("--dft-backend must be qe or vasp", code=2)
    if kspacing_from_mesh is not None:
        from matsim_agents.workflows.active_learning.campaign_init import kspacing_of_mesh

        if reference_kspacing is not None:
            fail("Give --reference-kspacing or --kspacing-from-mesh, not both", code=2)
        side, n = kspacing_from_mesh
        reference_kspacing = kspacing_of_mesh([side] * 3, [n] * 3)
    settings = None
    if dft_settings is not None:
        import yaml

        settings = yaml.safe_load(dft_settings.read_text())
        if not isinstance(settings, dict):
            fail(f"{dft_settings}: expected a mapping", code=2)
    try:
        proposal = write_campaign(
            campaign_dir,
            mlip_config=mlip_config,
            dft_backend=dft_backend,
            seed_paths=seeds,
            compositions=compositions,
            extra_elements=extra_elements,
            stress_tolerance_eV_per_A3=stress_tolerance,
            reference_kpts=reference_kpts,
            reference_kspacing_inv_A=reference_kspacing,
            relaxation_force_tolerance_eV_per_A=force_tolerance,
            magnetic_elements=magnetic_elements or None,
            ecutwfc_ry=ecutwfc_ry,
            ecutrho_ry=ecutrho_ry,
            kpts=kpts,
            dft_settings=settings,
            preset=preset,
        )
    except (OSError, ValueError) as error:
        fail(str(error))
    console.print(
        f"Wrote {campaign_dir}/al.yaml and {campaign_dir}/elemental-phases.yaml", markup=False
    )
    for note in proposal.review:
        console.print(f"  review: {note}", markup=False)
    for element, reason in proposal.unresolved.items():
        console.print(f"  UNRESOLVED {element}: {reason}", markup=False)
    console.print(
        "Nothing is approved. Review the phase plan, set the approvals and placeholders in "
        "al.yaml, then run `matsim-agents al validate-config`.",
        markup=False,
    )
