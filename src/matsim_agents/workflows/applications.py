"""Dry-run checks of an application against the workflow it names.

Each check loads the application's workflow config, confirms it agrees with the
application target, and returns a summary. Nothing is computed or submitted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from matsim_agents.core.application import ApplicationConfig


def _config_path(app: ApplicationConfig):
    path = app.resolve(app.workflow_config)
    if path is None:
        raise ValueError(f"application {app.name!r} has no workflow_config")
    return path


def _full_supercell(app: ApplicationConfig) -> dict[str, Any]:
    from matsim_agents.workflows.full_supercell_campaign import FullSupercellCampaign

    cfg = FullSupercellCampaign.from_yaml(_config_path(app))
    p = cfg.parent_supercell
    if app.target is not None:
        t = app.target
        if dict(p.composition) != dict(t.composition):
            raise ValueError("workflow_config composition differs from the application target")
        if tuple(p.conventional_repeats) != tuple(t.supercell):
            raise ValueError("workflow_config repeats differ from the application target")
        if p.expected_parent_atoms != t.expected_sites:
            raise ValueError("workflow_config site count differs from the application target")
        if p.lattice != t.parent_lattice:
            raise ValueError("workflow_config lattice differs from the application target")
    return {
        "campaign_id": cfg.campaign_id,
        "sites": p.expected_parent_atoms,
        "repeats": list(p.conventional_repeats),
        "lattice": p.lattice,
    }


def _hierarchy(app: ApplicationConfig) -> dict[str, Any]:
    from matsim_agents.workflows.hierarchy.config import load_campaign_config

    cfg = load_campaign_config(_config_path(app))
    if app.target is not None and dict(cfg.target.composition) != dict(app.target.composition):
        raise ValueError("workflow_config composition differs from the application target")
    return {
        "campaign_id": cfg.campaign.campaign_id,
        "elements": list(cfg.elements),
        "orders": list(cfg.orders),
        "dry_run": cfg.execution.dry_run,
    }


def _composition_exploration(app: ApplicationConfig) -> dict[str, Any]:
    from matsim_agents.core.composition import parse_composition

    bad = [f for f in app.compositions if parse_composition(f) is None]
    if bad:
        raise ValueError(f"unparseable compositions: {bad}")
    return {"compositions": list(app.compositions)}


CHECKS: dict[str, Callable[[ApplicationConfig], dict[str, Any]]] = {
    "full_supercell_campaign": _full_supercell,
    "hierarchy": _hierarchy,
    "composition_exploration": _composition_exploration,
}


def _target_parent(app: ApplicationConfig) -> dict[str, Any]:
    """Build the target's parent supercell to show the structure generator supports it."""
    from matsim_agents.science.structures.alloys import AlloySpec, build_supercell

    t = app.target
    if t is None:
        return {}
    spec = AlloySpec(
        crystal_structure=t.parent_lattice,
        counts=dict(t.composition),
        repeats=tuple(t.supercell),
        seed=0,
    )
    n_built = len(build_supercell(spec))
    if n_built != t.expected_sites:
        raise ValueError(f"built {n_built} parent sites, target expects {t.expected_sites}")
    return {"parent_sites_built": n_built}


def check_application(app: ApplicationConfig) -> dict[str, Any]:
    """Validate an application against its workflow; raise if they disagree."""
    try:
        check = CHECKS[app.workflow]
    except KeyError:
        raise ValueError(f"unknown workflow {app.workflow!r}; known: {sorted(CHECKS)}") from None
    return {"name": app.name, "workflow": app.workflow, **check(app), **_target_parent(app)}


__all__ = ["CHECKS", "check_application"]
