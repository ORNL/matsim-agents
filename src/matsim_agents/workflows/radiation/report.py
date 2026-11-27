"""Machine-readable report of one radiation-damage run.

The report restates what the run established and what it did not. It never reports a
qualified result: ``qualification`` is copied from the run, and the standing caveats are always
present.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.workflows.radiation.damage import RadiationDamageRun

REPORT_FILE = "radiation-damage-report.json"
REPORT_SCHEMA = "matsim-radiation-damage-report/1"
STANDING_CAVEATS = (
    "kinetics use an unqualified barrier and a user-supplied attempt frequency",
    "the model is harmonic transition-state theory on a single hop, not a damage cascade",
    "no scientific qualification has been granted to any result in this report",
)


def build_report(run: RadiationDamageRun) -> dict[str, Any]:
    """A JSON-serializable summary of ``run``."""
    state = run.state
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "directory": run.directory,
        "config_hash": state.config_hash,
        "transition_id": state.transition_id,
        "mode": "dry_run" if state.dry_run else "dispatched",
        "stages": [{"name": s.name, "status": s.status, "detail": s.detail} for s in state.stages],
        "qualification": run.qualification,
        "blockers": list(run.blockers),
        "caveats": list(STANDING_CAVEATS),
        "kinetics": None,
        "network": None,
    }
    if run.network_report is not None:
        net = run.network_report
        report["network"] = {
            "ready_for_exploratory_kmc": net.ready_for_exploratory_kmc,
            "ready_for_qualified_kmc": net.ready_for_qualified_kmc,
            "issues": list(net.issues),
        }
    if run.ensemble is not None:
        ens = run.ensemble
        report["kinetics"] = {
            "temperature_K": ens.temperature_K,
            "n_trajectories": ens.n_trajectories,
            "seed": ens.seed,
            "horizon_s": ens.horizon_s,
            "outcomes": {
                kind: {
                    "count": o.count,
                    "probability": o.probability,
                    "wilson_low": o.wilson_low,
                    "wilson_high": o.wilson_high,
                }
                for kind, o in ens.outcomes.items()
            },
            "censored_fraction": ens.censored_fraction,
            "mean_first_passage_s": ens.mean_first_passage_s,
            "first_passage_biased": ens.first_passage_biased,
            "warnings": list(ens.warnings),
            "status": "exploratory",
        }
    return report


def write_report(run: RadiationDamageRun) -> Path:
    """Atomically write the report into the run directory and return its path."""
    text = json.dumps(build_report(run), indent=2, allow_nan=False) + "\n"
    return atomic_write_text(Path(run.directory) / REPORT_FILE, text)


def read_report(directory: str | os.PathLike) -> dict[str, Any]:
    """The saved report of a run directory."""
    path = Path(directory) / REPORT_FILE
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} does not exist; run the workflow first") from None
    if not isinstance(data, dict) or data.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"{path} is not a {REPORT_SCHEMA} report")
    return data


__all__ = [
    "REPORT_FILE",
    "REPORT_SCHEMA",
    "STANDING_CAVEATS",
    "build_report",
    "read_report",
    "write_report",
]
