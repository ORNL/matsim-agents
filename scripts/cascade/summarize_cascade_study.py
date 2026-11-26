"""Summarize the records of a cascade study directory into ``summary.json`` and a short report.

Literature values are read from ``--reference`` (a JSON file with the value, its unit and a
source string); nothing is built in. Estimates are measurements of the supplied potential under
the stated numerics and are reported next to, never as, a validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matsim_agents.science.cascade.damage_functions import arc_dpa_displacements, nrt_displacements
from matsim_agents.science.cascade.engines import size_convergence
from matsim_agents.workflows.radiation.cascade_study import (
    displacement_probability,
    load_records,
    mean_sem,
    threshold_estimates,
)


def summarize_threshold(records: list[dict], reference: dict | None) -> dict:
    out: dict = {"groups": {}}
    for sr in sorted({r["short_range"] for r in records}):
        for d in sorted({r["direction"] for r in records}):
            sel = [r for r in records if r["short_range"] == sr and r["direction"] == d]
            if not sel:
                continue
            prob = displacement_probability(sel)
            out["groups"][f"{sr}/{d}"] = {
                "probability": {str(e): {"p": p, "n": n} for e, (p, n) in prob.items()},
                **threshold_estimates(prob),
            }
        pooled = displacement_probability([r for r in records if r["short_range"] == sr])
        out["groups"][f"{sr}/all"] = {
            "probability": {str(e): {"p": p, "n": n} for e, (p, n) in pooled.items()},
            **threshold_estimates(pooled),
        }
    if reference:
        out["reference"] = reference
    return out


def summarize_cascades(records: list[dict], tolerance: float) -> dict:
    out: dict = {"sizes": {}}
    by_size: dict[int, list[dict]] = {}
    for r in records:
        by_size.setdefault(int(r["n_atoms"]), []).append(r)
    for n, sel in sorted(by_size.items()):
        fp_mean, fp_sem = mean_sem([r["n_frenkel_pairs"] for r in sel])
        out["sizes"][str(n)] = {
            "realizations": len(sel),
            "frenkel_pairs": [r["n_frenkel_pairs"] for r in sel],
            "mean": fp_mean,
            "sem": fp_sem,
            "boundary_ok": [r["boundary_ok"] for r in sel],
            "unstable": sum(not r["numerically_stable"] for r in sel),
            "max_abs_energy_drift_eV": max(r["max_abs_energy_drift_eV"] for r in sel),
        }
    values = {n: [float(r["n_frenkel_pairs"]) for r in s] for n, s in by_size.items()}
    conv = size_convergence(values, tolerance=tolerance)
    out["size_convergence"] = {
        "converged": conv.converged,
        "relative_change": conv.relative_change,
        "tolerance": conv.tolerance,
        "statement": conv.statement,
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("--reference", help="JSON with sourced literature values")
    p.add_argument("--tolerance", type=float, default=0.15)
    args = p.parse_args()
    d = Path(args.directory)
    records = load_records(d)
    if not records:
        raise SystemExit("no task records found")
    study = records[0]["study"]
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    summary = (
        summarize_threshold(records, reference)
        if study == "threshold"
        else summarize_cascades(records, args.tolerance)
    )
    summary["study"] = study
    summary["n_records"] = len(records)
    summary["n_unstable"] = sum(not r["numerically_stable"] for r in records)
    summary["n_boundary_flagged"] = sum(not r["boundary_ok"] for r in records)
    if study != "threshold" and reference and "arc_dpa" in reference:
        a = reference["arc_dpa"]
        energy = float(records[0]["energy_eV"])
        summary["literature_estimate"] = {
            "damage_energy_assumed_eV": energy,
            "nrt": nrt_displacements(energy, a["ed_eV"]),
            "arc_dpa": arc_dpa_displacements(energy, a["ed_eV"], a["b"], a["c"]),
            "source": a["source"],
            "note": "damage energy taken equal to the PKA energy (no electronic loss)",
        }
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
