#!/usr/bin/env python3
"""Aggregate warm-start benchmark runs and evaluate acceptance thresholds.

Thresholds implemented (from manuscript):
1) same-minimum agreement: |delta_energy_ev| <= energy tolerance
2) no increase in failed-convergence rate for warm vs cold
3) median warm effort lower than cold for BFGS and/or total SCF
"""

import argparse
import json
import statistics
from pathlib import Path


def _iter_comparison_files(runs_root, fixture):
    pattern = "qe-warmstart-*/qe-warmstart/**/{}/comparison.json".format(fixture)
    for path in sorted(runs_root.glob(pattern)):
        yield path


def _load_record(path):
    data = json.loads(path.read_text())
    cold = data.get("cold") or {}
    warm = data.get("warm") or {}

    required = [
        cold.get("converged"),
        warm.get("converged"),
        cold.get("bfgs_steps"),
        warm.get("bfgs_steps"),
        cold.get("scf_iterations_total"),
        warm.get("scf_iterations_total"),
        data.get("delta_energy_ev"),
    ]
    if any(v is None for v in required):
        return None

    run_root = path
    while run_root.name and not run_root.name.startswith("qe-warmstart-"):
        run_root = run_root.parent

    return {
        "run_dir": str(run_root),
        "case_dir": str(path.parent),
        "cold_converged": bool(cold["converged"]),
        "warm_converged": bool(warm["converged"]),
        "cold_bfgs": int(cold["bfgs_steps"]),
        "warm_bfgs": int(warm["bfgs_steps"]),
        "cold_scf": int(cold["scf_iterations_total"]),
        "warm_scf": int(warm["scf_iterations_total"]),
        "delta_energy_ev": float(data["delta_energy_ev"]),
    }


def _median(xs):
    return float(statistics.median(xs)) if xs else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, help="Path containing qe-warmstart-* run dirs")
    parser.add_argument("--fixture", default="MoNbTaW_HEA", help="Fixture name to aggregate")
    parser.add_argument("--energy-tol-ev", type=float, default=0.05)
    parser.add_argument("--json-out", default="", help="Optional path to write machine-readable summary")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        raise SystemExit("runs root does not exist: {}".format(runs_root))

    files = list(_iter_comparison_files(runs_root, args.fixture))
    records = [r for r in (_load_record(f) for f in files) if r is not None]

    if not records:
        raise SystemExit(
            "no usable comparison.json found for fixture {!r} under {}".format(
                args.fixture, runs_root
            )
        )

    n = len(records)
    same_min_flags = [abs(r["delta_energy_ev"]) <= args.energy_tol_ev for r in records]
    cold_fail_rate = sum(0 if r["cold_converged"] else 1 for r in records) / float(n)
    warm_fail_rate = sum(0 if r["warm_converged"] else 1 for r in records) / float(n)

    warm_minus_cold_bfgs = [float(r["warm_bfgs"] - r["cold_bfgs"]) for r in records]
    warm_minus_cold_scf = [float(r["warm_scf"] - r["cold_scf"]) for r in records]

    med_wm_bfgs = _median(warm_minus_cold_bfgs)
    med_wm_scf = _median(warm_minus_cold_scf)

    same_min_all = all(same_min_flags)
    fail_rate_ok = warm_fail_rate <= cold_fail_rate
    effort_ok = (med_wm_bfgs < 0.0) or (med_wm_scf < 0.0)
    passes = same_min_all and fail_rate_ok and effort_ok

    summary = {
        "fixture": args.fixture,
        "runs_root": str(runs_root),
        "num_runs": n,
        "energy_tolerance_ev": args.energy_tol_ev,
        "same_minimum_all": same_min_all,
        "same_minimum_fraction": sum(1 for x in same_min_flags if x) / float(n),
        "cold_fail_rate": cold_fail_rate,
        "warm_fail_rate": warm_fail_rate,
        "median_warm_minus_cold_bfgs": med_wm_bfgs,
        "median_warm_minus_cold_scf": med_wm_scf,
        "fail_rate_ok": fail_rate_ok,
        "effort_ok": effort_ok,
        "passes_thresholds": passes,
        "records": records,
    }

    print("Warm-start threshold check")
    print("  fixture: {}".format(args.fixture))
    print("  runs: {}".format(n))
    print("  same-minimum (all runs): {}".format(same_min_all))
    print(
        "  fail-rate cold/warm: {:.3f}/{:.3f} ({})".format(
            cold_fail_rate,
            warm_fail_rate,
            "OK" if fail_rate_ok else "NOT OK",
        )
    )
    print("  median (warm-cold) BFGS: {:.3f}".format(med_wm_bfgs))
    print("  median (warm-cold) SCF:  {:.3f}".format(med_wm_scf))
    print("  overall PASS: {}".format(passes))

    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print("  wrote JSON summary: {}".format(out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
