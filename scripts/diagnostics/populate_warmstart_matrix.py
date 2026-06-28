#!/usr/bin/env python3
"""Populate the warm-start backend-ablation table (tab:warmstart_matrix) from runs.

Scans a runs root for warm-start ``comparison.json`` files produced by the four
benchmark configurations and emits LaTeX body rows (and optionally a full
``table*`` environment) summarising cold vs. warm convergence per
fixture x backend combination, averaged over repeats.

Backend is inferred from the top-level run-directory prefix, because the MLIP
sub-dict inside ``comparison.json`` is always keyed ``hydragnn`` for backward
compatibility:

    qe-warmstart-*        -> HydraGNN + QE
    vasp-warmstart-*      -> HydraGNN + VASP
    uma-warmstart-*       -> UMA + QE
    uma-vasp-warmstart-*  -> UMA + VASP

The QE and VASP comparison schemas differ:
    QE   cold/warm: ``bfgs_steps``, ``scf_iterations_total``
    VASP cold/warm: ``n_ionic_steps``, ``scf_iterations_per_step`` (sum for total)
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Ordered so the emitted table matches the manuscript layout.
FIXTURES = ["MoNbTaW_HEA", "NbTaW_BCC", "NbMoW_BCC"]
FIXTURE_LABELS = {
    "MoNbTaW_HEA": r"MoNbTaW \acs{HEA}",
    "NbTaW_BCC": "NbTaW BCC",
    "NbMoW_BCC": "NbMoW BCC",
}

BACKENDS = [
    ("HydraGNN", "QE"),
    ("HydraGNN", "VASP"),
    ("UMA", "QE"),
    ("UMA", "VASP"),
]
BACKEND_LABELS = {
    ("HydraGNN", "QE"): (r"HydraGNN", r"\acs{QE}"),
    ("HydraGNN", "VASP"): (r"HydraGNN", r"\acs{VASP}"),
    ("UMA", "QE"): (r"UMA", r"\acs{QE}"),
    ("UMA", "VASP"): (r"UMA", r"\acs{VASP}"),
}

# Run-dir prefix -> (mlip, dft). Order matters: the UMA+VASP prefix must be
# tested before the bare ``vasp-warmstart-`` prefix.
PREFIX_BACKENDS = [
    ("uma-vasp-warmstart-", ("UMA", "VASP")),
    ("uma-warmstart-", ("UMA", "QE")),
    ("vasp-warmstart-", ("HydraGNN", "VASP")),
    ("qe-warmstart-", ("HydraGNN", "QE")),
]


def _classify_run_dir(name):
    """Return (mlip, dft) for a top-level run directory name, or None."""
    for prefix, backend in PREFIX_BACKENDS:
        if name.startswith(prefix):
            return backend
    return None


def _extract_run_and_fixture(path):
    """Return ((mlip, dft), fixture, job_id) for a comparison.json path, or None.

    The run directory is the first ancestor whose name matches a known prefix;
    the fixture is the name of the directory directly containing the file; the
    job id is the trailing integer of the run-directory name (used by --latest).
    """
    fixture = path.parent.name
    for ancestor in path.parents:
        backend = _classify_run_dir(ancestor.name)
        if backend is not None:
            job_id = _job_id(ancestor.name)
            return backend, fixture, job_id
    return None


def _job_id(run_dir_name):
    """Trailing integer of a run-directory name (e.g. ...-warmstart-55096521)."""
    tail = run_dir_name.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def _ionic_steps(side):
    """Number of ionic (BFGS) steps, tolerating QE and VASP schemas."""
    if side.get("bfgs_steps") is not None:
        return int(side["bfgs_steps"])
    if side.get("n_ionic_steps") is not None:
        return int(side["n_ionic_steps"])
    return None


def _total_scf(side):
    """Total SCF iterations, tolerating QE and VASP schemas."""
    if side.get("scf_iterations_total") is not None:
        return int(side["scf_iterations_total"])
    per_step = side.get("scf_iterations_per_step")
    if per_step:
        return int(sum(per_step))
    return None


def _load_record(path):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    cold = data.get("cold") or {}
    warm = data.get("warm") or {}

    values = {
        "cold_ionic": _ionic_steps(cold),
        "warm_ionic": _ionic_steps(warm),
        "cold_scf": _total_scf(cold),
        "warm_scf": _total_scf(warm),
    }
    if any(v is None for v in values.values()):
        return None
    # A side with zero total SCF iterations never ran a real DFT optimization
    # (process crashed/aborted before the first SCF cycle); such a record is a
    # failed run and must not pollute the averages.
    if values["cold_scf"] == 0 or values["warm_scf"] == 0:
        return None
    values["cold_converged"] = bool(cold.get("converged"))
    values["warm_converged"] = bool(warm.get("converged"))
    return values


def _fmt_mean(xs):
    if not xs:
        return "TBD"
    m = statistics.mean(xs)
    # Integer-valued metrics: show as integer when the mean is whole.
    if abs(m - round(m)) < 1e-9:
        return str(int(round(m)))
    return "{:.1f}".format(m)


def _fmt_delta_scf(cold_scf, warm_scf):
    if not cold_scf or not warm_scf:
        return "TBD"
    c = statistics.mean(cold_scf)
    w = statistics.mean(warm_scf)
    if c == 0:
        return "TBD"
    pct = 100.0 * (w - c) / c
    if abs(pct) < 0.5:
        return r"0\%"
    sign = "+" if pct >= 0 else "$-$"
    return "{}{:.0f}\\%".format(sign, abs(pct))


def collect(runs_root, require_converged=False, latest=0):
    """Return {(fixture, mlip, dft): {metric: [values...]}}.

    When ``latest`` > 0, only the ``latest`` newest runs (by job id) are kept
    per fixture x backend cell.
    """
    # First gather per-cell records tagged with their job id.
    raw = defaultdict(list)
    for path in runs_root.glob("**/comparison.json"):
        info = _extract_run_and_fixture(path)
        if info is None:
            continue
        backend, fixture, job_id = info
        if fixture not in FIXTURE_LABELS:
            continue
        rec = _load_record(path)
        if rec is None:
            continue
        if require_converged and not (
            rec["cold_converged"] and rec["warm_converged"]
        ):
            continue
        key = (fixture, backend[0], backend[1])
        raw[key].append((job_id, rec))

    buckets = defaultdict(lambda: defaultdict(list))
    for key, items in raw.items():
        # Newest job ids first; keep only ``latest`` when requested.
        items.sort(key=lambda pair: pair[0], reverse=True)
        if latest > 0:
            items = items[:latest]
        for _job_id_value, rec in items:
            bucket = buckets[key]
            bucket["cold_ionic"].append(rec["cold_ionic"])
            bucket["warm_ionic"].append(rec["warm_ionic"])
            bucket["cold_scf"].append(rec["cold_scf"])
            bucket["warm_scf"].append(rec["warm_scf"])
    return buckets


def render_rows(buckets):
    lines = []
    for fi, fixture in enumerate(FIXTURES):
        label = FIXTURE_LABELS[fixture]
        for bi, (mlip, dft) in enumerate(BACKENDS):
            mlip_lbl, dft_lbl = BACKEND_LABELS[(mlip, dft)]
            bucket = buckets.get((fixture, mlip, dft), {})
            cold_ionic = bucket.get("cold_ionic", [])
            warm_ionic = bucket.get("warm_ionic", [])
            cold_scf = bucket.get("cold_scf", [])
            warm_scf = bucket.get("warm_scf", [])
            n = len(cold_ionic)
            first = (
                "\\multirow{{4}}{{*}}{{{}}}".format(label) if bi == 0 else ""
            )
            row = " & ".join(
                [
                    first,
                    mlip_lbl,
                    dft_lbl,
                    _fmt_mean(cold_ionic),
                    _fmt_mean(warm_ionic),
                    _fmt_mean(cold_scf),
                    _fmt_mean(warm_scf),
                    _fmt_delta_scf(cold_scf, warm_scf),
                    str(n) if n else "TBD",
                ]
            )
            lines.append(row + r" \\")
        if fi != len(FIXTURES) - 1:
            lines.append(r"\cmidrule(lr){1-9}")
    return "\n".join(lines)


TABLE_TEMPLATE = r"""\begin{{table*}}[t]
\centering
\caption{{Cold vs.\ warm-start convergence for the factorial backend ablation
that pairs each \acs{{MLIP}} backend with each \acs{{DFT}} solver, across the three
alloy fixtures. \acs{{BFGS}} = ionic steps; \acs{{SCF}} = total
self-consistent-field iterations; values are averaged over the available repeats.
$\Delta_\text{{SCF}}<0$ indicates fewer total \acs{{SCF}} iterations for the warm
start. TBD entries are pending Perlmutter jobs.}}
\label{{tab:warmstart_matrix}}
\begin{{tabular}}{{lllcccccc}}
\toprule
\multirow{{2}}{{*}}{{Fixture}} & \multirow{{2}}{{*}}{{\acs{{MLIP}}}} & \multirow{{2}}{{*}}{{\acs{{DFT}}}}
 & \multicolumn{{2}}{{c}}{{\acs{{BFGS}} steps}} & \multicolumn{{2}}{{c}}{{Total \acs{{SCF}}}} & \multirow{{2}}{{*}}{{$\Delta_\text{{SCF}}$}} & \multirow{{2}}{{*}}{{Repeats}}\\
\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}
 & & & Cold & Warm & Cold & Warm & & \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        required=True,
        help="Path containing the *-warmstart-<jobid> run directories",
    )
    parser.add_argument(
        "--full-table",
        action="store_true",
        help="Emit the complete table* environment instead of body rows only",
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        help="Only aggregate runs where both cold and warm converged",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=0,
        metavar="N",
        help="Keep only the N newest runs (by job id) per fixture x backend cell",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to write the LaTeX output (default: stdout)",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        raise SystemExit("runs root does not exist: {}".format(runs_root))

    buckets = collect(
        runs_root,
        require_converged=args.require_converged,
        latest=args.latest,
    )
    rows = render_rows(buckets)
    output = TABLE_TEMPLATE.format(rows=rows) if args.full_table else rows + "\n"

    if args.out:
        Path(args.out).write_text(output)
        print("wrote {}".format(args.out))
    else:
        print(output)


if __name__ == "__main__":
    main()
