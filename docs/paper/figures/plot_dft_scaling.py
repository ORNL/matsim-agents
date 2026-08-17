#!/usr/bin/env python3
"""Strong-scaling of the AL DFT-labelling stage vs. node allocation.

Reads the per-(N, repeat) scaling runs produced by
``scripts/advanced/perlmutter/submit-al-dft-scaling-sweep.sh`` (RUN_TAG
``hea-fcc-scaling-N<nodes>-r<rep>``). Each repeat draws an independent random
16-job workload (held fixed across node counts within the repeat) and labels it
with ``<nodes>``-way VASP concurrency. For each node count we form the
work-normalized DFT throughput (converged frames per DFT-hour) per repeat and
average over repeats, so per-draw variability is dampened. Plots (a) mean DFT
throughput and (b) parallel efficiency vs. node count, both with +/- std error
bars, against the ideal-linear reference.

Usage
-----
    fairchem_venv/bin/python plot_dft_scaling.py \
        --runs-root /global/cfs/projectdirs/m5216/mlupopa/runs \
        --out al_dft_scaling.pdf

Run from ``docs/paper/figures``. Also prints the numbers cited in the text.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Match only the randomized-repeat sweep tags hea-fcc-scaling-N<nodes>-r<rep>.
# The trailing anchor excludes the legacy single-run baseline dirs (no -r suffix,
# same seed as r1) and any moved-aside ".timeout-<jobid>" partial dirs, so the
# aggregate reflects exactly the R independent random draws per node count.
TAG_RE = re.compile(r"hea-fcc-scaling-N(\d+)-r(\d+)$")


def collect(runs_root: str) -> list[tuple[int, int, int, float]]:
    """Return sorted [(nodes, rep, n_converged, dft_seconds), ...] over runs."""
    rows: list[tuple[int, int, int, float]] = []
    for d in glob.glob(os.path.join(runs_root, "hea-fcc-scaling-N*")):
        m = TAG_RE.search(os.path.basename(d))
        if not m:
            continue
        nodes = int(m.group(1))
        rep = int(m.group(2)) if m.group(2) else 1
        states = sorted(glob.glob(os.path.join(d, "iteration_*", "state.json")))
        conv = 0
        dft = 0.0
        for s in states:
            try:
                j = json.load(open(s))
            except Exception:  # noqa: BLE001
                continue
            conv += j.get("n_dft_converged", 0)
            dft += j.get("timings_sec", {}).get("dft", 0.0)
        if conv and dft > 0:
            rows.append((nodes, rep, conv, dft))
    return sorted(rows)


def aggregate(rows: list[tuple[int, int, int, float]]):
    """Collapse per-(N, rep) rows into per-N throughput mean/std over repeats."""
    per_node: dict[int, list[float]] = defaultdict(list)
    reps_per_node: dict[int, int] = defaultdict(int)
    for nodes, _rep, conv, dft in rows:
        per_node[nodes].append(conv / (dft / 3600.0))  # frames per hour
        reps_per_node[nodes] += 1
    nodes = np.array(sorted(per_node), dtype=float)
    tput_mean = np.array([np.mean(per_node[int(n)]) for n in nodes])
    tput_std = np.array([np.std(per_node[int(n)], ddof=0) for n in nodes])
    n_reps = np.array([reps_per_node[int(n)] for n in nodes], dtype=int)
    return nodes, tput_mean, tput_std, n_reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", default="al_dft_scaling.pdf")
    args = ap.parse_args()

    rows = collect(args.runs_root)
    if not rows:
        raise SystemExit(
            "No scaling runs found (expected RUN_TAG hea-fcc-scaling-N<nodes>-r<rep>). "
            "Submit scripts/advanced/perlmutter/submit-al-dft-scaling-sweep.sh first."
        )

    nodes, throughput, tput_std, n_reps = aggregate(rows)

    base = throughput[0]  # 1-node mean throughput (nodes sorted, first = min)
    base_nodes = nodes[0]
    ideal = base * nodes / base_nodes
    efficiency = 100.0 * (throughput / nodes) / (base / base_nodes)
    # Propagate the throughput std into the efficiency (linear scaling by nodes).
    eff_std = 100.0 * (tput_std / nodes) / (base / base_nodes)

    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(8.6, 3.3))

    ax_t.plot(nodes, ideal, "k--", lw=1.2, label="ideal (linear)")
    ax_t.errorbar(
        nodes, throughput, yerr=tput_std, fmt="o-", color="#3b6fb0",
        lw=1.6, ms=6, capsize=3, label="measured (mean +/- std)",
    )
    ax_t.set_xlabel("Nodes (= concurrent VASP jobs)")
    ax_t.set_ylabel("DFT throughput (frames / h)")
    ax_t.set_title("(a) DFT-labelling throughput", fontsize=9)
    ax_t.set_xticks(nodes)
    ax_t.legend(fontsize=8)

    ax_e.axhline(100.0, color="k", ls="--", lw=1.2)
    ax_e.errorbar(
        nodes, efficiency, yerr=eff_std, fmt="s-", color="#e08214",
        lw=1.6, ms=6, capsize=3,
    )
    ax_e.set_xlabel("Nodes (= concurrent VASP jobs)")
    ax_e.set_ylabel("Parallel efficiency (%)")
    ax_e.set_title("(b) Strong-scaling efficiency", fontsize=9)
    ax_e.set_xticks(nodes)
    ax_e.set_ylim(0, 115)
    for xi, e in zip(nodes, efficiency):
        ax_e.text(xi, e, f"{e:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

    print("\nnodes, repeats, throughput_mean(frames/h), throughput_std, speedup, efficiency%")
    for i in range(len(nodes)):
        print(
            f"  N={int(nodes[i]):2d}  reps={int(n_reps[i]):2d}  "
            f"tput={throughput[i]:7.1f}+/-{tput_std[i]:5.1f}  "
            f"speedup={throughput[i]/base:4.2f}x  eff={efficiency[i]:5.1f}%"
        )


if __name__ == "__main__":
    main()
