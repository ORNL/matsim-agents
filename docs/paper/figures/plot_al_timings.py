#!/usr/bin/env python3
"""Per-iteration stage-timing breakdown for the end-to-end AL runs.

Reads every ``iter_*/state.json`` under the given RUN_TAG directories and plots,
for each case, the mean wall time spent in each loop stage. The message is that
DFT labelling dominates the wall clock while the agentic orchestration
(calculator build, MD sampling, acquisition, dataset append, retrain hook) is a
small, near-constant overhead -- i.e. the framework adds negligible cost on top
of the first-principles work.

Usage
-----
    fairchem_venv/bin/python plot_al_timings.py \
        --runs-root /global/cfs/projectdirs/m5216/mlupopa/runs \
        --out al_timing_breakdown.pdf

Run from ``docs/paper/figures``. Also prints the numbers cited in the text.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# RUN_TAG -> display label (order = plot order, left to right)
CASES = [
    ("phosphorene-2d-al-001", "Phosphorene"),
    ("lifepo4-al-001", r"LiFePO$_4$"),
    ("cantor-fcc-al-001", "Cantor FCC HEA"),
    ("hea-bcc-al-001", "BCC HEA"),
    ("zn-formate-mof-uma-al-001", "Zn-formate MOF"),
]

# stage key -> (legend label, colour). Order = stack order (bottom to top).
STAGES = [
    ("dft", "DFT labelling", "#3b6fb0"),
    ("md_sampling", "MLIP-MD sampling", "#e08214"),
    ("acquisition", "Acquisition (UQ)", "#8073ac"),
    ("build_calculators", "Calculator build", "#5aae61"),
    ("append_dataset", "Dataset append", "#d6604d"),
    ("retrain", "Retrain hook", "#999999"),
]


def load_case(runs_root: str, tag: str) -> tuple[int, dict[str, float]]:
    """Return (n_iters, mean-seconds-per-iteration per stage) for one RUN_TAG."""
    states = sorted(glob.glob(os.path.join(runs_root, tag, "iteration_*", "state.json")))
    per_stage: dict[str, list[float]] = {k: [] for k, _, _ in STAGES}
    for s in states:
        try:
            j = json.load(open(s))
        except Exception:  # noqa: BLE001
            continue
        t = j.get("timings_sec", {})
        for k, _, _ in STAGES:
            per_stage[k].append(float(t.get(k, 0.0)))
    n = len(states)
    means = {k: (float(np.mean(v)) if v else 0.0) for k, v in per_stage.items()}
    return n, means


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", default="al_timing_breakdown.pdf")
    args = ap.parse_args()

    labels: list[str] = []
    niters: list[int] = []
    stacks: dict[str, list[float]] = {k: [] for k, _, _ in STAGES}
    for tag, label in CASES:
        n, means = load_case(args.runs_root, tag)
        if n == 0:
            continue
        labels.append(label)
        niters.append(n)
        for k, _, _ in STAGES:
            stacks[k].append(means[k])

    x = np.arange(len(labels))
    fig, (ax_abs, ax_ovh) = plt.subplots(
        1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )

    # --- Panel (a): absolute mean per-iteration wall, stacked ---------------
    bottom = np.zeros(len(labels))
    for k, leg, col in STAGES:
        vals = np.array(stacks[k])
        ax_abs.bar(x, vals, bottom=bottom, label=leg, color=col, width=0.62)
        bottom += vals
    ax_abs.set_ylabel("Mean wall time / iteration (s)")
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax_abs.set_title("(a) Per-iteration stage cost", fontsize=9)
    ax_abs.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    for xi, tot in zip(x, bottom):
        ax_abs.text(xi, tot, f"{tot:,.0f}s", ha="center", va="bottom", fontsize=6.5)

    # --- Panel (b): orchestration overhead as % of wall --------------------
    overhead_pct = []
    for i in range(len(labels)):
        tot = sum(stacks[k][i] for k, _, _ in STAGES)
        dft = stacks["dft"][i]
        overhead_pct.append(100.0 * (tot - dft) / tot if tot > 0 else 0.0)
    ax_ovh.bar(x, overhead_pct, color="#e08214", width=0.62)
    ax_ovh.set_ylabel("Non-DFT overhead (% of wall)")
    ax_ovh.set_xticks(x)
    ax_ovh.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax_ovh.set_title("(b) Orchestration + MLIP overhead", fontsize=9)
    ax_ovh.set_ylim(0, max(overhead_pct) * 1.35 if overhead_pct else 1)
    for xi, p in zip(x, overhead_pct):
        ax_ovh.text(xi, p, f"{p:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

    # --- print numbers for the manuscript text -----------------------------
    print("\ncase, n_iter, dft%, non-dft%, mean_dft_s, mean_md_s")
    for i, lab in enumerate(labels):
        tot = sum(stacks[k][i] for k, _, _ in STAGES)
        dft = stacks["dft"][i]
        print(
            f"{lab:16s} n={niters[i]:2d}  dft={100*dft/tot:5.1f}%  "
            f"nondft={100*(tot-dft)/tot:4.1f}%  "
            f"dft={dft:8.1f}s  md={stacks['md_sampling'][i]:6.1f}s"
        )


if __name__ == "__main__":
    main()
