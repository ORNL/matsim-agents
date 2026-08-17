#!/usr/bin/env python3
"""Fine-tuning behaviour of HydraGNN and UMA on AL-collected DFT labels.

Reads the per-iteration evaluation metrics produced by
``python -m matsim_agents.active_learning.evaluate`` (one ``iter<N>.json`` per
AL iteration, plus optional ``iter<N>_parity.npz``) for one or more backends
and produces a 2x2 figure:

* (a) force MAE vs. AL iteration,
* (b) energy MAE (offset-corrected, eV/atom) vs. AL iteration,
* (c) force parity before fine-tuning (iteration 0),
* (d) force parity after fine-tuning (final iteration).

Panels (a,b) show that error drops as AL-collected DFT labels are folded back
into each model; (c,d) visualise the tightening of the predicted-vs-DFT force
scatter for each backend.

Usage
-----
    fairchem_venv/bin/python plot_al_finetune.py \
        --series HydraGNN=/path/to/runs/al-hea-hydragnn/eval \
        --series UMA=/path/to/runs/al-zn-formate-uma/eval \
        --out al_finetune.pdf

Each ``eval`` directory holds ``iter<N>.json`` (required) and
``iter<N>_parity.npz`` (optional; needed for panels c,d). Run from
``docs/paper/figures``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ITER_JSON_RE = re.compile(r"iter(\d+)\.json$")

# Stable per-backend colours so (a,b) lines and (c,d) scatters agree.
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _load_series(eval_dir: str) -> dict:
    """Load all iter<N>.json metrics from a directory, sorted by iteration."""
    records = []
    for path in glob.glob(os.path.join(eval_dir, "iter*.json")):
        m = ITER_JSON_RE.search(os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            data = json.load(f)
        it = data.get("iteration")
        if it is None:
            it = int(m.group(1))
        data["_iteration"] = int(it)
        data["_dir"] = eval_dir
        records.append(data)
    records.sort(key=lambda d: d["_iteration"])
    return {"records": records}


def _parity_npz(eval_dir: str, iteration: int) -> dict | None:
    path = os.path.join(eval_dir, f"iter{iteration}_parity.npz")
    if not os.path.isfile(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _parity_scatter(ax, series: dict, colors: dict, which: str) -> None:
    """Force parity scatter for the first (before) or last (after) iteration."""
    lo, hi = np.inf, -np.inf
    plotted = False
    for label, s in series.items():
        recs = s["records"]
        if not recs:
            continue
        rec = recs[0] if which == "before" else recs[-1]
        par = _parity_npz(rec["_dir"], rec["_iteration"])
        if par is None or par["f_ref_eV_per_A"].size == 0:
            continue
        fr = par["f_ref_eV_per_A"]
        fp = par["f_pred_eV_per_A"]
        ax.scatter(fr, fp, s=3, alpha=0.25, color=colors[label], label=label, rasterized=True)
        lo = min(lo, float(fr.min()), float(fp.min()))
        hi = max(hi, float(fr.max()), float(fp.max()))
        plotted = True
    if plotted and np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    else:
        ax.text(0.5, 0.5, "no parity data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(r"DFT force (eV/$\mathrm{\AA}$)")
    ax.set_ylabel(r"MLIP force (eV/$\mathrm{\AA}$)")


def make_figure(series: dict[str, dict], out_path: str) -> None:
    colors = {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(series)}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5))
    (ax_f, ax_e), (ax_pb, ax_pa) = axes

    # (a) Force MAE vs iteration; (b) Energy MAE (shifted) vs iteration.
    for label, s in series.items():
        recs = s["records"]
        if not recs:
            continue
        its = [r["_iteration"] for r in recs]
        f_mae = [r.get("force_mae_eV_per_A", np.nan) for r in recs]
        e_mae = [r.get("energy_mae_eV_per_atom_shifted", np.nan) for r in recs]
        ax_f.plot(its, f_mae, "o-", color=colors[label], label=label)
        ax_e.plot(its, e_mae, "o-", color=colors[label], label=label)

    ax_f.set_xlabel("AL iteration")
    ax_f.set_ylabel(r"Force MAE (eV/$\mathrm{\AA}$)")
    ax_f.set_title("(a) Force error vs. fine-tuning iteration")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend()

    ax_e.set_xlabel("AL iteration")
    ax_e.set_ylabel("Energy MAE (eV/atom, offset-corrected)")
    ax_e.set_title("(b) Energy error vs. fine-tuning iteration")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend()

    _parity_scatter(ax_pb, series, colors, "before")
    ax_pb.set_title("(c) Force parity before fine-tuning")
    ax_pb.legend(markerscale=3)

    _parity_scatter(ax_pa, series, colors, "after")
    ax_pa.set_title("(d) Force parity after fine-tuning")
    ax_pa.legend(markerscale=3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Wrote {out_path}")

    # Echo the headline numbers cited in the text.
    for label, s in series.items():
        recs = s["records"]
        if not recs:
            continue
        first, last = recs[0], recs[-1]
        print(
            f"{label}: iter {first['_iteration']}->{last['_iteration']}  "
            f"F_MAE {first.get('force_mae_eV_per_A', float('nan')):.4f}->"
            f"{last.get('force_mae_eV_per_A', float('nan')):.4f} eV/A  "
            f"E_MAE(shift) {first.get('energy_mae_eV_per_atom_shifted', float('nan')):.4f}->"
            f"{last.get('energy_mae_eV_per_atom_shifted', float('nan')):.4f} eV/atom"
        )


def _parse_series(items: list[str]) -> dict[str, dict]:
    series: dict[str, dict] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--series expects LABEL=DIR, got {item!r}")
        label, eval_dir = item.split("=", 1)
        series[label] = _load_series(eval_dir)
    return series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL=EVAL_DIR",
        help="Backend label and its eval directory (repeatable).",
    )
    parser.add_argument("--out", default="al_finetune.pdf", help="Output figure path.")
    args = parser.parse_args(argv)

    series = _parse_series(args.series)
    make_figure(series, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
