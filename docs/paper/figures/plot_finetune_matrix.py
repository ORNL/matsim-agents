#!/usr/bin/env python3
"""Two-column heatmap of held-out fine-tuning errors (replaces Table V).

Renders the AL-corrected held-out test errors for every MLIP backend and
fine-tuning variant as a pair of annotated, log-scaled heatmaps:

* left panel:  force MAE (eV/A),
* right panel: reference-shifted per-atom energy MAE (meV/atom).

Rows are grouped by backend, with each backend's zero-shot baseline drawn
first (bold label) followed by its fine-tuning variants; columns are the five
paper cases. Colour is log-scaled so the order-of-magnitude drop from zero-shot
to fine-tuned is visible at a glance, while every cell is still annotated with
its exact value so the figure is a faithful, reviewer-checkable replacement for
the table.

The numbers below are the final harvested values (identical to the commented-out
Table V in ``main.tex``). Regenerate them with::

    python scripts/utils/harvest_finetune_eval.py --tsv

Usage
-----
    <venv>/bin/python plot_finetune_matrix.py --out finetune_matrix.pdf
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Case columns (follow Table II): LFP, Cantor, NbTaV, Phos, MOF.
CASES = ["LFP", "Cantor", "NbTaV", "Phos", "MOF"]

# (row label, is_zero_shot, group_id). group_id starts a new backend block.
ROWS = [
    ("HydraGNN  zero-shot", True, "HydraGNN"),
    ("HydraGNN  routed", False, "HydraGNN"),
    ("HydraGNN  unfrozen", False, "HydraGNN"),
    ("HydraGNN  frozen", False, "HydraGNN"),
    ("HydraGNN  scratch", False, "HydraGNN"),
    ("UMA  zero-shot", True, "UMA"),
    ("UMA  full FT", False, "UMA"),
    ("UMA  frozen", False, "UMA"),
    ("UMA  LoRA", False, "UMA"),
    ("MACE-MP-S  zero-shot", True, "MACE-MP-S"),
    ("MACE-MP-S  naive FT", False, "MACE-MP-S"),
    ("MACE-MP-S  LoRA", False, "MACE-MP-S"),
    ("MACE-MP-M  zero-shot", True, "MACE-MP-M"),
    ("MACE-MP-M  naive FT", False, "MACE-MP-M"),
    ("MACE-MP-M  LoRA", False, "MACE-MP-M"),
    ("MACE-MP-L  zero-shot", True, "MACE-MP-L"),
    ("MACE-MP-L  naive FT", False, "MACE-MP-L"),
    ("MACE-MP-L  LoRA", False, "MACE-MP-L"),
]

# Force MAE (eV/A): rows follow ROWS, columns follow CASES.
FORCE_MAE = np.array([
    [22.97, 6.43, 3.32, 2.84, 5.38],
    [3.09, 6.18, 2.02, 1.58, 2.49],
    [2.69, 3.43, 1.37, 0.68, 2.21],
    [4.98, 5.10, 1.46, 0.57, 3.26],
    [2.50, 2.14, 1.37, 0.42, 2.12],
    [0.73, 0.66, 0.18, 0.17, 1.03],
    [1.82, 0.65, 0.00, 0.36, 1.92],
    [0.73, 0.66, 0.18, 0.17, 1.03],
    [0.73, 0.66, 0.18, 0.22, 1.50],
    [1.30, 1.36, 0.48, 0.21, 0.61],
    [0.07, 0.02, 0.00, 0.05, 0.07],
    [0.42, 0.19, 0.01, 0.11, 0.34],
    [1.04, 1.07, 0.49, 0.31, 0.55],
    [0.04, 0.02, 0.00, 0.04, 0.05],
    [0.33, 0.20, 0.01, 0.09, 0.36],
    [1.25, 1.06, 0.66, 0.27, 0.43],
    [0.02, 0.01, 0.00, 0.04, 0.03],
    [0.35, 0.15, 0.01, 0.10, 0.29],
])

# Reference-shifted per-atom energy MAE (meV/atom).
ENERGY_MAE = np.array([
    [1917, 2248, 559, 136, 216],
    [662, 1108, 121, 165, 116],
    [612, 418, 28, 45, 117],
    [430, 843, 107, 59, 216],
    [604, 450, 49, 33, 120],
    [130, 150, 5, 15, 78],
    [445, 123, 0, 29, 115],
    [130, 150, 5, 15, 78],
    [130, 150, 5, 13, 178],
    [177, 154, 112, 31, 32],
    [18, 8, 0, 2, 11],
    [89, 190, 6, 8, 14],
    [117, 158, 19, 45, 27],
    [11, 9, 0, 3, 6],
    [68, 197, 5, 7, 20],
    [97, 177, 45, 46, 48],
    [9, 3, 0, 2, 11],
    [96, 171, 4, 6, 13],
], dtype=float)


def _draw_panel(ax, data, title, cbar_label, value_fmt, color_floor):
    """Log-scaled heatmap with per-cell value annotations."""
    labels = [r[0] for r in ROWS]
    n_rows, n_cols = data.shape

    # Floor for colour only (LogNorm needs strictly positive); annotations
    # still print the true value, including exact zeros.
    color_data = np.clip(data, color_floor, None)
    vmax = float(color_data.max())
    cmap = plt.get_cmap("RdYlGn_r")
    norm = LogNorm(vmin=color_floor, vmax=vmax)

    im = ax.imshow(color_data, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(CASES, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=7)
    # Bold the zero-shot baseline rows.
    for tick, row in zip(ax.get_yticklabels(), ROWS):
        if row[1]:
            tick.set_fontweight("bold")
    ax.set_title(title, fontsize=9)
    ax.tick_params(length=0)

    # Annotate every cell; pick text colour from background luminance.
    for i in range(n_rows):
        for j in range(n_cols):
            rgba = cmap(norm(color_data[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "black" if lum > 0.55 else "white"
            ax.text(
                j, i, value_fmt(data[i, j]), ha="center", va="center",
                fontsize=6.2, color=txt_color,
            )

    # Thin white group separators between backends.
    for i in range(1, n_rows):
        if ROWS[i][2] != ROWS[i - 1][2]:
            ax.axhline(i - 0.5, color="white", lw=2.2)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def make_figure(out_path: str) -> None:
    fig, (ax_f, ax_e) = plt.subplots(1, 2, figsize=(7.2, 6.6))

    _draw_panel(
        ax_f, FORCE_MAE,
        r"(a) Force MAE (eV/$\mathrm{\AA}$)",
        r"eV/$\mathrm{\AA}$ (log)",
        lambda v: "<0.01" if v < 0.01 else f"{v:.2f}", color_floor=0.01,
    )
    _draw_panel(
        ax_e, ENERGY_MAE,
        "(b) Energy MAE (meV/atom)",
        "meV/atom (log)",
        lambda v: "<1" if v < 1 else f"{v:.0f}", color_floor=1.0,
    )
    # Row labels already carry the backend/variant; hide the right panel's to
    # save width since it shares the row ordering with the left panel.
    ax_e.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="finetune_matrix.pdf", help="Output figure path.")
    args = parser.parse_args(argv)
    make_figure(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
