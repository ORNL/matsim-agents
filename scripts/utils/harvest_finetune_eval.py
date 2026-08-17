#!/usr/bin/env python3
"""Collect fine-tune-eval results into the unified manuscript table.

Scans ``<runs>/finetune-eval/<variant>/<case>/eval/iter*.json`` (written by
``matsim_agents.active_learning.finetune_eval``) and emits, for every method
row of ``tab:finetune``, the held-out force MAE (eV/A) and reference-shifted
per-atom energy MAE (meV/atom) across the five paper cases.

For each run the *before* endpoint is ``iter0.json`` and the *after* endpoint is
the highest-numbered ``iter<N>.json``. HydraGNN's zero-shot baseline row is the
common ``before`` shared by its fine-tune strategies; UMA and each MACE size
carry their own zero-shot baseline.

Usage:
    python scripts/utils/harvest_finetune_eval.py [--runs <runs_root>] [--tsv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Fixed column order of tab:finetune.
CASES = [
    "lifepo4-al-001",
    "cantor-fcc-al-001",
    "hea-bcc-al-001",
    "phosphorene-2d-al-001",
    "zn-formate-mof-uma-al-001",
]

# (row label, variant directory, endpoint) where endpoint is "before" (iter0)
# or "after" (last iter). Variant dirs follow the launcher's VARIANT_TAG scheme.
ROWS = [
    ("HydraGNN zero-shot", "hydragnn", "before"),
    ("\\quad routed", "hydragnn", "after"),
    ("\\quad unfrozen", "hydragnn-unfrozen", "after"),
    ("\\quad frozen", "hydragnn-frozen", "after"),
    ("\\quad scratch", "hydragnn-scratch", "after"),
    ("UMA zero-shot", "uma", "before"),
    ("\\quad full FT", "uma", "after"),
    ("\\quad frozen", "uma-frozen", "after"),
    ("\\quad LoRA", "uma-lora", "after"),
    ("MACE-MP-S zero-shot", "mace-small", "before"),
    ("\\quad naive FT", "mace-small", "after"),
    ("\\quad LoRA", "mace-small-lora", "after"),
    ("MACE-MP-M zero-shot", "mace", "before"),
    ("\\quad naive FT", "mace", "after"),
    ("\\quad LoRA", "mace-lora", "after"),
    ("MACE-MP-L zero-shot", "mace-large", "before"),
    ("\\quad naive FT", "mace-large", "after"),
    ("\\quad LoRA", "mace-large-lora", "after"),
]

DEFAULT_RUNS = "/global/cfs/projectdirs/m5216/mlupopa/runs"


def _read_endpoint(eval_dir: Path, endpoint: str) -> dict | None:
    if not eval_dir.is_dir():
        return None
    iters = sorted(
        (p for p in eval_dir.glob("iter*.json")),
        key=lambda p: int(p.stem[4:]) if p.stem[4:].isdigit() else -1,
    )
    if not iters:
        return None
    src = iters[0] if endpoint == "before" else iters[-1]
    try:
        return json.loads(src.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _fmt(val: float | None, scale: float, digits: int) -> str:
    if val is None:
        return "--"
    return f"{val * scale:.{digits}f}"


def harvest(runs_root: Path) -> list[tuple[str, list[str], list[str]]]:
    fte = runs_root / "finetune-eval"
    out = []
    for label, variant, endpoint in ROWS:
        force_cells, energy_cells = [], []
        for case in CASES:
            m = _read_endpoint(fte / variant / case / "eval", endpoint)
            f = m.get("force_mae_eV_per_A") if m else None
            e = m.get("energy_mae_eV_per_atom_shifted") if m else None
            force_cells.append(_fmt(f, 1.0, 2))
            energy_cells.append(_fmt(e, 1000.0, 0))  # eV/atom -> meV/atom
        out.append((label, force_cells, energy_cells))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="runs root (holds finetune-eval/)")
    ap.add_argument("--tsv", action="store_true", help="emit TSV instead of LaTeX rows")
    args = ap.parse_args()

    rows = harvest(Path(args.runs))
    if args.tsv:
        header = ["method"] + [f"F:{c}" for c in CASES] + [f"E:{c}" for c in CASES]
        print("\t".join(header))
        for label, fc, ec in rows:
            print("\t".join([label] + fc + ec))
        return

    for label, fc, ec in rows:
        print(f"{label} & " + " & ".join(fc) + " & " + " & ".join(ec) + " \\\\")


if __name__ == "__main__":
    main()
