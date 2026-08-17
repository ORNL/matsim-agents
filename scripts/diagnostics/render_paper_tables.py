#!/usr/bin/env python3
"""Render paper LaTeX table fragments from collected benchmark data.

Inputs:
- research/paper/manuscript/results/paper_results_master.csv

Outputs (default under research/paper/manuscript/results/tex):
- uq_table_rows.tex
- warmstart_qe_table_rows.tex
- singlepass_table_rows.tex

These files contain only table row lines, ready for \input{} in LaTeX.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List


UQ_LABEL_MAP = {
    "si": ("Si (diamond)", 8),
    "mgo": ("MgO (rocksalt)", 8),
    "monbtaw_hea": ("MoNbTaW (BCC \\acs{HEA})", 16),
}

CASE_LABEL_MAP = {
    "lifepo4": "LiFePO\\textsubscript{4} olivine",
    "hea_bcc": "NbTaVHfZrTi BCC \\acs{HEA}",
    "hea_fcc": "CrMnFeCoNi Cantor FCC \\acs{HEA}",
    "phosphorene": "Phosphorene",
    "cu_bht": "Cu-BHT conductive \\acs{MOF}",
    "zn_formate": "Zn(HCOO)\\textsubscript{2} \\acs{MOF}",
}


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r") as fh:
        return list(csv.DictReader(fh))


def _as_float(v: str | None):
    try:
        return float(v) if v not in (None, "") else None
    except Exception:
        return None


def _as_int(v: str | None):
    try:
        return int(v) if v not in (None, "") else None
    except Exception:
        return None


def _sci_tex(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "--"
    if x == 0:
        return "0"
    s = f"{x:.{digits}e}"
    mant, exp = s.split("e")
    exp_i = int(exp)
    return f"${mant} \\times 10^{{{exp_i}}}$"


def _dedupe_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for r in rows:
        key = (
            r.get("workflow"),
            r.get("job_id"),
            r.get("case"),
            r.get("fixture"),
            r.get("structure"),
            r.get("source_file"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _select_best_unique(rows: List[Dict[str, str]], key_fields: List[str]) -> List[Dict[str, str]]:
    """Keep one row per key_fields, preferring newest job_id when numeric."""
    best = {}
    for r in rows:
        key = tuple(r.get(k, "") for k in key_fields)
        curr = best.get(key)
        if curr is None:
            best[key] = r
            continue
        jid_new = _as_int(r.get("job_id")) or -1
        jid_old = _as_int(curr.get("job_id")) or -1
        if jid_new >= jid_old:
            best[key] = r
    return list(best.values())


def render_uq(rows: List[Dict[str, str]]) -> List[str]:
    uq_rows = [r for r in rows if r.get("workflow") == "uq"]
    uq_rows = _select_best_unique(uq_rows, ["structure"])

    lines = []
    for r in sorted(uq_rows, key=lambda x: x.get("structure", "")):
        structure_path = (r.get("structure") or "").lower()
        base = os.path.basename(structure_path)
        key = os.path.splitext(base)[0]
        label, atoms = UQ_LABEL_MAP.get(key, (base or "Unknown", "--"))

        steps = _as_int(r.get("num_steps"))
        fmax = _as_float(r.get("final_max_force_eV_per_A"))
        wt = _as_float(r.get("mean_top_weight"))
        ent = _as_float(r.get("mean_entropy"))

        steps_s = str(steps) if steps is not None else "--"
        fmax_s = _sci_tex(fmax, digits=1)
        wt_s = f"{wt:.4f}" if wt is not None else "--"
        ent_s = f"{ent:.3f}" if ent is not None else "--"

        lines.append(f"{label} & {atoms} & {steps_s} & {fmax_s} & {wt_s} & {ent_s} \\\\")
    return lines


def _extract_qe_scf_seq(comparison_json: str, mode: str) -> str:
    if not comparison_json or not os.path.isfile(comparison_json):
        return "---"
    try:
        with open(comparison_json, "r") as fh:
            data = json.load(fh)
        block = data.get(mode) or {}
        seq = block.get("scf_iterations_per_step")
        if isinstance(seq, list):
            return "[" + ", ".join(str(int(x)) for x in seq) + "]"
    except Exception:
        pass
    return "---"


def render_warmstart_qe(rows: List[Dict[str, str]]) -> List[str]:
    ws = [r for r in rows if r.get("workflow") == "warmstart_qe"]
    ws = _select_best_unique(ws, ["fixture"])
    lines = []

    for r in sorted(ws, key=lambda x: x.get("fixture", "")):
        fixture = (r.get("fixture") or "").lower()
        if fixture != "si":
            continue

        cold_steps = _as_int(r.get("cold_steps"))
        warm_steps = _as_int(r.get("warm_steps"))
        cold_scf = _as_int(r.get("cold_scf_total"))
        warm_scf = _as_int(r.get("warm_scf_total"))
        cold_wall = _as_float(r.get("cold_wall_time_sec"))
        warm_wall = _as_float(r.get("warm_wall_time_sec"))

        source = r.get("source_file") or ""
        cold_seq = _extract_qe_scf_seq(source, "cold")
        warm_seq = _extract_qe_scf_seq(source, "warm")

        delta_steps = (cold_steps - warm_steps) if (cold_steps is not None and warm_steps is not None) else None
        delta_scf = (cold_scf - warm_scf) if (cold_scf is not None and warm_scf is not None) else None
        delta_wall = (warm_wall - cold_wall) if (warm_wall is not None and cold_wall is not None) else None
        cold_wall_s = f"{cold_wall:.2f}" if cold_wall is not None else "--"
        warm_wall_s = f"{warm_wall:.2f}" if warm_wall is not None else "--"
        delta_wall_s = f"${delta_wall:+.2f}$" if delta_wall is not None else "--"

        lines.append(
            "Cold & {} & {} & {} & {} \\\\".format(
                cold_steps if cold_steps is not None else "--",
                cold_seq,
                cold_scf if cold_scf is not None else "--",
                cold_wall_s,
            )
        )
        lines.append(
            "Warm & {} & {} & {} & {} \\\\".format(
                warm_steps if warm_steps is not None else "--",
                warm_seq,
                warm_scf if warm_scf is not None else "--",
                warm_wall_s,
            )
        )
        lines.append("\\midrule")
        lines.append(
            "$\\Delta$ & {} & --- & {} & {} \\\\".format(
                delta_steps if delta_steps is not None else "--",
                delta_scf if delta_scf is not None else "--",
                delta_wall_s,
            )
        )
    return lines


def render_singlepass(rows: List[Dict[str, str]]) -> List[str]:
    sp = [r for r in rows if r.get("workflow") == "singlepass"]
    sp = _select_best_unique(sp, ["case"])
    lines = []

    for r in sorted(sp, key=lambda x: x.get("case", "")):
        case = r.get("case") or ""
        label = CASE_LABEL_MAP.get(case, case)
        status = (r.get("status") or "--").upper()
        n_relax = _as_int(r.get("n_relaxations"))
        e = _as_float(r.get("lowest_energy_eV"))
        e_pa = _as_float(r.get("energy_per_atom_eV"))
        fmax = _as_float(r.get("best_final_max_force_eV_per_A"))
        e_s = f"{e:.4f}" if e is not None else "--"
        e_pa_s = _sci_tex(e_pa, digits=2) if e_pa is not None else "--"

        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                label,
                n_relax if n_relax is not None else "--",
                status,
                e_s,
                e_pa_s,
                _sci_tex(fmax, digits=1),
            )
        )
    return lines


def _write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("% Auto-generated by scripts/diagnostics/render_paper_tables.py\n")
        for line in lines:
            fh.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render LaTeX table rows for the paper.")
    parser.add_argument(
        "--input-csv",
        default="research/paper/manuscript/results/paper_results_master.csv",
        help="Path to consolidated CSV produced by collect_paper_results.py",
    )
    parser.add_argument(
        "--output-dir",
        default="research/paper/manuscript/results/tex",
        help="Directory where .tex row fragments are written.",
    )
    args = parser.parse_args()

    rows = _dedupe_rows(_read_csv(args.input_csv))

    uq_lines = render_uq(rows)
    ws_lines = render_warmstart_qe(rows)
    sp_lines = render_singlepass(rows)

    uq_path = os.path.join(args.output_dir, "uq_table_rows.tex")
    ws_path = os.path.join(args.output_dir, "warmstart_qe_table_rows.tex")
    sp_path = os.path.join(args.output_dir, "singlepass_table_rows.tex")

    _write_lines(uq_path, uq_lines)
    _write_lines(ws_path, ws_lines)
    _write_lines(sp_path, sp_lines)

    print("Rendered LaTeX table fragments:")
    print("  - {} ({} rows)".format(os.path.abspath(uq_path), len(uq_lines)))
    print("  - {} ({} rows)".format(os.path.abspath(ws_path), len(ws_lines)))
    print("  - {} ({} rows)".format(os.path.abspath(sp_path), len(sp_lines)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
