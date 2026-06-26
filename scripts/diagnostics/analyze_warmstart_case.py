#!/usr/bin/env python3
"""Analyze one warm-start case directory for geometry updates and QE impact.

Example:
  /path/to/hydragnn_venv/bin/python scripts/diagnostics/analyze_warmstart_case.py \
    --case-dir /global/cfs/projectdirs/m5216/mlupopa/runs/qe-warmstart-54710481/qe-warmstart/test_hydragnn_warmstart_helps_0/MoNbTaW_HEA
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def _norm3(dx: float, dy: float, dz: float) -> float:
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _pairwise_displacements(init_positions, new_positions) -> Tuple[float, float, float]:
    if len(init_positions) != len(new_positions):
        raise ValueError("Position arrays have different lengths")

    mags: List[float] = []
    for p0, p1 in zip(init_positions, new_positions):
        dx = float(p1[0]) - float(p0[0])
        dy = float(p1[1]) - float(p0[1])
        dz = float(p1[2]) - float(p0[2])
        mags.append(_norm3(dx, dy, dz))

    if not mags:
        return 0.0, 0.0, 0.0

    mean = sum(mags) / len(mags)
    rms = math.sqrt(sum(m * m for m in mags) / len(mags))
    mx = max(mags)
    return mean, rms, mx


def _read_structure(path: Path):
    try:
        from ase.io import read  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ASE is required. Run with the HydraGNN venv python where ASE is installed."
        ) from exc

    return read(str(path))


def _analyze_geometry(case_dir: Path) -> None:
    initial = case_dir / f"{case_dir.name}.vasp"
    opt = case_dir / "hydragnn" / f"{case_dir.name}_optimized_structure.vasp"
    traj_path = case_dir / "hydragnn" / f"{case_dir.name}_optimization.traj"

    if not initial.is_file() or not opt.is_file() or not traj_path.is_file():
        print("[geometry] Missing one or more required structure files")
        print(f"  initial: {initial}")
        print(f"  optimized: {opt}")
        print(f"  traj: {traj_path}")
        return

    init_atoms = _read_structure(initial)
    opt_atoms = _read_structure(opt)

    try:
        from ase.io import read  # type: ignore

        frames = read(str(traj_path), index=":")
    except Exception as exc:
        print(f"[geometry] Could not read trajectory: {exc}")
        return

    mean, rms, mx = _pairwise_displacements(init_atoms.positions, opt_atoms.positions)
    print("[geometry] Initial -> HydraGNN optimized displacement (Angstrom)")
    print(f"  atoms: {len(init_atoms.positions)}")
    print(f"  mean: {mean:.6e}")
    print(f"  rms:  {rms:.6e}")
    print(f"  max:  {mx:.6e}")

    print("[geometry] Per-frame displacement from initial (Angstrom)")
    for idx, frame in enumerate(frames, start=1):
        _, r, m = _pairwise_displacements(init_atoms.positions, frame.positions)
        print(f"  frame {idx:>2}: rms={r:.6e}, max={m:.6e}")


def _analyze_hydragnn_csv(case_dir: Path) -> None:
    csv_path = case_dir / "hydragnn" / f"{case_dir.name}_optimization.csv"
    if not csv_path.is_file():
        print(f"[hydragnn] Missing CSV: {csv_path}")
        return

    rows = _read_csv_rows(csv_path)
    print(f"[hydragnn] Optimization CSV rows: {len(rows)}")
    for row in rows:
        step = row.get("step", "?")
        energy = row.get("energy_eV", "?")
        fmax = row.get("max_force_eV_per_A", "?")
        top_branch = row.get("top_branch", "?")
        top_weight = row.get("top_weight", "?")
        print(
            f"  step {step}: energy={energy}, max_force={fmax}, "
            f"top_branch={top_branch}, top_weight={top_weight}"
        )


def _analyze_comparison(case_dir: Path) -> None:
    comp_path = case_dir / "comparison.json"
    if not comp_path.is_file():
        print(f"[qe] Missing comparison file: {comp_path}")
        return

    data = json.loads(comp_path.read_text())
    cold = data.get("cold", {})
    warm = data.get("warm", {})

    print("[qe] Cold vs warm summary")
    print(
        "  cold: bfgs_steps={0}, scf_total={1}, scf_per_step={2}, converged={3}".format(
            cold.get("bfgs_steps"),
            cold.get("scf_iterations_total"),
            cold.get("scf_iterations_per_step"),
            cold.get("converged"),
        )
    )
    print(
        "  warm: bfgs_steps={0}, scf_total={1}, scf_per_step={2}, converged={3}".format(
            warm.get("bfgs_steps"),
            warm.get("scf_iterations_total"),
            warm.get("scf_iterations_per_step"),
            warm.get("converged"),
        )
    )
    print(
        "  delta: bfgs={0}, scf={1}, dE_eV={2}".format(
            data.get("delta_bfgs_steps"),
            data.get("delta_scf_iterations"),
            data.get("delta_energy_ev"),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, help="Path to case directory containing comparison.json")
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    if not case_dir.is_dir():
        raise SystemExit(f"Case directory does not exist: {case_dir}")

    print(f"Analyzing case: {case_dir}")
    _analyze_geometry(case_dir)
    _analyze_hydragnn_csv(case_dir)
    _analyze_comparison(case_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
