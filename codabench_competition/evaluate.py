#!/usr/bin/env python3
"""
evaluate.py — Run a submitted AtomisticCalculator over all test structures.

Loads a submitted Calculator subclass, runs energy+force single-points (and
optionally ASE relaxations) over every structure in the test set, and writes a
standardised predictions/ directory that score.py can consume.

Submission convention
---------------------
The submission ZIP must contain:
    model.py            — defines a class named `AtomisticCalculator`
    requirements.txt    — (optional) Python deps
    checkpoint/         — model weights / config directory

The class must follow the interface in starting_kit/MODEL_INTERFACE.md.
evaluate.py imports it as:
    sys.path.insert(0, <submission_dir>)
    from model import AtomisticCalculator
    calc = AtomisticCalculator.from_checkpoint("checkpoint/", device=<device>)

Usage
-----
    python evaluate.py \\
        --submission   <path to unzipped submission dir>  \\
        --structures   public_data/structures_metadata.csv \\
        --struct-dir   public_data/structures/ \\
        --output       predictions/ \\
        [--relax]      # also run ASE relaxation for Tasks 3 & 4 \\
        [--device cpu|cuda|xpu] \\
        [--fmax 0.05]  # force convergence threshold for relaxation (eV/Å)
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a submitted AtomisticCalculator.")
    p.add_argument("--submission",  required=True,
                   help="Path to the unzipped submission directory (must contain model.py).")
    p.add_argument("--structures",  default="public_data/structures_metadata.csv",
                   help="Path to structures_metadata.csv.")
    p.add_argument("--struct-dir",  default="public_data/structures/",
                   help="Root directory containing the per-class .xyz files.")
    p.add_argument("--output",      default="predictions/",
                   help="Output directory for prediction files.")
    p.add_argument("--relax",       action="store_true",
                   help="Also run ASE LBFGS relaxation (needed for Tasks 3 & 4).")
    p.add_argument("--device",      default="cpu",
                   help="Compute device: cpu, cuda, xpu, etc.")
    p.add_argument("--fmax",        type=float, default=0.05,
                   help="Force convergence threshold for relaxation (eV/Å).")
    p.add_argument("--steps",       type=int,   default=500,
                   help="Maximum relaxation steps.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dynamic model loading
# ---------------------------------------------------------------------------

def load_calculator_class(submission_dir: str):
    """Import AtomisticCalculator from <submission_dir>/model.py."""
    model_path = Path(submission_dir) / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"model.py not found in submission directory: {submission_dir}\n"
            "The submission must contain a file named model.py that defines "
            "a class named AtomisticCalculator."
        )
    spec = importlib.util.spec_from_file_location("submitted_model", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(submission_dir)))
    spec.loader.exec_module(module)
    if not hasattr(module, "AtomisticCalculator"):
        raise AttributeError(
            "model.py must define a class named 'AtomisticCalculator'. "
            f"Found: {[x for x in dir(module) if not x.startswith('_')]}"
        )
    return module.AtomisticCalculator


# ---------------------------------------------------------------------------
# Prediction writers
# ---------------------------------------------------------------------------

def write_energy_row(writer, structure_id: str, energy: float, n_atoms: int) -> None:
    writer.writerow({"structure_id": structure_id,
                     "energy_eV": f"{energy:.8f}",
                     "energy_eV_per_atom": f"{energy / n_atoms:.8f}",
                     "n_atoms": n_atoms})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- load structures list ---
    meta_path = Path(args.structures)
    struct_root = Path(args.struct_dir)
    if not meta_path.is_file():
        sys.exit(f"structures_metadata.csv not found: {meta_path}")

    with open(meta_path, newline="") as f:
        structures = list(csv.DictReader(f))
    print(f"Loaded {len(structures)} structures from {meta_path}")

    # --- load submitted calculator ---
    CalcClass = load_calculator_class(args.submission)
    checkpoint = str(Path(args.submission) / "checkpoint")
    print(f"Loading calculator from {checkpoint} on device={args.device} ...")
    calc = CalcClass.from_checkpoint(checkpoint, device=args.device)
    print(f"  Calculator: {CalcClass.__name__}")

    # --- prepare output dirs ---
    out = Path(args.output)
    forces_dir = out / "forces"
    relaxed_dir = out / "relaxed"
    forces_dir.mkdir(parents=True, exist_ok=True)
    if args.relax:
        relaxed_dir.mkdir(parents=True, exist_ok=True)

    energy_file = out / "energies.csv"
    energy_fields = ["structure_id", "energy_eV", "energy_eV_per_atom", "n_atoms"]

    t0 = time.perf_counter()
    n_ok, n_fail = 0, 0

    with open(energy_file, "w", newline="") as ef:
        energy_writer = csv.DictWriter(ef, fieldnames=energy_fields)
        energy_writer.writeheader()

        for row in structures:
            sid  = row["structure_id"]
            fpath = struct_root / row["file_path"]

            if not fpath.is_file():
                print(f"  [SKIP] {sid}: file not found ({fpath})")
                n_fail += 1
                continue

            try:
                atoms = read(str(fpath), format="extxyz")
                atoms.calc = calc

                # --- single-point ---
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()

                write_energy_row(energy_writer, sid, energy, len(atoms))
                np.save(str(forces_dir / f"{sid}.npy"), forces.astype(np.float32))

                # --- optional relaxation ---
                if args.relax:
                    atoms_r = atoms.copy()
                    atoms_r.calc = calc
                    opt = LBFGS(atoms_r, logfile=None)
                    opt.run(fmax=args.fmax, steps=args.steps)
                    write(str(relaxed_dir / f"{sid}.xyz"), atoms_r, format="extxyz")

                n_ok += 1
                if n_ok % 20 == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"  {n_ok}/{len(structures)} done  ({elapsed:.1f}s)")

            except Exception:
                print(f"  [FAIL] {sid}")
                traceback.print_exc()
                n_fail += 1

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {n_ok} OK, {n_fail} failed  ({elapsed:.1f}s)")
    print(f"Predictions written to: {out}/")


if __name__ == "__main__":
    main()
