#!/usr/bin/env python3
"""
run_baselines.py — Run MACE-MP-0, HydraGNN, UMA, and AllScAIP over all test
structures and write predictions/ directories. These serve as leaderboard seeds
and pipeline-validation baselines.

Usage
-----
# MACE-MP-0 only:
    python run_baselines.py --model mace --mace-size medium --device cpu

# HydraGNN only:
    python run_baselines.py --model hydragnn --hydragnn-logdir /path/to/logdir

# UMA (requires fairchem-core + HF authentication):
    python run_baselines.py --model uma --uma-model uma-s-1p2 --device cuda

# AllScAIP (requires fairchem-core + HF authentication):
    python run_baselines.py --model allscaip --allscaip-model facebook/AllScAIP --device cuda

# MACE + HydraGNN only:
    python run_baselines.py --model both --mace-size medium \\
                            --hydragnn-logdir /path/to/logdir

# All four models:
    python run_baselines.py --model all --mace-size medium \\
                            --hydragnn-logdir /path/to/logdir --device cuda

Outputs
-------
    predictions/mace_mp0/       — energies.csv + forces/
    predictions/hydragnn/       — energies.csv + forces/
    predictions/uma/            — energies.csv + forces/
    predictions/allscaip/       — energies.csv + forces/
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).parent
STRUCT_META  = HERE / "public_data" / "structures_metadata.csv"
STRUCT_ROOT  = HERE / "public_data" / "structures"
PRED_ROOT    = HERE / "predictions"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run competition baselines.")
    p.add_argument("--model",
                   choices=["mace", "hydragnn", "uma", "allscaip", "both", "all"],
                   default="both",
                   help="Which baseline(s) to run. 'both' = mace+hydragnn; 'all' = all four.")
    # MACE
    p.add_argument("--mace-size", default="medium",
                   help="MACE-MP-0 size: small | medium | large (or path to .model).")
    # HydraGNN
    p.add_argument("--hydragnn-logdir", default=None,
                   help="Path to a trained HydraGNN logdir (contains config.json).")
    # UMA
    p.add_argument("--uma-model", default="uma-s-1p2",
                   help="UMA model name: uma-s-1p2 | uma-m-1p1 | uma-s-1p1.")
    p.add_argument("--uma-task", default="omat",
                   help="fairchem task name for UMA (default: omat for inorganic bulk).")
    # AllScAIP
    p.add_argument("--allscaip-model", default="allscaip-md-conserving-all-omol",
                   help="AllScAIP model name: allscaip-md-conserving-all-omol | allscaip-md-direct-all-omol.")
    p.add_argument("--allscaip-task", default="omol",
                   help="fairchem task name for AllScAIP (default: omol; trained on OMol25).")
    # Common
    p.add_argument("--device",  default="cpu",
                   help="Compute device: cpu, cuda, xpu.")
    p.add_argument("--relax",   action="store_true",
                   help="Also run ASE LBFGS relaxation.")
    p.add_argument("--fmax",    type=float, default=0.05)
    p.add_argument("--steps",   type=int,   default=500)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Prediction runner
# ---------------------------------------------------------------------------

def run_predictions(calc, label: str, device: str,
                    relax: bool = False, fmax: float = 0.05, steps: int = 500) -> None:
    out = PRED_ROOT / label
    forces_dir  = out / "forces"
    relaxed_dir = out / "relaxed"
    forces_dir.mkdir(parents=True, exist_ok=True)
    if relax:
        relaxed_dir.mkdir(parents=True, exist_ok=True)

    with open(STRUCT_META, newline="") as f:
        structures = list(csv.DictReader(f))

    energy_file   = out / "energies.csv"
    energy_fields = ["structure_id", "energy_eV", "energy_eV_per_atom", "n_atoms"]

    t0 = time.perf_counter()
    n_ok = n_fail = 0

    with open(energy_file, "w", newline="") as ef:
        writer = csv.DictWriter(ef, fieldnames=energy_fields)
        writer.writeheader()

        for row in structures:
            sid   = row["structure_id"]
            fpath = STRUCT_ROOT / row["file_path"]
            if not fpath.is_file():
                n_fail += 1
                continue
            try:
                atoms = read(str(fpath), format="extxyz")
                atoms.calc = calc
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()

                writer.writerow({
                    "structure_id":      sid,
                    "energy_eV":         f"{energy:.8f}",
                    "energy_eV_per_atom": f"{energy / len(atoms):.8f}",
                    "n_atoms":           len(atoms),
                })
                np.save(str(forces_dir / f"{sid}.npy"), forces.astype(np.float32))

                if relax:
                    from ase.optimize import LBFGS
                    from ase.io import write as ase_write
                    atoms_r = atoms.copy()
                    atoms_r.calc = calc
                    LBFGS(atoms_r, logfile=None).run(fmax=fmax, steps=steps)
                    ase_write(str(relaxed_dir / f"{sid}.xyz"), atoms_r, format="extxyz")

                n_ok += 1
                if n_ok % 25 == 0:
                    print(f"  [{label}] {n_ok}/{len(structures)}  "
                          f"({time.perf_counter()-t0:.0f}s)")
            except Exception:
                print(f"  [FAIL] {sid}")
                traceback.print_exc()
                n_fail += 1

    elapsed = time.perf_counter() - t0
    print(f"[{label}] Done: {n_ok} OK, {n_fail} failed  ({elapsed:.1f}s)")
    print(f"  → {out}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_mace      = args.model in ("mace",     "both", "all")
    run_hydragnn  = args.model in ("hydragnn", "both", "all")
    run_uma       = args.model in ("uma",              "all")
    run_allscaip  = args.model in ("allscaip",         "all")

    if run_mace:
        print(f"\n=== MACE-MP-0 ({args.mace_size}) ===")
        try:
            sys.path.insert(0, str(HERE / "baselines" / "mace_mp0"))
            from model import AtomisticCalculator as MACE
            calc_mace = MACE.from_checkpoint(args.mace_size, device=args.device)
            run_predictions(calc_mace, "mace_mp0",
                            args.device, args.relax, args.fmax, args.steps)
        except ImportError:
            print("  mace-torch not installed. Run: pip install mace-torch")
        finally:
            sys.path.pop(0)

    if run_hydragnn:
        if not args.hydragnn_logdir:
            print("\n[HydraGNN] --hydragnn-logdir required. Skipping.")
        else:
            print(f"\n=== HydraGNN ({args.hydragnn_logdir}) ===")
            try:
                sys.path.insert(0, str(HERE / "baselines" / "hydragnn"))
                from model import AtomisticCalculator as HydraGNN
                calc_hgnn = HydraGNN.from_checkpoint(
                    args.hydragnn_logdir, device=args.device
                )
                run_predictions(calc_hgnn, "hydragnn",
                                args.device, args.relax, args.fmax, args.steps)
            except Exception:
                traceback.print_exc()
            finally:
                sys.path.pop(0)

    if run_uma:
        print(f"\n=== UMA ({args.uma_model}, task={args.uma_task}) ===")
        print("  Requires: pip install fairchem-core  +  huggingface-cli login")
        try:
            sys.path.insert(0, str(HERE / "baselines" / "uma"))
            from model import AtomisticCalculator as UMA  # type: ignore[import]
            calc_uma = UMA.from_checkpoint(
                args.uma_model, device=args.device, task_name=args.uma_task
            )
            run_predictions(calc_uma, "uma",
                            args.device, args.relax, args.fmax, args.steps)
        except ImportError:
            print("  fairchem-core not installed. Run: pip install fairchem-core")
        except Exception:
            traceback.print_exc()
        finally:
            sys.path.pop(0)

    if run_allscaip:
        print(f"\n=== AllScAIP ({args.allscaip_model}, task={args.allscaip_task}) ===")
        print("  Requires: pip install fairchem-core  +  huggingface-cli login")
        print("  Accept license at: https://huggingface.co/facebook/AllScAIP")
        try:
            sys.path.insert(0, str(HERE / "baselines" / "allscaip"))
            from model import AtomisticCalculator as AllScAIP  # type: ignore[import]
            calc_allscaip = AllScAIP.from_checkpoint(
                args.allscaip_model, device=args.device, task_name=args.allscaip_task
            )
            run_predictions(calc_allscaip, "allscaip",
                            args.device, args.relax, args.fmax, args.steps)
        except ImportError:
            print("  fairchem-core not installed. Run: pip install fairchem-core")
        except Exception:
            traceback.print_exc()
        finally:
            sys.path.pop(0)

    print("\nAll baselines complete.")
    print("Next: populate reference_data/ with DFT labels, then run score.py.")


if __name__ == "__main__":
    main()
