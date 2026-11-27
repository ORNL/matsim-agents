"""Step 2 perturbation diagnostic for the HydraGNN BEST6 fp64 MLFF.

Question being tested: when an "already relaxed" structure (reported as
converged at FIRE step 1 with |F|max < fmax) is given a controlled random
displacement, does the MLFF's force field drive it back to a minimum, or
does it sit on a flat numerical-noise plateau?

For each input structure we:
  (1) read the structure;
  (2) record the model's energy + |F|max at the unperturbed geometry;
  (3) apply an isotropic random displacement of fixed RMS amplitude;
  (4) run a tight relaxation (fmax=1e-4 eV/A, maxiter=500) and record
      the per-step energy and force trajectory;
  (5) compare the recovered structure to the original (atomic RMSD).

A bona-fide local minimum should:
  - have nontrivial |F|max immediately after the perturbation
    (~ k * displacement, with k the local stiffness);
  - decrease that |F|max monotonically during relaxation;
  - recover the original geometry within ~10x machine precision in RMSD.

Numerical-noise plateaus instead show:
  - |F|max staying ~constant or growing during relaxation;
  - high RMSD between recovered and original geometries.

Outputs a JSON summary alongside per-case CSVs that the per-seed
relaxation tool writes by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from matsim_agents.backends.mlip.arguments import add_mlip_arguments, mlip_config_from_args
from matsim_agents.backends.mlip.config import HydraGNNConfig
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, relax_with_mlip


def _atoms_rmsd(a, b) -> float:
    pa = np.asarray(a.get_positions())
    pb = np.asarray(b.get_positions())
    return float(np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=1))))


def _initial_energy_force(structure_path: str, mlip: HydraGNNConfig) -> tuple[float, float]:
    """Single-point energy + |F|max at the input geometry (no optimization)."""
    args = RelaxStructureInput(
        structure_path=structure_path,
        mlip=mlip,
        optimizer="FIRE",
        maxiter=0,  # ASE FIRE with maxiter=0 → single eval
        fmax=1e9,  # accept any |F|max
        random_displacement=False,
        output_dir=None,
    )
    r = relax_with_mlip(args)
    return r.final_energy_eV, r.final_max_force_eV_per_A


def run_case(
    name: str,
    structure_path: str,
    mlip: HydraGNNConfig,
    output_root: Path,
    displacement_scale: float,
    seed: int,
) -> dict:
    print(f"\n=== {name} ===", flush=True)
    print(f"  source : {structure_path}", flush=True)
    case_dir = output_root / name
    case_dir.mkdir(parents=True, exist_ok=True)

    from ase.io import read

    initial_atoms = read(structure_path)
    n_atoms = len(initial_atoms)
    print(f"  n_atoms: {n_atoms}", flush=True)

    # (a) unperturbed single-point
    E0, F0 = _initial_energy_force(structure_path, mlip)
    print(f"  unperturbed: E={E0:+.6e} eV   |F|max={F0:.3e} eV/A", flush=True)

    # (b) perturbed relaxation
    perturbed_path = case_dir / f"{name}_perturbed_input.vasp"
    rng = np.random.default_rng(seed)
    disp = rng.normal(scale=displacement_scale, size=(n_atoms, 3))
    perturbed_atoms = initial_atoms.copy()
    perturbed_atoms.set_positions(perturbed_atoms.get_positions() + disp)
    rms_disp = float(np.sqrt(np.mean(np.sum(disp**2, axis=1))))
    print(f"  applied displacement RMS = {rms_disp:.4f} A", flush=True)
    perturbed_atoms.write(str(perturbed_path), format="vasp")

    args = RelaxStructureInput(
        structure_path=str(perturbed_path),
        mlip=mlip,
        optimizer="FIRE",
        maxiter=500,
        fmax=1e-4,
        random_displacement=False,  # we already perturbed above
        relative_increase_threshold=10.0,  # disable rollback for diagnosis
        output_dir=str(case_dir),
    )
    r = relax_with_mlip(args)
    print(
        f"  relaxed:    E={r.final_energy_eV:+.6e} eV  |F|max={r.final_max_force_eV_per_A:.3e} eV/A"
        f"  steps={r.num_steps}",
        flush=True,
    )

    # (c) inspect the per-step CSV that relax_with_mlip writes
    csv_candidates = sorted(case_dir.glob("*_optimization.csv"))
    assert csv_candidates, f"no optimization CSV produced in {case_dir}"
    csv_path = csv_candidates[-1]
    import csv as _csv

    with open(csv_path) as fh:
        rows = list(_csv.DictReader(fh))

    energies = [float(row["energy_eV"]) for row in rows]
    fmax = [float(row["max_force_eV_per_A"]) for row in rows]

    # (d) RMSD between recovered structure and the original input
    recovered_candidates = sorted(case_dir.glob("*_optimized_structure.vasp"))
    recovered = read(str(recovered_candidates[-1]))
    rmsd_to_original = _atoms_rmsd(initial_atoms, recovered)

    summary = {
        "name": name,
        "input_structure": structure_path,
        "n_atoms": n_atoms,
        "displacement_scale_A": displacement_scale,
        "applied_rms_displacement_A": rms_disp,
        "unperturbed": {
            "energy_eV": E0,
            "energy_per_atom_meV": (E0 / n_atoms) * 1000.0,
            "fmax_eV_per_A": F0,
        },
        "post_relaxation": {
            "energy_eV": r.final_energy_eV,
            "energy_per_atom_meV": (r.final_energy_eV / n_atoms) * 1000.0,
            "fmax_eV_per_A": r.final_max_force_eV_per_A,
            "num_steps": r.num_steps,
        },
        "trajectory": {
            "energy_initial_eV": energies[0] if energies else None,
            "energy_final_eV": energies[-1] if energies else None,
            "energy_min_eV": min(energies) if energies else None,
            "fmax_initial_eV_per_A": fmax[0] if fmax else None,
            "fmax_final_eV_per_A": fmax[-1] if fmax else None,
            "fmax_max_eV_per_A": max(fmax) if fmax else None,
        },
        "rmsd_recovered_vs_original_A": rmsd_to_original,
        "csv_path": str(csv_path),
        "recovered_structure": str(recovered_candidates[-1]),
    }
    print(f"  RMSD(recovered, original) = {rmsd_to_original:.4f} A", flush=True)
    print(f"  fmax: {fmax[0]:.3e} -> {fmax[-1]:.3e} eV/A over {len(fmax)} step(s)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_mlip_arguments(ap)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--displacement-scale",
        type=float,
        default=0.10,
        help="per-coordinate sigma in Angstrom (default 0.10).",
    )
    ap.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="name:path pairs, e.g. BaTiO3:/path/to/struct.vasp",
    )
    args = ap.parse_args()
    mlip = mlip_config_from_args(ap, args)
    if not isinstance(mlip, HydraGNNConfig):
        ap.error("--mlip-config must select backend: hydragnn (the diagnostic targets HydraGNN)")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in args.cases:
        name, _, path = spec.partition(":")
        if not name or not path:
            sys.exit(f"bad --cases spec '{spec}', expected NAME:PATH")
        summaries.append(
            run_case(
                name=name,
                structure_path=path,
                mlip=mlip,
                output_root=out_root,
                displacement_scale=args.displacement_scale,
                seed=args.seed,
            )
        )

    summary_path = out_root / "step2_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "cases": summaries,
                "displacement_scale_A": args.displacement_scale,
                "seed": args.seed,
            },
            fh,
            indent=2,
        )
    print(f"\nWrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
