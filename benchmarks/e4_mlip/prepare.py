"""CPU-only model audit, inventory freezing and dry-run PBS template generation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write

from benchmarks.e4_mlip.run import DEVICES, Structure, geometry_hash, load_inventory, sha256

ROOT = Path(__file__).resolve().parents[2]


def generate_bcc_candidates(parent: Path, output: Path, *, seed: int = 20261108) -> Path:
    """Freeze 120 reproducible site occupations, without asserting checkpoint coverage."""
    from benchmarks.e4_mlip.inventory import family_for

    atoms = read(parent)
    composition = dict(Counter(atoms.get_chemical_symbols()))
    if (
        len(atoms) != 128
        or family_for(composition) != "bcc_refractory"
        or not atoms.pbc.all()
        or atoms.get_volume() <= 0
        or not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
    ):
        raise ValueError("Candidate parent must be a finite periodic 128-atom refractory alloy")
    output.mkdir(parents=True, exist_ok=False)
    seen = {geometry_hash(atoms)}
    sources = []
    for index in range(120):
        rng = np.random.default_rng(seed + index)
        candidate = atoms.copy()
        candidate.calc = None
        candidate.numbers = rng.permutation(atoms.numbers)
        digest = geometry_hash(candidate)
        if digest in seen:
            raise ValueError(f"Duplicate occupation generated at index {index}")
        seen.add(digest)
        identifier = f"E4B-BCC-generated-{index + 1:04d}"
        path = output / f"{identifier}.extxyz"
        write(path, candidate, format="extxyz")
        sources.append(
            {
                "structure_id": identifier,
                "family": "bcc_refractory",
                "path": str(path.resolve()),
                "source": "Reproducible site permutation of staged E4 BCC parent; no DFT labels",
                "initial_status": "initial",
                "provenance": {
                    "parent": str(parent.resolve()),
                    "parent_sha256": sha256(parent),
                    "parent_geometry_sha256": geometry_hash(atoms),
                    "algorithm": "numpy.default_rng(seed).permutation(parent.numbers)",
                    "numpy_version": np.__version__,
                    "random_seed": seed + index,
                    "cell_and_positions_preserved": True,
                    "scientific_coverage": "unverified; candidate workload only",
                },
            }
        )
    source_path = output / "sources.json"
    source_path.write_text(json.dumps(sources, indent=2) + "\n")
    rows = freeze_inventory(source_path, output / "structures.json")
    (output / "pilot-representatives.json").write_text(
        json.dumps([rows[0].model_dump(mode="json")], indent=2) + "\n"
    )
    (output / "candidate-status.json").write_text(
        json.dumps(
            {
                "status": "FROZEN_CANDIDATES_NOT_SCIENTIFICALLY_QUALIFIED",
                "count": 120,
                "composition": composition,
                "source": str(parent.resolve()),
                "checkpoint_coverage_established": False,
                "production_execution_authorized": False,
                "note": "Distinct site occupations; not a symmetry-canonical uniqueness proof.",
            },
            indent=2,
        )
        + "\n"
    )
    return output / "structures.json"


def audit(example: Path, output: Path) -> dict:
    logdir = example / "multidataset_hpo-BEST6-fp64"
    paths = {
        "checkpoint": logdir / "multidataset_hpo-BEST6-fp64_epoch_97.pk",
        "branch_checkpoint": example / "mlp_branch_weights.pt",
        "config": logdir / "config.json",
    }
    revision = subprocess.check_output(
        ["git", "-C", str(example), "rev-parse", "HEAD"], text=True
    ).strip()
    frozen = json.loads((Path(__file__).parent / "frozen-source.json").read_text())
    if revision != frozen["commit"]:
        raise ValueError(f"HydraGNN revision differs from frozen source: {revision}")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        metadata = {
            "status": "NO-GO",
            "backend": "hydragnn",
            "hydragnn_revision": revision,
            "hydragnn_root": str(example.resolve().parents[1]),
            "missing_checkpoint_files": missing,
            "note": "Awaiting checkpoint upload; hashes and model compatibility are unverified",
        }
        output.write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
    config = json.loads(paths["config"].read_text())
    metadata = {
        "backend": "hydragnn",
        "model_version": "BEST6-fp64 epoch 97",
        "architecture": config["NeuralNetwork"]["Architecture"],
        "precision": "fp64",
        "accelerator_backend": "xpu",
        "prediction_targets": ["energy", "forces"],
        "analytical_forces": True,
        "force_evidence": "inference_fused.py uses negative torch.autograd.grad of energy",
        "input_requirements": (
            "Periodic ASE atoms; atomic numbers, coordinates, cell, graph attributes"
        ),
        "supported_species": [],
        "species_verified": False,
        "species_evidence": None,
        "numerical_tolerances_approved": False,
        "energy_convention_verified": False,
        "energy_units": None,
        "force_units": None,
        "energy_reference_evidence": None,
        "proposed_tolerances": {"energy_eV": 1e-6, "force_eV_A": 1e-5, "position_A": 1e-4},
        "relaxation_interface": "relax_with_mlip, ASE FIRE, fixed cell",
        "hydragnn_revision": revision,
        "hydragnn_root": str(example.resolve().parents[1]),
        "software": {
            name: importlib.metadata.version(name) for name in ("torch", "hydragnn", "ase", "numpy")
        },
        **{
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "optional_backends": {
            "uma": "Installed isolated environment; checkpoint/species/XPU operations not verified",
            "mace": "Isolated environment; checkpoint/species/XPU operations not verified",
        },
    }
    output.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def freeze_inventory(sources: Path, output: Path) -> list[Structure]:
    """Freeze an explicit reviewed source list; never generate or duplicate geometries."""
    rows = []
    for entry in json.loads(sources.read_text()):
        path = Path(entry["path"]).resolve()
        atoms = read(path)
        rows.append(
            Structure(
                structure_id=entry["structure_id"],
                family=entry["family"],
                path=path,
                sha256=sha256(path),
                geometry_sha256=geometry_hash(atoms),
                atom_count=len(atoms),
                species=sorted(set(atoms.get_chemical_symbols())),
                source=entry["source"],
                composition=dict(Counter(atoms.get_chemical_symbols())),
                initial_status=entry.get("initial_status", "unknown"),
                provenance=entry.get("provenance", {}),
            ).model_dump(mode="json")
        )
    temporary = output.with_suffix(".pending.json")
    temporary.write_text(json.dumps(rows, indent=2) + "\n")
    try:
        validated = load_inventory(temporary)
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
        return validated
    finally:
        temporary.unlink(missing_ok=True)


def pbs_templates(output: Path) -> None:
    for devices in DEVICES:
        (output / f"devices-{devices:02d}.pbs").write_text(
            f"""#!/bin/bash
#PBS -N e4b-d{devices}
#PBS -A HydraGNN
#PBS -q capacity
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
set -eo pipefail
: "${{REPO:?}}" "${{MANIFEST:?}}" "${{MODEL_METADATA:?}}" "${{OUTPUT_ROOT:?}}"
source "${{REPO}}/deployments/aurora/setup/aurora-module-stack.sh"
load_aurora_xpu_modules
source "${{MATSIM_AURORA_VENV:-${{REPO}}/.venv}}/bin/activate"
export PYTHONPATH="${{HYDRAGNN_EXAMPLE:?}}:${{HYDRAGNN_ROOT:?}}:"\\
"${{REPO}}/src:${{REPO}}:${{PYTHONPATH:-}}"
export PYTHONNOUSERSITE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export OMP_NUM_THREADS=8
cd "${{REPO}}"
echo "job=${{PBS_JOBID:-dryrun}} allocation=HydraGNN queue=capacity host=$(hostname)"
module -t list 2>&1
git rev-parse HEAD
sha256sum "${{MANIFEST}}" "${{MODEL_METADATA}}"
python --version
ARGS=()
LAUNCH=()
if [[ "${{E4B_EXECUTE:-0}}" == 1 ]]; then
  : "${{PBS_JOBID:?Execution requires an approved PBS allocation}}"
  ARGS=(--execute)
  LAUNCH=(mpiexec -n 1 --ppn 1)
fi
"${{LAUNCH[@]}}" python -m benchmarks.e4_mlip.run \\
  --manifest "${{MANIFEST}}" --metadata "${{MODEL_METADATA}}" \\
  --devices {devices} --mode "${{MODE:-inference}}" --repetition "${{REP:-1}}" \\
  --output "${{OUTPUT_ROOT}}/${{MODE:-inference}}-d{devices}-r${{REP:-1}}-${{PBS_JOBID:-dryrun}}" \\
  "${{ARGS[@]}}"
"""
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument(
        "--bcc-parent",
        type=Path,
        help="Generate a provisional 120-case site-permutation workload from a staged BCC parent",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sources and args.bcc_parent:
        parser.error("--sources and --bcc-parent are mutually exclusive")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = audit(args.example, args.output / "model-metadata.json")
    pbs_templates(args.output)
    if args.sources:
        freeze_inventory(args.sources, args.output / "structures.json")
    if args.bcc_parent:
        generate_bcc_candidates(args.bcc_parent, args.output / "bcc-candidates")
    blockers = [
        "Checkpoint species support unverified",
        "Numerical consistency tolerances not approved",
        "Energy/force units and energy-reference conventions unverified",
        "Real XPU memory and synchronized timing pilot not authorized",
    ]
    if not args.sources and not args.bcc_parent:
        blockers.insert(0, "No reviewed 120-structure source list provided")
    if args.bcc_parent:
        blockers.insert(0, "Generated BCC inventory is provisional pending checkpoint coverage")
    if metadata.get("missing_checkpoint_files"):
        blockers.insert(0, "Frozen checkout awaits checkpoint upload")
    (args.output / "readiness.json").write_text(
        json.dumps(
            {
                "status": "NO-GO",
                "blockers": blockers,
                "gpu_initialized": False,
                "submission_authorized": False,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
