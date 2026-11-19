#!/usr/bin/env python3
"""
run_baselines.py — Run MLIP (MACE, HydraGNN, UMA) and AllScAIP baselines over
all test structures and write predictions/ directories. These serve as
leaderboard seeds and pipeline-validation baselines.

Each MLIP is selected by ``--mlip-config FILE``: a YAML file with an ``mlip:``
block (``backend: hydragnn | uma | mace``, that backend's settings, and its
``device``). The option is repeatable and accepts several files at once; every
file writes ``predictions/<file stem>/``. Curated selections live in ``mlip/``
and read their device from ``MLIP_DEVICE`` (default ``cpu``).

Usage
-----
# One MACE model, or all 14 bulk-material MACE models:
    python run_baselines.py --mlip-config mlip/materials/mace_omat_medium.yaml
    MLIP_DEVICE=cuda python run_baselines.py --mlip-config mlip/materials/*.yaml

# HydraGNN (HYDRAGNN_LOGDIR is the trained logdir):
    HYDRAGNN_LOGDIR=/path/to/logdir python run_baselines.py --mlip-config mlip/hydragnn.yaml

# UMA (requires fairchem-core + HF authentication):
    python run_baselines.py --mlip-config mlip/uma.yaml

# AllScAIP (requires fairchem-core + HF authentication):
    python run_baselines.py --allscaip --allscaip-device cuda

# Every curated model (MACE is dispatched to .venv-mace; the others to .venv):
    python run_baselines.py --mlip-config mlip/materials/*.yaml mlip/hydragnn.yaml \\
                            mlip/uma.yaml --allscaip

Every invocation also requires ``--elemental-reference-manifest``.

Outputs
-------
    predictions/<mlip-config stem>/ — energies.csv + forces/ (e.g. mace_omat_medium/)
    predictions/allscaip/           — energies.csv + forces/
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from ase.io import read

if (Path(__file__).parent / "energy_references.py").is_file():
    from energy_references import predict_elemental_references
else:
    from matsim_agents.datasets.energy_references import predict_elemental_references

HERE = Path(__file__).parent
STRUCT_META = HERE / "public_data" / "structures_metadata.csv"
STRUCT_ROOT = HERE / "public_data" / "structures"
PRED_ROOT = HERE / "predictions"

# Interpreter environments: MACE pins e3nn==0.4.4; FairChem/HydraGNN need e3nn>=0.5.
MACE_ENVIRONMENT, BASE_ENVIRONMENT = "mace", "base"


def load_baseline_class(name: str):
    """Load one baseline without reusing Python's cached ``model`` module."""
    model_path = HERE / "baselines" / name / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Baseline module not found: {model_path}")
    module_name = f"matsim_codabench_baseline_{name}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AtomisticCalculator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run competition baselines.")
    p.add_argument(
        "--mlip-config",
        type=Path,
        action="extend",
        nargs="+",
        default=[],
        metavar="FILE",
        help=(
            "YAML file with an mlip block (backend: hydragnn | uma | mace, its settings, "
            "and device). Repeatable; each file writes predictions/<file stem>/."
        ),
    )
    # AllScAIP (a FairChem model outside the mlip backends)
    p.add_argument("--allscaip", action="store_true", help="Also run the AllScAIP baseline.")
    p.add_argument(
        "--allscaip-model",
        default="allscaip-md-conserving-all-omol",
        help="AllScAIP model name (conserving or direct OMol variant).",
    )
    p.add_argument(
        "--allscaip-task",
        default="omol",
        help="fairchem task name for AllScAIP (default: omol; trained on OMol25).",
    )
    p.add_argument(
        "--allscaip-device", default="cpu", help="AllScAIP compute device: cpu, cuda, xpu."
    )
    # Common
    p.add_argument("--elemental-reference-manifest", required=True, type=Path)
    p.add_argument("--relax", action="store_true", help="Also run ASE LBFGS relaxation.")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=500)
    return p


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[tuple]]:
    """Parse the CLI and load every ``--mlip-config`` as ``(label, path, config)``."""
    from matsim_agents.backends.mlip.config import load_mlip_config

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.mlip_config and not args.allscaip:
        parser.error("select at least one model with --mlip-config FILE and/or --allscaip")
    selections = []
    for path in args.mlip_config:
        try:
            cfg = load_mlip_config(path)
        except (OSError, ValueError) as error:
            parser.error(f"--mlip-config {path}: {error}")
        selections.append((path.stem, path.expanduser().resolve(), cfg))
    labels = [label for label, _path, _cfg in selections] + (["allscaip"] if args.allscaip else [])
    if duplicates := sorted({label for label in labels if labels.count(label) > 1}):
        parser.error(f"prediction labels must be unique (repeated: {', '.join(duplicates)})")
    return args, selections


# ---------------------------------------------------------------------------
# Prediction runner
# ---------------------------------------------------------------------------


def run_predictions(
    calc,
    label: str,
    device: str,
    relax: bool = False,
    fmax: float = 0.05,
    steps: int = 500,
    *,
    elemental_reference_manifest: Path | None = None,
) -> None:
    out = PRED_ROOT / label
    forces_dir = out / "forces"
    relaxed_dir = out / "relaxed"
    forces_dir.mkdir(parents=True, exist_ok=True)
    if relax:
        relaxed_dir.mkdir(parents=True, exist_ok=True)

    with open(STRUCT_META, newline="") as f:
        structures = list(csv.DictReader(f))
    if elemental_reference_manifest is None:
        raise ValueError("predictions require a DFT-labelled elemental reference manifest")
    required_elements = set()
    for row in structures:
        required_elements.update(read(STRUCT_ROOT / row["file_path"]).get_chemical_symbols())
    references = predict_elemental_references(
        elemental_reference_manifest, calc, required_elements=required_elements
    )
    (out / "elemental_reference_predictions.json").write_text(
        json.dumps(references.provenance, indent=2), encoding="utf-8"
    )

    energy_file = out / "energies.csv"
    energy_fields = ["structure_id", "energy_eV", "energy_eV_per_atom", "n_atoms"]

    t0 = time.perf_counter()
    n_ok = n_fail = 0

    with (
        open(energy_file, "w", newline="") as ef,
        open(out / "formation_energies.csv", "w", newline="") as ff,
    ):
        writer = csv.DictWriter(ef, fieldnames=energy_fields)
        writer.writeheader()
        formation_writer = csv.DictWriter(
            ff, fieldnames=["structure_id", "formation_energy_eV_per_atom", "n_atoms"]
        )
        formation_writer.writeheader()

        for row in structures:
            sid = row["structure_id"]
            fpath = STRUCT_ROOT / row["file_path"]
            if not fpath.is_file():
                n_fail += 1
                continue
            try:
                atoms = read(str(fpath), format="extxyz")
                atoms.calc = calc
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                formation_energy = references.formation_energy(atoms, energy, model=True)

                writer.writerow(
                    {
                        "structure_id": sid,
                        "energy_eV": f"{energy:.8f}",
                        "energy_eV_per_atom": f"{energy / len(atoms):.8f}",
                        "n_atoms": len(atoms),
                    }
                )
                np.save(str(forces_dir / f"{sid}.npy"), forces.astype(np.float32))
                formation_writer.writerow(
                    {
                        "structure_id": sid,
                        "formation_energy_eV_per_atom": f"{formation_energy:.8f}",
                        "n_atoms": len(atoms),
                    }
                )

                if relax:
                    from ase.io import write as ase_write
                    from ase.optimize import LBFGS

                    atoms_r = atoms.copy()
                    atoms_r.calc = calc
                    LBFGS(atoms_r, logfile=None).run(fmax=fmax, steps=steps)
                    ase_write(str(relaxed_dir / f"{sid}.xyz"), atoms_r, format="extxyz")

                n_ok += 1
                if n_ok % 25 == 0:
                    print(
                        f"  [{label}] {n_ok}/{len(structures)}  ({time.perf_counter() - t0:.0f}s)"
                    )
            except Exception:
                print(f"  [FAIL] {sid}")
                traceback.print_exc()
                n_fail += 1

    elapsed = time.perf_counter() - t0
    print(f"[{label}] Done: {n_ok} OK, {n_fail} failed  ({elapsed:.1f}s)")
    print(f"  → {out}/")
    if n_fail:
        raise RuntimeError(f"{label} failed for {n_fail} of {len(structures)} structures")


def environment_of(cfg) -> str:
    """Interpreter environment that owns an mlip block's backend."""
    return MACE_ENVIRONMENT if cfg.backend == "mace" else BASE_ENVIRONMENT


def _common_arguments(args: argparse.Namespace) -> list[str]:
    common = [
        "--elemental-reference-manifest",
        str(args.elemental_reference_manifest.expanduser().resolve()),
        "--fmax",
        str(args.fmax),
        "--steps",
        str(args.steps),
    ]
    return common + (["--relax"] if args.relax else [])


def dispatch_environments(args: argparse.Namespace, selections: list[tuple]) -> bool:
    """Run selections in their owning interpreters when they need more than one.

    Returns ``False`` (nothing dispatched) when one interpreter serves every selection.
    """
    groups: dict[str, list[str]] = {}
    for _label, path, cfg in selections:
        groups.setdefault(environment_of(cfg), []).append(str(path))
    if args.allscaip:
        groups.setdefault(BASE_ENVIRONMENT, [])
    if len(groups) < 2:
        return False

    repo = HERE.parents[1]
    pythons = {
        MACE_ENVIRONMENT: Path(
            os.environ.get("MATSIM_MACE_PYTHON", repo / ".venv-mace/bin/python")
        ),
        BASE_ENVIRONMENT: Path(os.environ.get("MATSIM_BASE_PYTHON", repo / ".venv/bin/python")),
    }
    for environment in (MACE_ENVIRONMENT, BASE_ENVIRONMENT):
        python = pythons[environment]
        if not python.is_file():
            raise SystemExit(
                f"{environment} interpreter not found: {python}. Run the facility installer "
                f"with INSTALL_MACE=1 and/or INSTALL_UMA=1 first."
            )
    for environment in (MACE_ENVIRONMENT, BASE_ENVIRONMENT):
        python = pythons[environment]
        child_args = _common_arguments(args)
        if groups[environment]:
            child_args += ["--mlip-config", *groups[environment]]
        if environment == BASE_ENVIRONMENT and args.allscaip:
            child_args += [
                "--allscaip",
                "--allscaip-model",
                args.allscaip_model,
                "--allscaip-task",
                args.allscaip_task,
                "--allscaip-device",
                args.allscaip_device,
            ]
        print(f"\n=== dispatch {environment} environment via {python} ===", flush=True)
        subprocess.run([str(python), str(Path(__file__).resolve()), *child_args], check=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args, selections = parse_args(argv)
    if dispatch_environments(args, selections):
        return

    MLIP = load_baseline_class("mlip") if selections else None
    for label, path, cfg in selections:
        print(f"\n=== {cfg.backend} ({cfg.model_reference}, device={cfg.device}) from {path} ===")
        try:
            calc = MLIP.from_mlip_config(cfg)
            run_predictions(
                calc,
                label,
                cfg.device,
                args.relax,
                args.fmax,
                args.steps,
                elemental_reference_manifest=args.elemental_reference_manifest,
            )
        except Exception:
            traceback.print_exc()
            raise

    if args.allscaip:
        print(f"\n=== AllScAIP ({args.allscaip_model}, task={args.allscaip_task}) ===")
        print("  Requires: pip install fairchem-core  +  huggingface-cli login")
        print("  Accept license at: https://huggingface.co/facebook/AllScAIP")
        try:
            AllScAIP = load_baseline_class("allscaip")
            calc_allscaip = AllScAIP.from_checkpoint(
                args.allscaip_model, device=args.allscaip_device, task_name=args.allscaip_task
            )
            run_predictions(
                calc_allscaip,
                "allscaip",
                args.allscaip_device,
                args.relax,
                args.fmax,
                args.steps,
                elemental_reference_manifest=args.elemental_reference_manifest,
            )
        except ImportError:
            print("  fairchem-core not installed. Run: pip install fairchem-core")
            raise
        except Exception:
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
