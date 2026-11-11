"""Score frozen E3 MLIPs on Codabench public DFT labels only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

REPO = Path(__file__).resolve().parents[2]
WORKTREE_SOURCE = (REPO / "src").resolve()
ORIGINAL_CHECKOUT_SOURCE = (REPO.parent / "matsim-agents" / "src").resolve()
# The shared HPC venv is editable-installed from a sibling checkout; prefer this checkout.
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != ORIGINAL_CHECKOUT_SOURCE
]
sys.path.insert(0, str(WORKTREE_SOURCE))

_mlip_predict = importlib.import_module("benchmarks.e3.mlip_predict")
MODELS = _mlip_predict.MODELS
calculator = _mlip_predict.calculator

HYDRAGNN_MODEL = "hydragnn-omat24"
HYDRAGNN_HEAD = "OMat24"
HYDRAGNN_HEAD_INDEX = 7
HYDRAGNN_CHECKPOINT = "multidataset_hpo-BEST6-fp64_epoch_97.pk"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_structures": 0, "status": "no_successful_predictions"}
    errors = [np.asarray(row["force_error_eV_A"]) for row in rows]
    flat = np.concatenate([error.reshape(-1) for error in errors])
    formation_errors = [row["formation_energy_error_eV_atom"] for row in rows]
    return {
        "n_structures": len(rows),
        "force_component_mae_eV_A": float(np.mean(np.abs(flat))),
        "force_component_rmse_eV_A": float(np.sqrt(np.mean(flat**2))),
        "structure_macro_force_component_mae_eV_A": float(
            np.mean([np.mean(np.abs(error)) for error in errors])
        ),
        "formation_energy_mae_eV_atom": float(np.mean(np.abs(formation_errors))),
        "formation_energy_rmse_eV_atom": float(np.sqrt(np.mean(np.square(formation_errors)))),
    }


def build_calculator(
    model_name: str,
    device: str,
    hydragnn_logdir: Path | None,
    hydragnn_root: Path | None,
):
    if model_name != HYDRAGNN_MODEL:
        return calculator(model_name, device), MODELS[model_name]
    if hydragnn_logdir is None or hydragnn_root is None:
        raise ValueError("HydraGNN requires explicit logdir and source root")
    config_path = hydragnn_logdir / "config.json"
    checkpoint_path = hydragnn_logdir / HYDRAGNN_CHECKPOINT
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("HydraGNN config/checkpoint is missing")
    config = json.loads(config_path.read_text())
    heads = config["NeuralNetwork"]["Architecture"]["output_heads"]["graph"]
    if len(heads) != 16 or heads[HYDRAGNN_HEAD_INDEX]["type"] != f"branch-{HYDRAGNN_HEAD_INDEX}":
        raise ValueError("HydraGNN checkpoint does not expose the expected 16-head layout")
    os.environ["HYDRAGNN_ROOT"] = str(hydragnn_root.resolve())
    from matsim_agents.backends.mlip.calculator import build_hydragnn_calculator
    from matsim_agents.backends.mlip.config import HydraGNNConfig

    cfg = HydraGNNConfig(
        logdir=hydragnn_logdir,
        checkpoint=HYDRAGNN_CHECKPOINT,
        inference_head=HYDRAGNN_HEAD,
        mlp_device=device,
    )
    model_calculator = build_hydragnn_calculator(cfg)
    model_config = {
        "backend": "hydragnn",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "config_sha256": sha256(config_path),
        "inference_head": HYDRAGNN_HEAD,
        "inference_head_index": HYDRAGNN_HEAD_INDEX,
        "branch_weight_mlp_used": False,
        "checkpoint_load_report": getattr(
            model_calculator, "hydragnn_checkpoint_load_report", None
        ),
    }
    return model_calculator, model_config


def benchmark(
    codabench_root: Path,
    reference_root: Path,
    model_name: str,
    output: Path,
    device: str,
    hydragnn_logdir: Path | None = None,
    hydragnn_root: Path | None = None,
) -> dict[str, Any]:
    codabench_root = codabench_root.resolve()
    reference_root = reference_root.resolve()
    data = reference_root / "anon_reference_data"
    split = codabench_root / "reference_data"
    public_ids = set((split / "public_ids.txt").read_text().split())

    metadata = {row["structure_id"]: row for row in read_csv(data / "structures_metadata.csv")}
    labels = {}
    with (data / "formation_energies.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["structure_id"] in public_ids:
                labels[row["structure_id"]] = row
    ids = sorted(labels)
    if set(ids) != public_ids:
        raise ValueError("public benchmark IDs do not have a complete label set")
    if any(sid not in metadata for sid in ids):
        raise ValueError("public benchmark metadata is incomplete")

    calc, model_config = build_calculator(model_name, device, hydragnn_logdir, hydragnn_root)
    element_root = reference_root / "dft_reference/vasp/elemental_refs/calculations"
    elements = sorted({
        symbol
        for sid in ids
        for symbol in read(data / "relaxed" / f"{sid}.xyz").get_chemical_symbols()
    })
    model_refs: dict[str, float] = {}
    reference_failures: dict[str, str] = {}
    started = time.perf_counter()
    for symbol in elements:
        path = element_root / symbol / "CONTCAR"
        if not path.is_file():
            reference_failures[symbol] = "missing elemental CONTCAR"
            continue
        try:
            atoms = read(path, format="vasp")
            atoms.calc = calc
            model_refs[symbol] = float(atoms.get_potential_energy()) / len(atoms)
        except Exception as exc:
            reference_failures[symbol] = f"{type(exc).__name__}: {exc}"

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for sid in ids:
        try:
            atoms = read(data / "relaxed" / f"{sid}.xyz")
            reference_forces = np.load(data / "forces" / f"{sid}.npy", allow_pickle=False)
            if reference_forces.shape != (len(atoms), 3):
                raise ValueError("reference force shape does not match structure")
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            if not np.isfinite(energy) or not np.isfinite(forces).all():
                raise ValueError("model returned non-finite energy or force")
            missing = set(atoms.get_chemical_symbols()) - model_refs.keys()
            if missing:
                raise ValueError(f"missing model elemental references: {sorted(missing)}")
            formation_energy = (
                energy - sum(model_refs[symbol] for symbol in atoms.get_chemical_symbols())
            ) / len(atoms)
            predictions.append({
                "structure_id": sid,
                "material_class": metadata[sid]["material_class"],
                "formula": metadata[sid]["formula"],
                "n_atoms": len(atoms),
                "formation_energy_prediction_eV_atom": formation_energy,
                "formation_energy_reference_eV_atom": float(
                    labels[sid]["formation_energy_eV_per_atom"]
                ),
                "formation_energy_error_eV_atom": formation_energy
                - float(labels[sid]["formation_energy_eV_per_atom"]),
                "force_error_eV_A": (forces - reference_forces).tolist(),
            })
        except Exception as exc:
            failures.append({"structure_id": sid, "error": f"{type(exc).__name__}: {exc}"})

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["material_class"]].append(row)
    result = {
        "schema_version": 1,
        "status": "public_screen_only_not_E3_qualified_or_final_test",
        "model": model_name,
        "model_config": model_config,
        "device": device,
        "public_label_digest": hashlib.sha256(json.dumps(
            [labels[sid] for sid in ids], sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "dataset": {
            "public_ids_only": True,
            "public_label_count": len(ids),
            "private_labels_in_metrics_or_output": False,
            "public_ids_sha256": sha256(split / "public_ids.txt"),
            "geometry": "DFT-relaxed public structures; single-point MLIP evaluation",
            "DFT_force_labels": "saved force arrays at relaxed geometry",
            "DFT_formation_energy_source": "Codabench public formation-energy labels",
            "predicted_formation_energy_method": {
                "formula": "(E_model(compound) - sum_i n_i * E_model(pure_element_i)) / N",
                "same_model_and_head_for_compound_and_elements": True,
                "benchmark_labels_used_for_correction": False,
                "composition_linear_baseline_cancels_for_residual_models": True,
            },
            "e3_reference_qualification": "not_assumed; class-specific method audit required",
        },
        "elapsed_seconds_including_model_elemental_references": time.perf_counter() - started,
        "model_elemental_reference_failures": reference_failures,
        "successful_predictions": len(predictions),
        "failed_predictions": failures,
        "overall": summarize(predictions),
        "by_material_class": {
            material_class: summarize(rows) for material_class, rows in sorted(grouped.items())
        },
        "predictions": predictions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark result: {output}")
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    repo = REPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codabench-root", type=Path,
        default=repo.parent / "matsim-agents" / "benchmarks" / "codabench",
    )
    parser.add_argument(
        "--reference-root", type=Path, default=repo.parent / "matsim-agents-dft-reference"
    )
    parser.add_argument("--model", choices=sorted([*MODELS, HYDRAGNN_MODEL]), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hydragnn-logdir", type=Path)
    parser.add_argument("--hydragnn-root", type=Path)
    args = parser.parse_args()
    if args.model == HYDRAGNN_MODEL:
        args.hydragnn_logdir = args.hydragnn_logdir or (
            repo.parent
            / "matsim-agents"
            / "models"
            / "hydragnn-sc26"
            / "multidataset_hpo-BEST6-fp64"
        )
        args.hydragnn_root = args.hydragnn_root or (
            repo.parent / "HydraGNN-e4b-frozen-81d5b883"
        )
    result = benchmark(
        args.codabench_root,
        args.reference_root,
        args.model,
        args.output,
        args.device,
        args.hydragnn_logdir,
        args.hydragnn_root,
    )
    print(json.dumps({
        "model": result["model"],
        "successful_predictions": result["successful_predictions"],
        "failed_predictions": len(result["failed_predictions"]),
        "overall": result["overall"],
        "by_material_class": result["by_material_class"],
        "elapsed_seconds_including_model_elemental_references": result[
            "elapsed_seconds_including_model_elemental_references"
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())