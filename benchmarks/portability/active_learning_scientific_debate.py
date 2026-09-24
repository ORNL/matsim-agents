#!/usr/bin/env python3
"""Ground an all-model scientific debate in real active-learning evidence.

Closes the gap between two separately-tested portability tracks:

* Real active learning: UMA MD sampling -> acquisition -> QE DFT labelling
  of the benchmark Si cell (:mod:`matsim_agents.active_learning.loop`).
* Real all-model debate: every first-class local LLM debates one hypothesis
  (:mod:`benchmarks.portability.all_model_scientific_debate`).

This module runs the AL loop first, turns its labelled dataset into a
evidence summary, and hands that evidence to the exact same debate machinery
used by ``all_model_scientific_debate.py`` -- it does not reimplement any
catalog, exclusion, or verdict-validation logic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.portability.all_model_scientific_debate import (  # noqa: E402
    CATALOG,
    DEFAULT_MODELS_ROOT,
    execute_all_model_scientific_debate,
)
from matsim_agents.active_learning.config import ALConfig  # noqa: E402
from matsim_agents.active_learning.loop import run_active_learning  # noqa: E402
from matsim_agents.backends.llm.provider import get_chat_model  # noqa: E402

DEFAULT_AL_CONFIG = ROOT / "benchmarks" / "portability" / "config" / "active_learning" / "al-si-uma-qe.yaml"

HYPOTHESIS_TEMPLATE = (
    "An active-learning loop just generated candidate configurations of the fixed "
    "benchmark silicon cell by running molecular-dynamics sampling under the UMA "
    "foundation MLIP, then labelled a subset of those candidates with real DFT "
    "single-point calculations. Given the evidence below, assess whether this "
    "active-learning iteration produced physically reasonable, mutually consistent "
    "silicon configurations (plausible energies and forces, no signs of unphysical "
    "structures), and state whether that evidence is sufficient to trust the MLIP "
    "for autonomous candidate generation on this material without further DFT "
    "labelling, or whether more iterations are required first.\n\n{evidence}"
)


def _load_iteration_state(al_out_dir: Path, iteration: int = 0) -> dict[str, Any]:
    state_path = al_out_dir / f"iteration_{iteration:04d}" / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"active-learning iteration state not found: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def _dataset_frame_summary(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.is_file():
        return []
    from ase.io import read

    frames = read(str(dataset_path), index=":")
    summary = []
    for frame in frames:
        forces = frame.arrays.get("forces")
        max_force = float(np.max(np.linalg.norm(forces, axis=1))) if forces is not None else None
        summary.append(
            {
                "candidate_id": Path(str(frame.info.get("source_work_dir", "?"))).name,
                "dft_backend": frame.info.get("dft_backend"),
                "energy_eV": frame.info.get("energy"),
                "max_force_eV_per_A": max_force,
            }
        )
    return summary


def build_evidence(state: dict[str, Any], frames: list[dict[str, Any]]) -> str:
    lines = [
        "Real active-learning loop evidence for the benchmark Si cell "
        f"(UMA MD sampling -> acquisition -> {state.get('dft_backend') or 'DFT'} labelling):",
        f"- MD candidates generated: {state.get('n_candidates')}",
        f"- Candidates selected for DFT labelling: {state.get('n_selected')}",
        f"- DFT labels converged: {state.get('n_dft_converged')} "
        f"(failed: {state.get('n_dft_failed')})",
    ]
    if state.get("score_mean") is not None:
        lines.append(
            "- Acquisition uncertainty score min/mean/max: "
            f"{state['score_min']:.4g}/{state['score_mean']:.4g}/{state['score_max']:.4g}"
        )
    for frame in frames:
        energy = frame["energy_eV"]
        max_force = frame["max_force_eV_per_A"]
        energy_str = f"{energy:.6f} eV" if energy is not None else "unavailable"
        force_str = f"{max_force:.4f} eV/A" if max_force is not None else "unavailable"
        lines.append(
            f"  - frame {frame['candidate_id']}: DFT energy {energy_str}, max |force| {force_str}"
        )
    return "\n".join(lines)


def execute_active_learning_scientific_debate(
    *,
    output: Path,
    al_config: Path = DEFAULT_AL_CONFIG,
    rounds: int,
    catalog: Path = CATALOG,
    models_root: Path = DEFAULT_MODELS_ROOT,
    excluded_models: set[str] | None = None,
    environment: dict[str, str] | None = None,
    model_factory=get_chat_model,
    al_runner=run_active_learning,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)

    al_out_dir = output / "al-loop"
    os.environ["MATSIM_AL_OUT_DIR"] = str(al_out_dir.resolve())
    al_cfg = ALConfig.from_yaml(al_config)
    al_runner(al_cfg)

    state = _load_iteration_state(al_out_dir)
    frames = _dataset_frame_summary(al_out_dir / "dataset.extxyz")
    evidence = build_evidence(state, frames)
    al_passed = state.get("status") == "complete" and bool(state.get("n_dft_converged"))

    debate = execute_all_model_scientific_debate(
        output=output / "debate",
        rounds=rounds,
        catalog=catalog,
        models_root=models_root,
        excluded_models=excluded_models,
        environment=environment,
        model_factory=model_factory,
        hypothesis=HYPOTHESIS_TEMPLATE.format(evidence=evidence),
    )

    payload = {
        "schema_version": 1,
        "benchmark": "active-learning-scientific-debate",
        "status": "passed" if al_passed and debate["status"] == "passed" else "failed",
        "active_learning": {
            "config": str(al_config.resolve()),
            "out_dir": str(al_out_dir.resolve()),
            "state": state,
            "labelled_frames": frames,
            "evidence": evidence,
        },
        "debate": debate,
    }
    (output / "active_learning_scientific_debate_result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--al-config", type=Path, default=DEFAULT_AL_CONFIG)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(os.environ.get("MODEL_ROOT", DEFAULT_MODELS_ROOT)),
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help="catalog name or model ID to exclude for this facility run (repeatable)",
    )
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least 2")
    result = execute_active_learning_scientific_debate(
        output=args.output,
        al_config=args.al_config,
        rounds=args.rounds,
        catalog=args.catalog,
        models_root=args.models_root,
        excluded_models=set(args.exclude_model),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
