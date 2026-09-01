"""Executable, deterministic scientific portability checks.

These checks exercise the production workflow contracts without requiring a
licensed DFT code or downloading a model. Optional live LLM inference is
enabled explicitly; numerical facility qualification remains a separate run
with the installed production backends.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def relaxation_contract(structure: Path, output: Path) -> dict[str, Any]:
    """Exercise the complete relaxation workflow with a deterministic backend."""

    from matsim_agents.orchestration.state import RelaxationResult
    from matsim_agents.workflows.relaxation import ScientificRelaxationConfig, run_relaxation

    def fake_runner(args):
        destination = Path(args.output_dir) / "optimized.vasp"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.structure_path, destination)
        return RelaxationResult(
            structure_path=args.structure_path,
            optimized_structure_path=str(destination),
            trajectory_path=str(Path(args.output_dir) / "trajectory.traj"),
            log_csv_path=str(Path(args.output_dir) / "relaxation.csv"),
            final_energy_eV=-5.40,
            final_max_force_eV_per_A=0.01,
            num_steps=3,
            converged=True,
        )

    cfg = ScientificRelaxationConfig(
        mode="mlip",
        structure_path=str(structure),
        output_root=str(output / "runs"),
        mlip={
            "mlip_backend": "uma",
            "uma_model_name": "portability-deterministic",
        },
    )
    result = run_relaxation(cfg, mlip_runner=fake_runner)
    return {
        "status": result.status,
        "run_directory": result.run_directory,
        "converged": bool(result.stages and result.stages[-1].converged),
        "stage_count": len(result.stages),
    }


def active_learning_contract(output: Path) -> dict[str, Any]:
    """Validate a fixed 4-to-2 acquisition and immutable dataset manifest."""

    import numpy as np
    from ase import Atoms

    from matsim_agents.active_learning.dataset_governance import (
        validate_labelled_frames,
        write_dataset_manifest,
    )
    from matsim_agents.active_learning.trainer import LabelledFrame, append_frames_to_extxyz

    candidates = []
    for index in range(4):
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1.30 + index * 0.01, 1.30, 1.30]])
        atoms.set_cell([5.5, 5.5, 5.5])
        atoms.set_pbc(True)
        candidates.append(
            LabelledFrame(
                atoms=atoms,
                energy_eV=-10.0 + index * 0.1,
                forces_eV_per_A=np.zeros((2, 3)),
                stress_eV_per_A3=None,
                source_work_dir=f"candidate-{index}",
                iteration=0,
                backend="qe",
            )
        )
    # Fixed scores make candidate identity portable across accelerator stacks.
    scores = [0.1, 0.9, 0.4, 0.8]
    selected_ids = sorted(range(4), key=lambda index: (-scores[index], index))[:2]
    accepted, validation = validate_labelled_frames(candidates[index] for index in selected_ids)
    dataset = output / "datasets" / "dataset.extxyz"
    appended = append_frames_to_extxyz(accepted, dataset)
    manifest = write_dataset_manifest(
        dataset,
        dft_backend="qe",
        energy_reference="portability-deterministic",
        validation=validation,
    )
    return {
        "status": "complete",
        "candidate_count": len(candidates),
        "selected_candidate_ids": selected_ids,
        "accepted": validation.accepted,
        "appended": appended,
        "retrain": False,
        "promote_model": False,
        "dataset_manifest": str(manifest),
    }


def _phase_result(composition: str, output_dir: str):
    from matsim_agents.discovery.composition import parse_composition
    from matsim_agents.discovery.seeds import PhaseCandidate
    from matsim_agents.discovery.wrapper import CompositionExplorationResult
    from matsim_agents.workflows.phase_exploration import PhaseExplorationWorkflowResult

    parsed = parse_composition(composition)
    if parsed is None:
        raise ValueError(f"invalid benchmark composition: {composition}")
    candidate_path = Path(output_dir) / "Si-portability.vasp"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text("deterministic benchmark candidate\n", encoding="utf-8")
    initial = CompositionExplorationResult(
        composition=parsed,
        phase_candidates=[
            PhaseCandidate(
                formula=parsed.formula,
                structure_path=str(candidate_path),
                prototype_id="portability-diamond",
                num_atoms=8,
            )
        ],
    )
    return PhaseExplorationWorkflowResult(composition=parsed.formula, initial=initial)


def investigation_contract(output: Path, *, live_llm: bool = False) -> dict[str, Any]:
    """Run hypothesis, critique/revision, and phase-dispatch persistence."""

    from matsim_agents.workflows.investigation import (
        InvestigationConfig,
        PropertyTask,
        ScientificHypothesis,
        run_investigation,
    )

    transcript: list[dict[str, str]] = []

    def hypothesis_builder(objective, previous):
        if live_llm:
            from langchain_core.messages import HumanMessage

            from matsim_agents.backends.llm.provider import get_chat_model

            model = get_chat_model(
                provider=os.environ.get("MATSIM_LLM_PROVIDER", "vllm"),
                model=os.environ.get("MATSIM_LLM_MODEL"),
                base_url=os.environ.get("MATSIM_VLLM_BASE_URL"),
                temperature=0.0,
            )
            proposal = str(model.invoke([HumanMessage(content=objective)]).content)
            critique_prompt = f"Critique this materials hypothesis concisely:\n{proposal}"
            critique = str(model.invoke([HumanMessage(content=critique_prompt)]).content)
        else:
            proposal = "Diamond Si should remain the lowest-energy candidate in this fixed set."
            critique = (
                "Compare identical inputs and energy references before accepting the ranking."
            )
        transcript.extend(
            [
                {"role": "proposer", "content": proposal},
                {"role": "critic", "content": critique},
                {
                    "role": "revision",
                    "content": "Test Si with a fixed phase exploration and retain provenance.",
                },
            ]
        )
        return ScientificHypothesis(
            objective=objective,
            hypothesis=proposal,
            proposed_compositions=["Si"],
            scientific_rationale=critique,
            property_tasks=[
                PropertyTask(
                    property_name="relative phase energy",
                    method="fixed portability exploration",
                    success_criterion="one persisted ranked candidate",
                    required_fidelity="mlip_relaxation",
                    rationale="exercise the complete investigation handoff",
                )
            ],
        )

    cfg = InvestigationConfig(
        objective="Identify a stable elemental silicon phase.",
        output_root=str(output / "investigations"),
    )
    result = run_investigation(
        cfg,
        hypothesis_builder=hypothesis_builder,
        phase_runner=lambda composition, _policy, directory: _phase_result(composition, directory),
    )
    transcript_path = Path(result.run_directory) / "llm_discussion.json"
    transcript_path.write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")
    return {
        "status": result.status,
        "run_directory": result.run_directory,
        "discussion_turns": len(transcript),
        "composition_count": len(result.explorations),
        "live_llm": live_llm,
        "transcript": str(transcript_path),
    }


def execute_contract_suite(
    suite: str, *, structure: Path, output: Path, live_llm: bool = False
) -> dict[str, Any]:
    if suite == "relaxation":
        return relaxation_contract(structure, output)
    if suite == "active-learning":
        return active_learning_contract(output)
    if suite in {"phase-exploration", "llm-discussion"}:
        return investigation_contract(output, live_llm=live_llm)
    raise ValueError(f"unsupported contract suite: {suite}")


__all__ = [
    "active_learning_contract",
    "execute_contract_suite",
    "investigation_contract",
    "relaxation_contract",
]
