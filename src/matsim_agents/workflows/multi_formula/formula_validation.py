"""Driver-side checks of relaxed candidates: uncertainty, cross-model scores, robustness."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read as ase_read
from ase.io import write as ase_write

from matsim_agents.backends.mlip.calculator import build_ensemble, make_mlip_calculator
from matsim_agents.backends.mlip.relaxation import (
    RelaxStructureInput,
    relax_with_mlip,
)
from matsim_agents.core.relaxation import RelaxationResult
from matsim_agents.core.seeding import numpy_rng, stream_seed
from matsim_agents.science.thermodynamics.surrogate_hull import evaluate_surrogate_hull
from matsim_agents.workflows.active_learning.config import ALConfig
from matsim_agents.workflows.active_learning.md_sampler import Candidate
from matsim_agents.workflows.active_learning.uncertainty import select_candidates
from matsim_agents.workflows.multi_formula.formula_al import _model_identifier
from matsim_agents.workflows.multi_formula.formula_config import CampaignFormulaExecutionConfig


def _cross_model_scores(
    exploration: Any,
    configs: list[ALConfig],
    *,
    config_paths: list[Path | None] | None = None,
    validation_python: Path | None = None,
    validation_pythons: list[Path | None] | None = None,
    reference_manifest: Path | None = None,
    unary_max_steps: int = 200,
    unary_fmax_eV_per_A: float = 0.02,
    unary_maxstep_A: float = 0.01,
    minimum_unique_unary: int = 2,
    calculator_factory: Callable[..., Any] = make_mlip_calculator,
) -> dict[str, Any]:
    """Score converged structures and compare within-model phase orderings."""
    structures = [item for item in exploration.relaxations if item.converged]
    if not structures or not configs:
        return {"models": {}, "ranking_disagreement": False, "rankings": {}}
    references: list[tuple[str, str, Path]] = []
    if reference_manifest is not None:
        raw_manifest = json.loads(reference_manifest.read_text(encoding="utf-8"))
        raw_phases = raw_manifest.get("phases", raw_manifest)
        for phase_id, spec in raw_phases.items():
            if isinstance(spec, str):
                references.append((str(phase_id), str(phase_id), Path(spec)))
            else:
                references.append(
                    (
                        str(spec.get("phase_id", phase_id)),
                        str(spec.get("formula", phase_id)),
                        Path(spec["path"]),
                    )
                )
    models: dict[str, Any] = {}
    rankings: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    paths = config_paths or [None] * len(configs)
    if len(paths) != len(configs):
        raise ValueError("config_paths must align with configs")
    pythons = validation_pythons or [None] * len(configs)
    if len(pythons) != len(configs):
        raise ValueError("validation_pythons must align with configs")
    for index, (cfg, config_path, config_python) in enumerate(
        zip(configs, paths, pythons, strict=True)
    ):
        base_identifier = _model_identifier(cfg)
        identifier = base_identifier
        suffix = 2
        while identifier in models or identifier in failures:
            identifier = f"{base_identifier}#{suffix}"
            suffix += 1
        try:
            external_python = config_python or validation_python
            if index > 0 and external_python is not None:
                if config_path is None:
                    raise ValueError("external validation requires a config path")
                completed = subprocess.run(
                    [
                        str(external_python),
                        "-m",
                        "matsim_agents.workflows.multi_formula.mlip_validation_worker",
                    ],
                    input=json.dumps(
                        {
                            "config": str(config_path),
                            "formula": exploration.composition.formula,
                            "structure_paths": [
                                item.optimized_structure_path for item in structures
                            ],
                            "reference_manifest": (
                                str(reference_manifest) if reference_manifest else None
                            ),
                            "model_identifier": identifier,
                            "unary_max_steps": unary_max_steps,
                            "unary_fmax_eV_per_A": unary_fmax_eV_per_A,
                            "unary_maxstep_A": unary_maxstep_A,
                            "minimum_unique_unary": minimum_unique_unary,
                        }
                    ),
                    text=True,
                    capture_output=True,
                    check=True,
                )
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
                models[identifier] = payload["labels"]
                rankings[identifier] = payload["ranking"]
                if payload.get("surrogate_hull") is not None:
                    models[identifier + ":surrogate_hull"] = payload["surrogate_hull"]
                continue
            calculator = calculator_factory(cfg.mlip)
            labels: list[dict[str, Any]] = []
            for relaxation in structures:
                atoms = ase_read(relaxation.optimized_structure_path)
                atoms.calc = calculator
                forces = np.asarray(atoms.get_forces(), dtype=float)
                labels.append(
                    {
                        "structure_path": relaxation.optimized_structure_path,
                        "energy_eV": float(atoms.get_potential_energy()),
                        "energy_per_atom_eV": float(atoms.get_potential_energy()) / len(atoms),
                        "max_force_eV_per_A": float(np.linalg.norm(forces, axis=1).max()),
                    }
                )
            models[identifier] = labels
            rankings[identifier] = [
                label["structure_path"]
                for label in sorted(labels, key=lambda item: item["energy_per_atom_eV"])
            ]
            if references:
                coverage = evaluate_surrogate_hull(
                    reference_manifest,
                    exploration.composition.formula,
                    labels,
                    calculator,
                    model_identifier=identifier,
                    unary_max_steps=unary_max_steps,
                    unary_fmax_eV_per_A=unary_fmax_eV_per_A,
                    unary_maxstep_A=unary_maxstep_A,
                    minimum_unique_unary=minimum_unique_unary,
                )
                models[identifier + ":surrogate_hull"] = coverage
                models[identifier] = labels
        except Exception as exc:  # noqa: BLE001
            failures[identifier] = repr(exc)
    complete_rankings = list(rankings.values())
    return {
        "models": models,
        "failures": failures,
        "rankings": rankings,
        "ranking_disagreement": bool(
            len(complete_rankings) > 1
            and any(ranking != complete_rankings[0] for ranking in complete_rankings[1:])
        ),
    }


def _perturbation_robustness(
    exploration: Any,
    al_cfg: ALConfig,
    config: CampaignFormulaExecutionConfig,
    output_root: Path,
    *,
    relaxation_runner: Callable[[RelaxStructureInput], RelaxationResult] = relax_with_mlip,
) -> dict[str, Any]:
    """Rerelax seeded perturbations of the lowest-energy converged minimum."""
    converged = [item for item in exploration.relaxations if item.converged]
    if not converged or config.perturbation_trials == 0:
        return {"trials": [], "robust_fraction": None}
    reference = min(
        converged,
        key=lambda item: item.final_energy_eV / len(ase_read(item.optimized_structure_path)),
    )
    trials: list[dict[str, Any]] = []
    for index in range(config.perturbation_trials):
        trial_dir = output_root / f"trial-{index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        perturbed_path = trial_dir / "perturbed.extxyz"
        atoms = ase_read(reference.optimized_structure_path)
        trial_seed = stream_seed(al_cfg.random_seed, "perturbation", index)
        rng = numpy_rng(trial_seed)
        atoms.positions += rng.normal(scale=config.perturbation_scale_A, size=atoms.positions.shape)
        ase_write(perturbed_path, atoms)
        try:
            result = relaxation_runner(
                RelaxStructureInput(
                    structure_path=str(perturbed_path),
                    output_dir=str(trial_dir),
                    maxiter=int(config.exploration_kwargs.get("maxiter", 100)),
                    fmax=float(config.exploration_kwargs.get("fmax", 0.02)),
                    mlip=al_cfg.mlip,
                )
            )
            trials.append(
                {
                    "seed": trial_seed,
                    "converged": result.converged,
                    "energy_per_atom_eV": result.final_energy_eV / len(atoms),
                    "max_force_eV_per_A": result.final_max_force_eV_per_A,
                    "optimized_structure_path": result.optimized_structure_path,
                }
            )
        except Exception as exc:  # noqa: BLE001
            trials.append({"seed": trial_seed, "converged": False, "error": repr(exc)})
    return {
        "reference_structure_path": reference.optimized_structure_path,
        "displacement_scale_A": config.perturbation_scale_A,
        "trials": trials,
        "robust_fraction": sum(bool(trial["converged"]) for trial in trials) / len(trials),
    }


def _score_relaxed_candidate_uncertainty(
    exploration: Any,
    al_cfg: ALConfig,
) -> dict[str, float]:
    """Score the relaxed phase candidates in the registry's candidate-ID space."""
    if al_cfg.acquisition.strategy == "random":
        return {}
    relaxation_by_path = {item.structure_path: item for item in exploration.relaxations}
    candidates: list[Candidate] = []
    for index, phase in enumerate(exploration.phase_candidates):
        relaxation = relaxation_by_path.get(phase.structure_path)
        if relaxation is None or not relaxation.converged:
            continue
        candidate_id = phase.candidate_id or f"{phase.formula}-{phase.source[0].upper()}{index:04d}"
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                atoms=ase_read(relaxation.optimized_structure_path),
                seed_path=phase.structure_path,
                md_step=0,
            )
        )
    if not candidates:
        return {}
    enable_dropout = al_cfg.acquisition.strategy in {"mc_dropout", "ensemble_then_dropout"}
    primary = make_mlip_calculator(al_cfg.mlip, enable_mc_dropout=enable_dropout)
    ensemble = (
        build_ensemble(al_cfg.mlip, enable_mc_dropout=enable_dropout)
        if al_cfg.mlip.ensemble_models
        else []
    )
    _selected, scores = select_candidates(
        candidates,
        al_cfg.acquisition,
        primary_calculator=primary,
        ensemble_calculators=ensemble or None,
        seed=stream_seed(al_cfg.random_seed, "candidate-uncertainty"),
    )
    return {
        candidate.candidate_id: float(score)
        for candidate, score in zip(candidates, scores, strict=True)
        if np.isfinite(score)
    }
