"""Bind one campaign formula to phase exploration and active learning."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase.io import read as ase_read
from ase.io import write as ase_write
from pydantic import BaseModel, Field, model_validator

from matsim_agents.active_learning.calculator import build_ensemble, make_mlip_calculator
from matsim_agents.active_learning.candidates import Candidate
from matsim_agents.active_learning.config import (
    HYDRAGNN_DATASET_HEADS,
    ALConfig,
    resolve_hydragnn_inference_head,
)
from matsim_agents.active_learning.loop import run_active_learning
from matsim_agents.active_learning.uncertainty import select_candidates
from matsim_agents.backends.mlip.relaxation import (
    RelaxStructureInput,
)
from matsim_agents.backends.mlip.relaxation import (
    _run as run_mlip_relaxation,
)
from matsim_agents.campaign.registry import (
    CandidateSelectionPolicy,
    select_dft_refinement_candidates,
    structure_content_hash,
)
from matsim_agents.campaign.state import CampaignState
from matsim_agents.campaign.surrogate_hull import evaluate_surrogate_hull
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.stability import (
    ElementalReferenceEntry,
    RankingMode,
    ReferenceCompletenessPolicy,
    ReferenceEnergySet,
    ReferencePhaseEntry,
    score_stability,
)
from matsim_agents.orchestration.state import RelaxationResult
from matsim_agents.workflows.phase_exploration import (
    PhaseExplorationPolicy,
    PhaseExplorationWorkflowResult,
    run_phase_exploration,
)
from matsim_agents.workflows.relaxation import (
    DFTBackendConfig,
    GeometryControls,
    RelaxationMode,
    ScientificRelaxationConfig,
    ScientificRelaxationResult,
    run_relaxation,
)


class CampaignFormulaExecutionConfig(BaseModel):
    """Reusable numerical configuration for every formula in a campaign."""

    active_learning_config: Path
    phase_policy: PhaseExplorationPolicy
    exploration_kwargs: dict[str, Any] = Field(default_factory=dict)
    compute_nodes: int = Field(1, ge=1)
    execution_mode: Literal["dft", "uma_only"] = "dft"
    retraining: CampaignRetrainingConfig | None = None
    dft_refinement: CampaignDFTRefinementConfig | None = None
    model_override: str | None = None
    validation_configs: list[Path] = Field(default_factory=list)
    validation_python: Path | None = None
    validation_pythons: list[Path] = Field(default_factory=list)
    perturbation_trials: int = Field(0, ge=0)
    perturbation_scale_A: float = Field(0.05, gt=0)
    perturbation_seed: int = 0
    surrogate_reference_structures: Path | None = None
    surrogate_unary_max_steps: int = Field(200, ge=1)
    surrogate_unary_fmax_eV_per_A: float = Field(0.02, gt=0)
    surrogate_unary_maxstep_A: float = Field(0.01, gt=0)
    surrogate_minimum_unique_unary: int = Field(2, ge=1)

    @model_validator(mode="after")
    def _validate_execution_mode(self) -> CampaignFormulaExecutionConfig:
        missing = [str(path) for path in self.validation_configs if not path.is_file()]
        if missing:
            raise ValueError(f"validation config files do not exist: {missing}")
        if self.validation_python is not None and not self.validation_python.is_file():
            raise ValueError(f"validation Python does not exist: {self.validation_python}")
        missing_pythons = [str(path) for path in self.validation_pythons if not path.is_file()]
        if missing_pythons:
            raise ValueError(f"validation Python interpreters do not exist: {missing_pythons}")
        if self.validation_pythons and len(self.validation_pythons) not in {
            1,
            len(self.validation_configs),
        }:
            raise ValueError(
                "validation_pythons must contain one shared interpreter or align "
                "with validation_configs"
            )
        if (
            self.surrogate_reference_structures is not None
            and not self.surrogate_reference_structures.is_file()
        ):
            raise ValueError(
                "surrogate reference manifest does not exist: "
                f"{self.surrogate_reference_structures}"
            )
        if self.execution_mode == "uma_only":
            if self.phase_policy.active_learning or self.phase_policy.retrain_mlip:
                raise ValueError("uma_only execution disables active learning and retraining")
            if self.retraining is not None or self.dft_refinement is not None:
                raise ValueError("uma_only execution cannot configure retraining or DFT refinement")
        return self


def _model_identifier(cfg: ALConfig) -> str:
    if cfg.mlip.backend == "uma" and cfg.mlip.uma is not None:
        return f"uma:{cfg.mlip.uma.model_name}:{cfg.mlip.uma.task_name}"
    if cfg.mlip.backend == "mace" and cfg.mlip.mace is not None:
        return f"mace:{cfg.mlip.mace.family}:{cfg.mlip.mace.model}"
    if cfg.mlip.hydragnn is not None:
        identifier = f"hydragnn:{cfg.mlip.hydragnn.logdir}"
        head_index = resolve_hydragnn_inference_head(cfg.mlip.hydragnn.inference_head)
        if head_index is not None:
            identifier += f":head={HYDRAGNN_DATASET_HEADS[head_index]}"
        return identifier
    return cfg.mlip.backend


def _apply_model_override(cfg: ALConfig, model_path: str) -> None:
    """Point an AL configuration at a promoted backend artifact."""
    if cfg.mlip.backend == "uma" and cfg.mlip.uma is not None:
        cfg.mlip.uma.model_name = model_path
    elif cfg.mlip.backend == "mace" and cfg.mlip.mace is not None:
        cfg.mlip.mace.family = "checkpoint"
        cfg.mlip.mace.model = model_path
    elif cfg.mlip.hydragnn is not None:
        cfg.mlip.hydragnn.logdir = Path(model_path)


def _promoted_model_path(result: PhaseExplorationWorkflowResult) -> str | None:
    evidence = result.active_learning_result or {}
    overrides = evidence.get("exploration_kwargs", {})
    promoted = (
        overrides.get("uma_model_name")
        or overrides.get("mace_model")
        or overrides.get("checkpoint")
        or overrides.get("logdir")
    )
    return str(promoted) if promoted else None


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
                        "matsim_agents.campaign.mlip_validation_worker",
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
    relaxation_runner: Callable[[RelaxStructureInput], RelaxationResult] = run_mlip_relaxation,
) -> dict[str, Any]:
    """Rerelax seeded perturbations of the lowest-energy converged minimum."""
    converged = [item for item in exploration.relaxations if item.converged]
    if not converged or config.perturbation_trials == 0:
        return {"trials": [], "robust_fraction": None}
    reference = min(
        converged,
        key=lambda item: item.final_energy_eV / len(ase_read(item.optimized_structure_path)),
    )
    backend_kwargs = _exploration_kwargs(al_cfg, {})
    trials: list[dict[str, Any]] = []
    for index in range(config.perturbation_trials):
        trial_dir = output_root / f"trial-{index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        perturbed_path = trial_dir / "perturbed.extxyz"
        atoms = ase_read(reference.optimized_structure_path)
        rng = np.random.default_rng(config.perturbation_seed + index)
        atoms.positions += rng.normal(scale=config.perturbation_scale_A, size=atoms.positions.shape)
        ase_write(perturbed_path, atoms)
        try:
            result = relaxation_runner(
                RelaxStructureInput(
                    structure_path=str(perturbed_path),
                    output_dir=str(trial_dir),
                    maxiter=int(config.exploration_kwargs.get("maxiter", 100)),
                    fmax=float(config.exploration_kwargs.get("fmax", 0.02)),
                    **backend_kwargs,
                )
            )
            trials.append(
                {
                    "seed": config.perturbation_seed + index,
                    "converged": result.converged,
                    "energy_per_atom_eV": result.final_energy_eV / len(atoms),
                    "max_force_eV_per_A": result.final_max_force_eV_per_A,
                    "optimized_structure_path": result.optimized_structure_path,
                }
            )
        except Exception as exc:  # noqa: BLE001
            trials.append(
                {"seed": config.perturbation_seed + index, "converged": False, "error": repr(exc)}
            )
    return {
        "reference_structure_path": reference.optimized_structure_path,
        "displacement_scale_A": config.perturbation_scale_A,
        "trials": trials,
        "robust_fraction": sum(bool(trial["converged"]) for trial in trials) / len(trials),
    }


class CampaignRetrainingConfig(BaseModel):
    """Explicit training and promotion controls for campaign MLIPs."""

    train_script: Path
    train_launcher: Path | None = None
    epochs: int = Field(5, ge=1)
    promote_model: bool = False
    promotion_approved: bool = False
    validation_set: Path | None = None
    validation_reference_set: Path | None = None
    promotion_max_energy_mae_eV_per_atom: float = Field(0.1, gt=0)
    promotion_max_force_mae_eV_per_A: float = Field(0.2, gt=0)
    promotion_max_relative_regression: float = Field(0.05, ge=0)
    promotion_min_evaluated_frames: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _validate_training_paths_and_approval(self) -> CampaignRetrainingConfig:
        if not self.train_script.is_file():
            raise ValueError(f"training script does not exist: {self.train_script}")
        if self.train_launcher is not None and not self.train_launcher.is_file():
            raise ValueError(f"training launcher does not exist: {self.train_launcher}")
        if self.promote_model and not self.promotion_approved:
            raise ValueError("model promotion requires explicit approval")
        if self.promote_model and self.validation_set is None:
            raise ValueError("model promotion requires a held-out validation_set")
        return self


class ReferenceStructureSpec(BaseModel):
    """A structure that must be calculated as part of the reference hull."""

    phase_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    formula: str
    structure_path: Path
    relax_cell: bool | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    source: str = "user_supplied"
    provenance: dict[str, str] = Field(default_factory=dict)
    energy_correction_eV_per_atom: float = 0.0


class CampaignDFTRefinementConfig(BaseModel):
    """DFT relaxation and compatible reference generation for hull ranking."""

    method_signature: str
    reference_structures: dict[str, Path] = Field(default_factory=dict)
    reference_relax_cell: dict[str, bool] = Field(default_factory=dict)
    reference_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    reference_phases: list[ReferenceStructureSpec] = Field(default_factory=list)
    reference_completeness_policy: ReferenceCompletenessPolicy = Field(
        default_factory=ReferenceCompletenessPolicy
    )
    reference_energies: ReferenceEnergySet | None = None
    max_candidates: int = Field(1, ge=1)
    relax_cell: bool = True
    max_steps: int = Field(100, ge=1)
    force_tolerance_eV_per_A: float = Field(0.02, gt=0)
    candidate_acquisition: CandidateSelectionPolicy = Field(
        default_factory=CandidateSelectionPolicy
    )


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
        if al_cfg.mlip.ensemble_paths
        else []
    )
    _selected, scores = select_candidates(
        candidates,
        al_cfg.acquisition,
        primary_calculator=primary,
        ensemble_calculators=ensemble or None,
        seed=0,
    )
    return {
        candidate.candidate_id: float(score)
        for candidate, score in zip(candidates, scores, strict=True)
        if np.isfinite(score)
    }

    @model_validator(mode="after")
    def _validate_references(self) -> CampaignDFTRefinementConfig:
        missing = [str(path) for path in self.reference_structures.values() if not path.is_file()]
        if missing:
            raise ValueError(f"reference structures do not exist: {missing}")
        unknown_controls = set(self.reference_relax_cell) - set(self.reference_structures)
        unknown_settings = set(self.reference_settings) - set(self.reference_structures)
        if unknown_controls or unknown_settings:
            raise ValueError(
                "reference controls lack structures for "
                f"{sorted(unknown_controls | unknown_settings)}"
            )
        if (
            self.reference_energies is not None
            and self.reference_energies.method_signature != self.method_signature
        ):
            raise ValueError("reference energies and DFT refinement must share method_signature")
        return self


def _validate_dft_inputs(formula: str, cfg: ALConfig) -> None:
    composition = parse_composition(formula)
    if composition is None:
        raise ValueError(f"Could not parse campaign formula {formula!r}")
    if cfg.dft.backend == "vasp" and cfg.dft.vasp is not None:
        vasp = cfg.dft.vasp
        required_files = [vasp.vasp_bin, vasp.vasp_wrapper, vasp.incar_template]
        if vasp.kpoints_template is not None:
            required_files.append(vasp.kpoints_template)
        missing_files = [str(path) for path in required_files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"VASP input files do not exist: {missing_files}")
        if not vasp.potcar_dir.is_dir():
            raise FileNotFoundError(f"VASP POTCAR directory does not exist: {vasp.potcar_dir}")
        return
    if cfg.dft.backend != "qe" or cfg.dft.qe is None:
        return
    qe = cfg.dft.qe
    if qe.pseudopotentials is None:
        return
    missing_mappings = set(composition.elements) - set(qe.pseudopotentials)
    if missing_mappings:
        raise ValueError(
            f"QE pseudopotential mapping lacks elements {sorted(missing_mappings)} for {formula}"
        )
    missing_files = [
        str(qe.pseudo_dir / qe.pseudopotentials[element])
        for element in composition.elements
        if not (qe.pseudo_dir / qe.pseudopotentials[element]).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"QE pseudopotential files do not exist: {missing_files}")


def _iteration_states(root: Path) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for state_path in sorted(root.glob("iteration_*/state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            raise RuntimeError(f"active-learning iteration did not complete: {state_path}")
        states.append(state)
    if not states:
        raise RuntimeError(f"active-learning run produced no iteration state under {root}")
    return states


def _exploration_kwargs(cfg: ALConfig, overrides: dict[str, Any]) -> dict[str, Any]:
    values = dict(overrides)
    values.setdefault("mlip_backend", cfg.mlip.backend)
    if cfg.mlip.backend == "uma" and cfg.mlip.uma is not None:
        values.setdefault("uma_model_name", cfg.mlip.uma.model_name)
        values.setdefault("uma_task", cfg.mlip.uma.task_name)
        values.setdefault("mlp_device", cfg.mlip.uma.device)
        values.setdefault("precision", cfg.mlip.uma.precision)
    elif cfg.mlip.backend == "mace" and cfg.mlip.mace is not None:
        values.setdefault("mace_family", cfg.mlip.mace.family)
        values.setdefault("mace_model", cfg.mlip.mace.model)
        values.setdefault("mace_dispersion", cfg.mlip.mace.dispersion)
        values.setdefault("mlp_device", cfg.mlip.mace.device)
        values.setdefault("precision", cfg.mlip.mace.precision)
    elif cfg.mlip.hydragnn is not None:
        values.setdefault("logdir", str(cfg.mlip.hydragnn.logdir))
        values.setdefault("checkpoint", cfg.mlip.hydragnn.checkpoint)
        values.setdefault("hydragnn_inference_head", cfg.mlip.hydragnn.inference_head)
        values.setdefault(
            "hydragnn_branch_mlp_checkpoint",
            (
                str(cfg.mlip.hydragnn.hydragnn_branch_mlp_checkpoint)
                if cfg.mlip.hydragnn.hydragnn_branch_mlp_checkpoint is not None
                else None
            ),
        )
        values.setdefault("mlp_device", cfg.mlip.hydragnn.mlp_device)
        values.setdefault("precision", cfg.mlip.hydragnn.precision)
    return values


def _vasp_template_settings(al_cfg: ALConfig) -> dict[str, Any]:
    vasp = al_cfg.dft.vasp
    if vasp is None:
        raise ValueError("VASP refinement requires a dft.vasp configuration")
    try:
        from pymatgen.io.vasp.inputs import Incar
    except ImportError as exc:  # pragma: no cover - hull ranking already requires pymatgen
        raise RuntimeError("VASP refinement requires pymatgen to parse INCAR settings") from exc

    text = vasp.incar_template.read_text(encoding="utf-8").format(natoms=1)
    incar = dict(Incar.from_str(text))
    incar.update(vasp.extra_incar)
    field_map = {
        "ENCUT": "encut_ev",
        "PREC": "prec",
        "EDIFF": "ediff",
        "ISYM": "isym",
        "ISMEAR": "ismear",
        "SIGMA": "sigma",
        "KSPACING": "kspacing",
        "KGAMMA": "kgamma",
        "ISPIN": "ispin",
        "LREAL": "lreal",
        "LWAVE": "lwave",
        "LCHARG": "lcharg",
        "ALGO": "algo",
        "NELM": "nelm",
        "NELMIN": "nelmin",
    }
    settings = {field_map[key]: value for key, value in incar.items() if key in field_map}
    relaxation_keys = {"IBRION", "ISIF", "NSW", "EDIFFG"}
    settings["extra_incar"] = {
        key: value
        for key, value in incar.items()
        if key not in field_map and key not in relaxation_keys
    }
    if vasp.kpoints_template is not None:
        settings["kpoints_template"] = str(vasp.kpoints_template)
        settings.pop("kspacing", None)
    return settings


def _merge_dft_settings(
    settings: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    merged = dict(settings)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _dft_relaxation_config(
    structure_path: str,
    output_root: Path,
    al_cfg: ALConfig,
    refinement: CampaignDFTRefinementConfig,
    settings_overrides: dict[str, Any] | None = None,
) -> ScientificRelaxationConfig:
    if al_cfg.dft.backend == "vasp":
        vasp = al_cfg.dft.vasp
        if vasp is None:
            raise ValueError("campaign VASP refinement requires a dft.vasp block")
        settings = _merge_dft_settings(_vasp_template_settings(al_cfg), settings_overrides)
        return ScientificRelaxationConfig(
            mode=RelaxationMode.DFT,
            structure_path=structure_path,
            output_root=str(output_root),
            geometry=GeometryControls(relax_cell=refinement.relax_cell),
            dft=DFTBackendConfig(
                backend="vasp",
                launcher=[
                    "bash",
                    str(vasp.vasp_wrapper),
                    ".",
                    str(vasp.vasp_bin),
                    str(vasp.nodes_per_job),
                    str(vasp.ranks_per_node),
                    str(vasp.threads_per_rank),
                ],
                potcar_dir=str(vasp.potcar_dir),
                settings=settings,
                timeout_sec=vasp.timeout_sec,
            ),
            dft_approved=True,
            max_steps=refinement.max_steps,
            force_tolerance_eV_per_A=refinement.force_tolerance_eV_per_A,
        )

    qe = al_cfg.dft.qe
    if al_cfg.dft.backend != "qe" or qe is None:
        raise ValueError(f"unsupported campaign DFT refinement backend: {al_cfg.dft.backend}")
    if qe.pw_template is not None:
        raise ValueError("campaign DFT refinement does not support qe.pw_template")
    settings = {
        "ecutwfc_ry": qe.ecutwfc_ry,
        "ecutrho_ry": qe.ecutrho_ry,
        "kpts": qe.kpts,
        "koffset": qe.koffset,
        "occupations": qe.occupations,
        "smearing": qe.smearing,
        "degauss_ry": qe.degauss_ry,
        "pseudopotentials": qe.pseudopotentials,
        "extra_control": qe.extra_control,
        "extra_system": qe.extra_system,
        "extra_electrons": qe.extra_electrons,
    }
    settings = _merge_dft_settings(settings, settings_overrides)
    return ScientificRelaxationConfig(
        mode=RelaxationMode.DFT,
        structure_path=structure_path,
        output_root=str(output_root),
        geometry=GeometryControls(relax_cell=refinement.relax_cell),
        dft=DFTBackendConfig(
            backend="qe",
            launcher=[
                "bash",
                str(qe.pw_wrapper),
                "{work_dir}",
                str(qe.pw_bin),
                "{input}",
                str(qe.nodes_per_job),
                str(qe.ranks_per_node),
                str(qe.threads_per_rank),
            ],
            pseudo_dir=str(qe.pseudo_dir),
            settings={key: value for key, value in settings.items() if value is not None},
            timeout_sec=qe.timeout_sec,
        ),
        dft_approved=True,
        max_steps=refinement.max_steps,
        force_tolerance_eV_per_A=refinement.force_tolerance_eV_per_A,
    )


def _converged_dft_relaxation(
    source: RelaxationResult,
    result: ScientificRelaxationResult,
) -> RelaxationResult:
    if not result.stages:
        raise RuntimeError("DFT relaxation produced no stage result")
    stage = result.stages[-1]
    if not stage.converged or stage.energy_eV is None or stage.optimized_structure_path is None:
        raise RuntimeError(stage.failure_reason or "DFT relaxation did not converge")
    if stage.max_force_eV_per_A is None:
        raise RuntimeError("DFT relaxation did not report a residual force")
    return RelaxationResult(
        structure_path=source.structure_path,
        optimized_structure_path=stage.optimized_structure_path,
        trajectory_path=stage.artifacts.get("stdout", ""),
        log_csv_path=stage.artifacts.get("stdout", ""),
        final_energy_eV=stage.energy_eV,
        final_max_force_eV_per_A=stage.max_force_eV_per_A,
        num_steps=stage.steps,
        converged=True,
        notes=f"converged DFT relaxation; run_id={result.run_id}",
    )


def _generate_reference_energies(
    al_cfg: ALConfig,
    refinement: CampaignDFTRefinementConfig,
    output_root: Path,
    relaxation_runner: Callable[[ScientificRelaxationConfig], ScientificRelaxationResult],
) -> tuple[ReferenceEnergySet, int, float]:
    references = refinement.reference_energies or ReferenceEnergySet(
        identifier=f"campaign-{refinement.method_signature}",
        method_signature=refinement.method_signature,
        backend=al_cfg.dft.backend,
        elemental_energies_eV_per_atom={},
        completeness_policy=refinement.reference_completeness_policy,
    )
    if references.backend is not None and references.backend != al_cfg.dft.backend:
        raise ValueError(
            f"reference energies use {references.backend}, but refinement uses {al_cfg.dft.backend}"
        )
    if references.backend is None and (
        references.elemental_energies_eV_per_atom
        or references.competing_phases
        or references.phase_entries
    ):
        raise ValueError(
            "populated reference energies lack backend provenance; regenerate or migrate them"
        )
    references.backend = al_cfg.dft.backend
    legacy_specs = [
        ReferenceStructureSpec(
            phase_id=formula,
            formula=formula,
            structure_path=path,
            relax_cell=refinement.reference_relax_cell.get(formula),
            settings=refinement.reference_settings.get(formula, {}),
            source="legacy_manifest",
        )
        for formula, path in refinement.reference_structures.items()
    ]
    specs = [*legacy_specs, *refinement.reference_phases]
    phase_ids = [spec.phase_id for spec in specs]
    if len(phase_ids) != len(set(phase_ids)):
        raise ValueError("reference structure phase IDs must be unique")
    existing_phase_ids = {entry.phase_id for entry in references.phase_entries}
    pending: list[tuple[ReferenceStructureSpec, Any]] = []
    for spec in specs:
        composition = parse_composition(spec.formula)
        if composition is None:
            raise ValueError(f"Could not parse reference formula {spec.formula!r}")
        if len(composition.elements) == 1:
            element = next(iter(composition.elements))
            if element in references.elemental_energies_eV_per_atom:
                continue
        elif spec.phase_id in existing_phase_ids or (
            spec.source == "legacy_manifest" and spec.formula in references.competing_phases
        ):
            continue
        pending.append((spec, composition))
    pending.sort(key=lambda item: len(item[1].elements))

    calculations = 0
    elapsed_seconds = 0.0
    for spec, composition in pending:
        started = time.monotonic()
        result = relaxation_runner(
            _dft_relaxation_config(
                str(spec.structure_path),
                output_root / "references" / spec.phase_id,
                al_cfg,
                refinement.model_copy(
                    update={
                        "relax_cell": (
                            spec.relax_cell
                            if spec.relax_cell is not None
                            else refinement.relax_cell
                        )
                    }
                ),
                spec.settings,
            )
        )
        elapsed_seconds += time.monotonic() - started
        calculations += 1
        source = RelaxationResult(
            structure_path=str(spec.structure_path),
            optimized_structure_path=str(spec.structure_path),
            trajectory_path="",
            log_csv_path="",
            final_energy_eV=0.0,
            final_max_force_eV_per_A=0.0,
            num_steps=0,
            converged=True,
        )
        relaxed = _converged_dft_relaxation(source, result)
        from ase.io import read
        from pymatgen.core import Composition as PymatgenComposition

        atoms = read(relaxed.optimized_structure_path)
        energy_per_atom = relaxed.final_energy_eV / len(atoms)
        if len(composition.elements) == 1:
            element = next(iter(composition.elements))
            corrected_energy = energy_per_atom + spec.energy_correction_eV_per_atom
            references.elemental_energies_eV_per_atom[element] = corrected_energy
            references.elemental_entries[element] = ElementalReferenceEntry(
                element=element,
                phase_id=spec.phase_id,
                reference_formula=spec.formula,
                energy_eV_per_atom=corrected_energy,
                method_signature=refinement.method_signature,
                backend=al_cfg.dft.backend,
                structure_path=relaxed.optimized_structure_path,
                structure_hash=structure_content_hash(relaxed.optimized_structure_path),
                total_energy_eV=relaxed.final_energy_eV,
                source=spec.source,
                provenance=spec.provenance,
                corrections={"energy_correction_eV_per_atom": spec.energy_correction_eV_per_atom},
            )
            continue
        cell_composition = PymatgenComposition(atoms.get_chemical_formula())
        declared_composition = PymatgenComposition(spec.formula)
        if cell_composition.reduced_composition != declared_composition.reduced_composition:
            raise ValueError(
                f"relaxed reference composition {cell_composition.formula} does not match "
                f"declared formula {spec.formula}"
            )
        cell_amounts = cell_composition.get_el_amt_dict()
        missing = set(cell_amounts) - set(references.elemental_energies_eV_per_atom)
        if missing:
            raise ValueError(
                f"reference structures lack elemental references for {sorted(missing)}"
            )
        elemental_total = sum(
            amount * references.elemental_energies_eV_per_atom[element]
            for element, amount in cell_amounts.items()
        )
        formation_energy = (relaxed.final_energy_eV - elemental_total) / len(
            atoms
        ) + spec.energy_correction_eV_per_atom
        references.phase_entries.append(
            ReferencePhaseEntry(
                phase_id=spec.phase_id,
                formula=spec.formula,
                formation_energy_eV_per_atom=formation_energy,
                method_signature=refinement.method_signature,
                backend=al_cfg.dft.backend,
                structure_path=relaxed.optimized_structure_path,
                structure_hash=structure_content_hash(relaxed.optimized_structure_path),
                total_energy_eV=relaxed.final_energy_eV,
                energy_per_atom_eV=energy_per_atom,
                source=spec.source,
                provenance=spec.provenance,
                corrections={"energy_correction_eV_per_atom": spec.energy_correction_eV_per_atom},
            )
        )
    references = ReferenceEnergySet.model_validate(references.model_dump())
    refinement.reference_energies = references
    return references, calculations, elapsed_seconds


def run_formula_with_active_learning(
    formula: str,
    output_dir: str,
    *,
    config: CampaignFormulaExecutionConfig,
    al_runner: Callable[[ALConfig], None] = run_active_learning,
    phase_runner: Callable[..., PhaseExplorationWorkflowResult] = run_phase_exploration,
    relaxation_runner: Callable[
        [ScientificRelaxationConfig], ScientificRelaxationResult
    ] = run_relaxation,
    mlip_relaxation_runner: Callable[[RelaxStructureInput], RelaxationResult] = run_mlip_relaxation,
    calculator_factory: Callable[..., Any] = make_mlip_calculator,
) -> PhaseExplorationWorkflowResult:
    """Explore and label one formula using a fresh copy of the AL template."""

    al_cfg = ALConfig.from_yaml(config.active_learning_config)
    if config.execution_mode == "uma_only" and al_cfg.mlip.backend != "uma":
        raise ValueError(
            f"uma_only execution requires mlip.backend='uma', got {al_cfg.mlip.backend!r}"
        )
    if config.model_override:
        _apply_model_override(al_cfg, config.model_override)
    if config.execution_mode == "dft":
        _validate_dft_inputs(formula, al_cfg)
    if config.phase_policy.retrain_mlip:
        if config.retraining is None:
            raise ValueError("phase policy requests retraining but no retraining config is set")
        al_cfg.trainer.enabled = True
        al_cfg.trainer.train_script = config.retraining.train_script
        al_cfg.trainer.train_launcher = config.retraining.train_launcher
        al_cfg.trainer.epochs_per_iter = config.retraining.epochs
        al_cfg.trainer.promote_model = config.retraining.promote_model
        al_cfg.trainer.promotion_approved = config.retraining.promotion_approved
        al_cfg.trainer.validation_set = config.retraining.validation_set
        al_cfg.trainer.validation_reference_set = config.retraining.validation_reference_set
        al_cfg.trainer.promotion_max_energy_mae_eV_per_atom = (
            config.retraining.promotion_max_energy_mae_eV_per_atom
        )
        al_cfg.trainer.promotion_max_force_mae_eV_per_A = (
            config.retraining.promotion_max_force_mae_eV_per_A
        )
        al_cfg.trainer.promotion_max_relative_regression = (
            config.retraining.promotion_max_relative_regression
        )
        al_cfg.trainer.promotion_min_evaluated_frames = (
            config.retraining.promotion_min_evaluated_frames
        )
        if config.phase_policy.reevaluate_after_retraining and not al_cfg.trainer.promote_model:
            raise ValueError("post-retraining reevaluation requires model promotion")
    else:
        al_cfg.trainer.enabled = False
        al_cfg.trainer.promote_model = False

    formula_root = Path(output_dir)
    al_root = formula_root / "active_learning"
    al_cfg.md.seed_source.kind = "compositions"
    al_cfg.md.seed_source.paths = []
    al_cfg.md.seed_source.prompt = None
    al_cfg.md.seed_source.compositions = [formula]
    al_cfg.loop.out_dir = al_root

    def active_learning_runner(
        composition: str,
        phase_output_dir: str,
        retrain: bool,
    ) -> dict[str, Any]:
        if composition != formula:
            raise ValueError(f"phase runner requested {composition!r}, expected {formula!r}")
        if retrain != al_cfg.trainer.enabled:
            raise ValueError("phase retraining policy and AL trainer configuration disagree")
        al_runner(al_cfg)
        states = _iteration_states(al_root)
        total_seconds = sum(
            float(state.get("timings_sec", {}).get("total", 0.0)) for state in states
        )
        promoted = [state for state in states if state.get("model_promoted")]
        candidate_uncertainty = {
            str(candidate_id): float(score)
            for state in states
            for candidate_id, score in state.get("candidate_uncertainty", {}).items()
        }
        result: dict[str, Any] = {
            "n_dft_calculations": sum(
                int(state.get("n_dft_converged", 0)) + int(state.get("n_dft_failed", 0))
                for state in states
            ),
            "n_dft_converged": sum(int(state.get("n_dft_converged", 0)) for state in states),
            "n_active_learning_iterations": len(states),
            "node_hours": total_seconds * config.compute_nodes / 3600.0,
            "model_promoted": bool(promoted),
            "md_candidate_uncertainty": candidate_uncertainty,
            "iteration_states": states,
            "phase_output_dir": phase_output_dir,
        }
        if promoted:
            latest_model = promoted[-1].get("new_logdir")
            if latest_model:
                if al_cfg.mlip.backend == "uma":
                    result["exploration_kwargs"] = {"uma_model_name": latest_model}
                elif al_cfg.mlip.backend == "mace":
                    result["exploration_kwargs"] = {"checkpoint": latest_model}
                else:
                    result["exploration_kwargs"] = {"logdir": latest_model}
        return result

    phase_started = time.monotonic()
    result = phase_runner(
        formula,
        policy=config.phase_policy,
        output_dir=str(formula_root / "phase_exploration"),
        exploration_kwargs=_exploration_kwargs(al_cfg, config.exploration_kwargs),
        active_learning_runner=active_learning_runner,
    )
    phase_seconds = time.monotonic() - phase_started
    exploration = result.after_retraining or result.initial
    effective_al_cfg = al_cfg.model_copy(deep=True)
    promoted_model = _promoted_model_path(result)
    if promoted_model is not None:
        _apply_model_override(effective_al_cfg, promoted_model)
    candidate_uncertainty = _score_relaxed_candidate_uncertainty(exploration, effective_al_cfg)
    validation_configs = [
        effective_al_cfg,
        *(ALConfig.from_yaml(path) for path in config.validation_configs),
    ]
    cross_model = _cross_model_scores(
        exploration,
        validation_configs,
        config_paths=[None, *config.validation_configs],
        validation_python=config.validation_python,
        validation_pythons=(
            [
                None,
                *(
                    config.validation_pythons * len(config.validation_configs)
                    if len(config.validation_pythons) == 1
                    else config.validation_pythons
                ),
            ]
            if config.validation_pythons
            else None
        ),
        reference_manifest=config.surrogate_reference_structures,
        unary_max_steps=config.surrogate_unary_max_steps,
        unary_fmax_eV_per_A=config.surrogate_unary_fmax_eV_per_A,
        unary_maxstep_A=config.surrogate_unary_maxstep_A,
        minimum_unique_unary=config.surrogate_minimum_unique_unary,
        calculator_factory=calculator_factory,
    )
    robustness = _perturbation_robustness(
        exploration,
        effective_al_cfg,
        config,
        formula_root / "robustness",
        relaxation_runner=mlip_relaxation_runner,
    )
    if config.execution_mode == "uma_only":
        candidate_by_path = {
            candidate.structure_path: candidate for candidate in exploration.phase_candidates
        }
        labels = []
        for relaxation in exploration.relaxations:
            candidate = candidate_by_path.get(relaxation.structure_path)
            atom_count = candidate.num_atoms if candidate is not None else None
            labels.append(
                {
                    "candidate_id": candidate.candidate_id if candidate is not None else None,
                    "structure_path": relaxation.structure_path,
                    "optimized_structure_path": relaxation.optimized_structure_path,
                    "converged": relaxation.converged,
                    "final_energy_eV": relaxation.final_energy_eV,
                    "energy_per_atom_eV": (
                        relaxation.final_energy_eV / atom_count if atom_count else None
                    ),
                    "residual_force_eV_per_A": relaxation.final_max_force_eV_per_A,
                    "uncertainty": (
                        candidate_uncertainty.get(candidate.candidate_id)
                        if candidate is not None and candidate.candidate_id is not None
                        else None
                    ),
                }
            )
        model_identifier = None
        if al_cfg.mlip.uma is not None:
            model_identifier = al_cfg.mlip.uma.model_name
        result.active_learning_result = {
            "execution_mode": "uma_only",
            "evidence_level": "mlip_prediction",
            "backend": "uma",
            "model_identifier": model_identifier,
            "n_dft_calculations": 0,
            "n_active_learning_iterations": 0,
            "node_hours": phase_seconds * config.compute_nodes / 3600.0,
            "model_promoted": False,
            "candidate_uncertainty": candidate_uncertainty,
            "mlip_labels": labels,
            "cross_model_validation": cross_model,
            "perturbation_robustness": robustness,
        }
        return result
    if result.active_learning_result is not None:
        result.active_learning_result["candidate_uncertainty"] = candidate_uncertainty
    refinement = config.dft_refinement
    if refinement is None:
        return result
    if not config.phase_policy.dft_approved:
        raise PermissionError("campaign DFT refinement requires explicit DFT approval")
    candidates = [relaxation for relaxation in exploration.relaxations if relaxation.converged]
    if refinement.candidate_acquisition.enabled:
        selected, candidate_scores = select_dft_refinement_candidates(
            exploration,
            max_candidates=refinement.max_candidates,
            policy=refinement.candidate_acquisition,
            uncertainty_by_candidate=(result.active_learning_result or {}).get(
                "candidate_uncertainty", {}
            ),
        )
    else:
        candidates.sort(key=lambda relaxation: relaxation.final_energy_eV)
        selected = candidates[: refinement.max_candidates]
        candidate_scores = {}
    if not selected:
        raise RuntimeError("DFT refinement requires at least one converged MLIP relaxation")

    references, reference_calculations, reference_seconds = _generate_reference_energies(
        al_cfg,
        refinement,
        formula_root / "dft_refinement",
        relaxation_runner,
    )
    composition = parse_composition(formula)
    assert composition is not None
    missing = set(composition.elements) - set(references.elemental_energies_eV_per_atom)
    if missing:
        raise ValueError(f"reference-energy set lacks elemental references for {sorted(missing)}")

    refined: list[RelaxationResult] = []
    refinement_seconds = 0.0
    for index, candidate in enumerate(selected):
        started = time.monotonic()
        relaxation_result = relaxation_runner(
            _dft_relaxation_config(
                candidate.optimized_structure_path,
                formula_root / "dft_refinement" / "candidates" / f"candidate-{index:03d}",
                al_cfg,
                refinement,
            )
        )
        refinement_seconds += time.monotonic() - started
        refined.append(_converged_dft_relaxation(candidate, relaxation_result))
    exploration.stability = score_stability(
        formula,
        refined,
        force_tol_eV_per_A=refinement.force_tolerance_eV_per_A,
        candidates=exploration.phase_candidates,
        ranking_mode=RankingMode.CONVEX_HULL,
        reference_energies=references,
        method_signature=refinement.method_signature,
    )
    evidence = result.active_learning_result or {}
    evidence["n_dft_calculations"] = (
        int(evidence.get("n_dft_calculations", 0)) + len(refined) + reference_calculations
    )
    evidence["node_hours"] = (
        float(evidence.get("node_hours", 0.0))
        + (refinement_seconds + reference_seconds) * config.compute_nodes / 3600.0
    )
    evidence["dft_refinement"] = {
        "backend": al_cfg.dft.backend,
        "method_signature": refinement.method_signature,
        "reference_set_id": references.identifier,
        "reference_energy_set": references.model_dump(mode="json"),
        "reference_calculations": reference_calculations,
        "candidate_calculations": len(refined),
        "ranking_mode": RankingMode.CONVEX_HULL,
        "candidate_acquisition": {
            key: value.model_dump(mode="json") for key, value in candidate_scores.items()
        },
    }
    result.active_learning_result = evidence
    return result


def make_formula_runner(
    config: CampaignFormulaExecutionConfig,
) -> Callable[[str, str], PhaseExplorationWorkflowResult]:
    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        result = run_formula_with_active_learning(formula, output_dir, config=config)
        if result.model_promoted and result.active_learning_result:
            promoted = _promoted_model_path(result)
            if promoted:
                config.model_override = str(promoted)
        return result

    return runner


def latest_promoted_model(campaign: CampaignState) -> str | None:
    """Return the newest promoted checkpoint recorded in durable campaign evidence."""

    records = sorted(
        campaign.formula_runs.values(),
        key=lambda record: (record.iteration, record.attempts),
        reverse=True,
    )
    for record in records:
        if not record.model_promoted:
            continue
        for state in reversed(record.evidence.get("iteration_states", [])):
            checkpoint = state.get("new_logdir")
            if state.get("model_promoted") and checkpoint:
                return str(checkpoint)
    return None
