"""Bind one campaign formula to phase exploration and active learning."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from matsim_agents.active_learning.config import ALConfig
from matsim_agents.active_learning.loop import run_active_learning
from matsim_agents.campaign.state import CampaignState
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.stability import RankingMode, ReferenceEnergySet, score_stability
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
    retraining: CampaignRetrainingConfig | None = None
    dft_refinement: CampaignDFTRefinementConfig | None = None
    model_override: str | None = None


class CampaignRetrainingConfig(BaseModel):
    """Explicit training and promotion controls for campaign MLIPs."""

    train_script: Path
    train_launcher: Path | None = None
    epochs: int = Field(5, ge=1)
    promote_model: bool = False
    promotion_approved: bool = False

    @model_validator(mode="after")
    def _validate_training_paths_and_approval(self) -> CampaignRetrainingConfig:
        if not self.train_script.is_file():
            raise ValueError(f"training script does not exist: {self.train_script}")
        if self.train_launcher is not None and not self.train_launcher.is_file():
            raise ValueError(f"training launcher does not exist: {self.train_launcher}")
        if self.promote_model and not self.promotion_approved:
            raise ValueError("model promotion requires explicit approval")
        return self


class CampaignDFTRefinementConfig(BaseModel):
    """DFT relaxation and compatible reference generation for hull ranking."""

    method_signature: str
    reference_structures: dict[str, Path] = Field(default_factory=dict)
    reference_relax_cell: dict[str, bool] = Field(default_factory=dict)
    reference_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    reference_energies: ReferenceEnergySet | None = None
    max_candidates: int = Field(1, ge=1)
    relax_cell: bool = True
    max_steps: int = Field(100, ge=1)
    force_tolerance_eV_per_A: float = Field(0.02, gt=0)

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
        values.setdefault("checkpoint", cfg.mlip.mace.model)
        values.setdefault("mlp_device", cfg.mlip.mace.device)
        values.setdefault("precision", cfg.mlip.mace.precision)
    elif cfg.mlip.hydragnn is not None:
        values.setdefault("logdir", str(cfg.mlip.hydragnn.logdir))
        values.setdefault("checkpoint", cfg.mlip.hydragnn.checkpoint)
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
    settings = {
        field_map[key]: value for key, value in incar.items() if key in field_map
    }
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
            launcher=str(qe.pw_wrapper),
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
    )
    if references.backend is not None and references.backend != al_cfg.dft.backend:
        raise ValueError(
            f"reference energies use {references.backend}, but refinement uses "
            f"{al_cfg.dft.backend}"
        )
    if references.backend is None and (
        references.elemental_energies_eV_per_atom or references.competing_phases
    ):
        raise ValueError(
            "populated reference energies lack backend provenance; regenerate or migrate them"
        )
    references.backend = al_cfg.dft.backend
    pending: list[tuple[str, Any, Path]] = []
    for formula, path in refinement.reference_structures.items():
        composition = parse_composition(formula)
        if composition is None:
            raise ValueError(f"Could not parse reference formula {formula!r}")
        if len(composition.elements) == 1:
            element = next(iter(composition.elements))
            if element in references.elemental_energies_eV_per_atom:
                continue
        elif formula in references.competing_phases:
            continue
        pending.append((formula, composition, path))
    pending.sort(key=lambda item: len(item[1].elements))

    calculations = 0
    elapsed_seconds = 0.0
    for formula, composition, path in pending:
        started = time.monotonic()
        result = relaxation_runner(
            _dft_relaxation_config(
                str(path),
                output_root / "references" / formula,
                al_cfg,
                refinement.model_copy(
                    update={
                        "relax_cell": refinement.reference_relax_cell.get(
                            formula, refinement.relax_cell
                        )
                    }
                ),
                refinement.reference_settings.get(formula),
            )
        )
        elapsed_seconds += time.monotonic() - started
        calculations += 1
        source = RelaxationResult(
            structure_path=str(path),
            optimized_structure_path=str(path),
            trajectory_path="",
            log_csv_path="",
            final_energy_eV=0.0,
            final_max_force_eV_per_A=0.0,
            num_steps=0,
            converged=True,
        )
        relaxed = _converged_dft_relaxation(source, result)
        from ase.io import read

        atoms = read(relaxed.optimized_structure_path)
        energy_per_atom = relaxed.final_energy_eV / len(atoms)
        if len(composition.elements) == 1:
            element = next(iter(composition.elements))
            references.elemental_energies_eV_per_atom[element] = energy_per_atom
            continue
        missing = set(composition.elements) - set(references.elemental_energies_eV_per_atom)
        if missing:
            raise ValueError(
                f"reference structures lack elemental references for {sorted(missing)}"
            )
        elemental_total = sum(
            amount * references.elemental_energies_eV_per_atom[element]
            for element, amount in composition.elements.items()
        )
        references.competing_phases[formula] = (
            relaxed.final_energy_eV - elemental_total
        ) / composition.total_atoms
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
) -> PhaseExplorationWorkflowResult:
    """Explore and label one formula using a fresh copy of the AL template."""

    al_cfg = ALConfig.from_yaml(config.active_learning_config)
    if config.model_override:
        if al_cfg.mlip.backend == "uma" and al_cfg.mlip.uma is not None:
            al_cfg.mlip.uma.model_name = config.model_override
        elif al_cfg.mlip.backend == "mace" and al_cfg.mlip.mace is not None:
            al_cfg.mlip.mace.family = "checkpoint"
            al_cfg.mlip.mace.model = config.model_override
        elif al_cfg.mlip.hydragnn is not None:
            al_cfg.mlip.hydragnn.logdir = Path(config.model_override)
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
        result: dict[str, Any] = {
            "n_dft_calculations": sum(
                int(state.get("n_dft_converged", 0)) + int(state.get("n_dft_failed", 0))
                for state in states
            ),
            "n_dft_converged": sum(int(state.get("n_dft_converged", 0)) for state in states),
            "n_active_learning_iterations": len(states),
            "node_hours": total_seconds * config.compute_nodes / 3600.0,
            "model_promoted": bool(promoted),
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

    result = phase_runner(
        formula,
        policy=config.phase_policy,
        output_dir=str(formula_root / "phase_exploration"),
        exploration_kwargs=_exploration_kwargs(al_cfg, config.exploration_kwargs),
        active_learning_runner=active_learning_runner,
    )
    refinement = config.dft_refinement
    if refinement is None:
        return result
    if not config.phase_policy.dft_approved:
        raise PermissionError("campaign DFT refinement requires explicit DFT approval")
    exploration = result.after_retraining or result.initial
    candidates = [relaxation for relaxation in exploration.relaxations if relaxation.converged]
    candidates.sort(key=lambda relaxation: relaxation.final_energy_eV)
    selected = candidates[: refinement.max_candidates]
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
    evidence["n_dft_calculations"] = int(evidence.get("n_dft_calculations", 0)) + len(
        refined
    ) + reference_calculations
    evidence["node_hours"] = float(evidence.get("node_hours", 0.0)) + (
        refinement_seconds + reference_seconds
    ) * config.compute_nodes / 3600.0
    evidence["dft_refinement"] = {
        "backend": al_cfg.dft.backend,
        "method_signature": refinement.method_signature,
        "reference_set_id": references.identifier,
        "reference_calculations": reference_calculations,
        "candidate_calculations": len(refined),
        "ranking_mode": RankingMode.CONVEX_HULL,
    }
    result.active_learning_result = evidence
    return result


def make_formula_runner(
    config: CampaignFormulaExecutionConfig,
) -> Callable[[str, str], PhaseExplorationWorkflowResult]:
    def runner(formula: str, output_dir: str) -> PhaseExplorationWorkflowResult:
        result = run_formula_with_active_learning(formula, output_dir, config=config)
        if result.model_promoted and result.active_learning_result:
            overrides = result.active_learning_result.get("exploration_kwargs", {})
            promoted = (
                overrides.get("uma_model_name")
                or overrides.get("checkpoint")
                or overrides.get("logdir")
            )
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
