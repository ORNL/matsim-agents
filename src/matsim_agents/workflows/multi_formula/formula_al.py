"""Binding a formula to the active-learning template and reading its iteration states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matsim_agents.core.composition import parse_composition
from matsim_agents.core.hashing import sha256_file
from matsim_agents.workflows.active_learning.config import ALConfig
from matsim_agents.workflows.exploration.phase import (
    PhaseExplorationWorkflowResult,
)
from matsim_agents.workflows.multi_formula.formula_config import CampaignFormulaExecutionConfig
from matsim_agents.workflows.multi_formula.formula_references import (
    _pending_reference_calculation_count,
)


def _model_identifier(cfg: ALConfig) -> str:
    return cfg.mlip.model_identifier()


def _apply_model_override(cfg: ALConfig, model_path: str) -> None:
    """Point an AL configuration at a promoted backend artifact."""
    cfg.mlip = cfg.mlip.with_checkpoint(model_path)


def _promoted_model_path(result: PhaseExplorationWorkflowResult) -> str | None:
    promoted = (result.active_learning_result or {}).get("promoted_model_path")
    return str(promoted) if promoted else None


def _validate_dft_inputs(formula: str, cfg: ALConfig) -> None:
    composition = parse_composition(formula)
    if composition is None:
        raise ValueError(f"Could not parse campaign formula {formula!r}")
    if cfg.dft.backend == "vasp" and cfg.dft.vasp is not None:
        vasp = cfg.dft.vasp
        required_files = [vasp.vasp_bin, vasp.vasp_wrapper, vasp.incar_template]
        if vasp.vasp_env_setup is not None:
            required_files.append(vasp.vasp_env_setup)
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


def _iteration_states(root: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for state_path in sorted(root.glob("iteration_*/state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            raise RuntimeError(f"active-learning iteration did not complete: {state_path}")
        states.append(state)
    if not states and not allow_empty:
        raise RuntimeError(f"active-learning run produced no iteration state under {root}")
    return states


def _durable_iteration_states(root: Path) -> list[dict[str, Any]]:
    """Read all persisted iteration states, including a failed final iteration."""
    return [
        json.loads(state_path.read_text(encoding="utf-8"))
        for state_path in sorted(root.glob("iteration_*/state.json"))
    ]


def _exploration_kwargs(
    cfg: ALConfig, overrides: dict[str, Any], *, generation_seed: int
) -> dict[str, Any]:
    """Exploration arguments seeded with the attempt's *generation_seed*.

    A configured ``generation_action`` carries the seed itself, because
    ``explore_composition`` rejects an action combined with a separate seed.
    """
    kwargs = {"mlip": cfg.mlip, **overrides}
    action = kwargs.get("generation_action")
    if action is None:
        kwargs["random_seed"] = generation_seed
    else:
        kwargs["generation_action"] = action.model_copy(update={"random_seed": generation_seed})
        kwargs["random_seed"] = None
    return kwargs


def _prepare_al_template(
    formula: str, config: CampaignFormulaExecutionConfig, al_root: Path, *, random_seed: int
) -> ALConfig:
    """Load the AL template and apply this formula's model, training, budget, and seed.

    *random_seed* (the formula attempt's seed) replaces the template's seed, so every
    random stream of the attempt, including the AL loop's, derives from it.
    """

    al_cfg = ALConfig.from_yaml(config.active_learning_config, mlip=config.mlip)
    al_cfg.random_seed = random_seed
    if config.model_override:
        _apply_model_override(al_cfg, config.model_override)
    if config.execution_mode == "dft":
        _validate_dft_inputs(formula, al_cfg)
    if config.phase_policy.retrain_mlip:
        if config.retraining is None:
            raise ValueError("phase policy requests retraining but no retraining config is set")
        al_cfg.mlip.check_fine_tunable()
        al_cfg.trainer.enabled = True
        al_cfg.trainer.train_launcher = config.retraining.train_launcher
        al_cfg.trainer.epochs_per_iter = config.retraining.epochs
        al_cfg.trainer.promote_model = config.retraining.promote_model
        al_cfg.trainer.promotion_approved = config.retraining.promotion_approved
        al_cfg.trainer.validation_set = config.retraining.validation_set
        if config.retraining.validation_set is not None:
            al_cfg.trainer.validation_fraction = 0.0
        elif config.retraining.validation_fraction > 0:
            al_cfg.trainer.validation_fraction = config.retraining.validation_fraction
        al_cfg.trainer.validation_reference_set = config.retraining.validation_reference_set
        al_cfg.trainer.promotion_max_energy_mae_eV_per_atom = (
            config.retraining.promotion_max_energy_mae_eV_per_atom
        )
        al_cfg.trainer.promotion_max_force_mae_eV_per_A = (
            config.retraining.promotion_max_force_mae_eV_per_A
        )
        al_cfg.trainer.promotion_min_relative_improvement = (
            config.retraining.promotion_min_relative_improvement
        )
        al_cfg.trainer.promotion_min_evaluated_frames = (
            config.retraining.promotion_min_evaluated_frames
        )
        if config.phase_policy.reevaluate_after_retraining and not al_cfg.trainer.promote_model:
            raise ValueError("post-retraining reevaluation requires model promotion")
    else:
        al_cfg.trainer.enabled = False
        al_cfg.trainer.promote_model = False
    if (
        config.dft_refinement is not None
        and al_cfg.trainer.promote_model
        and not config.phase_policy.reevaluate_after_retraining
    ):
        raise ValueError(
            "DFT refinement after model promotion requires "
            "phase_policy.reevaluate_after_retraining=True"
        )

    if config.max_dft_calculations is not None:
        refinement_reserve = 0
        if config.dft_refinement is not None:
            refinement_reserve = (
                _pending_reference_calculation_count(config.dft_refinement)
                + config.dft_refinement.max_candidates
            )
        if refinement_reserve > config.max_dft_calculations:
            raise ValueError(
                "remaining DFT allowance is below required reference and refinement reserve"
            )
        al_cfg.loop.max_dft_calculations = config.max_dft_calculations - refinement_reserve
    al_cfg.md.seed_source.kind = "compositions"
    al_cfg.md.seed_source.paths = []
    al_cfg.md.seed_source.prompt = None
    al_cfg.md.seed_source.compositions = [formula]
    al_cfg.loop.out_dir = al_root
    return al_cfg


def _summarize_active_learning(
    al_root: Path, al_cfg: ALConfig, compute_nodes: int, phase_output_dir: str
) -> dict[str, Any]:
    """Collect durable AL iteration evidence after a successful loop run."""
    states = _iteration_states(
        al_root,
        allow_empty=al_cfg.loop.max_dft_calculations == 0,
    )
    total_seconds = sum(float(state.get("timings_sec", {}).get("total", 0.0)) for state in states)
    promoted = [state for state in states if state.get("model_promoted")]
    dataset_path = al_root / (
        "dataset.extxyz" if al_cfg.loop.dataset_format == "extxyz" else "dataset.db"
    )
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    training_data = None
    if dataset_path.is_file():
        training_data = {
            "path": str(dataset_path.resolve()),
            "sha256": sha256_file(dataset_path),
            "manifest_path": (str(manifest_path.resolve()) if manifest_path.is_file() else None),
        }
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            training_data.update(
                {
                    "dataset_id": manifest.get("dataset_id"),
                    "method_signature": manifest.get("method_signature"),
                    "dft_backend": manifest.get("dft_backend"),
                    "accepted_frames_in_last_append": manifest.get("validation", {}).get(
                        "accepted"
                    ),
                }
            )
    validation_dataset_path = al_root / "validation.extxyz"
    if validation_dataset_path.is_file():
        result_validation_data = {
            "path": str(validation_dataset_path.resolve()),
            "sha256": sha256_file(validation_dataset_path),
        }
    else:
        result_validation_data = None
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
        "node_hours": total_seconds * compute_nodes / 3600.0,
        "model_promoted": bool(promoted),
        "training_deferred": bool(
            not promoted and states and states[-1].get("training_status") == "deferred"
        ),
        "training_data": training_data,
        "held_out_validation_data": result_validation_data,
        "md_candidate_uncertainty": candidate_uncertainty,
        "iteration_states": states,
        "phase_output_dir": phase_output_dir,
    }
    if promoted:
        latest_model = promoted[-1].get("new_logdir")
        if latest_model:
            result["promoted_model_path"] = str(latest_model)
            result["exploration_kwargs"] = {
                "mlip": al_cfg.mlip.with_checkpoint(latest_model).model_dump(mode="json")
            }
    return result
