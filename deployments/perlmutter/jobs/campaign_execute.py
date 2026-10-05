#!/usr/bin/env python3
"""Resume a formula registry and execute a bounded real discovery campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matsim_agents.active_learning.config import ALConfig  # noqa: E402
from matsim_agents.campaign.acquisition import CampaignAcquisitionPolicy  # noqa: E402
from matsim_agents.campaign.execution import (  # noqa: E402
    CampaignDFTRefinementConfig,
    CampaignFormulaExecutionConfig,
    CampaignRetrainingConfig,
    ReferenceStructureSpec,
    latest_promoted_model,
    make_formula_runner,
)
from matsim_agents.campaign.orchestrator import CampaignRunPolicy, run_campaign  # noqa: E402
from matsim_agents.campaign.registry import CandidateSelectionPolicy  # noqa: E402
from matsim_agents.campaign.review import (  # noqa: E402
    CampaignDebateReviewConfig,
    make_debate_review_runner,
)
from matsim_agents.campaign.state import CampaignState  # noqa: E402
from matsim_agents.discovery.stability import (  # noqa: E402
    ReferenceCompletenessPolicy,
    ReferenceEnergySet,
)
from matsim_agents.execution.contracts import ApprovalPolicy, WorkflowStatus  # noqa: E402
from matsim_agents.workflows.debate import DebateParticipant  # noqa: E402
from matsim_agents.workflows.phase_exploration import PhaseExplorationPolicy  # noqa: E402


def _participants(specs: list[list[str]]) -> list[DebateParticipant]:
    return [
        DebateParticipant(
            name=name,
            provider=provider,
            model=model,
            base_url=base_url or None,
            role="independent reviewer of numerical campaign evidence",
        )
        for name, provider, model, base_url in specs
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-state", required=True, type=Path)
    parser.add_argument("--al-config", required=True, type=Path)
    parser.add_argument(
        "--model",
        dest="models",
        nargs=4,
        action="append",
        metavar=("NAME", "PROVIDER", "MODEL", "BASE_URL"),
        required=True,
    )
    parser.add_argument("--review-rounds", type=int, default=2)
    parser.add_argument("--minimum-review-agreement", type=float, default=1.0)
    parser.add_argument("--final-review", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=["dft", "uma-only"],
        default="dft",
    )
    parser.add_argument("--formulas-per-iteration", type=int, default=1)
    parser.add_argument(
        "--acquisition-mode",
        choices=["insertion-order", "random", "exploitation", "exploration", "adaptive"],
        default="insertion-order",
    )
    parser.add_argument("--acquisition-seed", type=int, default=0)
    parser.add_argument("--lambda-initial", type=float, default=0.5)
    parser.add_argument("--lambda-minimum", type=float, default=0.2)
    parser.add_argument("--lambda-maximum", type=float, default=0.8)
    parser.add_argument("--lambda-update-rate", type=float, default=0.15)
    parser.add_argument("--minimum-exploitation-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-exploration-fraction", type=float, default=0.2)
    parser.add_argument("--maximum-per-relaxed-family", type=int, default=1)
    parser.add_argument("--reserved-dft-per-formula", type=int, default=0)
    parser.add_argument("--reserved-node-hours-per-formula", type=float, default=0.0)
    parser.add_argument("--no-new-hull-vertex-iterations", type=int)
    parser.add_argument("--hull-energy-change-ev-per-atom", type=float)
    parser.add_argument("--minimum-formula-coverage", type=float)
    parser.add_argument("--require-low-uncertainty-near-hull", action="store_true")
    parser.add_argument("--low-uncertainty-threshold", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--max-dft-calculations", type=int, default=6)
    parser.add_argument("--max-al-iterations", type=int, default=3)
    parser.add_argument("--max-node-hours", type=float, default=8.0)
    parser.add_argument("--n-random", type=int, default=0)
    parser.add_argument("--relax-maxiter", type=int, default=100)
    parser.add_argument(
        "--validation-config",
        action="append",
        type=Path,
        default=[],
        help="Additional AL config whose MLIP rescoring provides cross-model validation.",
    )
    parser.add_argument(
        "--validation-python",
        action="append",
        default=[],
        type=Path,
        help=(
            "Independent Python interpreter for the corresponding --validation-config; "
            "repeat once per config, or provide once to use it for all configs."
        ),
    )
    parser.add_argument("--perturbation-trials", type=int, default=0)
    parser.add_argument("--perturbation-scale-A", type=float, default=0.05)
    parser.add_argument("--perturbation-seed", type=int, default=0)
    parser.add_argument("--surrogate-reference-structures", type=Path)
    parser.add_argument("--surrogate-unary-max-steps", type=int, default=200)
    parser.add_argument("--surrogate-unary-fmax", type=float, default=0.02)
    parser.add_argument("--surrogate-unary-maxstep", type=float, default=0.01)
    parser.add_argument("--surrogate-minimum-unique-unary", type=int, default=2)
    parser.add_argument("--degeneracy-tolerance-ev-per-atom", type=float, default=0.01)
    parser.add_argument(
        "--dft-reference-structures",
        type=Path,
        help="JSON mapping of elemental/competing formulas to compatible structure files.",
    )
    parser.add_argument("--dft-method-signature")
    parser.add_argument("--dft-refine-candidates", type=int, default=1)
    parser.add_argument("--dft-relax-max-steps", type=int, default=100)
    parser.add_argument("--dft-force-tolerance", type=float, default=0.02)
    parser.add_argument("--dft-relax-atoms-only", action="store_true")
    parser.add_argument("--approve-dft", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--approve-retraining", action="store_true")
    parser.add_argument(
        "--train-script",
        type=Path,
        default=ROOT / "src" / "matsim_agents" / "active_learning" / "finetune_uma.py",
    )
    parser.add_argument("--train-launcher", type=Path)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--promote-model", action="store_true")
    parser.add_argument("--approve-model-promotion", action="store_true")
    parser.add_argument("--promotion-validation-set", type=Path)
    parser.add_argument("--promotion-validation-reference-set", type=Path)
    parser.add_argument("--promotion-max-energy-mae", type=float, default=0.1)
    parser.add_argument("--promotion-max-force-mae", type=float, default=0.2)
    parser.add_argument("--promotion-max-relative-regression", type=float, default=0.05)
    parser.add_argument("--promotion-min-evaluated-frames", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-inconclusive", action="store_true")
    args = parser.parse_args(argv)
    if args.degeneracy_tolerance_ev_per_atom <= 0:
        parser.error("--degeneracy-tolerance-ev-per-atom must be positive")

    if args.execution_mode == "dft" and not args.approve_dft:
        parser.error("real campaign execution requires --approve-dft")
    if args.execution_mode == "uma-only":
        if args.approve_dft or args.dft_reference_structures or args.dft_method_signature:
            parser.error("uma-only execution forbids all DFT approval and reference options")
        if args.retrain or args.promote_model:
            parser.error("uma-only execution forbids retraining and model promotion")
    if args.retrain and not args.approve_retraining:
        parser.error("--retrain requires --approve-retraining")
    if args.promote_model and not args.retrain:
        parser.error("--promote-model requires --retrain")
    if args.promote_model and not args.approve_model_promotion:
        parser.error("--promote-model requires --approve-model-promotion")
    if args.promote_model and args.promotion_validation_set is None:
        parser.error("--promote-model requires --promotion-validation-set")
    if args.promotion_validation_set is not None and not args.promotion_validation_set.is_file():
        parser.error(f"promotion validation set does not exist: {args.promotion_validation_set}")
    if (
        args.promotion_validation_reference_set is not None
        and not args.promotion_validation_reference_set.is_file()
    ):
        parser.error(
            "promotion validation reference set does not exist: "
            f"{args.promotion_validation_reference_set}"
        )
    if not args.campaign_state.is_file():
        parser.error(f"campaign state does not exist: {args.campaign_state}")
    if bool(args.dft_reference_structures) != bool(args.dft_method_signature):
        parser.error(
            "--dft-reference-structures and --dft-method-signature must be provided together"
        )
    if args.dft_reference_structures and not args.dft_reference_structures.is_file():
        parser.error(
            f"DFT reference structure manifest does not exist: {args.dft_reference_structures}"
        )

    campaign = CampaignState.load(args.campaign_state)
    al_config = ALConfig.from_yaml(args.al_config)
    campaign.budget.max_candidates = args.max_candidates
    campaign.budget.max_dft_calculations = (
        args.max_dft_calculations if args.execution_mode == "dft" else None
    )
    campaign.budget.max_active_learning_iterations = (
        args.max_al_iterations if args.execution_mode == "dft" else None
    )
    campaign.budget.max_node_hours = args.max_node_hours
    refinement = None
    if args.dft_reference_structures:
        raw_references = json.loads(args.dft_reference_structures.read_text(encoding="utf-8"))
        if not isinstance(raw_references, dict) or not raw_references:
            parser.error("DFT reference structure manifest must be a non-empty JSON object")
        if "phases" in raw_references:
            raw_phases = raw_references["phases"]
            raw_completeness = raw_references.get("completeness", {})
        else:
            raw_phases = raw_references
            raw_completeness = {}
        if not isinstance(raw_phases, dict) or not raw_phases:
            parser.error("DFT reference manifest 'phases' must be a non-empty object")
        try:
            completeness_policy = ReferenceCompletenessPolicy.model_validate(raw_completeness)
        except ValueError as exc:
            parser.error(f"invalid reference completeness policy: {exc}")
        reference_paths = {}
        reference_relax_cell = {}
        reference_settings = {}
        reference_phases = []
        for phase_id, spec in raw_phases.items():
            if isinstance(spec, str):
                reference_paths[str(phase_id)] = Path(spec).expanduser().resolve()
                continue
            if not isinstance(spec, dict) or "path" not in spec:
                parser.error(
                    f"reference {phase_id!r} must be a path or an object containing 'path'"
                )
            path = Path(spec["path"]).expanduser().resolve()
            if not path.is_file():
                parser.error(f"reference structure does not exist: {path}")
            reference_phases.append(
                ReferenceStructureSpec(
                    phase_id=str(spec.get("phase_id", phase_id)),
                    formula=str(spec.get("formula", phase_id)),
                    structure_path=path,
                    relax_cell=bool(spec.get("relax_cell", True)),
                    settings=dict(spec.get("settings", {})),
                    source=str(spec.get("source", "manifest")),
                    provenance={
                        str(key): str(value)
                        for key, value in dict(spec.get("provenance", {})).items()
                    },
                    energy_correction_eV_per_atom=float(
                        spec.get("energy_correction_eV_per_atom", 0.0)
                    ),
                )
            )
        references = campaign.reference_energies or ReferenceEnergySet(
            identifier=f"campaign-{args.dft_method_signature}",
            method_signature=args.dft_method_signature,
            backend=al_config.dft.backend,
            elemental_energies_eV_per_atom={},
            completeness_policy=completeness_policy,
        )
        references.completeness_policy = completeness_policy
        refinement = CampaignDFTRefinementConfig(
            method_signature=args.dft_method_signature,
            reference_structures=reference_paths,
            reference_relax_cell=reference_relax_cell,
            reference_settings=reference_settings,
            reference_phases=reference_phases,
            reference_completeness_policy=completeness_policy,
            reference_energies=references,
            max_candidates=args.dft_refine_candidates,
            relax_cell=not args.dft_relax_atoms_only,
            max_steps=args.dft_relax_max_steps,
            force_tolerance_eV_per_A=args.dft_force_tolerance,
            candidate_acquisition=CandidateSelectionPolicy(
                enabled=args.acquisition_mode != "insertion-order",
                mode=(
                    "adaptive"
                    if args.acquisition_mode == "insertion-order"
                    else args.acquisition_mode
                ),
                lambda_value=args.lambda_initial,
                minimum_exploitation_fraction=args.minimum_exploitation_fraction,
                minimum_exploration_fraction=args.minimum_exploration_fraction,
                maximum_per_relaxed_family=args.maximum_per_relaxed_family,
            ),
        )
        campaign.reference_energies = references

    phase_policy = PhaseExplorationPolicy(
        active_learning=args.execution_mode == "dft",
        retrain_mlip=args.retrain,
        promote_model=args.promote_model,
        reevaluate_after_retraining=args.promote_model,
        approvals=ApprovalPolicy(
            before_dft=args.execution_mode == "dft",
            before_retraining=True,
            before_model_promotion=True,
        ),
        dft_approved=args.approve_dft,
        retraining_approved=args.approve_retraining,
        promotion_approved=args.approve_model_promotion,
        budget=campaign.budget,
    )
    formula_runner = make_formula_runner(
        CampaignFormulaExecutionConfig(
            active_learning_config=args.al_config,
            execution_mode=("uma_only" if args.execution_mode == "uma-only" else "dft"),
            phase_policy=phase_policy,
            exploration_kwargs={
                "n_random": args.n_random,
                "maxiter": args.relax_maxiter,
                "degeneracy_tol_eV_per_atom": args.degeneracy_tolerance_ev_per_atom,
            },
            compute_nodes=1,
            validation_configs=args.validation_config,
            validation_pythons=args.validation_python,
            perturbation_trials=args.perturbation_trials,
            perturbation_scale_A=args.perturbation_scale_A,
            perturbation_seed=args.perturbation_seed,
            surrogate_reference_structures=args.surrogate_reference_structures,
            surrogate_unary_max_steps=args.surrogate_unary_max_steps,
            surrogate_unary_fmax_eV_per_A=args.surrogate_unary_fmax,
            surrogate_unary_maxstep_A=args.surrogate_unary_maxstep,
            surrogate_minimum_unique_unary=args.surrogate_minimum_unique_unary,
            retraining=(
                CampaignRetrainingConfig(
                    train_script=args.train_script,
                    train_launcher=args.train_launcher,
                    epochs=args.train_epochs,
                    promote_model=args.promote_model,
                    promotion_approved=args.approve_model_promotion,
                    validation_set=args.promotion_validation_set,
                    validation_reference_set=args.promotion_validation_reference_set,
                    promotion_max_energy_mae_eV_per_atom=args.promotion_max_energy_mae,
                    promotion_max_force_mae_eV_per_A=args.promotion_max_force_mae,
                    promotion_max_relative_regression=args.promotion_max_relative_regression,
                    promotion_min_evaluated_frames=args.promotion_min_evaluated_frames,
                )
                if args.retrain
                else None
            ),
            dft_refinement=refinement,
            model_override=latest_promoted_model(campaign),
        )
    )
    participants = _participants(args.models)
    review_runner = make_debate_review_runner(
        CampaignDebateReviewConfig(
            participants=participants,
            output_root=str(args.campaign_state.parent / "reviews"),
            rounds=args.review_rounds,
            minimum_agreement_fraction=args.minimum_review_agreement,
        )
    )
    result = run_campaign(
        campaign,
        output_dir=args.campaign_state.parent,
        formula_runner=formula_runner,
        review_runner=review_runner,
        final_review_runner=review_runner if args.final_review else None,
        policy=CampaignRunPolicy(
            formulas_per_iteration=args.formulas_per_iteration,
            max_iterations=args.max_iterations,
            retry_failed=args.retry_failed,
            retry_inconclusive=args.retry_inconclusive,
            reserved_dft_calculations_per_formula=args.reserved_dft_per_formula,
            reserved_node_hours_per_formula=args.reserved_node_hours_per_formula,
            no_new_hull_vertex_iterations=args.no_new_hull_vertex_iterations,
            hull_energy_change_eV_per_atom=args.hull_energy_change_ev_per_atom,
            minimum_formula_coverage=args.minimum_formula_coverage,
            require_low_uncertainty_near_hull=args.require_low_uncertainty_near_hull,
            low_uncertainty_threshold=args.low_uncertainty_threshold,
            acquisition=CampaignAcquisitionPolicy(
                enabled=args.acquisition_mode != "insertion-order",
                mode=(
                    "adaptive"
                    if args.acquisition_mode == "insertion-order"
                    else args.acquisition_mode
                ),
                random_seed=args.acquisition_seed,
                lambda_initial=args.lambda_initial,
                lambda_minimum=args.lambda_minimum,
                lambda_maximum=args.lambda_maximum,
                update_rate=args.lambda_update_rate,
                minimum_exploitation_fraction=args.minimum_exploitation_fraction,
                minimum_exploration_fraction=args.minimum_exploration_fraction,
            ),
        ),
    )
    if refinement is not None:
        result.campaign.reference_energies = refinement.reference_energies
        result.campaign.save(args.campaign_state)
    summary = {
        "campaign_id": result.campaign.campaign_id,
        "status": result.campaign.status,
        "completed": result.formulas_completed,
        "failed": result.formulas_failed,
        "stop_reason": result.stop_reason,
        "state_path": result.state_path,
    }
    summary_path = args.campaign_state.parent / "campaign_result.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if result.campaign.status != WorkflowStatus.FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
