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
from matsim_agents.campaign.execution import (  # noqa: E402
    CampaignDFTRefinementConfig,
    CampaignFormulaExecutionConfig,
    CampaignRetrainingConfig,
    latest_promoted_model,
    make_formula_runner,
)
from matsim_agents.campaign.orchestrator import CampaignRunPolicy, run_campaign  # noqa: E402
from matsim_agents.campaign.review import (  # noqa: E402
    CampaignDebateReviewConfig,
    make_debate_review_runner,
)
from matsim_agents.campaign.state import CampaignState  # noqa: E402
from matsim_agents.discovery.stability import ReferenceEnergySet  # noqa: E402
from matsim_agents.execution.contracts import ApprovalPolicy  # noqa: E402
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
    parser.add_argument("--formulas-per-iteration", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--max-dft-calculations", type=int, default=6)
    parser.add_argument("--max-al-iterations", type=int, default=3)
    parser.add_argument("--max-node-hours", type=float, default=8.0)
    parser.add_argument("--n-random", type=int, default=0)
    parser.add_argument("--relax-maxiter", type=int, default=100)
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
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)

    if not args.approve_dft:
        parser.error("real campaign execution requires --approve-dft")
    if args.retrain and not args.approve_retraining:
        parser.error("--retrain requires --approve-retraining")
    if args.promote_model and not args.retrain:
        parser.error("--promote-model requires --retrain")
    if args.promote_model and not args.approve_model_promotion:
        parser.error("--promote-model requires --approve-model-promotion")
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
    campaign.budget.max_dft_calculations = args.max_dft_calculations
    campaign.budget.max_active_learning_iterations = args.max_al_iterations
    campaign.budget.max_node_hours = args.max_node_hours
    refinement = None
    if args.dft_reference_structures:
        raw_references = json.loads(args.dft_reference_structures.read_text(encoding="utf-8"))
        if not isinstance(raw_references, dict) or not raw_references:
            parser.error("DFT reference structure manifest must be a non-empty JSON object")
        reference_paths = {}
        reference_relax_cell = {}
        reference_settings = {}
        for formula, spec in raw_references.items():
            if isinstance(spec, str):
                reference_paths[str(formula)] = Path(spec).expanduser().resolve()
                continue
            if not isinstance(spec, dict) or "path" not in spec:
                parser.error(
                    f"reference {formula!r} must be a path or an object containing 'path'"
                )
            reference_paths[str(formula)] = Path(spec["path"]).expanduser().resolve()
            reference_relax_cell[str(formula)] = bool(spec.get("relax_cell", True))
            reference_settings[str(formula)] = dict(spec.get("settings", {}))
        references = campaign.reference_energies or ReferenceEnergySet(
            identifier=f"campaign-{args.dft_method_signature}",
            method_signature=args.dft_method_signature,
            backend=al_config.dft.backend,
            elemental_energies_eV_per_atom={},
        )
        refinement = CampaignDFTRefinementConfig(
            method_signature=args.dft_method_signature,
            reference_structures=reference_paths,
            reference_relax_cell=reference_relax_cell,
            reference_settings=reference_settings,
            reference_energies=references,
            max_candidates=args.dft_refine_candidates,
            relax_cell=not args.dft_relax_atoms_only,
            max_steps=args.dft_relax_max_steps,
            force_tolerance_eV_per_A=args.dft_force_tolerance,
        )
        campaign.reference_energies = references

    phase_policy = PhaseExplorationPolicy(
        active_learning=True,
        retrain_mlip=args.retrain,
        reevaluate_after_retraining=args.promote_model,
        approvals=ApprovalPolicy(
            before_dft=True,
            before_retraining=True,
            before_model_promotion=True,
        ),
        dft_approved=args.approve_dft,
        retraining_approved=args.approve_retraining,
        budget=campaign.budget,
    )
    formula_runner = make_formula_runner(
        CampaignFormulaExecutionConfig(
            active_learning_config=args.al_config,
            phase_policy=phase_policy,
            exploration_kwargs={
                "n_random": args.n_random,
                "maxiter": args.relax_maxiter,
            },
            compute_nodes=1,
            retraining=(
                CampaignRetrainingConfig(
                    train_script=args.train_script,
                    train_launcher=args.train_launcher,
                    epochs=args.train_epochs,
                    promote_model=args.promote_model,
                    promotion_approved=args.approve_model_promotion,
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
        policy=CampaignRunPolicy(
            formulas_per_iteration=args.formulas_per_iteration,
            max_iterations=args.max_iterations,
            retry_failed=args.retry_failed,
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
    return 0 if not result.formulas_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
