"""Interactive hypothesis-generation chat with auto-triggered atomistic exploration.

Prereqs:
    1. Install Ollama:  https://ollama.com  (or `brew install ollama`)
    2. Pull Qwen 2.5:   `ollama pull qwen2.5:14b`
    3. Have a HydraGNN logdir with a trained checkpoint and a
       BranchWeightMLP `.pt` file.

Run:
    python examples/discovery_chat.py \\
        --logdir /path/to/multidataset_hpo-BEST6-fp64 \\
        --mlp-checkpoint /path/to/mlp_branch_weights.pt

What happens:
    * You chat with Qwen 2.5 about a target property (e.g. "I want a
      Pb-free halide perovskite for photovoltaics").
    * Each time the conversation produces a new chemical formula, the
      user is asked whether to launch a HydraGNN-driven exploration.
    * If yes, the auxiliary wrapper enumerates plausible crystal phases
      (rocksalt, perovskite, zincblende, ...), relaxes each with the
      ASE/HydraGNN calculator, and reports chemical / dynamical
      stability proxies. Results are streamed back into the
      conversation so the LLM can refine its hypothesis.
    * If the relaxed phases show high uncertainty (low branch-weight
      confidence), discovery can hand off directly to the active-learning
      loop using a base AL YAML config (--al-config).
    * You can also trigger optional control actions from inside chat with
      slash commands: /relax <path_to_structure> (single-structure
      relaxation), /al [composition] (active-learning handoff for the
      given or most recently discussed composition), and /clear (reset
      the conversation and discovery state).
"""

from __future__ import annotations

import argparse

from matsim_agents.chat import DiscoveryChatConfig, run_chat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--mlp-checkpoint", required=True)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mlp-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default=None)
    parser.add_argument("--mlp-precision", default=None)
    parser.add_argument("--ase-structure-optimizer", default="FIRE")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--llm-provider", default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:14b")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--auto-confirm", action="store_true",
                        help="Skip the y/N prompt for every detected composition.")
    parser.add_argument(
      "--al-config",
      default=None,
      help=(
        "Path to a base active-learning YAML. When high-UQ is detected in discovery, "
        "the composition is handed off to AL by overriding seed_source to compositions."
      ),
    )
    parser.add_argument(
      "--no-al-handoff",
      action="store_true",
      help="Disable discovery->active-learning handoff even when UQ is high.",
    )
    parser.add_argument(
      "--al-dry-run",
      action="store_true",
      help="Plan and report AL handoff parameters without running the AL loop.",
    )
    parser.add_argument(
      "--uq-top-weight-threshold",
      type=float,
      default=0.6,
      help="High-UQ gate: trigger handoff when mean top branch weight drops below this.",
    )
    parser.add_argument(
      "--uq-min-unreliable-fraction",
      type=float,
      default=0.25,
      help="High-UQ gate: trigger handoff when this fraction of relaxations is low-confidence.",
    )
    parser.add_argument(
      "--uq-min-relaxations-for-handoff",
      type=int,
      default=3,
      help="Require at least this many relaxations before evaluating AL handoff UQ criteria.",
    )
    parser.add_argument(
      "--al-handoff-audit-path",
      default=None,
      help=(
        "Optional JSONL artifact path for discovery->AL handoff audit records "
        "(UQ metrics, trigger rationale, action)."
      ),
    )
    args = parser.parse_args()

    cfg = DiscoveryChatConfig(
        logdir=args.logdir,
        mlp_checkpoint=args.mlp_checkpoint,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        mlp_device=args.mlp_device,
        precision=args.precision,
        mlp_precision=args.mlp_precision,
        optimizer=args.ase_structure_optimizer,
        maxiter=args.maxiter,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        auto_confirm=args.auto_confirm,
        trigger_active_learning_on_high_uq=not args.no_al_handoff,
        active_learning_config=args.al_config,
        active_learning_dry_run=args.al_dry_run,
        uq_top_weight_threshold=args.uq_top_weight_threshold,
        uq_min_unreliable_fraction=args.uq_min_unreliable_fraction,
        uq_min_relaxations_for_handoff=args.uq_min_relaxations_for_handoff,
        al_handoff_audit_path=args.al_handoff_audit_path,
    )

    session = run_chat(cfg)
    print(f"\nSession finished. {len(session.explorations)} composition(s) explored.")


if __name__ == "__main__":
    main()
