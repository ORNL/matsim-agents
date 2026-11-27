"""Interactive hypothesis-generation chat with auto-triggered atomistic exploration.

Prereqs:
    1. Install Ollama:  https://ollama.com  (or `brew install ollama`)
    2. Pull Qwen 2.5:   `ollama pull qwen2.5:14b`
    3. Have a YAML file whose `mlip:` block selects the MLIP, e.g. one of
       examples/mlip/*.yaml (HydraGNN, UMA, or MACE).

Run:
    python examples/discovery_chat.py --mlip-config examples/mlip/uma-s-1p1-omat.yaml

What happens:
    * You chat with Qwen 2.5 about a target property (e.g. "I want a
      Pb-free halide perovskite for photovoltaics").
    * Each time the conversation produces a new chemical formula, the
      user is asked whether to launch an MLIP-driven exploration.
    * If yes, the auxiliary wrapper enumerates plausible crystal phases
      (rocksalt, perovskite, zincblende, ...), relaxes each with the
      ASE calculator of the selected MLIP, and reports chemical / dynamical
      stability proxies. Results are streamed back into the
      conversation so the LLM can refine its hypothesis.
    * You can also trigger optional control actions from inside chat with
      slash commands: /relax <path_to_structure> (single-structure
      relaxation), /al [composition] (active-learning handoff for the
      given or most recently discussed composition), and /clear (reset
      the conversation and discovery state).
"""

from __future__ import annotations

import argparse

from matsim_agents.backends.mlip.arguments import add_mlip_arguments, mlip_config_from_args
from matsim_agents.agents.interactive_chat import DiscoveryChatConfig, run_chat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_mlip_arguments(parser)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--ase-structure-optimizer", default="FIRE")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--llm-provider", default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:14b")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="Skip the y/N prompt for every detected composition.",
    )
    parser.add_argument(
        "--al-config",
        default=None,
        help=(
            "Path to a base active-learning YAML used by the /al command, which "
            "overrides seed_source to compositions=[formula]."
        ),
    )
    parser.add_argument(
        "--al-dry-run",
        action="store_true",
        help="Plan and report /al parameters without running the AL loop.",
    )
    args = parser.parse_args()
    mlip = mlip_config_from_args(parser, args)
    if mlip is None:
        parser.error("--mlip-config is required")

    cfg = DiscoveryChatConfig(
        mlip=mlip,
        output_dir=args.output_dir,
        optimizer=args.ase_structure_optimizer,
        maxiter=args.maxiter,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        auto_confirm=args.auto_confirm,
        active_learning_config=args.al_config,
        active_learning_dry_run=args.al_dry_run,
    )

    session = run_chat(cfg)
    print(f"\nSession finished. {len(session.explorations)} composition(s) explored.")


if __name__ == "__main__":
    main()
