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
    )

    session = run_chat(cfg)
    print(f"\nSession finished. {len(session.explorations)} composition(s) explored.")


if __name__ == "__main__":
    main()
