#!/usr/bin/env python3
"""Initialize a campaign with a real multi-LLM formula-discovery debate.

Exercises the campaign formula-generation slice (element-set-to-formula
enumeration + LLM-formula merge, see :mod:`matsim_agents.discovery.formula`,
:mod:`matsim_agents.discovery.formula_merge`, and
:mod:`matsim_agents.campaign`) against a *real* multi-LLM scientific debate
served on real vLLM endpoints -- no mocked LLM calls. This stage writes the
resumable formula registry consumed by ``campaign_execute.py``, which owns
per-formula exploration, active learning, DFT labelling, and evidence review.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matsim_agents.campaign.orchestrator import run_formula_discovery_stage  # noqa: E402
from matsim_agents.campaign.state import CampaignState  # noqa: E402
from matsim_agents.discovery.formula import FormulaGenerationPolicy  # noqa: E402
from matsim_agents.workflows.debate import (  # noqa: E402
    DebateParticipant,
    ScientificDebateConfig,
    run_scientific_debate,
)

DEFAULT_HYPOTHESIS_TEMPLATE = (
    "The available elements are {elements}. Propose chemically plausible binary "
    "and ternary stoichiometric formulas that should be included in a "
    "structure-discovery campaign. For every formula: provide the chemical "
    "rationale; state the assumed oxidation states; identify relevant known "
    "structure families; identify speculative formulas; propose calculations "
    "that could falsify the hypothesis; distinguish established knowledge from "
    "inference; identify missing information. Do not claim that a proposed "
    "formula is stable without numerical evidence."
)


def _parse_oxidation_states(specs: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for spec in specs:
        element, _, values = spec.partition(":")
        if not element or not values:
            raise ValueError(f"malformed --oxidation-state entry: {spec!r} (expected El:v1,v2,...)")
        result[element] = [int(v) for v in values.split(",")]
    return result


def _parse_participants(specs: list[list[str]]) -> list[DebateParticipant]:
    participants = []
    for name, provider, model, base_url in specs:
        participants.append(
            DebateParticipant(
                name=name,
                provider=provider,
                model=model,
                base_url=base_url or None,
                role="independent scientist evaluating candidate formulas",
            )
        )
    return participants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", nargs="+", required=True)
    parser.add_argument(
        "--oxidation-state",
        dest="oxidation_states",
        action="append",
        default=[],
        metavar="EL:V1,V2,...",
        help="Repeatable, e.g. --oxidation-state Nb:3,4,5 --oxidation-state O:-2",
    )
    parser.add_argument("--min-species", type=int, default=2)
    parser.add_argument("--max-species", type=int, default=3)
    parser.add_argument("--min-coefficient", type=int, default=1)
    parser.add_argument("--max-coefficient", type=int, default=6)
    parser.add_argument("--max-atoms", type=int, default=12)
    parser.add_argument(
        "--model",
        dest="models",
        nargs=4,
        action="append",
        metavar=("NAME", "PROVIDER", "MODEL", "BASE_URL"),
        required=True,
        help="Repeatable debate participant: NAME PROVIDER MODEL BASE_URL",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--campaign-id", default="campaign-e2e")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-root", default="./runs")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oxidation_states = _parse_oxidation_states(args.oxidation_states)
    policy = FormulaGenerationPolicy(
        elements=args.elements,
        minimum_species=args.min_species,
        maximum_species=args.max_species,
        minimum_coefficient=args.min_coefficient,
        maximum_coefficient=args.max_coefficient,
        maximum_atoms_in_reduced_formula=args.max_atoms,
        require_charge_balance=bool(oxidation_states),
        oxidation_states=oxidation_states,
    )

    participants = _parse_participants(args.models)
    hypothesis = DEFAULT_HYPOTHESIS_TEMPLATE.format(elements=", ".join(args.elements))
    debate_cfg = ScientificDebateConfig(
        hypothesis=hypothesis,
        participants=participants,
        rounds=args.rounds,
        output_root=args.output_root,
        debate_mode="equal",
        synthesis_method="independent_verdicts",
    )

    print(
        f"[{__name__}] running real multi-LLM debate ({len(participants)} participants, "
        f"{args.rounds} rounds) ..."
    )
    debate_result = run_scientific_debate(debate_cfg)
    print(
        f"[{__name__}] debate complete: run_id={debate_result.run_id}, "
        f"turns={len(debate_result.turns)}, verdicts={len(debate_result.verdicts)}"
    )

    campaign = CampaignState(
        campaign_id=args.campaign_id,
        element_set=args.elements,
        formula_policy=policy,
    )
    campaign = run_formula_discovery_stage(campaign, debate_result)

    active = campaign.active_formulas()
    llm_agreement = [c for c in active if c.llm_contributors]
    print(
        f"[{__name__}] formula discovery stage complete: "
        f"{len(campaign.formulas)} total formulas, {len(active)} active, "
        f"{len(llm_agreement)} with LLM attribution"
    )

    campaign_path = output_dir / "campaign_state.json"
    campaign.save(campaign_path)
    print(f"[{__name__}] wrote {campaign_path}")

    # Minimal end-to-end sanity gate: the debate actually produced dialogue,
    # and the merge stage produced at least one active formula.
    if len(debate_result.turns) != args.rounds * len(participants):
        print("[ERROR] debate turn count does not match rounds * participants", file=sys.stderr)
        return 1
    if not active:
        print("[ERROR] formula discovery stage produced zero active formulas", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
