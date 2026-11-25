"""SYNTHETIC end-to-end fusion PFC demo: score, acquisition advice, hypothesis revision.

Every number is invented (see ``matsim_agents.fusion.synthetic``). Nothing is run on
any machine, no LLM is called, and nothing here is a material property.

    PYTHONPATH=src python examples/fusion_pfc/synthetic_demo.py [OUTPUT_DIR]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from matsim_agents.campaign.budget import BudgetLedger
from matsim_agents.execution.ledger import EvidenceLedger
from matsim_agents.fusion import synthetic
from matsim_agents.fusion.acquisition import (
    FusionCalculation,
    plan_fusion_acquisition,
    render_fusion_plan,
)
from matsim_agents.fusion.policy import builtin_policy
from matsim_agents.fusion.revision import revise_fusion_hypothesis
from matsim_agents.fusion.scoring import render_report, score_candidates


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    policy = builtin_policy()
    evidence = synthetic.candidates()
    result = score_candidates(policy, synthetic.BENCHMARKS, synthetic.SCENARIO, evidence)
    print(render_report(result), end="\n\n")

    proposal = FusionCalculation(
        calc_id="C-dose",
        candidate_id="C",
        objective="low_activation_shutdown_dose",
        observable="shutdown_dose_rate",
        backend="fispact",
        cost_node_hours=5.0,
        expected_uncertainty=1e-4,
        prior_value=1e-3,
        prior_uncertainty=5e-4,
        prior_source="SYNTHETIC prior for the demo",
    )
    budget = BudgetLedger(total_node_hours=1000.0, allocation={"property": 50.0})
    plan = plan_fusion_acquisition(
        result, policy, synthetic.BENCHMARKS, [proposal], budget, draws=6, samples=200
    )
    print(render_fusion_plan(plan), end="\n\n")

    ledger = EvidenceLedger(out / "ledger.jsonl")
    leader = result.ranking[0] if result.ranking else "A"
    outcome = revise_fusion_hypothesis(ledger, "fusion-pfc-demo", leader, result, policy, evidence)
    print(f"hypothesis {outcome.record.hypothesis_id}: {outcome.record.status.value}")
    print(outcome.record.scientific_rationale)
    print(f"ledger: {ledger.path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="fpfc-")))
