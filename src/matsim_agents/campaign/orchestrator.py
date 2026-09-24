"""Formula-discovery stage of a multi-formula campaign.

Implements the first three lines of the campaign pseudocode: run the initial
multi-LLM debate, enumerate the deterministic formula grid, validate the
LLM-proposed formulas against the same deterministic constraints, and merge
the two into the campaign's formula registry. Later campaign stages (hull-
aware acquisition, DFT batch selection, repeated evidence-review debates) are
intentionally out of scope for this module.
"""

from __future__ import annotations

from matsim_agents.campaign.state import CampaignState
from matsim_agents.discovery.formula import enumerate_formulas
from matsim_agents.discovery.formula_merge import extract_llm_formula_proposals, merge_formulas
from matsim_agents.workflows.debate import ScientificDebateResult


def run_formula_discovery_stage(
    campaign: CampaignState,
    debate_result: ScientificDebateResult,
) -> CampaignState:
    """Merge deterministic and LLM-proposed formulas into ``campaign`` in place."""
    deterministic = enumerate_formulas(campaign.formula_policy)
    llm_proposals = extract_llm_formula_proposals(debate_result.verdicts, campaign.formula_policy)
    merged = merge_formulas(
        deterministic, llm_proposals, campaign.formula_policy, iteration=campaign.iteration
    )
    campaign.upsert_formulas(merged)
    campaign.debate_run_ids.append(debate_result.run_id)
    campaign.iteration += 1
    return campaign
