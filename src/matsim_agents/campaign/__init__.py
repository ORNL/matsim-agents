"""Public package for multi-formula discovery campaigns.

This layer coordinates *many* formulas at once (element-set-to-hull), on top
of the existing single-composition discovery machinery
(:mod:`matsim_agents.discovery`) and multi-LLM debate workflow
(:mod:`matsim_agents.workflows.debate`).
"""

from matsim_agents.campaign.execution import (
    CampaignDFTRefinementConfig,
    CampaignFormulaExecutionConfig,
    CampaignRetrainingConfig,
    latest_promoted_model,
    make_formula_runner,
    run_formula_with_active_learning,
)
from matsim_agents.campaign.orchestrator import (
    CampaignReviewDecision,
    CampaignRunPolicy,
    CampaignRunResult,
    run_campaign,
    run_formula_discovery_stage,
)
from matsim_agents.campaign.review import (
    CampaignDebateReviewConfig,
    make_debate_review_runner,
    run_campaign_debate_review,
)
from matsim_agents.campaign.state import CampaignReviewRecord, CampaignState, FormulaRunRecord

__all__ = [
    "CampaignReviewDecision",
    "CampaignReviewRecord",
    "CampaignDebateReviewConfig",
    "CampaignDFTRefinementConfig",
    "CampaignFormulaExecutionConfig",
    "CampaignRetrainingConfig",
    "CampaignRunPolicy",
    "CampaignRunResult",
    "CampaignState",
    "FormulaRunRecord",
    "latest_promoted_model",
    "run_campaign",
    "make_debate_review_runner",
    "make_formula_runner",
    "run_campaign_debate_review",
    "run_formula_with_active_learning",
    "run_formula_discovery_stage",
]
