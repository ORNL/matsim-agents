"""Public package for multi-formula discovery campaigns.

This layer coordinates *many* formulas at once (element-set-to-hull), on top
of the existing single-composition discovery machinery
(:mod:`matsim_agents.discovery`) and multi-LLM debate workflow
(:mod:`matsim_agents.workflows.debate`).
"""

from matsim_agents.campaign.state import CampaignState

__all__ = ["CampaignState"]
