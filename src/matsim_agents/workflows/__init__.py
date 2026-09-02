"""Composable scientific workflows built from lower-level backend services."""

from matsim_agents.workflows.debate import (
    DebateParticipant,
    ScientificDebateConfig,
    ScientificDebateResult,
    run_scientific_debate,
)
from matsim_agents.workflows.investigation import (
    HypothesisDiscussionConfig,
    InvestigationConfig,
    InvestigationResult,
    MultiLLMDebateDiscussionConfig,
    ScientificHypothesis,
    SingleLLMDiscussionConfig,
    build_hypothesis_from_discussion,
    run_investigation,
)
from matsim_agents.workflows.llm_check import LLMCheckConfig, LLMCheckResult, run_llm_check
from matsim_agents.workflows.relaxation import (
    RelaxationMode,
    ScientificRelaxationConfig,
    ScientificRelaxationResult,
    run_relaxation,
)

__all__ = [
    "DebateParticipant",
    "LLMCheckConfig",
    "LLMCheckResult",
    "HypothesisDiscussionConfig",
    "InvestigationConfig",
    "InvestigationResult",
    "MultiLLMDebateDiscussionConfig",
    "RelaxationMode",
    "ScientificRelaxationConfig",
    "ScientificRelaxationResult",
    "ScientificHypothesis",
    "SingleLLMDiscussionConfig",
    "build_hypothesis_from_discussion",
    "ScientificDebateConfig",
    "ScientificDebateResult",
    "run_relaxation",
    "run_scientific_debate",
    "run_llm_check",
    "run_investigation",
]
