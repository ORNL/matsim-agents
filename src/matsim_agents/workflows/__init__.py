"""Composable scientific workflows built from lower-level backend services."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Submodules import LLM and graph stacks; load them only when a name is used.
_EXPORTS = {
    "debate": (
        "DebateParticipant",
        "ScientificDebateConfig",
        "ScientificDebateResult",
        "run_scientific_debate",
    ),
    "exploration.api": (
        "ExplorationCheck",
        "ExplorationConfig",
        "ExplorationSummary",
        "check_exploration",
        "load_exploration_config",
        "resume_exploration",
        "run_exploration",
        "summarize_exploration",
    ),
    "investigation": (
        "HypothesisDiscussionConfig",
        "InvestigationConfig",
        "InvestigationResult",
        "MultiLLMDebateDiscussionConfig",
        "ScientificHypothesis",
        "SingleLLMDiscussionConfig",
        "build_hypothesis_from_discussion",
        "run_investigation",
    ),
    "llm_check": ("LLMCheckConfig", "LLMCheckResult", "run_llm_check"),
    "phase_exploration": (
        "HullFollowUpAssessment",
        "PhaseExplorationContext",
        "PhaseExplorationEvidenceRecord",
        "PhaseExplorationPolicy",
        "PhaseExplorationWorkflowResult",
        "run_phase_exploration",
    ),
    "relaxation": (
        "RelaxationMode",
        "ScientificRelaxationConfig",
        "ScientificRelaxationResult",
        "run_relaxation",
    ),
}
_MODULE_BY_NAME = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = [
    "DebateParticipant",
    "ExplorationCheck",
    "ExplorationConfig",
    "ExplorationSummary",
    "HullFollowUpAssessment",
    "HypothesisDiscussionConfig",
    "InvestigationConfig",
    "InvestigationResult",
    "LLMCheckConfig",
    "LLMCheckResult",
    "MultiLLMDebateDiscussionConfig",
    "PhaseExplorationContext",
    "PhaseExplorationEvidenceRecord",
    "PhaseExplorationPolicy",
    "PhaseExplorationWorkflowResult",
    "RelaxationMode",
    "ScientificDebateConfig",
    "ScientificDebateResult",
    "ScientificHypothesis",
    "ScientificRelaxationConfig",
    "ScientificRelaxationResult",
    "SingleLLMDiscussionConfig",
    "build_hypothesis_from_discussion",
    "check_exploration",
    "load_exploration_config",
    "resume_exploration",
    "run_exploration",
    "run_investigation",
    "run_llm_check",
    "run_phase_exploration",
    "run_relaxation",
    "run_scientific_debate",
    "summarize_exploration",
]
assert sorted(__all__) == sorted(_MODULE_BY_NAME)


def __getattr__(name: str) -> Any:
    module = _MODULE_BY_NAME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


if TYPE_CHECKING:
    from matsim_agents.workflows.debate import (
        DebateParticipant,
        ScientificDebateConfig,
        ScientificDebateResult,
        run_scientific_debate,
    )
    from matsim_agents.workflows.exploration.api import (
        ExplorationCheck,
        ExplorationConfig,
        ExplorationSummary,
        check_exploration,
        load_exploration_config,
        resume_exploration,
        run_exploration,
        summarize_exploration,
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
    from matsim_agents.workflows.phase_exploration import (
        HullFollowUpAssessment,
        PhaseExplorationContext,
        PhaseExplorationEvidenceRecord,
        PhaseExplorationPolicy,
        PhaseExplorationWorkflowResult,
        run_phase_exploration,
    )
    from matsim_agents.workflows.relaxation import (
        RelaxationMode,
        ScientificRelaxationConfig,
        ScientificRelaxationResult,
        run_relaxation,
    )
