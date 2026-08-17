"""Regression tests for the package-boundary migration."""

import matsim_agents.graph as legacy_graph
import matsim_agents.llm as legacy_llm
import matsim_agents.state as legacy_state
from matsim_agents.backends.llm import provider
from matsim_agents.orchestration import objective_graph, state


def test_legacy_orchestration_modules_alias_canonical_modules() -> None:
    assert legacy_graph is objective_graph
    assert legacy_state is state


def test_legacy_llm_module_aliases_canonical_provider() -> None:
    assert legacy_llm is provider
