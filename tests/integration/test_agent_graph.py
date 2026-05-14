"""Integration tests for the agent graph (planner → executor → analyst).

All tests use FakeListChatModel or bypass the LLM entirely so no GPU,
model weights, or network access is required.

Test categories:
  1. Individual node unit tests (planner, analyst) with scripted LLM responses.
  2. Full graph end-to-end with executor short-circuited (no HydraGNN).
  3. Routing logic (pending_tasks / iteration cap).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from matsim_agents.state import MatSimState, RelaxationResult, TaskSpec

# ── helpers ───────────────────────────────────────────────────────────────────


def _state(**kwargs) -> MatSimState:
    return MatSimState(**kwargs)


# ── planner node ──────────────────────────────────────────────────────────────


class TestPlannerNode:
    """planner_node() produces a plan from the objective."""

    def test_empty_objective_produces_empty_plan(self):
        from matsim_agents.agents.planner import planner_node

        result = planner_node(_state())
        assert result["plan"] == []
        assert result["pending_tasks"] == []

    def test_fallback_plan_when_llm_unavailable(self, si_vasp):
        """When the LLM raises, planner falls back to a file-based plan."""
        from matsim_agents.agents.planner import planner_node

        # Point at a provider that will fail to import
        state = _state(
            objective=f"Relax the structure at {si_vasp}",
            llm_provider="nonexistent_provider",
        )
        result = planner_node(state)
        # The fallback creates one task per file path found in the objective
        assert len(result["plan"]) == 1
        assert result["plan"][0].structure_path == si_vasp
        assert result["pending_tasks"] == result["plan"]

    def test_planner_with_structured_output_stub(self, si_vasp, monkeypatch):
        """Planner calls with_structured_output and uses the returned plan."""
        import matsim_agents.llm as llm_mod

        task = TaskSpec(structure_path=si_vasp)
        plan_obj = MagicMock()
        plan_obj.tasks = [task]
        plan_obj.rationale = "test rationale"

        fake_structured = MagicMock()
        fake_structured.invoke.return_value = plan_obj

        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured

        monkeypatch.setattr(llm_mod, "get_chat_model", lambda **_: fake_llm)

        state = _state(
            objective=f"Relax {si_vasp}",
            llm_provider="vllm",
            llm_model="test-model",
        )
        from matsim_agents.agents.planner import planner_node

        result = planner_node(state)
        assert len(result["plan"]) == 1
        assert result["plan"][0].structure_path == si_vasp
        assert "test rationale" in result["messages"][0].content

    def test_planner_message_recorded(self, si_vasp, monkeypatch):
        import matsim_agents.llm as llm_mod

        task = TaskSpec(structure_path=si_vasp)
        plan_obj = MagicMock(tasks=[task], rationale="")
        fake_structured = MagicMock()
        fake_structured.invoke.return_value = plan_obj
        fake_llm = MagicMock()
        fake_llm.with_structured_output.return_value = fake_structured
        monkeypatch.setattr(llm_mod, "get_chat_model", lambda **_: fake_llm)

        state = _state(objective=f"Relax {si_vasp}", llm_provider="vllm")
        from matsim_agents.agents.planner import planner_node

        result = planner_node(state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)


# ── analyst node ──────────────────────────────────────────────────────────────


class TestAnalystNode:
    """analyst_node() summarizes results deterministically and via LLM."""

    def test_no_results_returns_message(self):
        from matsim_agents.agents.analyst import analyst_node

        result = analyst_node(_state())
        assert "No relaxation results" in result["analysis"]

    def test_deterministic_summary_picks_best(self, fake_relaxation_result):
        from matsim_agents.agents.analyst import _deterministic_summary

        # Two results; summary should pick the lower energy one
        worse = RelaxationResult(
            structure_path="b.vasp",
            optimized_structure_path="b_opt.vasp",
            trajectory_path="b.traj",
            log_csv_path="b.csv",
            final_energy_eV=-1.0,
            final_max_force_eV_per_A=0.05,
            num_steps=10,
            converged=False,
        )
        state = _state(results=[fake_relaxation_result, worse])
        summary = _deterministic_summary(state)
        assert "-5.432" in summary  # best energy from fake_relaxation_result
        assert "2 relaxation" in summary

    def test_analyst_uses_llm_when_available(self, fake_relaxation_result, monkeypatch):
        import matsim_agents.llm as llm_mod

        fake_llm = FakeListChatModel(responses=["LLM-generated analysis."])
        monkeypatch.setattr(llm_mod, "get_chat_model", lambda **_: fake_llm)

        state = _state(results=[fake_relaxation_result], llm_provider="vllm")
        from matsim_agents.agents.analyst import analyst_node

        result = analyst_node(state)
        assert result["analysis"] == "LLM-generated analysis."
        assert "LLM-generated analysis." in result["messages"][0].content

    def test_analyst_fallback_on_llm_error(self, fake_relaxation_result, monkeypatch):
        import matsim_agents.llm as llm_mod

        def _fail(**_):
            raise RuntimeError("LLM unavailable")

        monkeypatch.setattr(llm_mod, "get_chat_model", _fail)
        state = _state(results=[fake_relaxation_result], llm_provider="vllm")
        from matsim_agents.agents.analyst import analyst_node

        result = analyst_node(state)
        # Should fall back to deterministic summary silently
        assert result["analysis"] is not None
        assert "-5.432" in result["analysis"]


# ── graph routing ─────────────────────────────────────────────────────────────


class TestGraphRouting:
    """The conditional edge routes correctly based on pending_tasks/iteration."""

    def test_routes_to_executor_when_pending(self):
        from matsim_agents.graph import _route_after_executor

        state = _state(pending_tasks=[TaskSpec(structure_path="a.vasp")], iteration=0)
        assert _route_after_executor(state) == "executor"

    def test_routes_to_analyst_when_no_pending(self):
        from matsim_agents.graph import _route_after_executor

        state = _state(pending_tasks=[], iteration=0)
        assert _route_after_executor(state) == "analyst"

    def test_routes_to_analyst_when_max_iterations_hit(self):
        from matsim_agents.graph import _route_after_executor

        state = _state(
            pending_tasks=[TaskSpec(structure_path="a.vasp")],
            iteration=5,
            max_iterations=5,
        )
        assert _route_after_executor(state) == "analyst"


# ── full graph end-to-end (executor mocked) ───────────────────────────────────


class TestFullGraphEndToEnd:
    """Run the compiled graph with the executor node mocked out."""

    @pytest.fixture
    def patched_graph(self, si_vasp, fake_relaxation_result, monkeypatch):
        """Build the graph with planner/analyst using fake LLM and executor mocked."""
        import matsim_agents.llm as llm_mod

        task = TaskSpec(structure_path=si_vasp)
        plan_obj = MagicMock(tasks=[task], rationale="Auto-planned relaxation.")
        fake_structured = MagicMock()
        fake_structured.invoke.return_value = plan_obj
        fake_planner_llm = MagicMock()
        fake_planner_llm.with_structured_output.return_value = fake_structured

        def _fake_get_chat_model(**kwargs):
            # Planner calls with_structured_output; analyst calls invoke directly.
            # Return the appropriate stub based on what's been called so far.
            return fake_planner_llm

        monkeypatch.setattr(llm_mod, "get_chat_model", _fake_get_chat_model)

        # Replace executor node with a stub that marks the task done
        def _fake_executor(state: MatSimState, config=None):
            if not state.pending_tasks:
                return {}
            return {
                "pending_tasks": state.pending_tasks[1:],
                "results": [fake_relaxation_result],
                "iteration": state.iteration + 1,
                "messages": [AIMessage(content="[executor] fake relaxation done.")],
            }

        from matsim_agents import graph as graph_mod

        monkeypatch.setattr(graph_mod, "executor_node", _fake_executor)

        from matsim_agents.graph import build_graph

        return build_graph()

    def test_full_graph_completes(self, patched_graph, si_vasp):
        """Graph runs from planner to analyst and produces a final analysis."""
        initial = _state(
            objective=f"Relax the structure at {si_vasp}",
            llm_provider="vllm",
        )
        config = {"configurable": {"thread_id": "test-1"}}
        final = patched_graph.invoke(initial.model_dump(), config=config)

        assert final["analysis"] is not None
        assert len(final["results"]) == 1
        assert final["iteration"] == 1

    def test_graph_messages_accumulate(self, patched_graph, si_vasp):
        """Each node appends messages; final state has messages from all nodes."""
        initial = _state(
            objective=f"Relax the structure at {si_vasp}",
            llm_provider="vllm",
        )
        config = {"configurable": {"thread_id": "test-2"}}
        final = patched_graph.invoke(initial.model_dump(), config=config)

        contents = [m["content"] if isinstance(m, dict) else m.content for m in final["messages"]]
        full_text = " ".join(contents)
        assert "planner" in full_text.lower()
        assert "executor" in full_text.lower() or "analyst" in full_text.lower()

    def test_graph_max_iterations_cap(self, si_vasp, fake_relaxation_result, monkeypatch):
        """Graph stops when max_iterations is reached even with pending tasks."""
        import matsim_agents.llm as llm_mod

        # Plan with 3 tasks but max_iterations=1
        tasks = [TaskSpec(structure_path=si_vasp) for _ in range(3)]
        plan_obj = MagicMock(tasks=tasks, rationale="3 tasks")
        fake_structured = MagicMock()
        fake_structured.invoke.return_value = plan_obj
        fake_planner_llm = MagicMock()
        fake_planner_llm.with_structured_output.return_value = fake_structured
        monkeypatch.setattr(llm_mod, "get_chat_model", lambda **_: fake_planner_llm)

        call_count = {"n": 0}

        import matsim_agents.graph as graph_mod

        def _counting_executor(state: MatSimState, config=None):
            call_count["n"] += 1
            if not state.pending_tasks:
                return {}
            return {
                "pending_tasks": state.pending_tasks[1:],
                "results": [fake_relaxation_result],
                "iteration": state.iteration + 1,
                "messages": [AIMessage(content="[executor] done.")],
            }

        monkeypatch.setattr(graph_mod, "executor_node", _counting_executor)

        from matsim_agents.graph import build_graph

        graph = build_graph()
        initial = _state(
            objective=f"Relax {si_vasp}",
            llm_provider="vllm",
            max_iterations=1,
        )
        config = {"configurable": {"thread_id": "test-cap"}}
        final = graph.invoke(initial.model_dump(), config=config)

        # Executor was called at most max_iterations times
        assert call_count["n"] <= 1
        assert final["analysis"] is not None
