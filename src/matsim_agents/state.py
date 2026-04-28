"""Typed shared state passed between LangGraph nodes."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RelaxationResult(BaseModel):
    """Outcome of a single atomistic structure relaxation."""

    structure_path: str
    optimized_structure_path: str
    trajectory_path: str
    log_csv_path: str
    final_energy_eV: float
    final_max_force_eV_per_A: float
    num_steps: int
    converged: bool
    top_branch: int | None = None
    top_branch_weight: float | None = None
    notes: str | None = None


class TaskSpec(BaseModel):
    """A single unit of work the planner emits for the executor."""

    kind: Literal["relax", "analyze", "report"] = "relax"
    structure_path: str
    optimizer: Literal["FIRE", "BFGS", "BFGSLineSearch"] = "FIRE"
    maxiter: int = 200
    maxstep: float = 1e-2
    charge: float = 0.0
    spin: float = 0.0
    random_displacement: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class MatSimState(BaseModel):
    """Shared state for the materials-discovery agent graph.

    Fields annotated with reducers (`add`, `add_messages`) accumulate across
    node returns; plain fields are overwritten.
    """

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    objective: str = ""
    plan: list[TaskSpec] = Field(default_factory=list)
    pending_tasks: list[TaskSpec] = Field(default_factory=list)
    results: Annotated[list[RelaxationResult], add] = Field(default_factory=list)
    analysis: str | None = None
    iteration: int = 0
    max_iterations: int = 5

    # LLM selection (None -> fall back to env / library defaults)
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None

    model_config = {"arbitrary_types_allowed": True}
