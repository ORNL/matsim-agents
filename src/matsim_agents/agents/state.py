"""Typed shared state passed between LangGraph nodes."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_serializer, model_validator

from matsim_agents.backends.mcp.records import MCPCallRecord, redact, redact_uri
from matsim_agents.core.relaxation import RelaxationResult


class TaskSpec(BaseModel):
    """A single unit of work the planner emits for the executor.

    ``mcp_tool`` and ``mcp_resource`` tasks name an external MCP server and the
    tool or resource to use; they need no structure or relaxation settings.
    """

    kind: Literal["relax", "analyze", "report", "mcp_tool", "mcp_resource"] = "relax"
    structure_path: str | None = None
    optimizer: Literal["FIRE", "BFGS", "BFGSLineSearch"] = "FIRE"
    maxiter: int = 200
    maxstep: float = 1e-2
    charge: float | None = None  # unset: the selected MLIP's configured charge
    spin: float | None = None  # unset: the selected MLIP's configured spin
    random_displacement: bool = False
    displacement_seed: int | None = None  # required with random_displacement
    extra: dict[str, Any] = Field(default_factory=dict)
    mcp_server: str | None = None
    mcp_tool: str | None = None
    mcp_resource_uri: str | None = None
    mcp_arguments: dict[str, Any] = Field(default_factory=dict)

    # Dumps feed checkpoints and `plan --json`; the executor reads the live fields.
    @field_serializer("mcp_arguments")
    def _redact_arguments(self, value: dict[str, Any]) -> dict[str, Any]:
        return redact(value)

    @field_serializer("mcp_resource_uri")
    def _redact_resource_uri(self, value: str | None) -> str | None:
        return redact_uri(value) if value else value

    @model_validator(mode="after")
    def _fields_for_kind(self) -> TaskSpec:
        if self.kind == "mcp_tool":
            if not self.mcp_server or not self.mcp_tool:
                raise ValueError("mcp_tool tasks require mcp_server and mcp_tool")
        elif self.kind == "mcp_resource":
            if not self.mcp_server or not self.mcp_resource_uri:
                raise ValueError("mcp_resource tasks require mcp_server and mcp_resource_uri")
        elif not self.structure_path:
            raise ValueError(f"{self.kind} tasks require structure_path")
        return self


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
    handoff_events: Annotated[list[str], add] = Field(default_factory=list)
    mcp_results: Annotated[list[MCPCallRecord], add] = Field(default_factory=list)
    analysis: str | None = None
    iteration: int = 0
    max_iterations: int = 5

    # LLM selection (None -> fall back to env / library defaults)
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None

    model_config = {"arbitrary_types_allowed": True}
