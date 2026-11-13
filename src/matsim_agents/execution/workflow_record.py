"""Human-facing records for native workflow invocations, not a scientific engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from matsim_agents.execution.run_directory import ScientificRunDirectory


class WorkflowInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workflow: str
    command: list[str]
    working_directory: Path
    state: Literal["running", "completed", "failed", "interrupted"]
    return_code: int | None = None
    failure_reason: str | None = None
    native_run_directory: str | None = None
    configuration_sha256: dict[str, str]
    execution_approved: bool = False

    @field_validator("working_directory")
    @classmethod
    def _absolute_working_directory(cls, path: Path) -> Path:
        if not path.is_absolute():
            raise ValueError("The native working directory must be absolute")
        return path


def read_invocation(directory: Path) -> WorkflowInvocation:
    return WorkflowInvocation.model_validate_json((directory / "workflow-state.json").read_text())


def write_invocation(run: ScientificRunDirectory, record: WorkflowInvocation) -> None:
    run.write_json("workflow-state.json", record)
    run.write_json("workflow-summary.json", record)
    report = [
        f"# {record.workflow} workflow report",
        "",
        f"- Invocation state: **{record.state}**",
        f"- Native command exit code: {record.return_code}",
        f"- Explicit execution approval: {record.execution_approved}",
        f"- Native working directory: `{record.working_directory}`",
        "",
        "## Scientific interpretation",
        "",
        "Invocation completion is not scientific convergence or thermodynamic stability.",
        "Read the native evidence and convergence criteria before making scientific claims.",
        "",
        "## Evidence",
        "",
        "- [Saved configuration](resolved_config.json)",
        "- [Invocation state](workflow-state.json)",
        "- [Native command output](workflow-output.log)",
        "",
    ]
    if record.native_run_directory:
        report.append(f"Native scientific run: `{record.native_run_directory}`")
    if record.failure_reason:
        report.extend(["", "## Failure", "", record.failure_reason])
    report.extend(
        [
            "",
            "## Next actions",
            "",
            "- Inspect the native output log for scientific stop reasons and artifact locations.",
            "- Use `workflow status RUN_DIRECTORY` or `workflow summary RUN_DIRECTORY`.",
            "- Use `workflow resume RUN_DIRECTORY` only where the catalog declares native resume.",
            "- Non-resumable workflows require a fresh invocation; "
            "completed failures are not retried here.",
            "",
            "Status reflects the last saved invocation record, "
            "not a scheduler query or liveness probe.",
        ]
    )
    run.write_text("report.md", "\n".join(report) + "\n")


def saved_workflow_config(directory: Path) -> dict:
    return json.loads((directory / "resolved_config.json").read_text())
