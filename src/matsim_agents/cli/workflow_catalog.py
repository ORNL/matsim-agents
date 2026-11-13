"""Native command adapters for the common workflow interface.

This catalog does not implement numerical execution or invent restart semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True)
class WorkflowAdapter:
    command: tuple[str, ...]
    config_argument: bool = False
    resume: Literal["directory", "config", "chat", "e1", "unsupported"] = "unsupported"
    description: str = ""


ADAPTERS = {
    "explore": WorkflowAdapter(("explore", "run"), True, "directory", "MACE phase exploration"),
    "relax": WorkflowAdapter(("relax",), True, description="MLIP/DFT relaxation"),
    "characterize": WorkflowAdapter(("characterize",), True, "config", "Characterization recipes"),
    "debate": WorkflowAdapter(("debate",), True, description="Scientific LLM debate"),
    "llm-check": WorkflowAdapter(("llm-check",), True, description="LLM readiness qualification"),
    "al": WorkflowAdapter(("al", "run"), True, "config", "Active learning"),
    "objective": WorkflowAdapter(("run",), description="Objective-driven agent graph"),
    "plan": WorkflowAdapter(("plan",), description="Objective planning"),
    "supervisor": WorkflowAdapter(("supervisor-run",), description="Composition supervisor"),
    "chat": WorkflowAdapter(("chat",), resume="chat", description="Interactive discovery"),
    "e1": WorkflowAdapter(("e1-development",), True, "e1", "Development screening"),
    "known-answers": WorkflowAdapter(
        ("benchmark", "known-answers"), description="Known-answer suite"
    ),
    "agent-eval": WorkflowAdapter(
        ("benchmark", "agent-eval"), True, description="Agent evaluation"
    ),
    "e3-mlip": WorkflowAdapter(
        (), resume="config", description="E3-MLIP three-round production benchmark"
    ),
    "e3-dft": WorkflowAdapter(
        (), description="Frozen E3/VASP batch validation or explicit submission"
    ),
}


class E3MLIPSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: Path
    output_root: Path
    material: Literal[
        "bcc_refractory", "fcc_cantor", "lifepo4", "phosphorene", "nb_ta_o", "zn_formate_mof"
    ]
    strategy: Literal["llm_guided_adaptation"] = "llm_guided_adaptation"
    seed: Literal[20261113] = 20261113
    workers: Literal[64] = 64
    workers_per_node: Literal[4] = 4
    worker_python: Path | None = None
    resource_state: Path
    hard_optimization_seconds: Literal[300] = 300


class E3DFTSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: Path
    calculation_ids: list[str] = Field(default_factory=list)


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow: str
    config: Path | None = None
    arguments: list[str] = Field(default_factory=list)
    output_root: Path = Path("./workflow-runs")
    settings: E3MLIPSettings | E3DFTSettings | None = None

    @model_validator(mode="after")
    def _consistent(self) -> WorkflowConfig:
        adapter = ADAPTERS.get(self.workflow)
        if adapter is None:
            raise ValueError(f"Unknown workflow {self.workflow!r}; use workflow list")
        if adapter.config_argument and self.config is None:
            raise ValueError(f"{self.workflow} requires config: path/to/native-config")
        if self.workflow.startswith("e3-"):
            expected = E3MLIPSettings if self.workflow == "e3-mlip" else E3DFTSettings
            if not isinstance(self.settings, expected):
                raise ValueError(f"{self.workflow} requires its campaign settings")
            if self.config is not None or self.arguments:
                raise ValueError("E3 presets use settings, not config/arguments")
        elif self.settings is not None:
            raise ValueError("settings is reserved for the E3 presets")
        elif not adapter.config_argument and self.config is not None:
            raise ValueError(f"{self.workflow} uses arguments, not config")
        return self


def load_workflow(path: Path) -> WorkflowConfig:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Workflow configuration must be a mapping")
    cfg = WorkflowConfig.model_validate(raw)
    for field in ("config", "output_root"):
        value = getattr(cfg, field)
        if value is not None:
            value = value.expanduser()
            setattr(cfg, field, (path.parent / value).resolve())
    if cfg.settings is not None:
        for field in ("manifest", "output_root", "worker_python", "resource_state"):
            value = getattr(cfg.settings, field, None)
            if value is not None:
                value = value.expanduser()
                setattr(cfg.settings, field, (path.parent / value).resolve())
    return cfg


def native_arguments(cfg: WorkflowConfig) -> list[str]:
    adapter = ADAPTERS[cfg.workflow]
    return [
        *adapter.command,
        *([str(cfg.config)] if adapter.config_argument else []),
        *cfg.arguments,
    ]
