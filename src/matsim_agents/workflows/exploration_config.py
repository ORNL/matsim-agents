"""Validated exploration settings, path resolution, and no-compute preflight."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.mlip.relaxation import RelaxStructureInput
from matsim_agents.discovery.seeds import SeedGenerationAction
from matsim_agents.execution.ledger import file_sha256


class ExplorationGenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    formula_units: int = Field(1, gt=0)
    max_aflow_candidates: int = Field(1, ge=0)
    pyxtal_count: int = Field(1, ge=0)
    space_groups: list[int] = Field(default_factory=list)
    random_seed: int = 0


class ExplorationRelaxationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer: Literal["FIRE", "BFGS", "BFGSLineSearch"] = "FIRE"
    maxiter: int = Field(500, gt=0)
    maxstep: float = Field(0.05, gt=0, allow_inf_nan=False)
    fmax: float = Field(0.02, gt=0, le=0.05, allow_inf_nan=False)
    max_optimization_seconds: float = Field(60, gt=0, allow_inf_nan=False)
    relative_increase_threshold: float | None = Field(0.2, ge=0, allow_inf_nan=False)
    relax_cell: bool = False
    cell_mask: tuple[bool, bool, bool, bool, bool, bool] | None = None
    stress_tolerance_eV_per_A3: float | None = Field(None, gt=0, allow_inf_nan=False)
    max_cell_step_strain: float | None = Field(None, gt=0, allow_inf_nan=False)
    max_cell_total_strain: float | None = Field(None, gt=0, lt=1, allow_inf_nan=False)
    require_native_stress: bool = False

    @model_validator(mode="after")
    def _validate_cell_options(self) -> ExplorationRelaxationConfig:
        RelaxStructureInput(
            structure_path="validation-only.extxyz",
            mlip_backend="mace",
            **self.model_dump(),
        )
        return self


class ExplorationConfig(BaseModel):
    """Bounded MLIP exploration; never launches LLM, DFT, or AL work."""

    model_config = ConfigDict(extra="forbid")

    formula: str
    checkpoint: Path
    checkpoint_sha256: str | None = Field(None, pattern=r"^[a-f0-9]{64}$")
    output_dir: Path = Path("./runs")
    device: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["fp32", "fp64"] = "fp64"
    max_relaxations: int = Field(2, gt=0)
    iterations: int = Field(1, ge=1, le=1000)
    selection: Literal["sequential", "diverse"] = "sequential"
    generation: ExplorationGenerationConfig = Field(default_factory=ExplorationGenerationConfig)
    relaxation: ExplorationRelaxationConfig = Field(default_factory=ExplorationRelaxationConfig)

    @model_validator(mode="after")
    def _validate_generation(self) -> ExplorationConfig:
        self.formula = self.generation_action().formula
        return self

    def generation_action(self) -> SeedGenerationAction:
        return SeedGenerationAction(
            action_id="exploration",
            formula=self.formula,
            **self.generation.model_dump(),
        )

    def relaxation_request(self, structure_path: str, output_dir: str) -> RelaxStructureInput:
        """Construct and validate the same scientific request for every attempt."""
        return RelaxStructureInput(
            structure_path=structure_path,
            output_dir=output_dir,
            mlip_backend="mace",
            mace_family="checkpoint",
            mace_model=str(self.checkpoint),
            mlp_device=self.device,
            precision=self.precision,
            **self.relaxation.model_dump(),
        )


class ExplorationCheck(BaseModel):
    ready: bool
    python: str
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    checkpoint_sha256: str | None = None


def load_exploration_config(path: str | Path) -> ExplorationConfig:
    """Resolve data paths relative to the YAML file, not the caller's directory."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Exploration configuration must be a YAML or JSON mapping")
    cfg = ExplorationConfig.model_validate(raw)
    for field in ("checkpoint", "output_dir"):
        value = getattr(cfg, field).expanduser()
        if not value.is_absolute():
            value = path.parent / value
        setattr(cfg, field, value.resolve())
    return cfg


def check_exploration(cfg: ExplorationConfig) -> ExplorationCheck:
    """Check configuration, imports, and checkpoint without importing/loading MACE."""
    errors = []
    for module in ("ase", "numpy", "torch", "mace", "pymatgen"):
        if importlib.util.find_spec(module) is None:
            errors.append(
                f"Missing {module} in {sys.executable}. Use the prepared .venv-mace "
                "interpreter or install the documented MACE dependencies."
            )
    if cfg.generation.pyxtal_count and importlib.util.find_spec("pyxtal") is None:
        errors.append("PyXtal generation requested but pyxtal is not installed.")
    digest = None
    if not cfg.checkpoint.is_file():
        errors.append(f"Checkpoint does not exist: {cfg.checkpoint}")
    else:
        digest = file_sha256(cfg.checkpoint)
        if cfg.checkpoint_sha256 is not None and digest != cfg.checkpoint_sha256:
            errors.append("Checkpoint SHA-256 does not match checkpoint_sha256.")
    notes = [
        "No model was loaded, structures generated, LLM called, or job submitted.",
        "Import availability is not an ABI, authentication, or device qualification.",
        "Use a compute allocation for extended workloads; this command does not schedule one.",
        "Only load a trusted checkpoint. PyTorch checkpoints may contain executable code.",
    ]
    if cfg.device == "cuda":
        notes.append("CUDA visibility and compatibility will be checked when MACE loads.")
    if cfg.checkpoint_sha256 is None:
        notes.append(
            "No expected checkpoint hash supplied; pin the reported hash for reproduction."
        )
    return ExplorationCheck(
        ready=not errors,
        python=sys.executable,
        errors=errors,
        notes=notes,
        checkpoint_sha256=digest,
    )
