"""Validated exploration settings, path resolution, and no-compute preflight."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.mlip.config import MLIPConfig, resolve_mlip_block
from matsim_agents.backends.mlip.relaxation import RelaxationSettings, RelaxStructureInput
from matsim_agents.core.hashing import sha256_file as file_sha256
from matsim_agents.core.yamlio import load_yaml_mapping
from matsim_agents.science.structures.seed_structures import SeedGenerationAction


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
        RelaxationSettings(**self.model_dump())
        return self


class ExplorationConfig(BaseModel):
    """Bounded MLIP exploration with any supported backend.

    The ``mlip`` block selects the model (``backend: hydragnn | uma | mace``
    plus that backend's settings, including device and precision), exactly as
    in active-learning configs. Never launches LLM, DFT, or AL work.
    """

    model_config = ConfigDict(extra="forbid")

    formula: str
    mlip: MLIPConfig
    checkpoint_sha256: str | None = Field(None, pattern=r"^[a-f0-9]{64}$")
    output_dir: Path = Path("./runs")
    max_relaxations: int = Field(2, gt=0)
    iterations: int = Field(1, ge=1, le=1000)
    selection: Literal["sequential", "diverse"] = "sequential"
    generation: ExplorationGenerationConfig = Field(default_factory=ExplorationGenerationConfig)
    relaxation: ExplorationRelaxationConfig = Field(default_factory=ExplorationRelaxationConfig)

    @model_validator(mode="after")
    def _validate_generation(self) -> ExplorationConfig:
        self.formula = self.generation_action().formula
        return self

    @property
    def backend(self) -> str:
        return self.mlip.backend

    def generation_action(self) -> SeedGenerationAction:
        return SeedGenerationAction(
            action_id="exploration",
            formula=self.formula,
            **self.generation.model_dump(),
        )

    def model_identifier(self) -> str:
        return self.mlip.model_identifier()

    def relaxation_request(self, structure_path: str, output_dir: str) -> RelaxStructureInput:
        """Construct and validate the same scientific request for every attempt."""
        return RelaxStructureInput(
            structure_path=structure_path,
            output_dir=output_dir,
            mlip=self.mlip,
            **self.relaxation.model_dump(),
        )


class ExplorationCheck(BaseModel):
    ready: bool
    python: str
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    backend: str
    model_identifier: str
    checkpoint_sha256: str | None = None
    # SHA-256 of every local file that determines the model's predictions.
    model_files: dict[str, str] = Field(default_factory=dict)


def load_exploration_config(path: str | Path) -> ExplorationConfig:
    """Resolve data paths relative to the YAML file, not the caller's directory."""
    path = Path(path).resolve()
    raw = load_yaml_mapping(path)
    resolve_mlip_block(raw, path.parent)
    cfg = ExplorationConfig.model_validate(raw)
    output_dir = cfg.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir
    cfg.output_dir = output_dir.resolve()
    return cfg


def check_exploration(cfg: ExplorationConfig) -> ExplorationCheck:
    """Check configuration, imports, and model inputs without importing/loading the MLIP."""
    errors = []
    backend = cfg.backend
    for module in ("ase", "numpy", "torch", "pymatgen", *cfg.mlip.required_modules):
        if importlib.util.find_spec(module) is None:
            errors.append(
                f"Missing {module} in {sys.executable}. Use the interpreter prepared for "
                f"the {backend} backend or install its documented dependencies."
            )
    if cfg.generation.pyxtal_count and importlib.util.find_spec("pyxtal") is None:
        errors.append("PyXtal generation requested but pyxtal is not installed.")
    missing = [path for path in cfg.mlip.required_paths() if not path.exists()]
    errors.extend(f"Required {backend} model file does not exist: {path}" for path in missing)
    if not missing:
        try:
            cfg.mlip.check_loadable()
        except ValueError as error:
            errors.append(f"Invalid {backend} model selection: {error}")
    digest = None
    model_files: dict[str, str] = {}
    try:
        artifact = cfg.mlip.model_artifact()
        files = cfg.mlip.model_files()
    except ValueError as error:
        errors.append(f"Model files cannot be identified: {error}")
        artifact, files = None, []
    for path in files:
        if path.is_file():
            model_files[str(path)] = file_sha256(path)
        elif not missing:
            errors.append(f"Required {backend} model file does not exist: {path}")
    if artifact is None:
        if cfg.checkpoint_sha256 is not None:
            errors.append(
                "checkpoint_sha256 requires a local model file; this model is selected by name."
            )
    elif artifact.is_file():
        digest = file_sha256(artifact)
        if cfg.checkpoint_sha256 is not None and digest != cfg.checkpoint_sha256:
            errors.append("Checkpoint SHA-256 does not match checkpoint_sha256.")
    notes = [
        "No model was loaded, structures generated, LLM called, or job submitted.",
        "Import availability is not an ABI, authentication, or device qualification.",
        "Use a compute allocation for extended workloads; this command does not schedule one.",
        "Only load a trusted checkpoint. PyTorch checkpoints may contain executable code.",
    ]
    device = cfg.mlip.device
    if device in {"cuda", "xpu"}:
        notes.append(f"{device.upper()} visibility and compatibility will be checked at load.")
    if artifact is None:
        notes.append(
            "The model is identified by its configuration; select a local model file to "
            "pin its SHA-256."
        )
    elif cfg.checkpoint_sha256 is None:
        notes.append(
            "No expected checkpoint hash supplied; pin the reported hash for reproduction."
        )
    return ExplorationCheck(
        ready=not errors,
        python=sys.executable,
        errors=errors,
        notes=notes,
        backend=backend,
        model_identifier=cfg.model_identifier(),
        checkpoint_sha256=digest,
        model_files=model_files,
    )
