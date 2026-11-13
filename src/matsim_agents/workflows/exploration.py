"""Small, config-driven MACE phase searches using the existing workflow engine."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.mlip.relaxation import (
    CalculatorCache,
    RelaxStructureInput,
    relax_with_mlip,
)
from matsim_agents.discovery.seeds import SeedGenerationAction
from matsim_agents.execution.ledger import file_sha256
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.workflows.phase_exploration import (
    PhaseExplorationEvidenceRecord,
    PhaseExplorationPolicy,
    run_phase_exploration,
)


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
        if self.selection == "diverse" and self.iterations == 1:
            raise ValueError("Diverse selection requires checkpointed rounds: set iterations > 1")
        self.formula = self.generation_action().formula
        return self

    def generation_action(self) -> SeedGenerationAction:
        return SeedGenerationAction(
            action_id="exploration",
            formula=self.formula,
            **self.generation.model_dump(),
        )


class ExplorationCheck(BaseModel):
    ready: bool
    python: str
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    checkpoint_sha256: str | None = None


class ExplorationSummary(BaseModel):
    run_directory: str
    status: Literal["completed", "failed"]
    outcome: str
    counts: dict[str, int]
    termination_reasons: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    calculator_loads: int | None = None
    rounds_completed: int = 0
    rounds_requested: int = 1


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


def summarize_exploration(run_directory: str | Path) -> ExplorationSummary:
    """Read the canonical persisted evidence, without importing a model."""
    run = ScientificRunDirectory.open(run_directory)
    if (run.path / "rounds-state.json").is_file():
        from matsim_agents.workflows.exploration_rounds import summarize_rounds

        return summarize_rounds(run)
    record = PhaseExplorationEvidenceRecord.model_validate_json(
        (run.path / "results.json").read_text(encoding="utf-8")
    )
    initial = record.result.initial if record.result is not None else None
    reasons: dict[str, int] = {}
    if initial is not None:
        for result in initial.relaxations:
            reason = result.termination_reason or "not_recorded"
            reasons[reason] = reasons.get(reason, 0) + 1
    errors = [record.failure_reason] if record.failure_reason else []
    notes = []
    if initial is not None:
        errors.extend(initial.failures)
        errors.extend(initial.callback_failures)
        if initial.ranking_failure:
            if initial.outcome_class == "ranking_failure":
                errors.append(initial.ranking_failure)
            else:
                notes.append(initial.ranking_failure)
    counts = initial.candidate_counts if initial is not None else {}
    failed = (
        record.status == "failed"
        or bool(errors)
        or (initial is not None and initial.outcome_class == "generation_failure")
    )
    if initial is not None and initial.outcome_class == "generation_failure":
        errors.append("No candidates were generated; review composition and generation settings.")
    summary = ExplorationSummary(
        run_directory=str(run.path),
        status="failed" if failed else "completed",
        outcome=initial.outcome_class if initial is not None else "workflow_failure",
        counts=counts,
        termination_reasons=reasons,
        errors=errors,
        notes=notes,
    )
    if (run.path / "exploration-summary.json").is_file():
        saved = ExplorationSummary.model_validate_json(
            (run.path / "exploration-summary.json").read_text(encoding="utf-8")
        )
        summary.calculator_loads = saved.calculator_loads
    return summary


def run_exploration(
    cfg: ExplorationConfig, *, on_progress: Callable[[str], None] | None = None
) -> ExplorationSummary:
    """Use one calculator cache across a bounded batch and persist an honest summary."""
    if cfg.iterations > 1:
        from matsim_agents.workflows.exploration_rounds import run_rounds

        return run_rounds(cfg, on_progress=on_progress)
    check = check_exploration(cfg)
    if not check.ready:
        raise ValueError("Exploration preflight failed:\n" + "\n".join(check.errors))
    cache = CalculatorCache(capacity=1)

    def relax(request: RelaxStructureInput):
        # Override all scientific options at this boundary, then validate again.
        # The discovery wrapper still owns generation, budgets, and ranking.
        request = RelaxStructureInput.model_validate(
            {
                **request.model_dump(),
                **cfg.relaxation.model_dump(),
                "precision": cfg.precision,
            }
        )
        return relax_with_mlip(request, calculators=cache)

    result = run_phase_exploration(
        cfg.formula,
        policy=PhaseExplorationPolicy(),
        output_dir=str(cfg.output_dir),
        request_config={
            "config": cfg.model_dump(mode="json"),
            "preflight": check.model_dump(mode="json"),
        },
        exploration_kwargs={
            "mlip_backend": "mace",
            "mace_family": "checkpoint",
            "mace_model": str(cfg.checkpoint),
            "mlp_device": cfg.device,
            "precision": cfg.precision,
            "generation_action": cfg.generation_action(),
            "max_relaxations": cfg.max_relaxations,
            "optimizer": cfg.relaxation.optimizer,
            "maxiter": cfg.relaxation.maxiter,
            "maxstep": cfg.relaxation.maxstep,
            "fmax": cfg.relaxation.fmax,
            "relax_cell": cfg.relaxation.relax_cell,
            "relative_increase_threshold": cfg.relaxation.relative_increase_threshold,
            "relax_fn": relax,
        },
    )
    if result.run_directory is None:
        raise RuntimeError("Phase exploration did not return its persisted run directory")
    run = ScientificRunDirectory.open(result.run_directory)
    run.write_json("exploration-config.json", cfg)
    run.write_json("exploration-preflight.json", check)
    summary = summarize_exploration(run.path)
    summary.calculator_loads = cache.loads
    run.write_json("exploration-summary.json", summary)
    return summary


def resume_exploration(
    run_directory: str | Path, *, on_progress: Callable[[str], None] | None = None
) -> ExplorationSummary:
    """Resume a checkpointed multi-round run without repeating completed attempts."""
    from matsim_agents.workflows.exploration_rounds import run_rounds

    return run_rounds(None, run_directory=run_directory, on_progress=on_progress)
