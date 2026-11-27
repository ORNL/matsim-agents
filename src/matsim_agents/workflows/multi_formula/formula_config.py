"""Numerical configuration shared by every formula in a campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.mlip.config import MLIPConfig
from matsim_agents.science.thermodynamics.stability import (
    ReferenceCompletenessPolicy,
    ReferenceEnergySet,
)
from matsim_agents.workflows.candidate_registry import (
    CandidateSelectionPolicy,
)
from matsim_agents.workflows.exploration.phase import (
    PhaseExplorationPolicy,
)


class CampaignFormulaExecutionConfig(BaseModel):
    """Reusable numerical configuration for every formula in a campaign."""

    # Unknown keys fail, so removed seed fields cannot be silently ignored.
    model_config = ConfigDict(extra="forbid")

    active_learning_config: Path
    # Replaces the AL config's mlip block, so any backend drives the campaign.
    mlip: MLIPConfig | None = None
    phase_policy: PhaseExplorationPolicy
    exploration_kwargs: dict[str, Any] = Field(default_factory=dict)
    compute_nodes: int = Field(1, ge=1)
    execution_mode: Literal["dft", "mlip_only"] = "dft"
    retraining: CampaignRetrainingConfig | None = None
    dft_refinement: CampaignDFTRefinementConfig | None = None
    model_override: str | None = None
    validation_configs: list[Path] = Field(default_factory=list)
    validation_python: Path | None = None
    validation_pythons: list[Path] = Field(default_factory=list)
    perturbation_trials: int = Field(0, ge=0)
    perturbation_scale_A: float = Field(0.05, gt=0)
    surrogate_reference_structures: Path | None = None
    surrogate_unary_max_steps: int = Field(200, ge=1)
    surrogate_unary_fmax_eV_per_A: float = Field(0.02, gt=0)
    surrogate_unary_maxstep_A: float = Field(0.01, gt=0)
    surrogate_minimum_unique_unary: int = Field(2, ge=1)
    max_dft_calculations: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _validate_execution_mode(self) -> CampaignFormulaExecutionConfig:
        if "random_seed" in self.exploration_kwargs:
            raise ValueError(
                "exploration_kwargs must not set random_seed: each formula attempt's seed "
                "comes from the campaign seed"
            )
        missing = [str(path) for path in self.validation_configs if not path.is_file()]
        if missing:
            raise ValueError(f"validation config files do not exist: {missing}")
        if self.validation_python is not None and not self.validation_python.is_file():
            raise ValueError(f"validation Python does not exist: {self.validation_python}")
        missing_pythons = [str(path) for path in self.validation_pythons if not path.is_file()]
        if missing_pythons:
            raise ValueError(f"validation Python interpreters do not exist: {missing_pythons}")
        if self.validation_pythons and len(self.validation_pythons) not in {
            1,
            len(self.validation_configs),
        }:
            raise ValueError(
                "validation_pythons must contain one shared interpreter or align "
                "with validation_configs"
            )
        if (
            self.surrogate_reference_structures is not None
            and not self.surrogate_reference_structures.is_file()
        ):
            raise ValueError(
                "surrogate reference manifest does not exist: "
                f"{self.surrogate_reference_structures}"
            )
        if self.execution_mode == "mlip_only":
            if self.phase_policy.active_learning or self.phase_policy.retrain_mlip:
                raise ValueError("mlip_only execution disables active learning and retraining")
            if self.retraining is not None or self.dft_refinement is not None:
                raise ValueError(
                    "mlip_only execution cannot configure retraining or DFT refinement"
                )
        return self


class CampaignRetrainingConfig(BaseModel):
    """Explicit training and promotion controls for campaign MLIPs."""

    model_config = ConfigDict(extra="forbid")

    train_launcher: Path | None = None
    epochs: int = Field(5, ge=1)
    promote_model: bool = False
    promotion_approved: bool = False
    validation_set: Path | None = None
    validation_fraction: float = Field(0.0, ge=0.0, lt=1.0)
    validation_reference_set: Path | None = None
    promotion_max_energy_mae_eV_per_atom: float = Field(0.1, gt=0)
    promotion_max_force_mae_eV_per_A: float = Field(0.2, gt=0)
    promotion_min_relative_improvement: float = Field(
        0.05,
        ge=0,
        le=1,
        description="Minimum relative MAE reduction in either metric; neither may regress.",
    )
    promotion_min_evaluated_frames: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _validate_training_paths_and_approval(self) -> CampaignRetrainingConfig:
        if self.train_launcher is not None and not self.train_launcher.is_file():
            raise ValueError(f"training launcher does not exist: {self.train_launcher}")
        if self.promote_model and not self.promotion_approved:
            raise ValueError("model promotion requires explicit approval")
        if self.validation_set is not None and self.validation_fraction > 0:
            raise ValueError("validation_set and validation_fraction are mutually exclusive")
        if self.promote_model and self.validation_set is None and self.validation_fraction <= 0:
            raise ValueError(
                "model promotion requires a held-out validation_set or validation_fraction"
            )
        return self


class ReferenceStructureSpec(BaseModel):
    """A structure that must be calculated as part of the reference hull."""

    model_config = ConfigDict(extra="forbid")

    phase_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    formula: str
    structure_path: Path
    relax_cell: bool | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    source: str = "user_supplied"
    provenance: dict[str, str] = Field(default_factory=dict)
    energy_correction_eV_per_atom: float = 0.0


class CampaignDFTRefinementConfig(BaseModel):
    """DFT relaxation and compatible reference generation for hull ranking."""

    model_config = ConfigDict(extra="forbid")

    method_signature: str
    reference_phases: list[ReferenceStructureSpec] = Field(default_factory=list)
    reference_completeness_policy: ReferenceCompletenessPolicy = Field(
        default_factory=ReferenceCompletenessPolicy
    )
    reference_energies: ReferenceEnergySet | None = None
    max_candidates: int = Field(1, ge=1)
    relax_cell: bool = True
    max_steps: int = Field(100, ge=1)
    force_tolerance_eV_per_A: float = Field(0.02, gt=0)
    candidate_acquisition: CandidateSelectionPolicy = Field(
        default_factory=CandidateSelectionPolicy
    )

    @model_validator(mode="after")
    def _validate_references(self) -> CampaignDFTRefinementConfig:
        missing = [
            str(spec.structure_path)
            for spec in self.reference_phases
            if not spec.structure_path.is_file()
        ]
        if missing:
            raise ValueError(f"reference structures do not exist: {missing}")
        if (
            self.reference_energies is not None
            and self.reference_energies.method_signature != self.method_signature
        ):
            raise ValueError("reference energies and DFT refinement must share method_signature")
        return self
