"""Pydantic configuration for the active-learning loop.

The full config is intentionally explicit: every knob that affects scientific
results (force cutoffs, MD temperatures, acquisition strategy) is exposed as
a top-level field so the YAML is self-documenting and trivially diffable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from matsim_agents.backends.dft.config import DFTConfig
from matsim_agents.backends.mlip.config import MLIPConfig
from matsim_agents.execution.placement import ResourcePlan


class MDConfig(BaseModel):
    """Parameters for the HydraGNN-driven MD sampler.

    The list of starting structures is supplied via :class:`SeedSourceConfig`
    (``seed_source``).
    """

    seed_source: SeedSourceConfig = Field(
        ...,
        description=(
            "Where the initial seed structures come from. Three modes: "
            "explicit ``paths``, ``compositions`` (formulas \u2192 prototype "
            "seeds), or ``prompt`` (LLM expands a free-text target into "
            "compositions, then prototype seeds)."
        ),
    )
    timestep_fs: float = 1.0
    n_steps: int = 200
    temperature_K: float = 600.0
    thermostat: Literal["langevin", "nvt-berendsen", "nve"] = "langevin"
    friction_inv_ps: float = 0.1
    sample_every: int = 10  # snapshot every N steps for the candidate pool
    max_force_threshold_eV_per_A: float = 50.0  # discard exploding frames
    max_displacement_A: float = 5.0  # discard frames moving > X Å from start
    random_seed: int | None = Field(
        default=None,
        description=(
            "If set, seed the MD RNG (Maxwell-Boltzmann velocities and the "
            "Langevin thermostat noise) so the sampled trajectories and the "
            "resulting candidate pool are reproducible across runs. Each seed "
            "structure gets its own stream derived from this value, so results "
            "do not depend on whether seeds run sequentially or concurrently. "
            "Leave unset for stochastic sampling (the default)."
        ),
    )


class LLMSeedConfig(BaseModel):
    """LLM provider used by ``seed_source.kind='prompt'``.

    Defaults match :func:`matsim_agents.backends.llm.provider.get_chat_model`. On Frontier the
    typical configuration is a vLLM server running e.g. Qwen2.5-72B-Instruct.
    """

    provider: Literal["ollama", "vllm", "openai", "anthropic", "huggingface"] = "ollama"
    model: str = "qwen2.5:14b"
    base_url: str | None = None
    temperature: float = 0.0


class SeedSourceConfig(BaseModel):
    """Where the MD seed structures come from."""

    kind: Literal["paths", "compositions", "prompt"] = "paths"

    # kind == "paths"
    paths: list[Path] = Field(
        default_factory=list,
        description="Explicit list of ASE-readable structure files.",
    )

    # kind == "compositions"
    compositions: list[str] = Field(
        default_factory=list,
        description="Reduced chemical formulas (e.g. ['LiCoO2', 'LiFePO4']).",
    )

    # kind == "prompt"
    prompt: str | None = Field(
        None,
        description=(
            "Free-text materials-discovery target. The LLM will expand it "
            "into a list of formulas which are then turned into prototype seeds."
        ),
    )
    llm: LLMSeedConfig | None = None
    max_compositions: int = Field(
        6,
        description="Cap on number of compositions accepted from the LLM (kind='prompt').",
    )

    # Phase-enumeration knobs (apply to kind='compositions' and 'prompt').
    max_phases_per_composition: int = Field(
        3,
        description=(
            "Per composition, keep at most N prototype-derived seeds. "
            "pyXtal random-search seeds (controlled by ``n_random``) are "
            "appended on top of this cap."
        ),
    )
    n_random: int = Field(
        0,
        description=(
            "Number of supplementary pyXtal random-search seeds per "
            "composition (requires the optional ``pyxtal`` dependency). "
            "Default 0: AL relies on prototype-derived seeds only."
        ),
    )
    random_seed: int = Field(0, description="Seed for pyXtal random search (per composition).")

    @model_validator(mode="after")
    def _check_required_fields(self) -> SeedSourceConfig:
        if self.kind == "paths" and not self.paths:
            raise ValueError("seed_source.kind='paths' requires non-empty 'paths'.")
        if self.kind == "compositions" and not self.compositions:
            raise ValueError("seed_source.kind='compositions' requires non-empty 'compositions'.")
        if self.kind == "prompt":
            if not self.prompt:
                raise ValueError("seed_source.kind='prompt' requires 'prompt'.")
            if self.llm is None:
                # Auto-fill with defaults so the user doesn't *have* to set llm:
                self.llm = LLMSeedConfig()
        return self


# Forward-ref resolution for MDConfig.seed_source.
MDConfig.model_rebuild()


class AcquisitionConfig(BaseModel):
    """Which uncertainty measure(s) to use to pick frames for VASP labelling."""

    strategy: Literal["ensemble", "mc_dropout", "random", "ensemble_then_dropout"] = "ensemble"
    n_select: int = 256  # how many candidates to label per AL iteration
    mc_dropout_passes: int = 8
    mc_dropout_p: float = 0.1
    diversity_filter: bool = True  # greedy-farthest-point on composition + UQ score
    min_uncertainty_eV_per_A: float = 0.0  # below this, skip labelling


class TrainingReferences(BaseModel):
    """Elemental DFT references every fine-tune trains against.

    HydraGNN trains on formation energies relative to them; UMA and MACE use them
    to put the DFT labels on their foundation model's energy scale. Give a
    verified manifest or an approved elemental reference calculation plan.
    """

    manifest: Path | None = None
    phase_plan: Path | None = None
    cache_dir: Path | None = None
    phases_approved: bool = False
    dft_approved: bool = False
    max_dft_calculations: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _reference_source(self) -> TrainingReferences:
        if (self.manifest is None) == (self.phase_plan is None):
            raise ValueError("Specify exactly one training reference manifest or phase_plan")
        source = self.manifest if self.manifest is not None else self.phase_plan
        if source is None or not source.is_file():
            raise ValueError(f"Training reference input does not exist: {source}")
        if self.phase_plan is not None and (self.cache_dir is None or not self.phases_approved):
            raise ValueError("Training reference phase_plan requires cache_dir and phases_approved")
        return self


class TrainerConfig(BaseModel):
    """How to retrain the MLIP at the end of each AL iteration.

    Every backend fine-tunes against elemental DFT references
    (``training_references``, or ``validation_reference_set``), which are
    checked when the configuration is loaded. Leave ``enabled: false`` to keep
    the foundation model and only accumulate labels.
    """

    enabled: bool = Field(
        False,
        description="Opt in to retraining; collecting DFT labels without it is the safe default.",
    )
    promote_model: bool = Field(
        False,
        description=(
            "Use the newly trained candidate in the next AL iteration. This is deliberately "
            "separate from enabled so retraining never silently replaces the deployed model."
        ),
    )
    promotion_approved: bool = Field(
        False,
        description=(
            "Explicit human/policy approval after reviewing candidate validation metrics."
        ),
    )
    validation_set: Path | None = Field(
        None,
        description="Held-out DFT-labelled extxyz required for automatic model promotion.",
    )
    validation_reference_set: Path | None = Field(
        None,
        description=(
            "DFT-labelled pure-element JSON manifest required for energy comparison. "
            "Each model evaluates the same declared geometries before subtracting "
            "its own elemental energies; complete element coverage is required."
        ),
    )
    training_references: TrainingReferences | None = Field(
        None,
        description=(
            "Elemental DFT references for training; validation_reference_set is used when "
            "unset. One of them is required when enabled=True."
        ),
    )
    finetune_mode: Literal["lora", "full"] = Field(
        "lora",
        description=(
            "How UMA and MACE fine-tunes adapt the model: low-rank adapters, which keep "
            "more of the pretraining on small AL datasets, or all weights. HydraGNN "
            "uses its fine-tuner's own strategy."
        ),
    )
    start_from: Literal["incumbent", "foundation"] = Field(
        "incumbent",
        description=(
            "Model each retrain starts from: the current (last promoted) model, or the "
            "configured foundation model, so successive fine-tunes do not compound."
        ),
    )
    validation_fraction: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Target held-out fraction of cumulative accepted labels within this AL dataset. "
            "Only new frames are assigned; existing training/validation membership is fixed."
        ),
    )
    validation_split_seed: int = 0
    compare_after_training: bool = Field(
        False,
        description="Compare incumbent and candidate models on held-out validation frames.",
    )
    promotion_max_energy_mae_eV_per_atom: float = Field(
        0.1, gt=0, description="Maximum held-out formation-energy MAE for promotion."
    )
    promotion_max_force_mae_eV_per_A: float = Field(0.2, gt=0)
    promotion_min_relative_improvement: float = Field(
        0.05,
        ge=0,
        le=1,
        description=(
            "Minimum relative MAE reduction in formation energy or forces for promotion; "
            "neither metric may regress. Zero still requires improvement beyond "
            "numerical tolerance."
        ),
    )
    promotion_min_evaluated_frames: int = Field(1, ge=1)
    train_script: Path | None = Field(
        None,
        description=(
            "Path to a backend training script. Required when enabled=True. "
            "HydraGNN scripts receive --dataset/--logdir/--resume_from in the "
            "direct Python path; UMA scripts receive --dataset/--output-dir/"
            "--base-model/--task-name. Launcher scripts may define their own "
            "site-specific command using the positional arguments passed by "
            "matsim-agents."
        ),
    )
    train_launcher: Path | None = Field(
        None,
        description=(
            "Optional bash launcher that wraps `srun python train.py ...`. If unset, "
            "the trainer falls back to running train_script in the current process."
        ),
    )
    epochs_per_iter: int = 5

    @model_validator(mode="before")
    @classmethod
    def _reject_placement(cls, data: Any) -> Any:
        moved = {
            "nodes_for_train": "resources.pools.training.nodes",
            "ranks_per_node": "resources.pools.training.devices",
        }
        if isinstance(data, dict) and (present := [key for key in moved if key in data]):
            targets = "; ".join(f"trainer.{key} -> {moved[key]}" for key in present)
            raise ValueError(f"placement settings moved to the resources: plan: {targets}")
        return data

    @model_validator(mode="after")
    def _check_train_script(self) -> TrainerConfig:
        for field_name in ("validation_set", "validation_reference_set"):
            validation_path = getattr(self, field_name)
            if validation_path is not None and not validation_path.is_file():
                raise ValueError(
                    f"trainer.{field_name} must be an existing file: {validation_path}"
                )
        if self.enabled and self.train_script is None:
            raise ValueError("trainer.enabled=True requires trainer.train_script.")
        if (
            self.enabled
            and self.training_references is None
            and self.validation_reference_set is None
        ):
            raise ValueError(
                "trainer.enabled=True requires trainer.training_references (or "
                "trainer.validation_reference_set): every fine-tune trains against "
                "elemental DFT references"
            )
        if self.promote_model and not self.enabled:
            raise ValueError("trainer.promote_model=True requires trainer.enabled=True.")
        if self.promote_model and not self.promotion_approved:
            raise ValueError(
                "trainer.promote_model=True requires trainer.promotion_approved=True "
                "after candidate validation."
            )
        if self.promote_model and self.validation_set is None and self.validation_fraction <= 0.0:
            raise ValueError(
                "trainer.promote_model=True requires trainer.validation_set or "
                "trainer.validation_fraction > 0"
            )
        if self.validation_set is not None and self.validation_fraction > 0.0:
            raise ValueError(
                "trainer.validation_set and trainer.validation_fraction are mutually exclusive"
            )
        if (
            self.compare_after_training
            and self.validation_set is None
            and self.validation_fraction <= 0.0
        ):
            raise ValueError(
                "trainer.compare_after_training=True requires trainer.validation_set or "
                "trainer.validation_fraction > 0"
            )
        return self


class LoopConfig(BaseModel):
    """Top-level loop control."""

    n_iterations: int = 10
    max_dft_calculations: int | None = Field(
        None, ge=0, description="Hard cap on DFT labels across all loop iterations."
    )
    out_dir: Path = Field(..., description="Root directory for all AL artefacts.")
    dataset_format: Literal["ase_db", "extxyz"] = "extxyz"
    resume: bool = True
    fail_fast: bool = False  # if True, abort on any VASP failure


# --------------------------------------------------------------------------- #
# Top-level config                                                            #
# --------------------------------------------------------------------------- #


class ALConfig(BaseModel):
    """Root config for the active-learning loop."""

    mlip: MLIPConfig
    md: MDConfig
    acquisition: AcquisitionConfig
    dft: DFTConfig
    trainer: TrainerConfig
    loop: LoopConfig
    resources: ResourcePlan = Field(
        ...,
        description=(
            "Placement of DFT and training work on the allocation; see "
            "docs/resource-dispatch.md. Requires a dft pool, and a training pool "
            "when trainer.enabled is true."
        ),
    )

    @model_validator(mode="after")
    def _check_resource_pools(self) -> ALConfig:
        if "dft" not in self.resources.pools:
            raise ValueError("resources.pools.dft is required: active learning labels with DFT")
        if self.trainer.enabled and "training" not in self.resources.pools:
            raise ValueError("trainer.enabled=True requires resources.pools.training")
        return self

    @model_validator(mode="after")
    def _check_ensemble_for_strategy(self) -> ALConfig:
        s = self.acquisition.strategy
        needs_ensemble = s in {"ensemble", "ensemble_then_dropout"}
        if needs_ensemble and not self.mlip.ensemble_paths:
            raise ValueError(
                f"acquisition.strategy={s!r} requires at least one additional model: "
                "set mlip.hydragnn.ensemble_paths (HydraGNN) or mlip.uma.ensemble_models (UMA)."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ALConfig:
        """Load and validate an active-learning config YAML.

        Supports shell-style variable substitution in **all string values**:

        * ``${VAR}``                  — required; raises if unset.
        * ``${VAR:-default}``          — falls back to ``default`` if unset.
        * ``${VAR:?error message}``    — raises with that message if unset.

        Variables are resolved in this order:

        1. ``os.environ``
        2. An optional top-level ``vars:`` mapping inside the YAML itself.

        The ``vars:`` block is consumed (stripped) before pydantic validation,
        so it never appears in the parsed :class:`ALConfig`. Substitution runs
        on the raw YAML text, so you can interpolate inside paths, lists, and
        even keys.
        """
        import yaml

        from matsim_agents.execution.variables import expand_variables, variable_defaults

        path = Path(path)
        raw_text = path.read_text()

        # First parse to extract the optional `vars:` block as fallback values.
        try:
            preview = yaml.safe_load(raw_text) or {}
        except yaml.YAMLError:
            preview = {}
        vars_block = preview.get("vars") if isinstance(preview, dict) else None
        defaults = variable_defaults(vars_block, source=path)
        substituted = expand_variables(raw_text, defaults, source=path)
        data = yaml.safe_load(substituted)
        if isinstance(data, dict):
            data.pop("vars", None)
        return cls.model_validate(data)
