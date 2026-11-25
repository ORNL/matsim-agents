"""Pydantic configuration for the active-learning loop.

The full config is intentionally explicit: every knob that affects scientific
results (force cutoffs, MD temperatures, acquisition strategy) is exposed as
a top-level field so the YAML is self-documenting and trivially diffable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.dft.config import DFTConfig
from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.backends.mlip.config import MLIPBackendConfig, MLIPConfig
from matsim_agents.execution.placement import ResourcePlan
from matsim_agents.science.neb.contracts import NebTransition


class MDConfig(BaseModel):
    """Parameters for the HydraGNN-driven MD sampler.

    The list of starting structures is supplied via :class:`SeedSourceConfig`
    (``seed_source``).
    """

    model_config = ConfigDict(extra="forbid")

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


class NebKineticsConfig(BaseModel):
    """Explicit assumptions for attaching a NEB barrier to exploratory kinetics."""

    model_config = ConfigDict(extra="forbid")

    attempt_frequency_hz: float = Field(gt=0, allow_inf_nan=False)
    temperature_K: float = Field(gt=0, allow_inf_nan=False)
    prefactor_note: str | None = None


class NebRefinementStage(BaseModel):
    """Optional climbing-image refinement after a converged standard-NEB stage."""

    model_config = ConfigDict(extra="forbid")

    optimizer: Literal["FIRE", "BFGS", "LBFGS"] = "FIRE"
    fmax_eV_per_A: float = Field(0.02, gt=0, allow_inf_nan=False)
    max_steps: int = Field(200, ge=1)
    climbing_image: bool = True

    @model_validator(mode="after")
    def _require_climbing_image(self) -> NebRefinementStage:
        if not self.climbing_image:
            raise ValueError("a NEB refinement_stage must enable climbing_image")
        return self


class NebSamplingConfig(BaseModel):
    """ASE NEB sampling settings for one pair of endpoint structures."""

    model_config = ConfigDict(extra="forbid")

    initial_path: Path
    final_path: Path
    transition_id: str | None = None
    transition: NebTransition | None = None
    kinetics: NebKineticsConfig | None = None
    n_images: int = Field(7, ge=4)
    interpolation: Literal["linear", "idpp"] = "linear"
    spring_constant_eV_per_A2: float = Field(0.1, gt=0, allow_inf_nan=False)
    optimizer: Literal["FIRE", "BFGS", "LBFGS"] = "FIRE"
    fmax_eV_per_A: float = Field(0.02, gt=0, allow_inf_nan=False)
    max_steps: int = Field(1000, ge=1)
    climbing_image: bool = False
    refinement_stage: NebRefinementStage | None = None
    record_every: int = Field(10, ge=1)
    min_distance_A: float = Field(1.0, gt=0, allow_inf_nan=False)
    max_step_A: float = Field(1.0, gt=0, allow_inf_nan=False)
    relax_endpoints: bool = False
    endpoint_fmax_eV_per_A: float = Field(0.02, gt=0, allow_inf_nan=False)
    endpoint_max_steps: int = Field(200, ge=1)
    uncertainty_interval_steps: int = Field(
        0,
        ge=0,
        description=(
            "Evaluate ensemble barrier uncertainty every N NEB steps; 0 disables interim checks."
        ),
    )
    max_pathway_uncertainty_eV: float | None = Field(None, gt=0, allow_inf_nan=False)
    resume: bool = True

    @model_validator(mode="after")
    def _check_uncertainty_gate(self) -> NebSamplingConfig:
        if self.max_pathway_uncertainty_eV is not None and not self.uncertainty_interval_steps:
            raise ValueError("max_pathway_uncertainty_eV requires uncertainty_interval_steps > 0")
        if (
            self.transition is not None
            and self.transition_id is not None
            and self.transition.transition_id != self.transition_id
        ):
            raise ValueError("transition_id must match transition.transition_id")
        if self.kinetics is not None and self.transition is None:
            raise ValueError("NEB kinetics output requires a typed transition")
        if self.refinement_stage is not None and self.climbing_image:
            raise ValueError("two-stage NEB requires climbing_image: false in the initial stage")
        if (
            self.transition is not None
            and self.transition.calculation is not None
            and self.transition.calculation.n_images != self.n_images
        ):
            raise ValueError("NEB n_images must match transition.calculation.n_images")
        return self


class NebEscalationConfig(BaseModel):
    """Acceptance and optional full-QE-NEB escalation policy."""

    model_config = ConfigDict(extra="forbid")

    policy: Literal["mlip_only", "uncertainty_gated", "always_dft_neb"] = "mlip_only"
    max_pathway_uncertainty_eV: float | None = Field(None, gt=0, allow_inf_nan=False)
    max_dft_neb_runs: int = Field(0, ge=0)
    dft_neb: QENebConfig | None = None
    allow_unrelaxed_endpoints: bool = False

    @model_validator(mode="after")
    def _check_policy(self) -> NebEscalationConfig:
        if (
            self.policy in {"mlip_only", "uncertainty_gated"}
            and self.max_pathway_uncertainty_eV is None
        ):
            raise ValueError(f"{self.policy} requires max_pathway_uncertainty_eV")
        if self.policy == "uncertainty_gated":
            if self.dft_neb is None or self.max_dft_neb_runs < 1:
                raise ValueError("uncertainty_gated requires dft_neb and max_dft_neb_runs >= 1")
        elif self.policy == "always_dft_neb":
            if self.dft_neb is None or self.max_dft_neb_runs < 1:
                raise ValueError("always_dft_neb requires dft_neb and max_dft_neb_runs >= 1")
        elif self.dft_neb is not None or self.max_dft_neb_runs:
            raise ValueError("mlip_only cannot configure DFT NEB runs")
        return self


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

    model_config = ConfigDict(extra="forbid")

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


class MDSamplingStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["md"]
    parameters: MDConfig


class NEBSamplingStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["neb"]
    parameters: NebSamplingConfig


SamplingStrategy = Annotated[
    MDSamplingStrategy | NEBSamplingStrategy, Field(discriminator="method")
]


class SamplingConfig(BaseModel):
    """One or more independent numerical strategies sharing the same MLIP and AL pipeline."""

    model_config = ConfigDict(extra="forbid")

    strategies: list[SamplingStrategy] = Field(min_length=1)


class AcquisitionConfig(BaseModel):
    """Which uncertainty measure(s) to use to pick frames for VASP labelling."""

    strategy: Literal["ensemble", "mc_dropout", "random", "ensemble_then_dropout"] = "ensemble"
    n_select: int = 256  # how many candidates to label per AL iteration
    mc_dropout_passes: int = Field(8, ge=2)
    mc_dropout_p: float = Field(0.1, gt=0, lt=1, allow_inf_nan=False)
    diversity_filter: bool = True  # greedy-farthest-point on composition + UQ score
    min_uncertainty_eV_per_A: float = 0.0  # below this, skip labelling
    max_per_neb_pathway: int | None = Field(
        2,
        ge=1,
        description="Cap selected NEB images per transition so one path cannot consume the budget.",
    )


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

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        False,
        description="Opt in to retraining; collecting DFT labels without it is the safe default.",
    )
    force_error_trigger_eV_per_A: float | None = Field(
        None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Retrain only if at least one converged single-point DFT audit has a "
            "force-component MAE above this tolerance against the incumbent. "
            "Audits still consume DFT budget below the tolerance."
        ),
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
    train_launcher: Path | None = Field(
        None,
        description=(
            "Optional site launcher for the fine-tune. It receives "
            "`<backend> <nodes> <ranks_per_node> -m <module> <args...>`, prepares that "
            "backend's environment, and runs `python` with the remaining arguments "
            "(e.g. under srun). If unset, the selected MLIP's fine-tuner "
            "(mlip.fine_tuner) runs in the current process."
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
    def _check_training_inputs(self) -> TrainerConfig:
        for field_name in ("validation_set", "validation_reference_set"):
            validation_path = getattr(self, field_name)
            if validation_path is not None and not validation_path.is_file():
                raise ValueError(
                    f"trainer.{field_name} must be an existing file: {validation_path}"
                )
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


class BenchmarkSet(BaseModel):
    """A frozen, DFT-labelled evaluation set scored after every iteration.

    Benchmarks give learning curves (error against cumulative DFT labels) and a
    common yardstick for comparing acquisition strategies. They never influence
    acquisition, training, or promotion; their frames must not be acquired.
    """

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    dataset: Path

    @model_validator(mode="after")
    def _exists(self) -> BenchmarkSet:
        if not self.dataset.is_file():
            raise ValueError(f"benchmark set {self.name!r} does not exist: {self.dataset}")
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
    benchmark_sets: list[BenchmarkSet] = Field(
        default_factory=list,
        description="Frozen evaluation sets scored with the model in use after each "
        "iteration's training and promotion decisions (learning curves).",
    )


# --------------------------------------------------------------------------- #
# Top-level config                                                            #
# --------------------------------------------------------------------------- #


class ALConfig(BaseModel):
    """Root config for the active-learning loop."""

    model_config = ConfigDict(extra="forbid")

    random_seed: int = Field(
        ...,
        description=(
            "Root seed of the run. Every random draw (pyXtal seed structures, MD "
            "velocities and thermostat noise, acquisition, MC-dropout masks, the "
            "validation split) comes from a named stream of this seed; see "
            "docs/reproducibility.md."
        ),
    )
    mlip: MLIPConfig
    md: MDConfig | None = None
    sampling: SamplingConfig | None = None
    acquisition: AcquisitionConfig
    dft: DFTConfig
    escalation: NebEscalationConfig | None = None
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
    def _check_sampling(self) -> ALConfig:
        if self.sampling is None and self.md is None:
            raise ValueError("specify legacy md configuration or sampling.strategies")
        if (
            self.md is not None
            and self.sampling is not None
            and any(strategy.method == "md" for strategy in self.sampling.strategies)
        ):
            raise ValueError(
                "configure MD either in sampling.strategies or the legacy md field, not both"
            )
        return self

    @model_validator(mode="after")
    def _check_neb_escalation(self) -> ALConfig:
        neb_strategies = [
            strategy.parameters
            for strategy in (self.sampling.strategies if self.sampling is not None else ())
            if isinstance(strategy, NEBSamplingStrategy)
        ]
        if self.escalation is None:
            if any(strategy.kinetics is not None for strategy in neb_strategies):
                raise ValueError("NEB kinetics output requires an explicit escalation policy")
            return self
        if not neb_strategies:
            raise ValueError("escalation requires at least one NEB sampling strategy")
        if self.escalation.policy in {"uncertainty_gated", "always_dft_neb"}:
            if self.dft.backend != "qe":
                raise ValueError("full DFT NEB escalation requires dft.backend='qe'")
            for strategy in neb_strategies:
                if strategy.transition is None:
                    raise ValueError(
                        "full DFT NEB escalation requires a typed transition for every NEB strategy"
                    )
                if self.escalation.dft_neb is not None and (
                    self.escalation.dft_neb.n_images != strategy.n_images
                ):
                    raise ValueError("escalation.dft_neb.n_images must match each NEB strategy")
        return self

    @model_validator(mode="after")
    def _check_resource_pools(self) -> ALConfig:
        if "dft" not in self.resources.pools:
            raise ValueError("resources.pools.dft is required: active learning labels with DFT")
        if self.trainer.enabled and "training" not in self.resources.pools:
            raise ValueError("trainer.enabled=True requires resources.pools.training")
        return self

    @model_validator(mode="after")
    def _check_benchmarks(self) -> ALConfig:
        names = [benchmark.name for benchmark in self.loop.benchmark_sets]
        if len(names) != len(set(names)):
            raise ValueError("loop.benchmark_sets names must be unique")
        if names and self.benchmark_reference_manifest() is None:
            raise ValueError(
                "loop.benchmark_sets compare formation energies and need elemental DFT "
                "references: trainer.validation_reference_set or "
                "trainer.training_references.manifest"
            )
        return self

    def benchmark_reference_manifest(self) -> Path | None:
        if self.trainer.validation_reference_set is not None:
            return self.trainer.validation_reference_set
        references = self.trainer.training_references
        return references.manifest if references is not None else None

    @model_validator(mode="after")
    def _check_fine_tunable(self) -> ALConfig:
        if self.trainer.enabled:
            self.mlip.check_fine_tunable()
        return self

    @model_validator(mode="after")
    def _check_ensemble_for_strategy(self) -> ALConfig:
        s = self.acquisition.strategy
        needs_ensemble = s in {"ensemble", "ensemble_then_dropout"}
        if needs_ensemble and not self.mlip.ensemble_models:
            raise ValueError(
                f"acquisition.strategy={s!r} requires at least one additional model: "
                "set mlip.ensemble_models."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path, *, mlip: MLIPBackendConfig | None = None) -> ALConfig:
        """Load and validate an active-learning config YAML.

        ``mlip`` replaces the file's ``mlip`` block, so one campaign definition
        runs with any backend (see ``--mlip-config``); the replaced block is
        neither expanded nor validated.

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
        return cls.model_validate(_load_yaml_mapping(path, mlip=mlip))


class SamplingRunConfig(BaseModel):
    """Standalone MD/NEB sampling through ``matsim-agents run`` without DFT or AL."""

    model_config = ConfigDict(extra="forbid")

    random_seed: int
    mlip: MLIPConfig
    sampling: SamplingConfig
    resources: ResourcePlan
    acquisition: AcquisitionConfig | None = None

    @model_validator(mode="after")
    def _check_standalone_uq(self) -> SamplingRunConfig:
        if self.acquisition is None:
            if any(
                isinstance(strategy, NEBSamplingStrategy)
                and strategy.parameters.max_pathway_uncertainty_eV is not None
                for strategy in self.sampling.strategies
            ):
                raise ValueError("NEB pathway uncertainty thresholds require acquisition settings")
            return self
        needs_ensemble = self.acquisition.strategy in {"ensemble", "ensemble_then_dropout"}
        if needs_ensemble and not self.mlip.ensemble_models:
            raise ValueError(
                f"acquisition.strategy={self.acquisition.strategy!r} requires mlip.ensemble_models"
            )
        if self.acquisition.strategy == "random" and any(
            isinstance(strategy, NEBSamplingStrategy)
            and strategy.parameters.max_pathway_uncertainty_eV is not None
            for strategy in self.sampling.strategies
        ):
            raise ValueError("random acquisition cannot provide a pathway uncertainty gate")
        return self

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, mlip: MLIPBackendConfig | None = None
    ) -> SamplingRunConfig:
        """Load a sampling config using the active-learning variable and MLIP rules."""
        return cls.model_validate(_load_yaml_mapping(path, mlip=mlip))


def _load_yaml_mapping(path: str | Path, *, mlip: MLIPBackendConfig | None) -> dict[str, Any]:
    """Expand a workflow YAML and optionally replace its selected MLIP backend."""
    import yaml

    from matsim_agents.execution.variables import expand_variables, variable_defaults

    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    try:
        preview = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError:
        preview = {}
    vars_block = preview.get("vars") if isinstance(preview, dict) else None
    defaults = variable_defaults(vars_block, source=path)
    if mlip is not None and isinstance(preview, dict):
        preview.pop("mlip", None)
        raw_text = yaml.safe_dump(preview, sort_keys=False)
    substituted = expand_variables(raw_text, defaults, source=path)
    data = yaml.safe_load(substituted)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    data.pop("vars", None)
    if mlip is not None:
        data["mlip"] = mlip
    return data
