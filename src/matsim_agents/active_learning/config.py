"""Pydantic configuration for the active-learning loop.

The full config is intentionally explicit: every knob that affects scientific
results (force cutoffs, MD temperatures, acquisition strategy) is exposed as
a top-level field so the YAML is self-documenting and trivially diffable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
# Sub-configs                                                                 #
# --------------------------------------------------------------------------- #


class HydraGNNConfig(BaseModel):
    """Inputs needed to load and (optionally) retrain a HydraGNN MLFF model."""

    # Accept both the field name and the YAML-facing alias 'mlp_checkpoint'.
    model_config = ConfigDict(populate_by_name=True)

    logdir: Path = Field(..., description="HydraGNN logdir with config.json + checkpoint.")
    checkpoint: str | None = Field(None, description="Specific checkpoint filename or path.")
    hydragnn_branch_mlp_checkpoint: Path | None = Field(
        None,
        alias="mlp_checkpoint",
        description="Optional auxiliary BranchWeightMLP checkpoint (YAML key: mlp_checkpoint).",
    )
    newhead_ft_config: Path | None = Field(
        None,
        description=(
            "For the 'drop-all-heads + new head' fine-tune: path to the "
            "``newhead.json`` produced by finetune_hydragnn_newhead. When set, "
            "the calculator rebuilds the backbone from ``config.json``, re-applies "
            "the single-branch head surgery, loads ``checkpoint`` and uses direct "
            "single-head inference (no BranchWeightMLP)."
        ),
    )
    ft_repo: Path | None = Field(
        None,
        description="Path to ORNL/HydraGNN_GFM_FineTuning4Materials (for update_model).",
    )
    radius: float | None = Field(None, description="Override cutoff radius (Å).")
    max_neighbours: int | None = Field(None, description="Override max neighbours.")
    charge: float = 0.0
    spin: float = 0.0
    precision: str | None = None  # "fp32" | "fp64" | "bf16"
    mlp_device: Literal["cuda", "cpu"] = "cuda"
    ensemble_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Paths to additional HydraGNN logdirs forming the ensemble. If "
            "non-empty, ensemble disagreement is available as an acquisition."
        ),
    )


class MCDropoutInjectionConfig(BaseModel):
    """Test-time dropout injection for backends without native dropout (e.g. UMA).

    UMA and most production foundation MLIPs are trained *without* dropout, so
    ``acquisition.strategy: mc_dropout`` would otherwise return zero variance.
    When ``enabled``, dropout modules are inserted via forward hooks on the
    selected layer types after the model is loaded. They stay **dormant**
    (identity) during normal energy/force prediction and relaxation, and only
    become active while MC-Dropout scoring toggles them into ``train()`` mode,
    so deterministic results (relaxation, MD) are unchanged.

    Caveat: injecting dropout into a model not trained with it yields a
    *heuristic* uncertainty (uncalibrated), not a Bayesian posterior. Prefer a
    deep ensemble where a calibrated signal is required.
    """

    enabled: bool = True
    p: float = Field(0.1, description="Dropout probability used at the injected layers.")
    target_layers: Literal["linear", "all"] = Field(
        "linear",
        description="Which layer outputs to apply dropout to: 'linear' = nn.Linear only.",
    )
    max_layers: int | None = Field(
        None, description="Cap the number of injected layers (None = every match)."
    )


class UMAConfig(BaseModel):
    """Inputs to load a UMA (fairchem) universal MLIP as an ASE calculator.

    By default UMA is used as a frozen foundation model, but the AL trainer can
    optionally call a user-supplied FairChem/UMA fine-tuning launcher after each
    DFT-labelled batch and use the resulting model path on the next iteration.
    """

    model_name: str = Field(
        "uma-s-1p1",
        description="fairchem pretrained model name (e.g. 'uma-s-1p1') or local checkpoint path.",
    )
    task_name: Literal["omat", "omol", "oc20", "odac", "omc"] = Field(
        "omat",
        description="UMA task head (omat = inorganic bulk materials; omol = molecules/MOFs).",
    )
    device: Literal["cuda", "cpu"] = "cuda"
    precision: Literal["fp32", "fp64", "bf16"] | None = None
    charge: float = 0.0
    spin: float = 0.0
    ensemble_models: list[str] = Field(
        default_factory=list,
        description=(
            "Additional UMA model names/checkpoints forming a deep ensemble. If "
            "non-empty, ensemble disagreement is available as an acquisition."
        ),
    )
    dropout: MCDropoutInjectionConfig = Field(
        default_factory=MCDropoutInjectionConfig,
        description="Test-time dropout injection settings for MC-Dropout acquisition.",
    )


class MACEConfig(BaseModel):
    """Inputs to load a MACE MLIP as an ASE calculator.

    Supports the foundation models shipped with ``mace-torch`` -- ``mace_mp``
    (Materials Project, inorganic) and ``mace_off`` (organic molecules) -- as
    well as a local fine-tuned ``.model`` checkpoint. ``model`` selects the
    size/variant (``small`` | ``medium`` | ``large``, or a release tag/URL) so
    multiple MACE versions are benchmarkable behind the same backend, matching
    the Frontier HydraGNN-vs-MACE-vs-UMA comparison pipeline.
    """

    model_config = ConfigDict(populate_by_name=True)

    family: Literal["mace_mp", "mace_off", "checkpoint"] = Field(
        "mace_mp",
        description=(
            "Which MACE model family to load: 'mace_mp' (Materials Project, "
            "inorganic), 'mace_off' (organic molecules), or 'checkpoint' (a local "
            ".model file given by `model`)."
        ),
    )
    model: str = Field(
        "medium",
        description=(
            "Variant for the foundation families ('small' | 'medium' | 'large', "
            "or a specific release tag/URL), or a path to a local .model checkpoint "
            "when family='checkpoint'."
        ),
    )
    device: Literal["cuda", "cpu"] = "cuda"
    precision: Literal["fp32", "fp64"] | None = Field(
        None,
        description="Calculator dtype -> MACE default_dtype (fp64 recommended for relaxation).",
    )
    dispersion: bool = Field(
        False, description="Add DFT-D3 dispersion correction (mace_mp / mace_off only)."
    )
    ensemble_models: list[str] = Field(
        default_factory=list,
        description=(
            "Additional MACE variants/checkpoints forming a deep ensemble. If "
            "non-empty, ensemble disagreement is available as an acquisition."
        ),
    )
    dropout: MCDropoutInjectionConfig = Field(
        default_factory=MCDropoutInjectionConfig,
        description=(
            "Test-time dropout injection for MC-Dropout acquisition (MACE has no native dropout)."
        ),
    )


class MLIPConfig(BaseModel):
    """Selects which ML potential predicts energies and forces.

    ``backend`` chooses the active surrogate. Both ``hydragnn`` and ``uma``
    sub-blocks may be present simultaneously; only the block matching
    ``backend`` must be populated (the other is ignored). This makes the
    surrogate switchable with a single environment variable when ``backend``
    is wired to one, e.g. ``backend: ${MLIP_BACKEND:-hydragnn}``::

        MLIP_BACKEND=uma matsim-agents al run <cfg>   # frozen UMA foundation model
        matsim-agents al run <cfg>                   # HydraGNN (default)

    For backward compatibility, a legacy top-level ``hydragnn:`` block (no
    ``mlip:`` block) is promoted to ``mlip: {backend: hydragnn, hydragnn: ...}``
    by the :class:`ALConfig` root validator.
    """

    backend: Literal["hydragnn", "uma", "mace"] = "hydragnn"
    hydragnn: HydraGNNConfig | None = None
    uma: UMAConfig | None = None
    mace: MACEConfig | None = None

    @model_validator(mode="after")
    def _check_backend_block(self) -> MLIPConfig:
        if self.backend == "hydragnn" and self.hydragnn is None:
            raise ValueError("mlip.backend='hydragnn' requires an mlip.hydragnn block.")
        if self.backend == "uma" and self.uma is None:
            raise ValueError("mlip.backend='uma' requires an mlip.uma block.")
        if self.backend == "mace" and self.mace is None:
            raise ValueError("mlip.backend='mace' requires an mlip.mace block.")
        return self

    @property
    def ensemble_paths(self) -> list:
        """Unified ensemble-member list used by the AL loop, regardless of backend."""
        if self.backend == "hydragnn" and self.hydragnn is not None:
            return list(self.hydragnn.ensemble_paths)
        if self.backend == "uma" and self.uma is not None:
            return list(self.uma.ensemble_models)
        if self.backend == "mace" and self.mace is not None:
            return list(self.mace.ensemble_models)
        return []


class MDConfig(BaseModel):
    """Parameters for the HydraGNN-driven MD sampler.

    The list of starting structures is supplied via :class:`SeedSourceConfig`
    (``seed_source``). For backward compatibility, ``seed_structures`` may
    still be set as a top-level list of paths; the ``ALConfig`` root
    validator promotes it to ``seed_source.kind='paths'``.
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
            "Langevin thermostat noise) so the sampled trajectory and the "
            "resulting candidate pool are reproducible across runs. Leave "
            "unset for stochastic sampling (the default)."
        ),
    )


class LLMSeedConfig(BaseModel):
    """LLM provider used by ``seed_source.kind='prompt'``.

    Defaults match :func:`matsim_agents.llm.get_chat_model`. On Frontier the
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


class VASPConfig(BaseModel):
    """How to run VASP single-point calculations."""

    vasp_bin: Path = Field(..., description="Path to vasp_std (or vasp_gam/vasp_ncl) on Frontier.")
    vasp_wrapper: Path = Field(
        ...,
        description=(
            "Bash wrapper that does `module reset && module load PrgEnv-cray ...` "
            "and execs `srun vasp_std`. Provided at "
            "scripts/launchers/frontier/_vasp-step-frontier.sh."
        ),
    )
    incar_template: Path = Field(..., description="INCAR template with Python str.format() slots.")
    kpoints_template: Path | None = Field(
        None, description="Optional KPOINTS template; otherwise auto KSPACING in INCAR."
    )
    potcar_dir: Path = Field(..., description="Directory containing per-element POTCAR files.")
    nodes_per_job: int = 1
    ranks_per_node: int = 8  # 8 GCDs per MI250X node
    threads_per_rank: int = 7
    timeout_sec: int = 7200
    extra_incar: dict[str, str] = Field(default_factory=dict)


class QEBackendConfig(BaseModel):
    """How to run Quantum ESPRESSO ``pw.x`` single-point SCF calculations.

    The Python side fills in element-aware defaults (plane-wave cutoffs,
    smearing, k-mesh) via :func:`matsim_agents.tools.qe_relax.recommend_settings`
    so most fields are optional. Pin them explicitly for production runs.
    """

    pw_bin: Path = Field(..., description="Path to pw.x on Frontier (e.g. install-gpu/bin/pw.x).")
    pw_wrapper: Path = Field(
        ...,
        description=(
            "Bash wrapper that does `module reset && module load PrgEnv-cray ...` "
            "and execs `srun pw.x -in <input>`. Provided at "
            "scripts/launchers/frontier/_qe-step-frontier.sh."
        ),
    )
    pseudo_dir: Path = Field(
        ...,
        description=(
            "Directory containing per-element UPF pseudopotentials. The QE "
            "backend auto-detects ``<symbol>.*.UPF`` per element."
        ),
    )
    pw_template: Path | None = Field(
        None,
        description=(
            "Optional pw.in namelist template (analogue of VASP's INCAR "
            "template). When set, the backend reads this file, runs "
            "``str.format(nat=, ntyp=, pseudo_dir=, prefix=, outdir=)`` on "
            "it, and APPENDS the auto-generated structure cards "
            "(ATOMIC_SPECIES, CELL_PARAMETERS, ATOMIC_POSITIONS, K_POINTS). "
            "When None, the backend generates the full pw.in programmatically "
            "via recommend_settings(). The two paths are mutually exclusive: "
            "the namelist-level pins below (ecutwfc_ry, occupations, "
            "extra_control, ...) are IGNORED when pw_template is set — put "
            "those values in the template directly."
        ),
    )
    # Optional explicit overrides (None ⇒ auto from element table).
    ecutwfc_ry: float | None = None
    ecutrho_ry: float | None = None
    kpts: tuple[int, int, int] | None = None
    koffset: tuple[int, int, int] | None = None
    occupations: Literal["smearing", "fixed", "tetrahedra"] | None = None
    smearing: str = "gaussian"
    degauss_ry: float = 0.01
    pseudopotentials: dict[str, str] | None = Field(
        None,
        description="Optional explicit {symbol: filename} mapping (else auto-detected).",
    )
    extra_control: dict[str, Any] = Field(default_factory=dict)
    extra_system: dict[str, Any] = Field(default_factory=dict)
    extra_electrons: dict[str, Any] = Field(default_factory=dict)
    nodes_per_job: int = 1
    ranks_per_node: int = 8
    threads_per_rank: int = 7
    timeout_sec: int = 7200


class DFTConfig(BaseModel):
    """Selects which DFT backend labels candidate structures.

    Exactly one of ``vasp`` / ``qe`` must be populated, matching ``backend``.
    The :class:`ALConfig` root validator also accepts a legacy top-level
    ``vasp:`` block (without a ``dft:`` block) for backward compatibility.
    """

    backend: Literal["vasp", "qe"] = "vasp"
    vasp: VASPConfig | None = None
    qe: QEBackendConfig | None = None

    @model_validator(mode="after")
    def _check_backend_block(self) -> DFTConfig:
        if self.backend == "vasp" and self.vasp is None:
            raise ValueError("dft.backend='vasp' requires a dft.vasp block.")
        if self.backend == "qe" and self.qe is None:
            raise ValueError("dft.backend='qe' requires a dft.qe block.")
        return self


class TrainerConfig(BaseModel):
    """How to retrain the MLIP at the end of each AL iteration.

    HydraGNN and UMA both use a user-supplied training script/launcher. For UMA,
    leave ``enabled: false`` to keep using the frozen foundation model and simply
    accumulate labels for offline fine-tuning.
    """

    enabled: bool = True
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
    nodes_for_train: int = 8
    ranks_per_node: int = 8

    @model_validator(mode="after")
    def _check_train_script(self) -> TrainerConfig:
        if self.enabled and self.train_script is None:
            raise ValueError("trainer.enabled=True requires trainer.train_script.")
        return self


class LoopConfig(BaseModel):
    """Top-level loop control."""

    n_iterations: int = 10
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

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_vasp_block(cls, data: Any) -> Any:
        """Accept the pre-multi-backend YAML schema.

        * Top-level ``hydragnn:`` (no ``mlip:``) → ``mlip: {backend: hydragnn,
          hydragnn: ...}`` so existing single-surrogate configs keep working.
        * Top-level ``vasp:`` (no ``dft:``) → ``dft: {backend: vasp, vasp: ...}``.
        * ``md.seed_structures: [...]`` → ``md.seed_source: {kind: paths, ...}``.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)  # don't mutate the caller's dict
        if "hydragnn" in data and "mlip" not in data:
            data["mlip"] = {"backend": "hydragnn", "hydragnn": data.pop("hydragnn")}
        if "vasp" in data and "dft" not in data:
            data["dft"] = {"backend": "vasp", "vasp": data.pop("vasp")}
        md = data.get("md")
        if isinstance(md, dict) and "seed_source" not in md and "seed_structures" in md:
            md = dict(md)
            md["seed_source"] = {"kind": "paths", "paths": md.pop("seed_structures")}
            data["md"] = md
        return data

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
        import os
        import re

        import yaml

        path = Path(path)
        raw_text = path.read_text()

        # First parse to extract the optional `vars:` block as fallback values.
        try:
            preview = yaml.safe_load(raw_text) or {}
        except yaml.YAMLError:
            preview = {}

        # ${VAR}, ${VAR:-default}, ${VAR:?msg}.  Nested braces not supported.
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:[-?][^}]*)?\}")

        defaults: dict[str, str] = {}
        if isinstance(preview, dict) and isinstance(preview.get("vars"), dict):
            # A `vars:` entry may reference the environment with its own name as
            # a fallback default, e.g.  ``RUNS_ROOT: ${RUNS_ROOT:-/some/path}``
            # ("use $RUNS_ROOT if exported, else this literal"). Resolve that
            # self-reference up front: without this, ``_resolve`` would look the
            # name up in ``defaults`` and hand back the entry's own raw string,
            # so it converges to the literal ``${RUNS_ROOT:-/some/path}`` and the
            # default path is never applied. References to *other* vars are left
            # untouched here and resolved by the main loop below.
            def _break_self_ref(key: str, value: str) -> str:
                def _sub(match: re.Match[str]) -> str:
                    name = match.group(1)
                    modifier = match.group(2) or ""
                    if os.environ.get(name):
                        return os.environ[name]
                    if name == key:
                        if modifier.startswith(":-"):
                            return modifier[2:]
                        if modifier.startswith(":?"):
                            msg = modifier[2:].strip() or f"required variable {name!r} is unset"
                            raise ValueError(f"{path}: {msg}")
                    return match.group(0)  # env var or cross-ref; resolve later

                return pattern.sub(_sub, value)

            defaults = {str(k): _break_self_ref(str(k), str(v)) for k, v in preview["vars"].items()}

        def _resolve(match: re.Match[str]) -> str:

            name = match.group(1)
            modifier = match.group(2) or ""
            if name in os.environ and os.environ[name] != "":
                return os.environ[name]
            if name in defaults:
                return defaults[name]
            if modifier.startswith(":-"):
                return modifier[2:]
            if modifier.startswith(":?"):
                msg = modifier[2:].strip() or f"required variable {name!r} is unset"
                raise ValueError(f"{path}: {msg}")
            raise ValueError(
                f"{path}: undefined variable ${{{name}}} "
                "(set it in the environment, in the YAML 'vars:' block, "
                "or use ${VAR:-default} syntax)."
            )

        substituted = raw_text
        for _ in range(10):  # iterate so vars can reference other vars
            new_text = pattern.sub(_resolve, substituted)
            if new_text == substituted:
                break
            substituted = new_text
        else:
            raise ValueError(
                f"{path}: variable substitution did not converge after 10 passes "
                "(possible circular reference in 'vars:')."
            )
        data = yaml.safe_load(substituted)
        if isinstance(data, dict):
            data.pop("vars", None)
        return cls.model_validate(data)
