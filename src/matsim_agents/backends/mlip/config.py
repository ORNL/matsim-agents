"""MLIP backend configurations.

Every MLIP is configured by one subclass of :class:`MLIPBackendConfig`. A
subclass owns everything specific to its backend: defaults, file paths, how the
ASE calculator is built, how a trained checkpoint replaces the foundation
model, its provenance identity, and the Python modules it needs. Capabilities
depend only on the base interface and accept :data:`MLIPConfig`, the
discriminated union of the subclasses, so all backends are interchangeable::

    mlip:
      backend: mace          # hydragnn | uma | mace
      model: medium-omat-0
      device: cpu
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

if TYPE_CHECKING:
    from matsim_agents.backends.mlip.relaxation import RelaxCalculator

UMATask = Literal["omat", "omol", "oc20", "odac", "omc"]
MACEFamily = Literal[
    "mace_mp",
    "mace_off",
    "mace_omol",
    "mace_polar",
    "mace_anicc",
    "checkpoint",
]

# Families trained on a fixed element set; other families are not restricted here.
MACE_FAMILY_ELEMENTS: dict[str, frozenset[str]] = {
    "mace_off": frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}),
    "mace_anicc": frozenset({"H", "C", "N", "O"}),
}

HYDRAGNN_DATASET_HEADS = (
    "Alexandria",
    "ANI1x",
    "MPTrj",
    "OC2020",
    "OC2022",
    "OC25",
    "ODAC23",
    "OMat24",
    "OMol25",
    "OMol25-neutral",
    "OMol25-non-neutral",
    "OPoly2026",
    "Nabla2DFT",
    "QCML",
    "QM7X",
    "transition1x",
)


def _compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _resolve_path(base: Path, value: object) -> object:
    if not isinstance(value, str | Path):
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def resolve_hydragnn_inference_head(value: str | int | None) -> int | None:
    """Resolve a dataset name, head index, or ``branch-N`` alias to its canonical index."""
    if value is None:
        return None
    if isinstance(value, int) or str(value).strip().isdigit():
        index = int(value)
        if 0 <= index < len(HYDRAGNN_DATASET_HEADS):
            return index
        raise ValueError(f"HydraGNN inference head index must be in [0, 15], got {index}")
    normalized = str(value).strip().casefold()
    if normalized.startswith("branch-") and normalized[7:].isdigit():
        index = int(normalized[7:])
        if 0 <= index < len(HYDRAGNN_DATASET_HEADS):
            return index
        raise ValueError(f"HydraGNN inference head index must be in [0, 15], got {index}")
    by_name = {name.casefold(): index for index, name in enumerate(HYDRAGNN_DATASET_HEADS)}
    if normalized not in by_name:
        raise ValueError(
            f"unknown HydraGNN inference head {value!r}; expected a branch index or one of "
            f"{', '.join(HYDRAGNN_DATASET_HEADS)}"
        )
    return by_name[normalized]


class MCDropoutInjectionConfig(BaseModel):
    """Test-time dropout injection for backends without native dropout (UMA, MACE).

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


# --------------------------------------------------------------------------- #
# Backend interface                                                           #
# --------------------------------------------------------------------------- #


class MLIPBackendConfig(BaseModel, ABC):
    """Interface every MLIP backend configuration implements."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    # Name of the built-in fine-tuning module (matsim_agents.backends.mlip.training.<fine_tuner>).
    fine_tuner: str
    device: Literal["cuda", "cpu", "xpu"] = "cuda"
    precision: str | None = None
    ensemble_models: list[str] = Field(
        default_factory=list,
        description=(
            "Additional models (names, checkpoints, or logdirs) forming a deep ensemble. "
            "If non-empty, ensemble disagreement is available as an acquisition."
        ),
    )

    # Importable modules this backend needs, beyond the shared ASE/Torch stack.
    required_modules: ClassVar[tuple[str, ...]] = ()
    # Fields holding file paths, resolved against the declaring YAML file.
    path_fields: ClassVar[tuple[str, ...]] = ()
    # Devices the built-in fine-tuner (fine_tuner) can train on.
    fine_tune_devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})

    @abstractmethod
    def build_calculator(self, *, enable_mc_dropout: bool = False) -> Any:
        """Load the model and return its ASE calculator."""

    @abstractmethod
    def with_checkpoint(self, path: str | Path) -> MLIPBackendConfig:
        """This configuration pointed at a trained or fine-tuned model."""

    @abstractmethod
    def ensemble_member(self, model: str) -> MLIPBackendConfig:
        """This configuration with ``model`` as the loaded model (an ensemble member)."""

    @abstractmethod
    def model_identifier(self) -> str:
        """Stable identity of the model and its prediction-relevant settings."""

    @property
    @abstractmethod
    def model_reference(self) -> str:
        """Short name or path of the model, for reports and evidence."""

    def model_artifact(self) -> Path | None:
        """Local model file whose SHA-256 identifies the model, if the backend uses one."""
        return None

    def model_files(self) -> list[Path]:
        """Every local file whose contents determine this model's predictions.

        Empty for a model selected by a released name. Raises ``ValueError`` when
        the files would only be chosen at load time, so they cannot be pinned.
        """
        artifact = self.model_artifact()
        return [artifact] if artifact is not None else []

    def unsupported_elements(self, elements: Iterable[str]) -> list[str]:
        """Elements the model is known not to cover; empty when coverage is not known."""
        return []

    def with_resolved_paths(self, base: Path) -> MLIPBackendConfig:
        """This configuration with relative local paths resolved against ``base``."""
        raw = self.model_dump(mode="json")
        return parse_mlip_config(type(self).resolve_paths(raw, Path(base)))

    def check_loadable(self) -> None:
        """Raise ``ValueError`` if inputs needed to load the model are missing."""

    def required_paths(self) -> list[Path]:
        """Local files and directories that must exist before the model can load."""
        artifact = self.model_artifact()
        return [artifact] if artifact is not None else []

    def check_fine_tunable(self) -> None:
        """Raise ``ValueError`` if the built-in fine-tuner cannot train this model."""
        if self.device not in self.fine_tune_devices:
            raise ValueError(
                f"{self.fine_tuner} does not support device={self.device!r} "
                f"(supported: {', '.join(sorted(self.fine_tune_devices))}); keep "
                "trainer.enabled false or select a supported device"
            )

    @property
    def fine_tuner_module(self) -> str:
        """Module run as ``python -m <module>`` to fine-tune this model."""
        return f"matsim_agents.backends.mlip.training.{self.fine_tuner}"

    @abstractmethod
    def fine_tune_args(
        self,
        *,
        dataset: Path,
        out_dir: Path,
        epochs: int,
        references: Path,
        finetune_mode: str,
        seed: int,
    ) -> list[str]:
        """Command-line arguments of :attr:`fine_tuner_module` for one fine-tune."""

    def ensemble_configs(self) -> list[MLIPBackendConfig]:
        """The primary model followed by every ensemble member."""
        return [self, *(self.ensemble_member(model) for model in self.ensemble_models)]

    def conditioning(self) -> tuple[float, float]:
        """The (charge, spin) this model is conditioned on; (0, 0) when it has none."""
        return (
            float(getattr(self, "charge", 0.0)),
            float(getattr(self, "spin", 0.0)),
        )

    def with_conditioning(self, charge: float, spin: float) -> MLIPBackendConfig:
        """This model conditioned on ``charge`` and ``spin``."""
        if "charge" in type(self).model_fields:
            return self.model_copy(update={"charge": charge, "spin": spin})
        if charge or spin:
            raise ValueError(
                f"{self.backend} models are not conditioned on charge or spin "
                f"(requested charge={charge}, spin={spin})"
            )
        return self

    def relaxation_calculator(self) -> RelaxCalculator:
        """Calculator for structure relaxation and its branch-weight UQ support."""
        from matsim_agents.backends.mlip.relaxation import RelaxCalculator

        return RelaxCalculator(
            self.build_calculator(),
            0,
            0,
            f"branch-weight UQ unavailable for {self.backend} backend",
        )

    @classmethod
    def resolve_paths(cls, raw: dict[str, Any], base: Path) -> dict[str, Any]:
        """Resolve this backend's file paths in a raw mapping against ``base``."""
        for field in cls.path_fields:
            if raw.get(field) is not None:
                raw[field] = _resolve_path(base, raw[field])
        return raw


# --------------------------------------------------------------------------- #
# Backends                                                                    #
# --------------------------------------------------------------------------- #


class HydraGNNConfig(MLIPBackendConfig):
    """A HydraGNN multi-head model loaded from a training logdir."""

    backend: Literal["hydragnn"] = "hydragnn"
    logdir: Path = Field(..., description="HydraGNN logdir with config.json + checkpoint.")
    hydragnn_root: Path | None = Field(
        None,
        description="HydraGNN source checkout used by built-in fine-tuners.",
    )
    checkpoint: str | None = Field(None, description="Specific checkpoint filename or path.")
    mlp_checkpoint: Path | None = Field(
        None,
        description="Optional auxiliary BranchWeightMLP checkpoint.",
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
    precision: str | None = Field(None, description="Model precision (fp32|fp64|bf16).")
    mlp_precision: str | None = Field(
        None, description="BranchWeightMLP precision override (fp32|fp64|bf16)."
    )
    inference_head: str | int | None = Field(
        None,
        description=(
            "Optional decoding head selected by canonical dataset name, index 0..15, "
            "or a 'branch-N' name. "
            "When set, inference bypasses BranchWeightMLP and uses only that head."
        ),
    )

    fine_tuner: Literal["finetune_hydragnn", "finetune_hydragnn_newhead"] = Field(
        "finetune_hydragnn",
        description=(
            "Fine-tuning recipe: finetune_hydragnn (routed multi-head, needs mlp_checkpoint) "
            "or finetune_hydragnn_newhead (drop all heads and train one new head)."
        ),
    )

    required_modules: ClassVar[tuple[str, ...]] = ("hydragnn", "torch_geometric")
    fine_tune_devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu", "xpu"})
    path_fields: ClassVar[tuple[str, ...]] = (
        "logdir",
        "hydragnn_root",
        "mlp_checkpoint",
        "newhead_ft_config",
        "ft_repo",
    )

    @model_validator(mode="after")
    def _validate_inference_head(self) -> HydraGNNConfig:
        resolve_hydragnn_inference_head(self.inference_head)
        if self.inference_head is not None and self.newhead_ft_config is not None:
            raise ValueError("inference_head cannot be combined with a single new-head model")
        return self

    def check_loadable(self) -> None:
        from matsim_agents.backends.mlip.hydragnn_energy import has_declared_hydragnn_checkpoint

        if (
            self.mlp_checkpoint is None
            and self.inference_head is None
            and not has_declared_hydragnn_checkpoint(str(self.logdir))
        ):
            raise ValueError(
                "HydraGNN requires mlp_checkpoint (BranchWeightMLP), an inference_head, "
                "or a declared checkpoint."
            )

    def required_paths(self) -> list[Path]:
        paths = [self.logdir, *super().required_paths()]
        if self.mlp_checkpoint is not None:
            paths.append(self.mlp_checkpoint)
        return paths

    def fine_tune_args(
        self,
        *,
        dataset: Path,
        out_dir: Path,
        epochs: int,
        references: Path,
        finetune_mode: str,
        seed: int,
    ) -> list[str]:
        args = [
            "--dataset", str(dataset),
            "--output-dir", str(out_dir),
            "--gfm-logdir", str(self.logdir),
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--elemental-reference-manifest", str(references),
            "--device", self.device,
            "--charge", str(self.charge),
            "--spin", str(self.spin),
        ]  # fmt: skip
        if self.fine_tuner == "finetune_hydragnn":
            if self.mlp_checkpoint is None:
                raise ValueError("finetune_hydragnn requires mlp_checkpoint (BranchWeightMLP)")
            args += ["--branch-mlp", str(self.mlp_checkpoint)]
        if self.checkpoint is not None:
            args += ["--gfm-checkpoint", self.checkpoint]
        if self.hydragnn_root is not None:
            args += ["--hydragnn-root", str(self.hydragnn_root)]
        if self.fine_tuner == "finetune_hydragnn_newhead" and self.ft_repo is not None:
            args += ["--ft-repo", str(self.ft_repo)]
        return args

    def build_calculator(self, *, enable_mc_dropout: bool = False) -> Any:
        from matsim_agents.backends.mlip.calculator import build_hydragnn_calculator

        return build_hydragnn_calculator(self)

    def with_checkpoint(self, path: str | Path) -> HydraGNNConfig:
        return self.model_copy(update={"logdir": Path(path), "checkpoint": None})

    def ensemble_member(self, model: str) -> HydraGNNConfig:
        return self.model_copy(update={"logdir": Path(model)})

    def model_identifier(self) -> str:
        identifier = f"hydragnn:{self.logdir}:checkpoint={self.checkpoint or 'auto/latest'}"
        head_index = resolve_hydragnn_inference_head(self.inference_head)
        if head_index is not None:
            identifier += f":head={HYDRAGNN_DATASET_HEADS[head_index]}"
        settings = {
            "branch_mlp_checkpoint": (
                str(self.mlp_checkpoint) if self.mlp_checkpoint is not None else None
            ),
            "charge": self.charge,
            "max_neighbours": self.max_neighbours,
            "newhead_ft_config": (
                str(self.newhead_ft_config) if self.newhead_ft_config is not None else None
            ),
            "precision": self.precision,
            "radius": self.radius,
            "spin": self.spin,
        }
        return f"{identifier}:settings={_compact_json(settings)}"

    @property
    def model_reference(self) -> str:
        return str(self.logdir)

    def model_artifact(self) -> Path | None:
        if not self.checkpoint:
            return None
        checkpoint = Path(self.checkpoint)
        return checkpoint if checkpoint.is_absolute() else self.logdir / checkpoint

    def model_files(self) -> list[Path]:
        convention_path = self.logdir / "energy-convention.json"
        convention = json.loads(convention_path.read_text()) if convention_path.is_file() else {}
        checkpoint = self.model_artifact()
        if checkpoint is None and "checkpoint" in convention:
            checkpoint = self.logdir / convention["checkpoint"]
        if checkpoint is None:
            raise ValueError(
                "HydraGNN checkpoint is chosen from the logdir at load time; set "
                "mlip.checkpoint so the model can be identified"
            )
        files = [self.logdir / "config.json", checkpoint]
        # Training metadata the loader applies (energy convention, routing, new head).
        for name in ("energy-convention.json", "routing.json", "newhead.json"):
            if (self.logdir / name).is_file():
                files.append(self.logdir / name)
        if "branch_mlp_checkpoint" in convention:
            files.append(self.logdir / convention["branch_mlp_checkpoint"])
        for path in (self.mlp_checkpoint, self.newhead_ft_config):
            if path is not None:
                files.append(path)
        return sorted(set(files))

    def relaxation_calculator(self) -> RelaxCalculator:
        from matsim_agents.backends.mlip.relaxation import hydragnn_relaxation_calculator

        return hydragnn_relaxation_calculator(self)


class UMAConfig(MLIPBackendConfig):
    """A UMA (FAIRChem) universal MLIP, pretrained by name or a fine-tuned checkpoint."""

    backend: Literal["uma"] = "uma"
    model_name: str = Field(
        "uma-s-1p1",
        description="fairchem pretrained model name (e.g. 'uma-s-1p1') or local checkpoint path.",
    )
    task_name: UMATask = Field(
        "omat",
        description="UMA task head (omat = inorganic bulk materials; omol = molecules/MOFs).",
    )
    precision: Literal["fp32", "fp64", "bf16"] | None = None
    charge: float = 0.0
    spin: float = 0.0
    dropout: MCDropoutInjectionConfig = Field(
        default_factory=MCDropoutInjectionConfig,
        description="Test-time dropout injection settings for MC-Dropout acquisition.",
    )

    fine_tuner: Literal["finetune_uma"] = Field(
        "finetune_uma", description="Fine-tuning recipe (fairchem UMA fine-tune)."
    )

    required_modules: ClassVar[tuple[str, ...]] = ("fairchem",)

    @classmethod
    def resolve_paths(cls, raw: dict[str, Any], base: Path) -> dict[str, Any]:
        # model_name is a registry name unless it names an existing file.
        name = raw.get("model_name")
        if isinstance(name, str):
            path = Path(name).expanduser()
            if not path.is_absolute() and (base / path).exists():
                raw["model_name"] = str((base / path).resolve())
        return raw

    def fine_tune_args(
        self,
        *,
        dataset: Path,
        out_dir: Path,
        epochs: int,
        references: Path,
        finetune_mode: str,
        seed: int,
    ) -> list[str]:
        args = [
            "--dataset", str(dataset),
            "--output-dir", str(out_dir),
            "--base-model", str(self.model_name),
            "--task-name", str(self.task_name),
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--elemental-reference-manifest", str(references),
            "--device", self.device,
        ]  # fmt: skip
        return args + (["--lora"] if finetune_mode == "lora" else [])

    def build_calculator(self, *, enable_mc_dropout: bool = False) -> Any:
        from matsim_agents.backends.mlip.calculator import build_uma_calculator

        return build_uma_calculator(self, enable_mc_dropout=enable_mc_dropout)

    def with_checkpoint(self, path: str | Path) -> UMAConfig:
        return self.model_copy(update={"model_name": str(path)})

    def ensemble_member(self, model: str) -> UMAConfig:
        return self.model_copy(update={"model_name": model})

    def model_identifier(self) -> str:
        settings = {"charge": self.charge, "precision": self.precision, "spin": self.spin}
        return f"uma:{self.model_name}:{self.task_name}:settings={_compact_json(settings)}"

    @property
    def model_reference(self) -> str:
        return self.model_name

    def local_checkpoint(self) -> Path | None:
        """The checkpoint loaded for a local ``model_name``; ``None`` for a registry name.

        A directory is a fine-tune output: fairchem writes
        ``<dir>/<run>/checkpoints/final/inference_ckpt.pt``, and the canonical
        location is searched first.
        """
        path = Path(self.model_name).expanduser()
        if not path.exists():
            return None
        if path.is_file():
            return path
        canonical = path / "checkpoints" / "final" / "inference_ckpt.pt"
        if canonical.is_file():
            return canonical
        matches = sorted(path.glob("**/inference_ckpt.pt"))
        if not matches:
            raise ValueError(f"UMA model directory has no inference_ckpt.pt: {path}")
        return matches[-1]

    def model_artifact(self) -> Path | None:
        return self.local_checkpoint()


class MACEConfig(MLIPBackendConfig):
    """A MACE foundation model (any mace-torch 0.3.16 loader) or local ``.model`` file.

    Materials variants such as MPA, OMAT, MATPES, and MH are ``mace_mp`` aliases.
    """

    backend: Literal["mace"] = "mace"
    family: MACEFamily = Field(
        "mace_mp",
        description=(
            "MACE calculator family: mace_mp, mace_off, mace_omol, mace_polar, "
            "mace_anicc, or checkpoint."
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
    precision: Literal["fp32", "fp64"] | None = Field(
        None,
        description="Calculator dtype -> MACE default_dtype (fp64 recommended for relaxation).",
    )
    dispersion: bool = Field(False, description="Add DFT-D3 dispersion correction (mace_mp only).")
    head: str | None = Field(
        None,
        description=(
            "Output head of a multi-head MACE model (e.g. 'omat_pbe' or 'mp_pbe_refit_add' "
            "for mh-0/mh-1); required for those models, ignored by single-head ones."
        ),
    )
    dropout: MCDropoutInjectionConfig = Field(
        default_factory=MCDropoutInjectionConfig,
        description=(
            "Test-time dropout injection for MC-Dropout acquisition (MACE has no native dropout)."
        ),
    )

    fine_tuner: Literal["finetune_mace"] = Field(
        "finetune_mace", description="Fine-tuning recipe (mace_run_train fine-tune)."
    )

    required_modules: ClassVar[tuple[str, ...]] = ("mace",)

    @model_validator(mode="after")
    def _validate_family_options(self) -> MACEConfig:
        if self.model == "medium":
            self.model = {
                "mace_omol": "extra_large",
                "mace_polar": "polar-1-m",
                "mace_anicc": "default",
            }.get(self.family, self.model)
        if self.family == "checkpoint" and not Path(self.model).is_file():
            raise ValueError("MACE checkpoint family requires model to be an existing file")
        if self.dispersion and self.family != "mace_mp":
            raise ValueError("MACE dispersion is supported only for family='mace_mp'")
        if self.head is not None and self.family not in {"mace_mp", "checkpoint"}:
            raise ValueError("MACE head selection is supported only for mace_mp and checkpoint")
        if self.family == "mace_mp" and self.model in {"mh-0", "mh-1"} and self.head is None:
            raise ValueError(f"multi-head MACE model {self.model!r} requires an explicit head")
        if self.family == "mace_anicc" and self.precision == "fp32":
            raise ValueError("MACE ANI-CC loader does not support precision='fp32'")
        return self

    @classmethod
    def resolve_paths(cls, raw: dict[str, Any], base: Path) -> dict[str, Any]:
        if raw.get("family") == "checkpoint" and raw.get("model"):
            raw["model"] = _resolve_path(base, raw["model"])
        return raw

    def fine_tune_args(
        self,
        *,
        dataset: Path,
        out_dir: Path,
        epochs: int,
        references: Path,
        finetune_mode: str,
        seed: int,
    ) -> list[str]:
        args = [
            "--dataset", str(dataset),
            "--output-dir", str(out_dir),
            "--family", str(self.family),
            "--base-model", str(self.model),
            "--precision", self.precision or "fp64",
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--elemental-reference-manifest", str(references),
            "--device", self.device,
        ]  # fmt: skip
        return args + (["--lora"] if finetune_mode == "lora" else [])

    def build_calculator(self, *, enable_mc_dropout: bool = False) -> Any:
        from matsim_agents.backends.mlip.calculator import build_mace_calculator

        return build_mace_calculator(self, enable_mc_dropout=enable_mc_dropout)

    def with_checkpoint(self, path: str | Path) -> MACEConfig:
        return self.model_copy(
            update={"family": "checkpoint", "model": str(path), "dispersion": False}
        )

    def ensemble_member(self, model: str) -> MACEConfig:
        values = self.model_dump()
        values["model"] = model
        values["ensemble_models"] = []
        if Path(model).is_file():
            values["family"] = "checkpoint"
            values["dispersion"] = False
        return MACEConfig.model_validate(values)

    def model_identifier(self) -> str:
        settings: dict[str, Any] = {"dispersion": self.dispersion, "precision": self.precision}
        if self.head is not None:
            settings["head"] = self.head
        return f"mace:{self.family}:{self.model}:settings={_compact_json(settings)}"

    @property
    def model_reference(self) -> str:
        return self.model if self.head is None else f"{self.model}:{self.head}"

    def model_artifact(self) -> Path | None:
        return Path(self.model) if self.family == "checkpoint" else None

    def unsupported_elements(self, elements: Iterable[str]) -> list[str]:
        covered = MACE_FAMILY_ELEMENTS.get(self.family)
        if covered is None:
            return []
        return sorted(set(elements) - covered)


# --------------------------------------------------------------------------- #
# Selection                                                                   #
# --------------------------------------------------------------------------- #

MLIP_BACKEND_CONFIGS: dict[str, type[MLIPBackendConfig]] = {
    "hydragnn": HydraGNNConfig,
    "uma": UMAConfig,
    "mace": MACEConfig,
}
MLIPBackend = Literal["hydragnn", "uma", "mace"]
MLIP_BACKENDS: tuple[str, ...] = get_args(MLIPBackend)

MLIPConfig = Annotated[HydraGNNConfig | UMAConfig | MACEConfig, Field(discriminator="backend")]
"""Any supported MLIP, selected by its ``backend`` field."""

_MLIP_ADAPTER: TypeAdapter[MLIPBackendConfig] = TypeAdapter(MLIPConfig)


def parse_mlip_config(value: Any) -> MLIPBackendConfig:
    """Validate a mapping (or an existing config) as the selected backend's config."""
    if isinstance(value, MLIPBackendConfig):
        return value
    return _MLIP_ADAPTER.validate_python(value)


def resolve_mlip_paths(block: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolve the selected backend's file paths in a raw ``mlip`` mapping against ``base``."""
    backend = MLIP_BACKEND_CONFIGS.get(str(block.get("backend")))
    return backend.resolve_paths(block, base) if backend is not None else block


def load_mlip_config(path: str | Path) -> MLIPBackendConfig:
    """Read the ``mlip`` block of a YAML file.

    ``${VAR}`` references inside the block are substituted from the environment
    and the file's ``vars:`` mapping; relative paths resolve against the file.
    """
    import yaml

    from matsim_agents.execution.variables import expand_variables, variable_defaults

    path = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: invalid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with an mlip block")
    block = data.get("mlip")
    if not isinstance(block, dict) or "backend" not in block:
        raise ValueError(f"{path}: mlip block must define backend: {' | '.join(MLIP_BACKENDS)}")
    defaults = variable_defaults(data.get("vars"), source=path)
    block = yaml.safe_load(expand_variables(yaml.safe_dump(block), defaults, source=path))
    return parse_mlip_config(resolve_mlip_paths(block, path.parent))
