"""Configuration models for MLIP calculators (HydraGNN, UMA, MACE)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


def resolve_hydragnn_inference_head(value: str | int | None) -> int | None:
    """Resolve a HydraGNN dataset name or branch index to the canonical head index."""
    if value is None:
        return None
    if isinstance(value, int) or str(value).strip().isdigit():
        index = int(value)
        if 0 <= index < len(HYDRAGNN_DATASET_HEADS):
            return index
        raise ValueError(f"HydraGNN inference head index must be in [0, 15], got {index}")
    normalized = str(value).strip().casefold()
    by_name = {name.casefold(): index for index, name in enumerate(HYDRAGNN_DATASET_HEADS)}
    if normalized not in by_name:
        raise ValueError(
            f"unknown HydraGNN inference head {value!r}; expected one of "
            f"{', '.join(HYDRAGNN_DATASET_HEADS)}"
        )
    return by_name[normalized]


# --------------------------------------------------------------------------- #
# Sub-configs                                                                 #
# --------------------------------------------------------------------------- #


class HydraGNNConfig(BaseModel):
    """Inputs needed to load and (optionally) retrain a HydraGNN MLFF model."""

    # Accept both the field name and the YAML-facing alias 'mlp_checkpoint'.
    model_config = ConfigDict(populate_by_name=True)

    logdir: Path = Field(..., description="HydraGNN logdir with config.json + checkpoint.")
    hydragnn_root: Path | None = Field(
        None,
        description="HydraGNN source checkout used by built-in fine-tuners.",
    )
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
    mlp_device: Literal["cuda", "cpu", "xpu"] = "cuda"
    inference_head: str | int | None = Field(
        None,
        description=(
            "Optional decoding head selected by canonical dataset name or index 0..15. "
            "When set, inference bypasses BranchWeightMLP and uses only that head."
        ),
    )
    ensemble_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Paths to additional HydraGNN logdirs forming the ensemble. If "
            "non-empty, ensemble disagreement is available as an acquisition."
        ),
    )

    @model_validator(mode="after")
    def _validate_inference_head(self) -> HydraGNNConfig:
        resolve_hydragnn_inference_head(self.inference_head)
        if self.inference_head is not None and self.newhead_ft_config is not None:
            raise ValueError("inference_head cannot be combined with a single new-head model")
        return self


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
    device: Literal["cuda", "cpu", "xpu"] = "cuda"
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


MACEFamily = Literal[
    "mace_mp",
    "mace_off",
    "mace_omol",
    "mace_polar",
    "mace_anicc",
    "checkpoint",
]


class MACEConfig(BaseModel):
    """Inputs to load a MACE MLIP as an ASE calculator.

    Supports every foundation-model loader shipped by mace-torch 0.3.16, plus
    local fine-tuned ``.model`` checkpoints. Materials variants such as MPA,
    OMAT, MATPES, and MH are selected as ``mace_mp`` model aliases.
    """

    model_config = ConfigDict(populate_by_name=True)

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
    device: Literal["cuda", "cpu", "xpu"] = "cuda"
    precision: Literal["fp32", "fp64"] | None = Field(
        None,
        description="Calculator dtype -> MACE default_dtype (fp64 recommended for relaxation).",
    )
    dispersion: bool = Field(False, description="Add DFT-D3 dispersion correction (mace_mp only).")
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
        if self.family == "mace_anicc" and self.precision == "fp32":
            raise ValueError("MACE ANI-CC loader does not support precision='fp32'")
        return self


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
