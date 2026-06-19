"""Lazy builders for the MLP surrogate ASE calculators.

Two backends are supported behind a single factory, :func:`make_mlip_calculator`:

* **HydraGNN** — loaded from a logdir (``config.json`` + checkpoint), wrapping
  the pattern in ``HydraGNN/examples/.../structure_optimization_ASE.py``.
* **UMA** — Meta FAIR's universal MLIP via ``fairchem`` (``FAIRChemCalculator``).
  Optionally has test-time dropout injected so MC-Dropout acquisition works on
  this otherwise-deterministic foundation model.

Heavy / optional dependencies (torch, hydragnn, fairchem) are imported lazily
so the rest of matsim-agents (and pytest collection) keeps working without them.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from matsim_agents.active_learning.config import HydraGNNConfig, MLIPConfig, UMAConfig

log = logging.getLogger(__name__)


def make_mlip_calculator(
    cfg: MLIPConfig,
    *,
    enable_mc_dropout: bool = False,
    logdir_override: str | Path | None = None,
):
    """Build the primary ASE calculator for whichever backend ``cfg`` selects.

    ``enable_mc_dropout`` only affects backends without native dropout (UMA):
    when True, dropout is injected so :func:`uncertainty.score_mc_dropout` can
    produce non-zero variance. HydraGNN ignores it (it has native dropout).
    ``logdir_override`` is used by the ensemble path (HydraGNN only).
    """
    if cfg.backend == "hydragnn":
        assert cfg.hydragnn is not None  # guaranteed by MLIPConfig validator
        return build_hydragnn_calculator(cfg.hydragnn, logdir_override=logdir_override)
    if cfg.backend == "uma":
        assert cfg.uma is not None
        return build_uma_calculator(cfg.uma, enable_mc_dropout=enable_mc_dropout)
    raise ValueError(f"Unknown mlip.backend: {cfg.backend!r}")


def build_hydragnn_calculator(cfg: HydraGNNConfig, logdir_override: str | Path | None = None):
    """Build a ready-to-use ASE calculator from a HydraGNN logdir.

    ``logdir_override`` is used by the ensemble path to load each member from
    its own logdir while reusing the rest of ``cfg``.
    """
    # Heavy imports kept inside the function so this module stays cheap to import.
    import json

    import torch
    from hydragnn.utils.model.load_existing_model import load_existing_model_config

    from matsim_agents.tools.relaxation import (
        _build_calculator,  # type: ignore[attr-defined]
    )

    # We replicate the loading sequence from tools/relaxation.py inline to
    # avoid coupling to the LangGraph @tool wrapper there. If/when that module
    # exposes a public loader, this should switch to it.

    logdir = Path(logdir_override) if logdir_override is not None else cfg.logdir
    config_path = logdir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"HydraGNN config.json not found in {logdir}")

    with open(config_path) as f:
        hcfg = json.load(f)

    radius = (
        cfg.radius if cfg.radius is not None else hcfg["NeuralNetwork"]["Architecture"]["radius"]
    )
    max_neighbours = (
        cfg.max_neighbours
        if cfg.max_neighbours is not None
        else hcfg["NeuralNetwork"]["Architecture"]["max_neighbours"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_existing_model_config(
        config=hcfg,
        model_name=hcfg["NeuralNetwork"].get("model_name", "EquivariantPNAStack"),
    )
    if cfg.checkpoint:
        ckpt_path = (
            cfg.checkpoint if os.path.isabs(cfg.checkpoint) else str(logdir / cfg.checkpoint)
        )
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.to(device).eval()

    mlp = None
    if cfg.hydragnn_branch_mlp_checkpoint is not None:
        mlp_state = torch.load(str(cfg.hydragnn_branch_mlp_checkpoint), map_location=cfg.mlp_device)
        # The MLP architecture lives with the upstream training script; we only
        # forward the state-dict — callers needing the MLP must rely on
        # tools/relaxation.py's loader once it's promoted to a public API.
        mlp = mlp_state

    autocast_ctx = torch.autocast(device_type=device.type, enabled=False)
    mlp_autocast_ctx = torch.autocast(device_type=cfg.mlp_device, enabled=False)

    return _build_calculator(
        model=model,
        mlp=mlp,
        radius=radius,
        max_neighbours=max_neighbours,
        param_dtype=torch.float32,
        autocast_ctx=autocast_ctx,
        device=device,
        num_branches=hcfg["NeuralNetwork"]["Architecture"].get("num_branches", 1),
        mlp_device=cfg.mlp_device,
        mlp_autocast_ctx=mlp_autocast_ctx,
        unified_mlp_gnn_stack=False,
        charge=cfg.charge,
        spin=cfg.spin,
    )


def build_ensemble(cfg: MLIPConfig, *, enable_mc_dropout: bool = False) -> list[Any]:
    """Return a list of calculators: the primary model plus all ensemble members.

    * HydraGNN: primary logdir + each ``ensemble_paths`` logdir.
    * UMA: primary ``model_name`` + each ``ensemble_models`` entry.
    """
    if cfg.backend == "hydragnn":
        assert cfg.hydragnn is not None
        calcs = [build_hydragnn_calculator(cfg.hydragnn)]
        for p in cfg.hydragnn.ensemble_paths:
            calcs.append(build_hydragnn_calculator(cfg.hydragnn, logdir_override=p))
        return calcs
    if cfg.backend == "uma":
        assert cfg.uma is not None
        calcs = [build_uma_calculator(cfg.uma, enable_mc_dropout=enable_mc_dropout)]
        for name in cfg.uma.ensemble_models:
            member = cfg.uma.model_copy(update={"model_name": name})
            calcs.append(build_uma_calculator(member, enable_mc_dropout=enable_mc_dropout))
        return calcs
    raise ValueError(f"Unknown mlip.backend: {cfg.backend!r}")


# --------------------------------------------------------------------------- #
# UMA (fairchem) backend                                                      #
# --------------------------------------------------------------------------- #


def _find_torch_module(*roots: Any):
    """Best-effort search for the underlying ``torch.nn.Module`` inside a
    fairchem calculator / predict-unit.

    The attribute layout differs across fairchem versions, so we breadth-first
    search a small set of common attribute names and return the first
    ``nn.Module`` found. Returns ``None`` if none is reachable.
    """
    import torch.nn as nn

    candidate_attrs = (
        "model",
        "module",
        "potential",
        "net",
        "backbone",
        "predictor",
        "predict_unit",
        "trainer",
        "_model",
    )
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        obj = stack.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, nn.Module):
            return obj
        for name in candidate_attrs:
            if hasattr(obj, name):
                stack.append(getattr(obj, name))
    return None


def build_uma_calculator(cfg: UMAConfig, *, enable_mc_dropout: bool = False):
    """Build an ASE calculator backed by a UMA (fairchem) universal MLIP.

    When ``enable_mc_dropout`` and ``cfg.dropout.enabled`` are both True, dropout
    is injected into the underlying torch model so MC-Dropout acquisition yields
    non-zero variance. The dropout is dormant for ordinary energy/force calls.
    """
    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "The UMA backend requires the 'fairchem-core' package. Install it with "
            "`pip install fairchem-core` (and accept the UMA model license on "
            "Hugging Face), then set mlip.backend: uma."
        ) from exc

    predictor = pretrained_mlip.get_predict_unit(cfg.model_name, device=cfg.device)
    calc = FAIRChemCalculator(predictor, task_name=cfg.task_name)

    # Expose the underlying torch module as `.model` so uncertainty.score_mc_dropout
    # can toggle dropout, and (optionally) inject dropout for MC-Dropout UQ.
    torch_model = _find_torch_module(calc, predictor)
    if torch_model is None:
        log.warning(
            "UMA calculator loaded but the underlying torch module could not be "
            "located; MC-Dropout acquisition will not work for this backend. "
            "Use acquisition.strategy: ensemble (mlip.uma.ensemble_models) or random."
        )
    else:
        try:
            calc.model = torch_model  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - some calculators forbid new attrs
            log.debug("Could not attach `.model` to the UMA calculator instance.")
        if enable_mc_dropout and cfg.dropout.enabled:
            from matsim_agents.active_learning.uncertainty import inject_inference_dropout

            n = inject_inference_dropout(
                torch_model,
                p=cfg.dropout.p,
                target_layers=cfg.dropout.target_layers,
                max_layers=cfg.dropout.max_layers,
            )
            log.info(
                "Injected test-time dropout into %d UMA layer(s) (p=%g, target=%s) "
                "for MC-Dropout acquisition.",
                n,
                cfg.dropout.p,
                cfg.dropout.target_layers,
            )
        torch_model.eval()  # keep injected dropout dormant until scoring

    return calc
