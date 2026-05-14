"""Lazy builder for a HydraGNN ASE calculator.

This wraps the pattern used in
``HydraGNN/examples/multidataset_hpo_sc26/structure_optimization_ASE.py`` so
the AL loop can load any HydraGNN model + (optional) BranchWeightMLP into an
``ase.calculators.calculator.Calculator``.

We import HydraGNN lazily so the rest of matsim-agents (and pytest collection)
keeps working without HydraGNN installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from matsim_agents.active_learning.config import HydraGNNConfig


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
    if cfg.mlp_checkpoint is not None:
        mlp_state = torch.load(str(cfg.mlp_checkpoint), map_location=cfg.mlp_device)
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


def build_ensemble(cfg: HydraGNNConfig) -> list[Any]:
    """Return a list of calculators: the primary model plus all ensemble members."""
    calcs = [build_hydragnn_calculator(cfg)]
    for p in cfg.ensemble_paths:
        calcs.append(build_hydragnn_calculator(cfg, logdir_override=p))
    return calcs
