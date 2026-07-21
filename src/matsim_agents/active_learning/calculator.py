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


def _build_single_head_calculator(
    model,
    *,
    radius,
    max_neighbours,
    param_dtype,
    device,
    charge,
    spin,
):
    """ASE calculator for a single-branch (``num_branches==1``) HydraGNN model.

    Bypasses the BranchWeightMLP entirely: the decoder uses ``branch-0``
    directly, so energy = ``model(data)[0]`` and forces = ``-dE/dx`` via autograd.
    Used for the 'drop-all-heads + new head' fine-tune models.
    """
    import torch
    from ase.calculators.calculator import Calculator, all_changes

    from matsim_agents.tools.relaxation import _atoms_to_graph  # type: ignore[attr-defined]

    class SingleHeadHydraGNNCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.graph_attr = torch.tensor([charge, spin], dtype=torch.float32)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            data = _atoms_to_graph(atoms, self.graph_attr, radius, max_neighbours).to(device)
            # Keep float tensors on the model precision regardless of the ambient
            # default dtype at call time.
            data.pos = data.pos.to(param_dtype)
            if hasattr(data, "cell") and data.cell is not None:
                data.cell = data.cell.to(param_dtype)
            data.x = data.x.to(param_dtype)
            data.pos.requires_grad_(True)
            with torch.enable_grad():
                pred = model(data)
                energy = pred[0] if isinstance(pred, (list, tuple)) else pred
                energy = energy.squeeze(-1).sum()
                forces = -torch.autograd.grad(energy, data.pos)[0]
            self.results["energy"] = float(energy.detach())
            self.results["forces"] = forces.detach().cpu().numpy()

    return SingleHeadHydraGNNCalculator()


def build_hydragnn_calculator(cfg: HydraGNNConfig, logdir_override: str | Path | None = None):
    """Build a ready-to-use ASE calculator from a HydraGNN logdir.

    ``logdir_override`` is used by the ensemble path to load each member from
    its own logdir while reusing the rest of ``cfg``.
    """
    # Heavy imports kept inside the function so this module stays cheap to import.
    import json
    import sys

    import torch

    from matsim_agents.tools.relaxation import (
        _build_calculator,  # type: ignore[attr-defined]
    )

    logdir = Path(logdir_override) if logdir_override is not None else cfg.logdir
    config_path = logdir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"HydraGNN config.json not found in {logdir}")

    # ``inference_fused`` lives in the HydraGNN example dir alongside the GFM
    # logdir (``.../multidataset_hpo_sc26/<gfm_logdir>``). Make the eval path
    # self-sufficient by putting that dir on sys.path here, so it does not rely
    # on the caller (or the fine-tune module) having imported it first. Also
    # honour HYDRAGNN_ROOT if set, matching finetune_hydragnn's resolution.
    _example_dirs = [logdir.parent]
    _root_env = os.environ.get("HYDRAGNN_ROOT")
    if _root_env:
        _example_dirs.append(Path(_root_env) / "examples" / "multidataset_hpo_sc26")
    for _d in _example_dirs:
        if (_d / "inference_fused.py").is_file() and str(_d) not in sys.path:
            sys.path.insert(0, str(_d))

    # --- new-head (single-branch) fine-tune models -----------------------------
    # When ``newhead_ft_config`` is set, the checkpoint was produced by
    # finetune_hydragnn_newhead: the 16-branch decoder was replaced by a single
    # fresh ``branch-0`` head. Reproduce that surgery here (build the 16-branch
    # backbone, apply ORNL ``update_architecture``) so the checkpoint loads into a
    # byte-identical model, then use direct single-head inference (no branch-MLP).
    # This bespoke path cannot go through ``load_fused_stack``, so it is handled
    # first with an inline model build.
    if cfg.newhead_ft_config is not None:
        from hydragnn.models.create import create_model_config
        from hydragnn.train.train_validate_test import resolve_precision

        from matsim_agents.active_learning.finetune_hydragnn_newhead import (
            apply_newhead_surgery,
        )

        with open(config_path) as f:
            hcfg = json.load(f)

        # vesin's neighbor-list builder (via RadiusGraphPBC) requires a strict
        # Python float cutoff; HydraGNN configs commonly store radius as an int.
        radius = float(
            cfg.radius if cfg.radius is not None else hcfg["NeuralNetwork"]["Architecture"]["radius"]
        )
        max_neighbours = (
            cfg.max_neighbours
            if cfg.max_neighbours is not None
            else hcfg["NeuralNetwork"]["Architecture"]["max_neighbours"]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The model precision is dictated by the training config (the fp64 GFM
        # must be run in fp64) so keep every tensor on the same dtype.
        precision_str = (
            cfg.precision
            if cfg.precision is not None
            else hcfg["NeuralNetwork"]["Training"].get("precision", "fp32")
        )
        _, param_dtype, _ = resolve_precision(precision_str)
        torch.set_default_dtype(param_dtype)

        model = create_model_config(
            config=hcfg["NeuralNetwork"],
            verbosity=hcfg.get("Verbosity", {}).get("level", 0),
        )
        # create_model_config may overwrite the default dtype; restore + cast.
        torch.set_default_dtype(param_dtype)
        model = model.to(dtype=param_dtype, device=device)

        with open(cfg.newhead_ft_config) as f:
            newhead = json.load(f)
        model = apply_newhead_surgery(
            model, newhead["ft_config"], ft_repo=cfg.ft_repo, freeze=False
        )
        torch.set_default_dtype(param_dtype)
        model = model.to(dtype=param_dtype, device=device)
        if cfg.checkpoint:
            ckpt_path = (
                cfg.checkpoint if os.path.isabs(cfg.checkpoint) else str(logdir / cfg.checkpoint)
            )
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("model_state_dict", state)
            state_dict = {
                (k[len("module.") :] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        return _build_single_head_calculator(
            model=model,
            radius=radius,
            max_neighbours=max_neighbours,
            param_dtype=param_dtype,
            device=device,
            charge=cfg.charge,
            spin=cfg.spin,
        )

    # --- normal (multi-branch fused-stack) path --------------------------------
    # Delegate to the authoritative fused-stack loader shipped alongside the
    # HydraGNN example (also used by tools/relaxation.py). It builds the model,
    # reconstructs the BranchWeightMLP from its checkpoint, and returns matching
    # dtypes/device/autocast contexts. Reimplementing this here previously drifted
    # from the proven path (wrong model-build API, unstripped DDP "module." keys,
    # int radius that vesin rejects, and a raw state-dict passed as the MLP).
    from inference_fused import load_fused_stack  # provided alongside the HydraGNN example

    mlp_checkpoint = (
        str(cfg.hydragnn_branch_mlp_checkpoint)
        if cfg.hydragnn_branch_mlp_checkpoint is not None
        else None
    )

    (
        model,
        mlp,
        config,
        device,
        autocast_ctx,
        param_dtype,
        num_branches,
        mlp_device,
        mlp_autocast_ctx,
        unified_mlp_gnn_stack,
        _gnn_prec,
        _mlp_prec,
    ) = load_fused_stack(
        str(logdir),
        cfg.checkpoint,
        mlp_checkpoint,
        cfg.precision,
        cfg.precision,
        cfg.mlp_device,
    )

    arch = config["NeuralNetwork"]["Architecture"]
    radius = float(cfg.radius) if cfg.radius is not None else float(arch.get("radius", 5.0))
    max_neighbours = (
        int(cfg.max_neighbours)
        if cfg.max_neighbours is not None
        else int(arch.get("max_neighbours", 20))
    )

    return _build_calculator(
        model=model,
        mlp=mlp,
        radius=radius,
        max_neighbours=max_neighbours,
        param_dtype=param_dtype,
        autocast_ctx=autocast_ctx,
        device=device,
        num_branches=num_branches,
        mlp_device=mlp_device,
        mlp_autocast_ctx=mlp_autocast_ctx,
        unified_mlp_gnn_stack=unified_mlp_gnn_stack,
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


def _resolve_finetuned_uma_checkpoint(model_name: str) -> Path | None:
    """If ``model_name`` points at a fine-tuned UMA checkpoint on disk, return
    the resolved ``.pt`` path; otherwise return ``None`` (a registered name).

    The AL trainer's UMA fine-tune writes its final checkpoint under the fairchem
    convention ``<output-dir>/<timestamp_id>/checkpoints/final/inference_ckpt.pt``,
    so a directory is accepted and the canonical location is searched first.
    """
    p = Path(str(model_name))
    if not p.exists():
        return None
    if p.is_file():
        return p
    canonical = p / "checkpoints" / "final" / "inference_ckpt.pt"
    if canonical.is_file():
        return canonical
    matches = sorted(p.glob("**/inference_ckpt.pt"))
    return matches[-1] if matches else None


def build_uma_calculator(cfg: UMAConfig, *, enable_mc_dropout: bool = False):
    """Build an ASE calculator backed by a UMA (fairchem) universal MLIP.

    ``cfg.model_name`` may be either a registered pretrained model name (e.g.
    ``uma-s-1p1``) or a path to a locally fine-tuned checkpoint produced by the
    AL trainer; the latter is loaded via ``load_predict_unit`` so the in-loop
    fine-tuned model is picked up on the next iteration.

    When ``enable_mc_dropout`` and ``cfg.dropout.enabled`` are both True, dropout
    is injected into the underlying torch model so MC-Dropout acquisition yields
    non-zero variance. The dropout is dormant for ordinary energy/force calls.
    """
    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.units.mlip_unit import load_predict_unit
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "The UMA backend requires the 'fairchem-core' package. Install it with "
            "`pip install fairchem-core` (and accept the UMA model license on "
            "Hugging Face), then set mlip.backend: uma."
        ) from exc

    ckpt = _resolve_finetuned_uma_checkpoint(cfg.model_name)
    if ckpt is not None:
        log.info("Loading fine-tuned UMA checkpoint from %s", ckpt)
        predictor = load_predict_unit(str(ckpt), device=cfg.device)
    else:
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
