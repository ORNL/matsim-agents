"""Lazy builders for the MLIP ASE calculators.

Each backend configuration (:mod:`matsim_agents.backends.mlip.config`) builds its
calculator through its ``build_calculator`` method, which delegates to the
builder for that backend here:

* **HydraGNN** — loaded from a logdir (``config.json`` + checkpoint), wrapping
  the pattern in ``HydraGNN/examples/.../structure_optimization_ASE.py``.
* **UMA** — Meta FAIR's universal MLIP via ``fairchem`` (``FAIRChemCalculator``).
* **MACE** — any mace-torch foundation loader or a local ``.model`` file.

UMA and MACE optionally have test-time dropout injected so MC-Dropout
acquisition works on these otherwise-deterministic models.

Heavy / optional dependencies (torch, hydragnn, fairchem) are imported lazily
so the rest of matsim-agents (and pytest collection) keeps working without them.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from matsim_agents.backends.mlip.config import (
    HYDRAGNN_DATASET_HEADS,
    HYDRAGNN_INFERENCE_HEAD_INDEX,
    HydraGNNConfig,
    MACEConfig,
    UMAConfig,
    parse_mlip_config,
)
from matsim_agents.backends.mlip.devices import default_torch_device
from matsim_agents.backends.mlip.uma_artifacts import (
    load_uma_predict_unit_from_bundle,
    resolve_uma_artifact_bundle,
)

log = logging.getLogger(__name__)


def make_mlip_calculator(cfg: Any, *, enable_mc_dropout: bool = False) -> Any:
    """Build the primary ASE calculator of any MLIP configuration.

    ``enable_mc_dropout`` affects only backends without native dropout (UMA,
    MACE): dropout is injected so :func:`uncertainty.score_mc_dropout` can
    produce non-zero variance.
    """
    return parse_mlip_config(cfg).build_calculator(enable_mc_dropout=enable_mc_dropout)


def _build_single_head_calculator(
    model,
    *,
    radius,
    max_neighbours,
    param_dtype,
    device,
    charge,
    spin,
    autocast_dtype=None,
    autocast_ctx=None,
):
    """ASE calculator for a single-branch HydraGNN whose only branch is (or derives from) branch-7.

    The model must already hold exactly one branch (see
    :func:`~matsim_agents.backends.mlip.hydragnn_newhead.keep_only_branch_7`).
    Energy = ``model(data)[0]`` and forces = ``-dE/dx`` via autograd.
    """
    import torch
    from ase.calculators.calculator import Calculator, all_changes

    from matsim_agents.backends.mlip.hydragnn_precision import hydragnn_autocast
    from matsim_agents.backends.mlip.relaxation import atoms_to_graph

    num_branches = getattr(model, "num_branches", 1)
    if num_branches != 1:
        raise ValueError(
            f"HydraGNN model has {num_branches} branches; prune it to branch-7 (OMat24) first"
        )

    class SingleHeadHydraGNNCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.model = model
            self.graph_attr = torch.tensor([charge, spin], dtype=torch.float32)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            data = atoms_to_graph(atoms, self.graph_attr, radius, max_neighbours).to(device)
            # Keep float tensors on the model precision regardless of the ambient
            # default dtype at call time.
            data.pos = data.pos.to(param_dtype)
            if hasattr(data, "cell") and data.cell is not None:
                data.cell = data.cell.to(param_dtype)
            data.x = data.x.to(param_dtype)
            data.pos.requires_grad_(True)
            ctx = (
                autocast_ctx
                if autocast_ctx is not None
                else hydragnn_autocast(device, autocast_dtype)
            )
            with torch.enable_grad(), ctx:
                pred = model(data)
                if isinstance(pred, (list, tuple)):
                    energy = pred[0]
                elif isinstance(pred, dict) and "graph" in pred:
                    energy = pred["graph"][0]
                else:
                    energy = pred
                energy = energy.squeeze(-1).sum()
                forces = -torch.autograd.grad(energy, data.pos)[0]
            self.results["energy"] = float(energy.detach())
            self.results["forces"] = forces.detach().cpu().numpy()

    return SingleHeadHydraGNNCalculator()


def _apply_energy_convention(
    cfg: HydraGNNConfig, logdir: Path
) -> tuple[HydraGNNConfig, dict | None]:
    """Verify the logdir's training artifacts and adopt its recorded inference settings."""
    import json

    from ase.calculators.calculator import Calculator

    from matsim_agents.backends.mlip.hydragnn_energy import restore_hydragnn_total_energy

    restore_hydragnn_total_energy(Calculator(), logdir, cfg.checkpoint)
    convention_path = logdir / "energy-convention.json"
    if not convention_path.exists():
        return cfg, None
    cfg = cfg.model_copy(deep=True)
    convention = json.loads(convention_path.read_text())
    if "branch_mlp_checkpoint" in convention or "routed_branches" in convention:
        raise ValueError(
            f"{convention_path} is a routed BranchWeightMLP model; matsim-agents uses only "
            "HydraGNN branch-7 (OMat24)"
        )
    if cfg.checkpoint is None:
        cfg.checkpoint = convention["checkpoint"]
    if (logdir / "newhead.json").exists():
        cfg.newhead_ft_config = logdir / "newhead.json"
    return cfg, convention


def _add_hydragnn_example_dirs(logdir: Path) -> None:
    """Put the HydraGNN example dir (with ``inference_fused``) on ``sys.path``.

    The scripts come from the checkout whose ``hydragnn`` is imported, never from the
    directory beside the logdir: the weights may sit in a different HydraGNN tree
    than the commit recorded for them (``verify_hydragnn_source`` checks that commit).
    """
    import sys

    from matsim_agents.backends.mlip.hydragnn_source import imported_hydragnn_root

    example_dirs = [imported_hydragnn_root() / "examples" / "multidataset_hpo_sc26"]
    for directory in example_dirs:
        if (directory / "inference_fused.py").is_file() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def _branch_7_hydragnn_calculator(cfg: HydraGNNConfig, logdir: Path):
    """Calculator for the pretrained GFM with every branch except branch-7 (OMat24) deleted."""
    from inference_random_structures import find_checkpoint, load_config_and_model

    from matsim_agents.backends.mlip.hydragnn_checkpoint import restore_hydragnn_checkpoint
    from matsim_agents.backends.mlip.hydragnn_energy import restore_hydragnn_total_energy
    from matsim_agents.backends.mlip.hydragnn_newhead import keep_only_branch_7

    model, config, device, autocast_ctx, param_dtype = load_config_and_model(
        str(logdir), cfg.checkpoint, cfg.precision
    )
    checkpoint_path = find_checkpoint(str(logdir), cfg.checkpoint)
    # Restore the full 16-branch checkpoint strictly (so nothing is silently
    # missing), then delete the other 15 branches before the model is ever run.
    restore_hydragnn_checkpoint(model, checkpoint_path, device=device)
    load_report = model._hydragnn_checkpoint_load_report
    model = keep_only_branch_7(model)
    arch = config["NeuralNetwork"]["Architecture"]
    calculator = _build_single_head_calculator(
        model,
        autocast_ctx=autocast_ctx,
        radius=float(cfg.radius if cfg.radius is not None else arch.get("radius", 5.0)),
        max_neighbours=int(
            cfg.max_neighbours if cfg.max_neighbours is not None else arch.get("max_neighbours", 20)
        ),
        param_dtype=param_dtype,
        device=device,
        charge=cfg.charge,
        spin=cfg.spin,
    )
    calculator.inference_head_index = HYDRAGNN_INFERENCE_HEAD_INDEX
    calculator.inference_head_name = HYDRAGNN_DATASET_HEADS[HYDRAGNN_INFERENCE_HEAD_INDEX]
    calculator.hydragnn_checkpoint_load_report = load_report
    return restore_hydragnn_total_energy(calculator, logdir, cfg.checkpoint)


def _newhead_hydragnn_calculator(cfg: HydraGNNConfig, logdir: Path):
    """Calculator for a finetune_hydragnn_newhead model (single head initialised from branch-7).

    Reproduces the head surgery (16-branch backbone + ORNL ``update_architecture``)
    so the checkpoint loads into a byte-identical model.
    """
    import json

    import torch

    from matsim_agents.backends.mlip.hydragnn_energy import restore_hydragnn_total_energy

    config_path = logdir / "config.json"
    from hydragnn.models.create import create_model_config
    from hydragnn.train.train_validate_test import resolve_precision

    from matsim_agents.backends.mlip.hydragnn_newhead import apply_newhead_surgery

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
    device = torch.device(default_torch_device())

    # The model precision is dictated by the training config (the fp64 GFM
    # must be run in fp64) so keep every tensor on the same dtype.
    precision_str = (
        cfg.precision
        if cfg.precision is not None
        else hcfg["NeuralNetwork"]["Training"].get("precision", "fp32")
    )
    _, param_dtype, autocast_dtype = resolve_precision(precision_str)
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
    if newhead.get("head_index") != HYDRAGNN_INFERENCE_HEAD_INDEX:
        raise ValueError(
            f"{cfg.newhead_ft_config}: head_index={newhead.get('head_index')!r}; only models "
            "fine-tuned from branch-7 (OMat24) are allowed"
        )
    model = apply_newhead_surgery(model, newhead["ft_config"], ft_repo=cfg.ft_repo, freeze=False)
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
        from matsim_agents.backends.mlip.hydragnn_checkpoint import load_hydragnn_state_dict

        load_hydragnn_state_dict(model, state_dict)
    model.to(device).eval()
    calculator = _build_single_head_calculator(
        model=model,
        radius=radius,
        max_neighbours=max_neighbours,
        param_dtype=param_dtype,
        device=device,
        charge=cfg.charge,
        spin=cfg.spin,
        autocast_dtype=autocast_dtype,
    )
    return restore_hydragnn_total_energy(calculator, logdir, cfg.checkpoint)


def build_hydragnn_calculator(cfg: HydraGNNConfig):
    """Build a ready-to-use ASE calculator from a HydraGNN logdir."""
    logdir = cfg.logdir
    if not (logdir / "config.json").is_file():
        raise FileNotFoundError(f"HydraGNN config.json not found in {logdir}")
    cfg, _ = _apply_energy_convention(cfg, logdir)
    _add_hydragnn_example_dirs(logdir)
    from matsim_agents.backends.mlip.hydragnn_source import verify_hydragnn_source

    verify_hydragnn_source(logdir)
    if cfg.newhead_ft_config is not None:
        return _newhead_hydragnn_calculator(cfg, logdir)
    return _branch_7_hydragnn_calculator(cfg, logdir)


def build_ensemble(cfg: Any, *, enable_mc_dropout: bool = False) -> list[Any]:
    """Calculators for the primary model followed by every ensemble member."""
    return [
        member.build_calculator(enable_mc_dropout=enable_mc_dropout)
        for member in parse_mlip_config(cfg).ensemble_configs()
    ]


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

    ckpt = cfg.local_checkpoint()
    if ckpt is not None:
        log.info("Loading fine-tuned UMA checkpoint from %s", ckpt)
        predictor = load_predict_unit(str(ckpt), device=cfg.device)
    else:
        bundle = resolve_uma_artifact_bundle(cfg.model_name)
        if bundle is not None:
            log.info("Loading pretrained UMA bundle from %s", bundle.checkpoint.parent)
            predictor = load_uma_predict_unit_from_bundle(bundle, device=cfg.device)
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
            "Use acquisition.strategy: ensemble (mlip.ensemble_models) or random."
        )
    else:
        try:
            calc.model = torch_model  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - some calculators forbid new attrs
            log.debug("Could not attach `.model` to the UMA calculator instance.")
        if enable_mc_dropout and cfg.dropout.enabled:
            from matsim_agents.backends.mlip.dropout import inject_inference_dropout

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


def build_mace_calculator(cfg: MACEConfig, *, enable_mc_dropout: bool = False):
    """Build an ASE calculator backed by a MACE MLIP.

    Loads any foundation family exposed by mace-torch 0.3.16, or a local
    fine-tuned ``.model`` checkpoint when ``cfg.family == 'checkpoint'``.

    When ``enable_mc_dropout`` and ``cfg.dropout.enabled`` are both True, dropout
    is injected into the underlying torch model so MC-Dropout acquisition yields
    non-zero variance. The dropout is dormant for ordinary energy/force calls.
    """
    from matsim_agents.backends.mlip.mace_inference import configure_mace_inference_imports

    configure_mace_inference_imports()
    try:
        from mace.calculators import (
            MACECalculator,
            mace_anicc,
            mace_mp,
            mace_off,
            mace_omol,
            mace_polar,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "The MACE backend requires the 'mace-torch' package. Install it with "
            "INSTALL_MACE=1 deployments/<facility>/setup/install.sh (into .venv-mace) "
            "or `pip install mace-torch`, then set mlip.backend: mace."
        ) from exc

    # MACE keeps energies/forces at the calculator dtype; fp64 is preferred for
    # relaxation, so an unset precision defaults to float64.
    dtype = {"fp32": "float32", "fp64": "float64", None: "float64"}[cfg.precision]
    head = {"head": cfg.head} if cfg.head is not None else {}

    if cfg.family == "checkpoint":
        ckpt = Path(cfg.model)
        if not ckpt.is_file():
            raise FileNotFoundError(f"MACE checkpoint not found: {ckpt}")
        log.info("Loading MACE checkpoint from %s", ckpt)
        calc = MACECalculator(
            model_paths=[str(ckpt)], device=cfg.device, default_dtype=dtype, **head
        )
    elif cfg.family == "mace_off":
        calc = mace_off(model=cfg.model, device=cfg.device, default_dtype=dtype)
    elif cfg.family == "mace_omol":
        calc = mace_omol(model=cfg.model, device=cfg.device, default_dtype=dtype)
    elif cfg.family == "mace_polar":
        calc = mace_polar(model=cfg.model, device=cfg.device, default_dtype=dtype)
    elif cfg.family == "mace_anicc":
        calc = mace_anicc(
            model_path=None if cfg.model == "default" else cfg.model,
            device=cfg.device,
        )
    else:
        calc = mace_mp(
            model=cfg.model,
            device=cfg.device,
            default_dtype=dtype,
            dispersion=cfg.dispersion,
            **head,
        )

    # Expose the underlying torch module as `.model` so uncertainty.score_mc_dropout
    # can toggle dropout, and (optionally) inject dropout for MC-Dropout UQ. MACE
    # calculators hold their nets in a `.models` list.
    torch_model = None
    models = getattr(calc, "models", None)
    if models:
        torch_model = models[0]
    elif getattr(calc, "model", None) is not None:
        torch_model = calc.model
    if torch_model is None:
        log.warning(
            "MACE calculator loaded but the underlying torch module could not be "
            "located; MC-Dropout acquisition will not work for this backend. "
            "Use acquisition.strategy: ensemble (mlip.ensemble_models) or random."
        )
    else:
        try:
            calc.model = torch_model  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - some calculators forbid new attrs
            log.debug("Could not attach `.model` to the MACE calculator instance.")
        if enable_mc_dropout and cfg.dropout.enabled:
            from matsim_agents.backends.mlip.dropout import inject_inference_dropout

            n = inject_inference_dropout(
                torch_model,
                p=cfg.dropout.p,
                target_layers=cfg.dropout.target_layers,
                max_layers=cfg.dropout.max_layers,
            )
            log.info(
                "Injected test-time dropout into %d MACE layer(s) (p=%g, target=%s) "
                "for MC-Dropout acquisition.",
                n,
                cfg.dropout.p,
                cfg.dropout.target_layers,
            )
        torch_model.eval()  # keep injected dropout dormant until scoring

    return calc
