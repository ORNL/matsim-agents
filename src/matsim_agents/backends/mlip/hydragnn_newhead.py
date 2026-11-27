"""HydraGNN decoder surgery: keep only branch-7 (OMat24), or grow a new single head."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

BRANCH_7 = "branch-7"
_PER_BRANCH_DICTS = (
    "graph_shared",
    "convs_node_hidden",
    "batch_norms_node_hidden",
    "convs_node_output",
    "batch_norms_node_output",
)


def keep_only_branch_7(model):
    """Delete every pretrained HydraGNN branch except branch-7 (OMat24), before any use.

    Branch-7's trained modules are kept by reference and re-keyed ``branch-0``;
    the other 15 branches are removed from the model, ``num_branches`` becomes 1,
    and HydraGNN's single-branch forward path evaluates branch-7 with no
    ``dataset_name`` routing. Returns ``model``.
    """
    import torch.nn as nn

    target = model.model if hasattr(getattr(model, "model", None), "graph_shared") else model
    if BRANCH_7 not in target.graph_shared:
        raise ValueError(
            f"HydraGNN model has no {BRANCH_7} (OMat24); branches: {list(target.graph_shared)}"
        )
    for name in _PER_BRANCH_DICTS:
        branches = getattr(target, name, None)
        if isinstance(branches, nn.ModuleDict) and BRANCH_7 in branches:
            setattr(target, name, nn.ModuleDict({"branch-0": branches[BRANCH_7]}))
    heads = nn.ModuleList()
    for head in target.heads_NN:
        if BRANCH_7 not in head:
            raise ValueError(f"HydraGNN head has no {BRANCH_7} (OMat24); branches: {list(head)}")
        heads.append(nn.ModuleDict({"branch-0": head[BRANCH_7]}))
    target.heads_NN = heads
    target.num_branches = 1
    for kind, specs in (getattr(target, "config_heads", None) or {}).items():
        kept = [dict(spec, type="branch-0") for spec in specs if spec.get("type") == BRANCH_7]
        if len(kept) != 1:
            raise ValueError(f"HydraGNN config_heads[{kind!r}] has no single {BRANCH_7} entry")
        target.config_heads[kind] = kept
    if target is not model:
        # Wrappers (MultiTaskModelMP) hold duplicate references to the full decoder.
        for name in ("graph_shared", "heads_NN"):
            model._modules.pop(name, None)
        model.num_branches = 1
    restored = getattr(model, "_hydragnn_checkpoint_state_keys", None)
    if restored is not None:
        # Every kept tensor must be a strictly restored checkpoint tensor (branch-7's
        # re-keyed as branch-0); record the pruned key set for later validation.
        kept = tuple(model.state_dict())
        restored = set(restored)
        foreign = [k for k in kept if k.replace(".branch-0.", f".{BRANCH_7}.") not in restored]
        if foreign:
            raise ValueError(f"Pruned HydraGNN tensors not from the checkpoint: {foreign[:3]}")
        model._hydragnn_checkpoint_state_keys = kept
    return model


def load_update_model(ft_repo: str | Path | None = None):
    """Import ORNL's ``utils/update_model.py`` by file path.

    The module name ``utils`` collides with HydraGNN's own ``utils`` on
    ``sys.path``, so we load it directly from the repo file to avoid shadowing.
    """
    source = ft_repo or os.environ.get("HYDRAGNN_FT_REPO")
    if not source:
        raise ValueError("Set ft_repo or HYDRAGNN_FT_REPO to the fine-tuning repository")
    repo = Path(source).expanduser().resolve()
    path = repo / "utils" / "update_model.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"ORNL update_model.py not found at {path}. Clone "
            "ORNL/HydraGNN_GFM_FineTuning4Materials or set HYDRAGNN_FT_REPO."
        )
    spec = importlib.util.spec_from_file_location("ornl_update_model", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def apply_newhead_surgery(model, ft_config: dict, *, ft_repo=None, freeze: bool = True):
    """Replace the decoder with a single fresh ``branch-0`` head (+ optional freeze).

    Uses ORNL ``update_architecture`` (and ``apply_freeze_mode`` when ``freeze``)
    directly so the surgery matches the reference implementation. Returns the
    updated model.
    """
    um = load_update_model(ft_repo)
    model = um.update_architecture(model, ft_config)
    if freeze:
        model = um.apply_freeze_mode(model, ft_config)
    return model
