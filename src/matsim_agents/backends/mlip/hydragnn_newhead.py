"""Single-head HydraGNN decoder surgery via the ORNL fine-tuning reference repo."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_DEFAULT_FT_REPO = "/global/cfs/projectdirs/m5216/mlupopa/HydraGNN_GFM_FineTuning4Materials"


def load_update_model(ft_repo: str | Path | None = None):
    """Import ORNL's ``utils/update_model.py`` by file path.

    The module name ``utils`` collides with HydraGNN's own ``utils`` on
    ``sys.path``, so we load it directly from the repo file to avoid shadowing.
    """
    repo = (
        Path(ft_repo or os.environ.get("HYDRAGNN_FT_REPO") or _DEFAULT_FT_REPO)
        .expanduser()
        .resolve()
    )
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
