"""Opt-in, process-local suppression of unused Transformers metrics for MACE."""

from __future__ import annotations

import importlib.abc
import logging
import os
import sys

log = logging.getLogger(__name__)
_installed = False


class _RejectTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "transformers" or fullname.startswith("transformers."):
            raise ModuleNotFoundError(
                "Transformers is disabled by MATSIM_MACE_SKIP_TRANSFORMERS=1. "
                "Use a separate process without this setting for Transformer workloads.",
                name=fullname,
            )
        return None


def configure_mace_inference_imports() -> None:
    """Apply the opt-in shim before importing MACE; leave other processes unchanged.

    MACE 0.3.16 imports training tools even for inference. These import
    TorchMetrics, whose optional text/image metrics import Transformers when
    installed. Transformers can recursively scan thousands of source files:
    on shared HPC storage this metadata I/O can dominate calculator startup.
    No Transformer model is involved in MACE energy/force/stress inference.

    This deliberately patches a private Lightning availability check, not MACE
    physics or checkpoint loading. Keep it opt-in and regression-test dependency
    upgrades. Do not use it in processes needing Hugging Face LLMs/text metrics.
    """
    global _installed
    setting = os.environ.get("MATSIM_MACE_SKIP_TRANSFORMERS")
    if setting is None:
        return
    if setting != "1":
        raise ValueError("MATSIM_MACE_SKIP_TRANSFORMERS must be unset or '1'")
    if _installed:
        return
    if any(name.split(".")[0] in {"mace", "torchmetrics", "transformers"} for name in sys.modules):
        raise RuntimeError(
            "Enable MACE inference isolation before importing mace, torchmetrics, "
            "or transformers; start a fresh process."
        )

    from lightning_utilities.core.imports import Requirement, RequirementCache

    original = RequirementCache._check_requirement

    def check_requirement(cache):
        # Returning False through TorchMetrics' existing optional-dependency
        # gate is essential: blocking imports alone would break its startup.
        if cache.requirement and Requirement(cache.requirement).name.lower() == "transformers":
            cache.available = False
            cache.message = "Transformers disabled for MACE inference by explicit opt-in"
            return
        original(cache)

    RequirementCache._check_requirement = check_requirement
    # Never substitute a fake Transformer implementation or silently continue
    # if another dependency actually requires it.
    sys.meta_path.insert(0, _RejectTransformers())
    _installed = True
    log.warning(
        "MACE inference isolation active: optional Transformers metrics are disabled "
        "for this process. Transformer-dependent workloads must run separately."
    )
