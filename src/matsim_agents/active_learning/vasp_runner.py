"""Back-compat shim: the VASP-specific runner is now an instance of the
generic :mod:`matsim_agents.active_learning.dft_runner`.

Existing imports of ``run_vasp_batch`` / ``VASPJobSpec`` continue to work but
are deprecated. New code should call ``run_dft_batch`` against a backend
returned by ``dft_backend.make_backend(cfg.dft)``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from matsim_agents.active_learning.config import VASPConfig
from matsim_agents.active_learning.dft_backend import DFTJobSpec, DFTResult
from matsim_agents.active_learning.dft_runner import run_dft_batch
from matsim_agents.backends.dft.vasp import VASPBackend

# Public aliases for back-compat. The old VASPJobSpec used ``extra_incar``;
# DFTJobSpec uses the generic name ``extra`` for the same purpose.
VASPJobSpec = DFTJobSpec
VASPResult = DFTResult

__all__ = ["VASPJobSpec", "VASPResult", "run_vasp_batch"]


def run_vasp_batch(
    specs: Iterable[DFTJobSpec],
    cfg: VASPConfig,
    max_workers: int | None = None,
    on_complete: Callable[[DFTJobSpec, DFTResult], None] | None = None,
) -> list[DFTResult]:
    """Deprecated. Use ``run_dft_batch`` with a ``VASPBackend`` instead."""
    backend = VASPBackend(cfg)
    return run_dft_batch(specs, backend, max_workers=max_workers, on_complete=on_complete)
