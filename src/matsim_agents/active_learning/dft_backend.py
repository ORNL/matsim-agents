"""Backend-agnostic DFT labelling abstraction for the AL loop.

For active learning we only need a single-point energy + forces + (optionally)
stress per candidate structure. Both VASP and Quantum ESPRESSO can deliver
that; the differences are entirely in input-deck format, pseudopotential
handling, and launcher invocation. This module defines the small Protocol
that lets the loop dispatch jobs without caring which code is underneath.

Energy reference warning
------------------------
VASP PAW totals and QE pseudopotential totals are NOT directly comparable.
Each ``DFTResult`` records the backend name in ``backend`` so that downstream
dataset writers can tag frames; never train one HydraGNN model on a mixed
VASP+QE dataset without an explicit per-backend energy offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ase import Atoms

# --------------------------------------------------------------------------- #
# Result + job spec                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class DFTResult:
    """Outcome of one single-point DFT calculation, regardless of backend."""

    backend: str  # "vasp" | "qe"
    work_dir: str
    return_code: int
    converged: bool
    energy_eV: float | None
    forces_eV_per_A: Any | None  # numpy array, shape (N, 3)
    stress_eV_per_A3: Any | None  # numpy array, shape (3, 3) or (6,)
    n_atoms: int | None
    wall_time_sec: float | None
    final_atoms: Atoms | None
    notes: str | None = None


@dataclass
class DFTJobSpec:
    """One DFT job to run."""

    job_id: str
    atoms: Atoms
    work_dir: str
    extra: dict[str, str] | None = None  # backend-specific per-job overrides
    assigned_nodes: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Backend Protocol                                                            #
# --------------------------------------------------------------------------- #


@runtime_checkable
class DFTBackend(Protocol):
    """Minimal interface every DFT backend must implement.

    Implementations should be cheap to construct (no GPU init, no MPI) so the
    loop can build them once per iteration on the login/launcher rank.
    """

    name: str
    nodes_per_job: int
    ranks_per_node: int
    threads_per_rank: int
    timeout_sec: int

    def run_one(self, spec: DFTJobSpec) -> DFTResult:
        """Prepare inputs, exec the wrapper script, parse outputs."""
        ...


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def make_backend(dft_cfg) -> DFTBackend:
    """Build the right backend object from a ``DFTConfig``.

    Imported lazily so this module stays free of backend-specific deps.
    """
    backend_name = dft_cfg.backend
    if backend_name == "vasp":
        from matsim_agents.backends.dft.vasp import VASPBackend

        if dft_cfg.vasp is None:
            raise ValueError("dft.backend='vasp' requires a dft.vasp block.")
        return VASPBackend(dft_cfg.vasp)
    if backend_name == "qe":
        from matsim_agents.backends.dft.qe import QEBackend

        if dft_cfg.qe is None:
            raise ValueError("dft.backend='qe' requires a dft.qe block.")
        return QEBackend(dft_cfg.qe)
    raise ValueError(f"Unknown DFT backend: {backend_name!r} (expected 'vasp' or 'qe').")
