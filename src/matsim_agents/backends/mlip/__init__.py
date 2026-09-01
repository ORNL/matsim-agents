"""Machine-learned interatomic-potential backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, relax_structure

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

    from matsim_agents.orchestration.state import RelaxationResult


@runtime_checkable
class MLIPBackend(Protocol):
    """Contract every MLIP backend must satisfy.

    ``as_calculator`` returns an ASE-compatible calculator used by the AL loop
    for fast MD sweeps and candidate scoring.  ``relax`` is the high-level
    single-shot convenience entry point.
    """

    name: str

    def as_calculator(self) -> Calculator:
        """Return an ASE Calculator for this backend."""
        ...

    def relax(
        self,
        atoms: Atoms,
        *,
        fmax: float = 0.05,
        max_steps: int = 200,
    ) -> RelaxationResult:
        """Relax *atoms* to *fmax* (eV/Å) in at most *max_steps* steps."""
        ...


__all__ = ["MLIPBackend", "RelaxStructureInput", "relax_structure"]
