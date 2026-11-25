"""Energy-reference labels shared by datasets and MLIP backends."""

from __future__ import annotations

FORMATION_ENERGY_REFERENCE = "dft:declared_elemental_formation_energy"


def native_total_energy(backend: str) -> str:
    """The ``energy_reference`` of raw DFT total energies from ``backend``."""
    return f"{backend}:native_total_energy"


__all__ = ["FORMATION_ENERGY_REFERENCE", "native_total_energy"]
