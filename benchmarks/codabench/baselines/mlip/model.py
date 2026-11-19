"""
Matsim-Agents MLIP baseline (HydraGNN, UMA, or MACE) following the competition interface.

The model is an ``mlip:`` YAML block (``backend: hydragnn | uma | mace`` plus that
backend's settings, e.g. ``mlip/materials/mace_omat_medium.yaml``); the calculator
is built by matsim-agents' ``MLIPBackendConfig.build_calculator``.

Install matsim-agents with the backend's dependencies (``requirements-mace.txt``
for MACE, ``requirements-fairchem.txt`` for UMA/HydraGNN).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class AtomisticCalculator(Calculator):
    """Any matsim-agents MLIP wrapped as the competition AtomisticCalculator."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, inner: Calculator, **kwargs):
        super().__init__(**kwargs)
        self._inner = inner

    @classmethod
    def from_mlip_config(cls, cfg: Any) -> AtomisticCalculator:
        """Build the calculator of a parsed ``mlip`` block on its own ``device``."""
        return cls(cfg.build_calculator())

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> AtomisticCalculator:
        """
        Load the MLIP selected by a YAML file with an ``mlip`` block.

        Args:
            checkpoint_path: That YAML file, or a directory containing ``mlip.yaml``.
            device: "cpu", "cuda", or "xpu"; replaces the block's ``device``.
        """
        from matsim_agents.backends.mlip.config import load_mlip_config, parse_mlip_config

        path = Path(checkpoint_path)
        cfg = load_mlip_config(path / "mlip.yaml" if path.is_dir() else path)
        return cls.from_mlip_config(parse_mlip_config({**cfg.model_dump(), "device": device}))

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        self._inner.calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(self._inner.results["energy"]),
            "forces": np.array(self._inner.results["forces"], dtype=np.float32),
        }
        if "stress" in self._inner.results:
            self.results["stress"] = self._inner.results["stress"]

    def predict_many(self, structures: list[Atoms]) -> list[dict]:
        """Batched prediction — delegates to per-structure calculate()."""
        results = []
        for atoms in structures:
            self.calculate(atoms)
            results.append(
                {"energy": self.results["energy"], "forces": self.results["forces"].copy()}
            )
        return results
