"""
HydraGNN baseline calculator following the Matsim-Agents competition interface.

`from_checkpoint` expects `checkpoint_path` to be the HydraGNN logdir —
the directory that contains config.json and the model checkpoint file.

The HydraGNN ASE calculator is built via matsim-agents'
`active_learning.calculator.build_hydragnn_calculator`.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class AtomisticCalculator(Calculator):
    """HydraGNN MLFF wrapped as the competition AtomisticCalculator."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._inner: Calculator | None = None

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "AtomisticCalculator":
        """
        Load a trained HydraGNN model.

        Args:
            checkpoint_path: Path to the HydraGNN logdir (must contain config.json).
            device: "cpu", "cuda", "xpu", etc.
        """
        from matsim_agents.active_learning.calculator import build_hydragnn_calculator
        from matsim_agents.active_learning.config import HydraGNNConfig

        cfg = HydraGNNConfig(logdir=checkpoint_path)
        inner = build_hydragnn_calculator(cfg)

        instance = cls()
        instance._inner = inner
        return instance

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if self._inner is None:
            raise RuntimeError("Call from_checkpoint() before calculate().")
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        self._inner.calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(self._inner.results["energy"]),
            "forces": np.array(self._inner.results["forces"], dtype=np.float32),
        }

    def predict_many(self, structures: list[Atoms]) -> list[dict]:
        results = []
        for atoms in structures:
            self.calculate(atoms)
            results.append({"energy": self.results["energy"],
                             "forces": self.results["forces"].copy()})
        return results
