"""
MACE-MP-0 baseline calculator following the Matsim-Agents competition interface.

Install:
    pip install mace-torch

The `from_checkpoint` argument can be:
    - "small", "medium", "large"  (downloads from MACE-MP release)
    - A local path to a .model file
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class AtomisticCalculator(Calculator):
    """MACE-MP-0 universal MLFF wrapped as the competition AtomisticCalculator."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, model_size: str = "medium", device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        from mace.calculators import mace_mp
        self._inner = mace_mp(
            model=model_size,
            dispersion=False,
            default_dtype="float32",
            device=device,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "AtomisticCalculator":
        """
        Load MACE-MP-0.

        Args:
            checkpoint_path: "small" | "medium" | "large" or path to a .model file.
            device: "cpu", "cuda", "xpu", etc.
        """
        return cls(model_size=checkpoint_path, device=device)

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
            results.append({"energy": self.results["energy"],
                             "forces": self.results["forces"].copy()})
        return results
