"""
MACE foundation-model calculator following the Matsim-Agents competition interface.

Install:
    pip install mace-torch

``family`` selects any loader included in mace-torch 0.3.16. ``checkpoint_path``
is that family's model alias, URL, or local ``.model`` path.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class AtomisticCalculator(Calculator):
    """MACE foundation MLFF wrapped as the competition AtomisticCalculator."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        model: str = "medium",
        family: str = "mace_mp",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        from mace.calculators import (
            MACECalculator,
            mace_anicc,
            mace_mp,
            mace_off,
            mace_omol,
            mace_polar,
        )

        common = {"device": device, "default_dtype": "float32"}
        if family == "checkpoint":
            self._inner = MACECalculator(model_paths=[model], **common)
        elif family == "mace_mp":
            self._inner = mace_mp(model=model, dispersion=False, **common)
        elif family == "mace_off":
            self._inner = mace_off(model=model, **common)
        elif family == "mace_omol":
            self._inner = mace_omol(model=model, **common)
        elif family == "mace_polar":
            self._inner = mace_polar(model=model, **common)
        elif family == "mace_anicc":
            self._inner = mace_anicc(
                model_path=None if model == "default" else model,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported MACE family: {family!r}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        family: str = "mace_mp",
    ) -> AtomisticCalculator:
        """
        Load a MACE foundation model or local checkpoint.

        Args:
            checkpoint_path: Family-specific alias, URL, or local model path.
            device: "cpu", "cuda", "xpu", etc.
            family: Native MACE loader family or ``checkpoint``.
        """
        return cls(model=checkpoint_path, family=family, device=device)

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
