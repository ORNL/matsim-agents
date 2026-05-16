"""
UMA (Universal Model for Atoms) baseline calculator.

UMA is a large mixture-of-linear-experts graph network model from Meta FAIR
Chemistry, trained on OC20, ODAC23, OMat24, OMC25, and OMol25 datasets.

Install
-------
    pip install fairchem-core

Authentication
--------------
UMA weights require a free Hugging Face account with accepted license terms:
    1. Visit https://huggingface.co/facebook/UMA and accept the FAIR Chemistry License
    2. Run: huggingface-cli login

Available model names
---------------------
    uma-s-1p2   — small model, fastest, ~6.6 M active params  (recommended)
    uma-s-1p1   — earlier small model
    uma-m-1p1   — medium model, most accurate, ~50 M active params

Task names (task_name argument)
-------------------------------
    omat        — inorganic bulk materials          (default for this competition)
    oc20        — surface catalysis with adsorbates
    oc22        — oxide surface catalysis
    oc25        — electrocatalysis
    omol        — molecules and polymers
    omc         — molecular crystals
    odac        — metal-organic frameworks

The competition dataset spans multiple domains.  "omat" is a reasonable default
for the bulk/slab/2D/alloy majority of the 159 structures; switch to "omol" for
molecular entries or "oc20" for adsorbate-on-surface entries.

Reference
---------
Wood et al., "UMA: A Family of Universal Models for Atoms", arXiv:2506.23971
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class AtomisticCalculator(Calculator):
    """UMA universal MLFF wrapped as the competition AtomisticCalculator."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        model_name: str = "uma-s-1p2",
        task_name: str = "omat",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
        self._inner = FAIRChemCalculator(predictor, task_name=task_name)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        task_name: str = "omat",
    ) -> "AtomisticCalculator":
        """
        Load a UMA model.

        Args:
            checkpoint_path: Model name ("uma-s-1p2", "uma-m-1p1", …) or
                             local path to a .pt checkpoint file.
            device:          "cpu", "cuda", "xpu", etc.
            task_name:       Domain task string (see module docstring).
        """
        return cls(model_name=checkpoint_path, task_name=task_name, device=device)

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
            results.append({
                "energy": self.results["energy"],
                "forces": self.results["forces"].copy(),
            })
        return results
