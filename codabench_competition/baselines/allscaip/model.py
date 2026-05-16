"""
AllScAIP (All-Scale Chemistry AI Potential) baseline calculator.

AllScAIP is an attention-based, energy-conserving MLIP from Meta FAIR Chemistry
that uses an all-to-all node attention component to capture long-range interactions.
It scales to O(100 million) training samples.

Install (same package as UMA)
-----------------------------
    pip install fairchem-core

Authentication
--------------
AllScAIP weights require a free Hugging Face account with accepted license terms:
    1. Visit https://huggingface.co/facebook/AllScAIP and accept the FAIR Chemistry License
    2. Run: huggingface-cli login

Available model names (fairchem-core 2.20+)
------------------------------------------
    allscaip-md-conserving-all-omol   — medium, energy-conserving forces, OMol25  (default)
    allscaip-md-direct-all-omol       — medium, direct forces, OMol25

Note: currently available AllScAIP checkpoints are trained on OMol25 (molecular
dataset). Use task_name="omol" (default). For inorganic/bulk structures the model
will still produce predictions but accuracy may be lower than for molecular systems.

Task names (task_name argument)
-------------------------------
    omol        — molecules and polymers             (default; matches training data)
    oc20        — surface catalysis with adsorbates
    oc25        — electrocatalysis
    omat        — inorganic bulk materials (experimental for AllScAIP)

Reference
---------
Qu et al., "A recipe for scalable attention-based MLIPs: unlocking long-range
accuracy with all-to-all node attention", arXiv:2603.06567 (2026).
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

# fairchem-core registered name for AllScAIP (energy-conserving forces = better for energy tasks)
_DEFAULT_MODEL_ID = "allscaip-md-conserving-all-omol"


class AtomisticCalculator(Calculator):
    """AllScAIP universal MLFF wrapped as the competition AtomisticCalculator."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_ID,
        task_name: str = "omol",
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
        checkpoint_path: str = _DEFAULT_MODEL_ID,
        device: str = "cpu",
        task_name: str = "omol",
    ) -> "AtomisticCalculator":
        """
        Load an AllScAIP model.

        Args:
            checkpoint_path: Registered fairchem-core model name
                             ("allscaip-md-conserving-all-omol",
                             "allscaip-md-direct-all-omol"), or a local
                             path to a .pt checkpoint file.
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
