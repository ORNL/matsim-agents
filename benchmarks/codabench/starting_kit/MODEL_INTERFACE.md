# Matsim-Agents Model Interface Specification

Participants must submit a Python module containing a class that subclasses the standard
[ASE `Calculator`](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html).
This keeps the interface backend-agnostic and directly compatible with matsim-agents'
relaxation, MD, and scoring tools without any glue code.

## Required Interface

```python
from __future__ import annotations
from abc import abstractmethod
from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class MyAtomisticCalculator(Calculator):
    """Replace this docstring with your model name and a one-line description."""

    # Declare which properties your model can predict.
    # Must include at least 'energy' and 'forces' for this competition.
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "MyAtomisticCalculator":
        """
        Instantiate the calculator from a checkpoint file.

        Args:
            checkpoint_path: Path to the model checkpoint or config directory.
            device: Compute device string — 'cpu', 'cuda', 'xpu', etc.

        Returns:
            A ready-to-use instance of this calculator.
        """
        calc = cls()
        # Load your model weights here and store on self
        # e.g. calc._model = torch.load(checkpoint_path, map_location=device)
        return calc

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """
        Run inference for the given ASE Atoms object and populate self.results.

        Results must include:
            self.results['energy']  — scalar float, eV
            self.results['forces']  — np.ndarray of shape (N_atoms, 3), eV/Å
        """
        if properties is None:
            properties = self.implemented_properties

        # Standard ASE bookkeeping
        Calculator.calculate(self, atoms, properties, system_changes)

        # --- Replace below with your model inference ---
        self.results["energy"] = 0.0                              # eV
        self.results["forces"] = np.zeros((len(atoms), 3))        # eV/Å
        # -----------------------------------------------

    # Optional — implement for efficient batched evaluation.
    # The harness will call this if present, otherwise falls back to per-structure calculate().
    def predict_many(self, structures: list[Atoms]) -> list[dict]:
        """
        Batched prediction over a list of ASE Atoms objects.

        Returns:
            List of dicts, each with keys 'energy' (float) and 'forces' (np.ndarray).
        """
        results = []
        for atoms in structures:
            self.calculate(atoms)
            results.append(dict(self.results))
        return results
```

## Loading Convention

The evaluation harness loads your calculator with:

```python
from your_package import MyAtomisticCalculator

calc = MyAtomisticCalculator.from_checkpoint(
    checkpoint_path="/submitted/checkpoint",
    device="cpu",           # or 'cuda' / 'xpu' depending on the eval node
)

# Single structure
atoms.calc = calc
energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # (N, 3) eV/Å

# Batch (if predict_many is implemented)
results = calc.predict_many(list_of_atoms)
```

The calculator is then passed directly to ASE optimizers and matsim-agents tools — no
additional wrappers needed.

## Submission Checklist

- [ ] Your class is importable as `from <your_package> import <ClassName>`.
- [ ] `from_checkpoint(checkpoint_path, device)` is implemented and works offline.
- [ ] `implemented_properties` lists at least `'energy'` and `'forces'`.
- [ ] `calculate()` populates `self.results['energy']` and `self.results['forces']`.
- [ ] A `requirements.txt` lists all Python dependencies.
- [ ] A short `README.md` documents any non-standard setup (e.g. custom CUDA kernels).

## Notes

- Units: **energy in eV**, **forces in eV/Å**, consistent with ASE conventions.
- Structures are passed as `ase.Atoms` objects with periodic boundary conditions set.
- You may optionally include `'stress'` (Voigt 6-component, eV/Å³) in `implemented_properties`
  and `self.results` for future task extensions.
- Do not rely on a network connection at evaluation time; all weights must be bundled with
  or loadable from `checkpoint_path`.
