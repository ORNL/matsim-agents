from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from ase import Atoms

from matsim_agents.training import finetune_uma

torch = pytest.importorskip("torch")

ENERGY_MEAN, ENERGY_RMSD = 0.25, 3.0
FORCE_RMSD = 0.5
ELEMENT_REFS = {26: -8.0, 24: -9.5}


class _Normalizer:
    def __init__(self, mean: float, rmsd: float):
        self.mean, self.rmsd = mean, rmsd

    def denorm(self, value):
        return value * self.rmsd + self.mean


class _ElementRefs:
    def undo_refs(self, data, value):
        refs = sum(ELEMENT_REFS[int(z)] for z in data["atomic_numbers"])
        return value + refs


class _Data(dict):
    def __iter__(self):
        return iter(list(self.items()))

    def to(self, _device):
        return self


class _Model(torch.nn.Module):
    """Raw head outputs are linear in one trainable weight, so gradients are checkable."""

    def __init__(self, energy_refs):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        self.tasks = {
            "omat_energy": SimpleNamespace(
                property="energy",
                normalizer=_Normalizer(ENERGY_MEAN, ENERGY_RMSD),
                element_references=energy_refs,
            ),
            "omat_forces": SimpleNamespace(
                property="forces",
                normalizer=_Normalizer(0.0, FORCE_RMSD),
                element_references=None,
            ),
        }

    def forward(self, data):
        pos = data["pos"]
        return {
            "omat_energy": {"energy": (self.w * pos.sum()).reshape(1)},
            "omat_forces": {"forces": self.w * pos},
        }


def _calc():
    def a2g(atoms):
        return _Data(
            pos=torch.tensor(atoms.positions, dtype=torch.float32),
            atomic_numbers=torch.tensor(atoms.numbers),
        )

    return SimpleNamespace(a2g=a2g)


def _atoms():
    return Atoms("FeCr", positions=[[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]])


def test_energy_is_denormalized_and_element_referenced():
    model = _Model(_ElementRefs())
    atoms = _atoms()
    energy, _ = finetune_uma.uma_energy_forces(model, _calc(), atoms, "omat")
    raw = 2.0 * atoms.positions.sum()
    expected = raw * ENERGY_RMSD + ENERGY_MEAN + ELEMENT_REFS[26] + ELEMENT_REFS[24]
    assert energy.dtype == torch.float64
    assert float(energy) == pytest.approx(expected)
    assert finetune_uma.uma_energy(model, _calc(), atoms, "omat") == pytest.approx(expected)


def test_energy_without_references_is_only_denormalized():
    energy, _ = finetune_uma.uma_energy_forces(_Model(None), _calc(), _atoms(), "omat")
    assert float(energy) == pytest.approx(
        2.0 * _atoms().positions.sum() * ENERGY_RMSD + ENERGY_MEAN
    )


def test_forces_are_scaled_without_element_references():
    atoms = _atoms()
    _, forces = finetune_uma.uma_energy_forces(_Model(_ElementRefs()), _calc(), atoms, "omat")
    expected = torch.tensor(atoms.positions, dtype=torch.float64) * 2.0 * FORCE_RMSD
    assert torch.allclose(forces, expected)


def test_physical_outputs_keep_gradients_to_the_model():
    model = _Model(_ElementRefs())
    atoms = _atoms()
    energy, forces = finetune_uma.uma_energy_forces(model, _calc(), atoms, "omat")
    (energy.sum() + forces.sum()).backward()
    total = atoms.positions.sum()
    assert model.w.grad is not None
    assert float(model.w.grad) == pytest.approx(total * ENERGY_RMSD + total * FORCE_RMSD)


def test_load_trainable_uma_uses_traineval_profile(monkeypatch):
    observed: dict[str, object] = {}
    model = object()
    predict_unit = SimpleNamespace(model=SimpleNamespace(module=model))

    def load_predict_unit(checkpoint, *, device, inference_settings):
        observed.update(
            checkpoint=checkpoint,
            device=device,
            inference_settings=inference_settings,
        )
        return predict_unit

    fairchem_core = ModuleType("fairchem.core")
    fairchem_core.FAIRChemCalculator = lambda unit, task_name: (unit, task_name)
    fairchem_core.pretrained_mlip = SimpleNamespace(load_predict_unit=load_predict_unit)
    monkeypatch.setitem(sys.modules, "fairchem.core", fairchem_core)
    monkeypatch.setattr(finetune_uma, "_resolve_base_checkpoint", lambda _: "uma.pt")

    unit, calculator, loaded_model = finetune_uma.load_trainable_uma("uma-s-1p1", "omat", "cuda")

    assert unit is predict_unit
    assert calculator == (predict_unit, "omat")
    assert loaded_model is model
    assert observed == {
        "checkpoint": "uma.pt",
        "device": "cuda",
        "inference_settings": "traineval",
    }
