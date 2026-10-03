from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from matsim_agents.active_learning import finetune_uma


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
