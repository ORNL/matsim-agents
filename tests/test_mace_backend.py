from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from matsim_agents.active_learning.calculator import build_mace_calculator
from matsim_agents.active_learning.config import MACEConfig
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput
from matsim_agents.campaign.execution import _exploration_kwargs
from matsim_agents.discovery.seeds import PhaseCandidate
from matsim_agents.discovery.wrapper import explore_composition


@pytest.fixture
def fake_mace(monkeypatch):
    calls: dict[str, dict] = {}
    calculators = ModuleType("mace.calculators")

    def loader(name):
        def load(**kwargs):
            calls[name] = kwargs
            return SimpleNamespace(models=[])

        return load

    calculators.MACECalculator = loader("checkpoint")
    for name in ("mace_mp", "mace_off", "mace_omol", "mace_polar", "mace_anicc"):
        setattr(calculators, name, loader(name))
    mace = ModuleType("mace")
    mace.calculators = calculators
    monkeypatch.setitem(sys.modules, "mace", mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    return calls


@pytest.mark.parametrize(
    ("family", "model", "loader", "model_key"),
    [
        ("mace_mp", "medium-omat-0", "mace_mp", "model"),
        ("mace_off", "large", "mace_off", "model"),
        ("mace_omol", "extra_large", "mace_omol", "model"),
        ("mace_polar", "polar-1-l", "mace_polar", "model"),
        ("mace_anicc", "default", "mace_anicc", "model_path"),
    ],
)
def test_mace_foundation_family_dispatch(fake_mace, family, model, loader, model_key):
    build_mace_calculator(MACEConfig(family=family, model=model, device="cpu"))

    assert loader in fake_mace
    expected_model = None if family == "mace_anicc" else model
    assert fake_mace[loader][model_key] == expected_model


def test_mace_checkpoint_dispatch(fake_mace, tmp_path):
    checkpoint = tmp_path / "fine-tuned.model"
    checkpoint.touch()

    build_mace_calculator(MACEConfig(family="checkpoint", model=str(checkpoint), device="cpu"))

    assert fake_mace["checkpoint"]["model_paths"] == [str(checkpoint)]


def test_mace_family_defaults_and_validation():
    assert MACEConfig(family="mace_omol").model == "extra_large"
    assert MACEConfig(family="mace_polar").model == "polar-1-m"
    assert MACEConfig(family="mace_anicc").model == "default"
    with pytest.raises(ValueError, match="dispersion is supported only"):
        MACEConfig(family="mace_off", dispersion=True)


def test_campaign_forwards_mace_selection():
    mace = MACEConfig(
        family="mace_mp",
        model="medium-omat-0",
        device="cpu",
        precision="fp64",
        dispersion=True,
    )
    cfg = SimpleNamespace(mlip=SimpleNamespace(backend="mace", mace=mace))

    values = _exploration_kwargs(cfg, {})

    assert values == {
        "mlip_backend": "mace",
        "mace_family": "mace_mp",
        "mace_model": "medium-omat-0",
        "mace_dispersion": True,
        "mlp_device": "cpu",
        "precision": "fp64",
    }


def test_relaxation_input_accepts_mace():
    args = RelaxStructureInput(
        structure_path="structure.cif",
        mlip_backend="mace",
        mace_family="mace_polar",
        mace_model="polar-1-s",
    )

    assert args.mace_family == "mace_polar"
    assert args.mace_model == "polar-1-s"


def test_exploration_wrapper_forwards_mace_selection(tmp_path, monkeypatch):
    seed = tmp_path / "seed.extxyz"
    seed.touch()
    candidate = PhaseCandidate(
        formula="Nb",
        source="prototype",
        structure_path=str(seed),
        prototype_id="A_cI2_229_a",
    )
    monkeypatch.setattr(
        "matsim_agents.discovery.wrapper.generate_seeds", lambda *args, **kwargs: [candidate]
    )
    captured = {}

    def relax(request):
        captured.update(request.model_dump())
        raise RuntimeError("stop after capturing request")

    explore_composition(
        "Nb",
        output_dir=str(tmp_path),
        mlip_backend="mace",
        mace_family="mace_omol",
        mace_model="extra_large",
        mace_dispersion=True,
        relax_fn=relax,
    )

    assert captured["mace_family"] == "mace_omol"
    assert captured["mace_model"] == "extra_large"
    assert captured["mace_dispersion"] is True
