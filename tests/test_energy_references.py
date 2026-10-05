import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write

from matsim_agents.active_learning.config import MLIPConfig, TrainerConfig, UMAConfig
from matsim_agents.active_learning.evaluate import evaluate_frames, evaluate_promotion_candidate
from matsim_agents.discovery.energy_references import (
    load_elemental_reference_manifest,
    predict_elemental_references,
    validate_dataset_reference_method,
)


class OffsetCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, *, compound_error=0.0, **kwargs):
        super().__init__(**kwargs)
        self.compound_error = compound_error
        self.calls = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        symbols = atoms.get_chemical_symbols()
        self.calls.append(set(symbols))
        baseline = sum({"Nb": 100.0, "O": -20.0}[symbol] for symbol in symbols)
        bonding = -3.0 if len(set(symbols)) > 1 else 0.0
        error = self.compound_error if bonding else 0.0
        self.results = {
            "energy": baseline + bonding + error,
            "forces": np.zeros((len(atoms), 3)),
        }


def test_formation_comparison_removes_only_elemental_offsets(elemental_manifest, monkeypatch):
    manifest = elemental_manifest({"Nb": -10.0, "O": -5.0})
    calculator = OffsetCalculator(compound_error=0.6)
    monkeypatch.setattr(
        "matsim_agents.active_learning.calculator.make_mlip_calculator",
        lambda _cfg: calculator,
    )
    frames = [
        Atoms("NbO2", info={"energy": -23.0}),
        Atoms("Nb2O", info={"energy": -28.0}),
    ]
    metrics, parity = evaluate_frames(
        MLIPConfig(backend="uma", uma=UMAConfig()),
        frames,
        elemental_reference_manifest=manifest,
    )
    assert calculator.calls[:2] == [{"Nb"}, {"O"}]
    assert metrics.energy_mae_eV_per_atom > 20.0
    assert metrics.formation_energy_mae_eV_per_atom == pytest.approx(0.2)
    assert metrics.energy_mae_eV_per_atom_shifted == pytest.approx(0.2)
    assert parity["formation_ref_eV_per_atom"] == pytest.approx([-1.0, -1.0])
    assert parity["formation_pred_eV_per_atom"] == pytest.approx([-0.8, -0.8])
    assert metrics.elemental_reference_provenance["references"]["Nb"][
        "baseline_error_eV_per_atom"
    ] == pytest.approx(110.0)


@pytest.mark.parametrize("defect", ["coverage", "hash", "mixed", "nonfinite", "backend"])
def test_invalid_elemental_references_fail_before_model_predictions(elemental_manifest, defect):
    manifest = elemental_manifest({"Nb": -10.0, "O": -5.0})
    payload = json.loads(manifest.read_text())
    if defect == "coverage":
        del payload["references"]["O"]
    elif defect == "hash":
        payload["references"]["Nb"]["structure_sha256"] = "incorrect"
    elif defect == "mixed":
        payload["references"]["Nb"]["structure_path"] = payload["references"]["O"]["structure_path"]
    elif defect == "nonfinite":
        payload["references"]["Nb"]["energy_eV"] = float("nan")
    else:
        payload["backend"] = "mlip"
    manifest.write_text(json.dumps(payload))
    calculator = OffsetCalculator()
    with pytest.raises(ValueError):
        predict_elemental_references(manifest, calculator, required_elements={"Nb", "O"})
    assert calculator.calls == []


def test_formation_reference_normalizes_molecular_atom_count(elemental_manifest):
    references = predict_elemental_references(
        elemental_manifest({"Nb": -10.0, "O": -5.0}),
        OffsetCalculator(),
        required_elements={"Nb", "O"},
    )
    assert references.dft_eV_per_atom["O"] == -5.0
    assert references.model_eV_per_atom["O"] == -20.0
    assert references.formation_energy(Atoms("NbO2"), 57.0, model=True) == -1.0
    assert references.formation_energy(Atoms("NbO2"), -23.0, model=False) == -1.0


def test_promotion_evaluates_each_models_elemental_baselines(
    tmp_path, elemental_manifest, monkeypatch
):
    validation = tmp_path / "held-out.extxyz"
    frame = Atoms("NbO2", info={"energy": -23.0})
    frame.arrays["forces"] = np.zeros((3, 3))
    write(validation, frame)
    calculators = []

    def build(cfg):
        calculator = OffsetCalculator(
            compound_error=0.6 if cfg.uma.model_name == "candidate" else 0
        )
        calculators.append(calculator)
        return calculator

    monkeypatch.setattr("matsim_agents.active_learning.calculator.make_mlip_calculator", build)
    cfg = SimpleNamespace(
        mlip=MLIPConfig(backend="uma", uma=UMAConfig()),
        trainer=TrainerConfig(
            validation_set=validation,
            validation_reference_set=elemental_manifest({"Nb": -10.0, "O": -5.0}),
        ),
        model_copy=lambda **_kwargs: SimpleNamespace(),
    )
    decision = evaluate_promotion_candidate(
        cfg, "candidate", iteration=1, training_set=tmp_path / "training.extxyz"
    )
    assert len(calculators) == 2
    assert all(calc.calls[:2] == [{"Nb"}, {"O"}] for calc in calculators)
    assert decision.incumbent_metrics["formation_energy_mae_eV_per_atom"] == 0
    assert decision.candidate_metrics["formation_energy_mae_eV_per_atom"] == pytest.approx(0.2)
    assert not decision.approved


def test_missing_manifest_is_not_replaced_by_fitted_offset():
    with pytest.raises(ValueError, match="elemental reference manifest"):
        evaluate_frames(
            MLIPConfig(backend="uma", uma=UMAConfig()),
            [Atoms("Nb", info={"energy": 0.0})],
        )


def test_force_only_evaluation_needs_neither_elemental_references_nor_model_energies(monkeypatch):
    class ForcesOnlyCalculator(Calculator):
        implemented_properties = ["forces"]

        def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"forces": np.full((len(atoms), 3), 0.2)}

    monkeypatch.setattr(
        "matsim_agents.active_learning.calculator.make_mlip_calculator",
        lambda _cfg: ForcesOnlyCalculator(),
    )
    frame = Atoms("NbO2")
    frame.arrays["forces"] = np.zeros((3, 3))
    metrics, parity = evaluate_frames(MLIPConfig(backend="uma", uma=UMAConfig()), [frame])
    assert metrics.force_mae_eV_per_A == pytest.approx(0.2)
    assert metrics.n_energy_frames_evaluated == 0
    assert np.isnan(metrics.formation_energy_mae_eV_per_atom)
    assert metrics.elemental_reference_provenance == {}
    assert parity["formation_pred_eV_per_atom"].size == 0


@pytest.mark.parametrize("defect", [None, "backend", "signature", "hash"])
def test_compound_sidecar_method_validation(tmp_path, defect):
    dataset = tmp_path / "compound.extxyz"
    write(dataset, Atoms("NbO2", info={"energy": -23.0}))
    metadata = {
        "dft_backend": "qe",
        "method_signature": "test-dft",
        "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    if defect == "backend":
        metadata["dft_backend"] = "vasp"
    elif defect == "signature":
        metadata["method_signature"] = "different-dft"
    elif defect == "hash":
        metadata["sha256"] = "wrong"
    dataset.with_suffix(".extxyz.manifest.json").write_text(json.dumps(metadata))
    if defect is None:
        validate_dataset_reference_method(
            dataset, reference_backend="qe", reference_method_signature="test-dft"
        )
    else:
        with pytest.raises(ValueError, match="different DFT methods|hash"):
            validate_dataset_reference_method(
                dataset, reference_backend="qe", reference_method_signature="test-dft"
            )


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"backend": "qe", "method_signature": 1, "references": {}},
        {"backend": "qe", "method_signature": "dft", "references": {"Nb": None}},
        {
            "backend": "qe",
            "method_signature": "dft",
            "references": {"Nb": {"structure_path": "Nb.extxyz", "energy_eV": True}},
        },
    ],
)
def test_malformed_manifest_has_explicit_validation_error(tmp_path, payload):
    manifest = tmp_path / "malformed.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_elemental_reference_manifest(manifest, required_elements=set())


def test_multiple_reference_geometries_are_rejected(elemental_manifest):
    from ase.io import write

    manifest = elemental_manifest({"Nb": -10.0})
    geometry = (
        manifest.parent / json.loads(manifest.read_text())["references"]["Nb"]["structure_path"]
    )
    write(geometry, [Atoms("Nb"), Atoms("Nb")])
    with pytest.raises(ValueError, match="exactly one geometry"):
        load_elemental_reference_manifest(manifest, required_elements={"Nb"})


def test_finetune_campaign_checks_reference_coverage_before_training(
    tmp_path, elemental_manifest, monkeypatch
):
    from matsim_agents.active_learning import finetune_eval

    dataset = tmp_path / "dataset.extxyz"
    write(dataset, [Atoms("NbO2", info={"energy": -23.0}) for _ in range(4)])
    monkeypatch.setattr(finetune_eval, "_enforce_device_visibility", lambda _device: None)
    with pytest.raises(ValueError, match="coverage"):
        finetune_eval.run_campaign(
            dataset,
            tmp_path / "campaign",
            backend="uma",
            device="cpu",
            elemental_reference_manifest=elemental_manifest({"Nb": -10.0}),
        )
    assert not (tmp_path / "campaign/ft").exists()
    assert not (tmp_path / "campaign/split/train.extxyz").exists()
