"""Input checks of the OpenMC runner; they fail before OpenMC is imported."""

from __future__ import annotations

import pytest

from matsim_agents.backends.external.openmc_activation.runner import run_activation
from matsim_agents.science.irradiation.nuclear_evidence import NuclearEvidenceError

GOOD = {
    "material_name": "Fe",
    "composition": {"Fe": 1.0},
    "density_g_cm3": 7.874,
    "energies_eV": [1e-5, 1e5, 2e7],
    "group_flux_cm2_s": [5e13, 5e13],
    "irradiation_s": 86400.0,
    "cooling_times_s": [3600.0],
}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"composition": {}}, "positive atom fractions"),
        ({"composition": {"Fe": -1.0}}, "positive atom fractions"),
        ({"density_g_cm3": 0.0}, "density_g_cm3"),
        ({"irradiation_s": float("nan")}, "irradiation_s"),
        ({"energies_eV": [1e-5, 2e7]}, "one more entry"),
        ({"energies_eV": [1e5, 1e-5, 2e7]}, "must increase"),
        ({"group_flux_cm2_s": [5e13, -1.0]}, "non-negative"),
        ({"group_flux_cm2_s": [0.0, 0.0]}, "total flux"),
        ({"cooling_times_s": [-5.0]}, "cooling times"),
    ],
)
def test_bad_inputs_are_rejected(tmp_path, change, message):
    with pytest.raises(NuclearEvidenceError, match=message):
        run_activation(**{**GOOD, **change}, output_dir=tmp_path)


def test_missing_data_files_are_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENMC_CHAIN_FILE", raising=False)
    monkeypatch.delenv("OPENMC_CROSS_SECTIONS", raising=False)
    with pytest.raises(NuclearEvidenceError, match="OPENMC_CHAIN_FILE"):
        run_activation(**GOOD, output_dir=tmp_path)
