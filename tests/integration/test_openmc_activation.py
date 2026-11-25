"""Real OpenMC depletion of cobalt-59, checked against closed-form results.

Skips unless OpenMC imports and OPENMC_CROSS_SECTIONS / OPENMC_CHAIN_FILE name existing files
(see docs/external/openmc.md). The one-group cross section is flat in energy, so this checks
depletion, decay and unit handling, not the nuclear data or a realistic spectrum.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import pytest

from matsim_agents.backends.external.openmc_activation.inventory import (
    read_inventory,
    select_step,
    to_activation_evidence,
)
from matsim_agents.backends.external.openmc_activation.runner import run_activation

XS = os.environ.get("OPENMC_CROSS_SECTIONS", "")
CHAIN = os.environ.get("OPENMC_CHAIN_FILE", "")
if not (Path(XS).is_file() and Path(CHAIN).is_file()):
    pytest.skip("OPENMC_CROSS_SECTIONS / OPENMC_CHAIN_FILE are not set", allow_module_level=True)

# Imported only after the data check: a failed openmc import (mpi4py) leaks into other tests.
try:
    import openmc
    import openmc.data
    import openmc.deplete as dep
except Exception as exc:  # missing package or unresolved libhdf5 / libopenmc
    pytest.skip(f"OpenMC is not usable here: {exc}", allow_module_level=True)

DAY = 86400.0
FLUX = 1.0e11  # n/cm2/s, low enough that burn-up of Co-59 and Co-60 is negligible
ENERGIES = [1.0e-5, 2.0e7]


@pytest.fixture(scope="module")
def cobalt(tmp_path_factory):
    out = tmp_path_factory.mktemp("openmc-co")
    inventory = run_activation(
        material_name="Co",
        composition={"Co": 1.0},
        density_g_cm3=8.9,
        energies_eV=ENERGIES,
        group_flux_cm2_s=[FLUX],
        irradiation_s=DAY,
        cooling_times_s=[DAY],
        output_dir=out,
        nuclides=["Co59"],
    )
    return inventory, out


def _atoms(step, name):
    return step.atoms.get(name, 0.0)


def test_co60_matches_the_analytic_inventory(cobalt):
    inventory, _ = cobalt
    openmc.config["chain_file"] = CHAIN
    micro = dep.MicroXS.from_multigroup_flux(
        energies=ENERGIES, multigroup_flux=[1.0], chain_file=CHAIN, nuclides=["Co59"]
    )
    sigma_cm2 = float(micro["Co59", "(n,gamma)"][0]) * 1e-24
    start = inventory.steps[0]
    end = select_step(inventory, 0.0)
    expected = _atoms(start, "Co59") * sigma_cm2 * FLUX * DAY
    got = _atoms(end, "Co60") + _atoms(end, "Co60_m1")
    assert got == pytest.approx(expected, rel=1e-2)


def test_cooling_follows_the_co60_half_life(cobalt):
    inventory, _ = cobalt
    end = select_step(inventory, 0.0)
    cooled = select_step(inventory, DAY)
    decay = math.exp(-math.log(2.0) * DAY / openmc.data.half_life("Co60"))
    # Co-60m (10.5 min) decays into Co-60 during the first minutes of cooling.
    before = _atoms(end, "Co60") + _atoms(end, "Co60_m1")
    assert _atoms(cooled, "Co60") / before == pytest.approx(decay, rel=1e-2)


def test_activity_is_the_decay_constant_times_atoms(cobalt):
    inventory, _ = cobalt
    cooled = select_step(inventory, DAY)
    expected = sum(
        n * math.log(2.0) / openmc.data.half_life(name)
        for name, n in cooled.atoms.items()
        if openmc.data.half_life(name)
    )
    assert cooled.activity_bq == pytest.approx(expected, rel=1e-3)


def test_evidence_carries_the_specific_activity_and_the_hashes(cobalt):
    inventory, out = cobalt
    cooled = select_step(inventory, DAY)
    evidence = to_activation_evidence(
        inventory, cooling_time_s=DAY, material_id="Co", irradiation_scenario_id="flux-1e11"
    )
    values = {q.name: q.value for q in evidence.quantities}
    assert values["specific_activity"] == pytest.approx(cooled.activity_bq / cooled.mass_kg)
    assert {"chain_endfb80_pwr.xml", "cross_sections.xml", "run_spec.json"} <= set(
        inventory.input_sha256
    )
    co59 = Path(XS).parent / "neutron" / "Co59.h5"
    assert inventory.input_sha256["Co59.h5"] == hashlib.sha256(co59.read_bytes()).hexdigest()
    assert set(inventory.output_sha256) == {"depletion_results.h5"}
    assert read_inventory(out / "inventory.json") == inventory
