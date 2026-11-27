"""Software verification of the hard-spectrum-inventory checks on constructed inventories."""

import pytest

from matsim_agents.backends.external.openmc_activation.inventory import (
    OpenMCInventory,
    OpenMCStep,
)
from matsim_agents.workflows.radiation.transmutation_inventory import (
    UNCERTAINTY,
    ChainDeclaration,
    ChainError,
    assess_inventory,
    require_declared_spectrum,
)

DECLARED = ChainDeclaration("hard", "constructed test citation")


def step(irr, cool, atoms, activity=1.0, heat=0.5):
    return OpenMCStep(
        irradiation_time_s=irr,
        cooling_time_s=cool,
        atoms=atoms,
        mass_kg=2.0,
        activity_bq=activity,
        decay_heat_w=heat,
    )


def inventory(chain="chain_hard_test.xml", steps=None):
    steps = steps or [
        step(0, 0, {"W184": 1e6}),
        step(100, 0, {"W184": 1e6, "Re187": 50.0, "H1": 10.0, "He4": 20.0, "Ta181": 1.0}),
        step(300, 0, {"W184": 1e6, "Re187": 90.0, "H1": 30.0, "He4": 40.0, "Hf178_m1": 2.0}),
        step(300, 1000, {"W184": 1e6, "H1": 30.0, "He4": 40.0}),
    ]
    return OpenMCInventory(
        openmc_version="0.0",
        nuclear_data_library="constructed",
        material_name="W",
        flux_cm2_s=1e14,
        temperature_K=300.0,
        input_sha256={chain: "0" * 64},
        steps=tuple(steps),
    )


def test_thermal_chain_is_rejected_even_with_a_hard_spectrum_declaration():
    out = assess_inventory(inventory("chain_endfb80_pwr.xml"), DECLARED)
    assert not out.chain.usable_for_declared_spectrum
    assert any("thermal-fission" in r for r in out.chain.reasons)
    with pytest.raises(ChainError, match="thermal-fission"):
        require_declared_spectrum(out)


def test_unknown_or_uncited_declarations_are_not_accepted():
    for decl, text in [
        (ChainDeclaration("unknown"), "not 'hard'"),
        (ChainDeclaration("hard", " "), "citation"),
    ]:
        out = assess_inventory(inventory(), decl)
        assert not out.chain.usable_for_declared_spectrum
        assert any(text in r for r in out.chain.reasons)


def test_missing_chain_record_is_reported():
    out = assess_inventory(inventory(chain="settings.xml"), DECLARED)
    assert not out.chain.usable_for_declared_spectrum
    assert any("no depletion chain" in r for r in out.chain.reasons)


def test_declared_hard_spectrum_chain_is_accepted_with_a_stated_caveat():
    out = require_declared_spectrum(assess_inventory(inventory(), DECLARED))
    assert out.chain.usable_for_declared_spectrum
    assert any("rests on the caller's declaration" in c for c in out.caveats)
    assert any("no hard-spectrum chain or inventory has been run" in c for c in out.caveats)


def test_coverage_reports_present_and_absent_products_for_tungsten():
    out = assess_inventory(inventory(), DECLARED)
    by = {c.channel: c for c in out.coverage}
    assert out.host == "W"
    assert by["(n,gamma) then beta decay"].present == ("Re",)
    assert by["second-order (n,gamma) and decay"].absent == ("Os",)
    assert by["(n,alpha)"].present == ("Hf",)
    assert by["(n,p)"].absent == ("Ta",)
    assert by["(n,p) and (n,alpha) gas"].present == ("H", "He")
    assert any("absent product" in c for c in out.caveats)


def test_gas_rates_are_appm_per_second_between_irradiated_steps():
    out = assess_inventory(inventory(), DECLARED)
    rates = out.gas_rates
    assert len(rates) == 1 and (rates[0].start_s, rates[0].end_s) == (100, 300)
    assert rates[0].appm_per_s["H"] == pytest.approx(20 / 1e6 * 1e6 / 200)
    assert rates[0].appm_per_s["He"] == pytest.approx(20 / 1e6 * 1e6 / 200)


def test_schedule_lists_cooling_activity_and_heat_per_kg():
    out = assess_inventory(inventory(), DECLARED)
    assert len(out.schedule) == 4
    assert out.schedule[-1].cooling_time_s == 1000
    assert out.schedule[-1].activity_bq_per_kg == pytest.approx(0.5)
    assert out.schedule[-1].decay_heat_w_per_kg == pytest.approx(0.25)


def test_uncertainty_is_unavailable_not_zero():
    out = assess_inventory(inventory(), DECLARED)
    assert out.uncertainty == UNCERTAINTY and "unavailable" in out.uncertainty


def test_single_irradiated_step_cannot_resolve_a_rate():
    steps = [step(0, 0, {"W184": 1e6}), step(100, 0, {"W184": 1e6, "H1": 1.0, "He4": 1.0})]
    out = assess_inventory(inventory(steps=steps), DECLARED)
    assert out.gas_rates == ()
    assert any("cannot be resolved" in c for c in out.caveats)


def test_first_step_must_be_unirradiated():
    steps = [step(10, 0, {"W184": 1e6}), step(20, 0, {"W184": 1e6})]
    with pytest.raises(ChainError, match="unirradiated"):
        assess_inventory(inventory(steps=steps), DECLARED)


def test_unknown_host_checks_only_gas_and_says_so():
    steps = [step(0, 0, {"Fe56": 1e6}), step(100, 0, {"Fe56": 1e6, "H1": 1.0, "He4": 1.0})]
    out = assess_inventory(inventory(steps=steps), DECLARED)
    assert out.host == "Fe"
    assert [c.channel for c in out.coverage] == ["(n,p) and (n,alpha) gas"]
    assert any("no expected-product table" in c for c in out.caveats)


def test_user_supplied_expected_products_override_the_table():
    out = assess_inventory(inventory(), DECLARED, expected_products={"custom": ("Re", "Xe")})
    custom = next(c for c in out.coverage if c.channel == "custom")
    assert custom.present == ("Re",) and custom.absent == ("Xe",)
