from __future__ import annotations

import pytest

from matsim_agents.backends.external.spectra_pka.isotopes import (
    IsotopeError,
    natural_abundance,
    parse_nuclide,
    resolve_isotopes,
)

pytest.importorskip("mendeleev")


def test_natural_abundance_tungsten():
    w = natural_abundance("W")
    assert set(w) == {180, 182, 183, 184, 186}
    assert sum(w.values()) == pytest.approx(1.0)
    assert w[184] == pytest.approx(0.3064, abs=1e-4)


def test_pure_element_resolves_to_natural_isotopes():
    res = resolve_isotopes({"W": 1.0})
    assert sum(res.isotope_fractions.values()) == pytest.approx(1.0)
    assert res.element_basis == {"W": "natural"}
    assert "mendeleev" in res.abundance_source


def test_alloy_isotope_fractions_scale_with_element_fraction():
    res = resolve_isotopes({"W": 0.6, "Ta": 0.4})
    assert res.element_fractions["W"] == pytest.approx(0.6)
    assert res.element_fractions["Ta"] == pytest.approx(0.4)
    assert res.isotope_fractions[("W", 184)] == pytest.approx(0.6 * natural_abundance("W")[184])
    assert sum(res.isotopic_composition()["W"].values()) == pytest.approx(1.0)


def test_override_replaces_natural_for_that_element_only():
    res = resolve_isotopes({"W": 0.5, "Fe": 0.5}, overrides={"W": {184: 1.0}})
    assert res.element_basis == {"Fe": "natural", "W": "override"}
    assert {k for k in res.isotope_fractions if k[0] == "W"} == {("W", 184)}
    assert res.isotope_fractions[("W", 184)] == pytest.approx(0.5)


def test_inventory_defines_elemental_fraction():
    res = resolve_isotopes({"Fe": 0.9}, inventory={"W184": 0.06, "W-186": 0.03, "Re187": 0.01})
    assert res.element_basis["W"] == "inventory"
    assert res.element_fractions["W"] == pytest.approx(0.09)
    assert res.isotope_fractions[("Re", 187)] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("composition", "kwargs"),
    [
        ({"W": 0.9}, {}),  # does not sum to 1
        ({"W": 1.0}, {"overrides": {"W": {184: 0.5}}}),  # override does not sum to 1
        ({"W": 1.0}, {"overrides": {"Fe": {56: 1.0}}}),  # override for absent element
        ({"W": 1.0}, {"inventory": {"W184": 1.0}}),  # element in both
        ({"Xx": 1.0}, {}),  # not an element
        ({"W": 1.0}, {"overrides": {"W": {184: 1.2, 186: -0.2}}}),  # negative
    ],
)
def test_invalid_inputs_raise(composition, kwargs):
    with pytest.raises(IsotopeError):
        resolve_isotopes(composition, **kwargs)


def test_parse_nuclide():
    assert parse_nuclide("W184") == ("W", 184)
    assert parse_nuclide("Fe-56") == ("Fe", 56)
    with pytest.raises(IsotopeError):
        parse_nuclide("tungsten")
