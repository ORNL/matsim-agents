from __future__ import annotations

import math

import pytest

from matsim_agents.discovery.gas_energetics import (
    EnergyTerm,
    FragmentTerm,
    GasEnergeticsError,
    ReferenceTerm,
    binding_energy,
    insertion_energy,
)

HOST_COMP = {"W": 1000, "Cr": 24}


def term(label, energy, composition, **kw):
    base = {
        "label": label,
        "config_id": label,
        "energy_eV": energy,
        "composition": composition,
        "n_lattice_sites": 1024,
        "method": "qe-pbe",
        "settings_fingerprint": "s1",
        "spin_polarized": False,
        "converged": True,
        "uncertainty_eV": 0.01,
    }
    base.update(kw)
    return EnergyTerm(**base)


def comp(**delta):
    out = dict(HOST_COMP)
    for k, v in delta.items():
        out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


@pytest.fixture
def host():
    return term("host", -1000.0, HOST_COMP)


def test_binding_sign_attractive(host):
    he = term("host+He", -996.0, comp(He=1))
    v1 = term("V1", -990.0, comp(W=-1))
    v1he = term("V1He", -986.5, comp(W=-1, He=1))
    result = binding_energy(v1he, [FragmentTerm(term=he), FragmentTerm(term=v1)], host)
    assert result.comparable
    assert result.value_eV == pytest.approx(0.5)
    assert "attractive" in result.sign_convention
    assert result.uncertainty_eV == pytest.approx(0.02)


def test_binding_uncertainty_quadrature(host):
    he = term("He", -996.0, comp(He=1), uncertainty_eV=0.02)
    v1 = term("V1", -990.0, comp(W=-1), uncertainty_eV=0.03)
    v1he = term("V1He", -986.0, comp(W=-1, He=1), uncertainty_eV=0.04)
    host = term("host", -1000.0, HOST_COMP, uncertainty_eV=0.05)
    result = binding_energy(v1he, [FragmentTerm(term=he), FragmentTerm(term=v1)], host)
    assert result.uncertainty_eV == pytest.approx(math.sqrt(0.02**2 + 0.03**2 + 0.04**2 + 0.05**2))


def test_unbalanced_reaction_rejected(host):
    he = term("He", -996.0, comp(He=1))
    v1 = term("V1", -990.0, comp(W=-1))
    wrong = term("V1He2", -985.0, comp(W=-1, He=2))
    with pytest.raises(GasEnergeticsError, match="conserve"):
        binding_energy(wrong, [FragmentTerm(term=he), FragmentTerm(term=v1)], host)


def test_multiplicity_balances_he2_binding(host):
    he = term("He", -996.0, comp(He=1))
    he2 = term("He2", -992.0, comp(He=2))
    result = binding_energy(he2, [FragmentTerm(term=he, multiplicity=2)], host)
    assert result.value_eV == pytest.approx(2 * -996.0 - -992.0 - -1000.0)


def test_single_fragment_rejected(host):
    with pytest.raises(GasEnergeticsError):
        binding_energy(host, [FragmentTerm(term=host)], host)


def test_incompatible_settings_give_no_number(host):
    he = term("He", -996.0, comp(He=1), settings_fingerprint="other")
    v1 = term("V1", -990.0, comp(W=-1))
    v1he = term("V1He", -986.0, comp(W=-1, He=1))
    result = binding_energy(v1he, [FragmentTerm(term=he), FragmentTerm(term=v1)], host)
    assert not result.comparable and result.value_eV is None
    assert any("settings" in p for p in result.problems)


def test_unconverged_term_blocks(host):
    he = term("He", -996.0, comp(He=1), converged=False)
    v1 = term("V1", -990.0, comp(W=-1))
    v1he = term("V1He", -986.0, comp(W=-1, He=1))
    result = binding_energy(v1he, [FragmentTerm(term=he), FragmentTerm(term=v1)], host)
    assert result.value_eV is None and any("not converged" in p for p in result.problems)


def test_magnetic_state_mismatch_blocks():
    kw = {"spin_polarized": True, "magnetic_config_id": "m"}
    host = term("host", -1000.0, HOST_COMP, magnetic_class="ferromagnetic", **kw)
    he = term("He", -996.0, comp(He=1), magnetic_class="ferromagnetic", **kw)
    v1 = term("V1", -990.0, comp(W=-1), magnetic_class="nonmagnetic", **kw)
    v1he = term("V1He", -986.0, comp(W=-1, He=1), magnetic_class="ferromagnetic", **kw)
    result = binding_energy(v1he, [FragmentTerm(term=he), FragmentTerm(term=v1)], host)
    assert not result.comparable and any("magnetic state" in p for p in result.problems)


def test_spin_treatment_mismatch_blocks(host):
    he = term("He", -996.0, comp(He=1), spin_polarized=True, magnetic_config_id="m")
    v1 = term("V1", -990.0, comp(W=-1))
    v1he = term("V1He", -986.0, comp(W=-1, He=1))
    result = binding_energy(v1he, [FragmentTerm(term=he), FragmentTerm(term=v1)], host)
    assert not result.comparable and any("spin treatment" in p for p in result.problems)


def test_spin_polarized_needs_magnetic_config_id():
    with pytest.raises(ValueError):
        term("x", 0.0, HOST_COMP, spin_polarized=True)


def test_missing_uncertainty_is_reported_not_invented(host):
    he = term("He", -996.0, comp(He=1), uncertainty_eV=None)
    v1 = term("V1", -990.0, comp(W=-1))
    v1he = term("V1He", -986.0, comp(W=-1, He=1))
    result = binding_energy(v1he, [FragmentTerm(term=he), FragmentTerm(term=v1)], host)
    assert result.comparable and result.uncertainty_eV is None
    assert any("no uncertainty" in n for n in result.numerical_qualification)
    assert result.qualification == "unqualified"


def test_insertion_energy_needs_explicit_reference(host):
    defect = term("host+He", -996.0, comp(He=1))
    with pytest.raises(GasEnergeticsError, match="reference"):
        insertion_energy(defect, host, {})


def test_insertion_energy_value_and_provenance(host):
    defect = term("host+H2", -1003.0, comp(H=2))
    mu = ReferenceTerm(
        species="H",
        mu_eV=-3.4,
        definition="half H2 molecule",
        source="calc-1",
        method="qe-pbe",
        uncertainty_eV=0.02,
    )
    result = insertion_energy(defect, host, {"H": mu})
    assert result.value_eV == pytest.approx(-1003.0 + 1000.0 + 2 * 3.4)
    assert result.reference_terms and "half H2 molecule" in result.reference_terms[0]
    assert result.uncertainty_eV == pytest.approx(math.sqrt(0.01**2 + 0.01**2 + (2 * 0.02) ** 2))


def test_reference_method_mismatch_blocks(host):
    defect = term("host+He", -996.0, comp(He=1))
    mu = ReferenceTerm(
        species="He", mu_eV=-0.0, definition="isolated atom", source="tab", method="other-code"
    )
    result = insertion_energy(defect, host, {"He": mu})
    assert not result.comparable and result.value_eV is None


def test_identical_composition_rejected(host):
    with pytest.raises(GasEnergeticsError):
        insertion_energy(term("same", -1.0, HOST_COMP), host, {})
