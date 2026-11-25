from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from matsim_agents.discovery.magnetism import generators as gen
from matsim_agents.discovery.magnetism.defects import check_neb_magnetic_compatibility
from matsim_agents.discovery.species_support import (
    NEW_SPECIES,
    SpeciesSupportError,
    assess_pseudopotentials,
    check_mlip_support,
    inserted_species_moments,
    override_moments,
    parse_upf_header,
    require_mlip_support,
    select_policy_for_host,
    spin_variants,
    starting_polarization,
)

UPF = (
    '<UPF version="2.0.1">\n<PP_INFO/>\n<PP_HEADER element=" {el}" pseudo_type="PAW" '
    'is_paw="true" has_so="false" core_correction="{cc}" functional=" SLA  PW   PBX  PBC" '
    'z_valence="{z}" wfc_cutoff="5.0e1" rho_cutoff="4.0e2"/>\n</UPF>\n'
)


def write_upf(directory, name, element, z, cc="true"):
    (directory / name).write_text(UPF.format(el=element, z=z, cc=cc))


@pytest.fixture
def parent() -> Atoms:
    atoms = bulk("W", "bcc", a=3.16, cubic=True).repeat((2, 2, 2))
    symbols = atoms.get_chemical_symbols()
    symbols[0] = "Cr"
    symbols[5] = "Cr"
    atoms.set_chemical_symbols(symbols)
    atoms.set_array("atom_id", np.arange(len(atoms)))
    return atoms


@pytest.fixture
def parent_cfg(parent):
    return gen.ferromagnetic(parent, "parent", {"Cr": 1.0})


@pytest.fixture
def gas_structure(parent) -> Atoms:
    atoms = parent.copy()
    symbols = atoms.get_chemical_symbols()
    symbols[3] = "Re"
    atoms.set_chemical_symbols(symbols)
    helium = Atoms("He", positions=[[1.0, 0.5, 0.25]])
    helium.set_array("atom_id", np.array([len(parent)]))
    atoms += helium
    return atoms


def test_registered_species():
    assert set(NEW_SPECIES) == {"Re", "Os", "H", "He"}


def test_policy_selected_from_host_only(gas_structure):
    policy, note = select_policy_for_host(gas_structure)
    assert policy is not None and policy.name == "refractory_w_ta_cr_v_hf_mo"
    assert "Re" in note and "He" in note


def test_inserted_default_is_documented_zero(parent_cfg, gas_structure):
    moments = inserted_species_moments(gas_structure, parent_cfg)
    assert set(moments) == {3, len(parent_cfg.atom_ids)}
    assert all(m.policy == "zero_documented" for m in moments.values())


def test_unknown_explicit_species_rejected(parent_cfg, gas_structure):
    with pytest.raises(SpeciesSupportError):
        inserted_species_moments(gas_structure, parent_cfg, {"Xe": 1.0})


def test_spin_variants_preserve_parent_moments(parent, parent_cfg, gas_structure):
    variants = spin_variants(
        parent_cfg, gas_structure, defect_structure_id="gas", defect_kind="re_he"
    )
    assert set(variants) == {"unpolarized", "polarized"}
    assert not variants["unpolarized"].spin_polarized
    polarized = variants["polarized"]
    moments = dict(zip(polarized.atom_ids, polarized.initial_moments_muB, strict=True))
    for atom_id, moment in zip(parent_cfg.atom_ids, parent_cfg.initial_moments_muB, strict=True):
        if atom_id != 3:
            assert moments[atom_id] == moment
    assert moments[3] == 0.0 and moments[len(parent)] == 0.0
    assert any("zero_documented" in a for a in polarized.assumptions)


def test_nonmagnetic_parent_has_no_polarized_variant(parent, gas_structure):
    nm = gen.nonmagnetic(parent, "parent")
    assert set(spin_variants(nm, gas_structure, defect_structure_id="g", defect_kind="x")) == {
        "unpolarized"
    }


def test_override_requires_documented_assumption(parent_cfg, gas_structure):
    polarized = spin_variants(
        parent_cfg, gas_structure, defect_structure_id="gas", defect_kind="re_he"
    )["polarized"]
    with pytest.raises(SpeciesSupportError):
        override_moments(polarized, {3: 0.4}, variant_label="re_local", assumption=" ")
    with pytest.raises(SpeciesSupportError):
        override_moments(polarized, {999: 0.4}, variant_label="re_local", assumption="a")


def test_override_is_explicit_and_traceable(parent_cfg, gas_structure):
    polarized = spin_variants(
        parent_cfg, gas_structure, defect_structure_id="gas", defect_kind="re_he"
    )["polarized"]
    alt = override_moments(
        polarized, {3: 0.4}, variant_label="re_local", assumption="trial induced Re moment"
    )
    moments = dict(zip(alt.atom_ids, alt.initial_moments_muB, strict=True))
    assert moments[3] == 0.4
    assert alt.provenance["override_of_config_id"] == polarized.config_id
    assert alt.qe_starting_magnetization is None
    assert any("trial induced Re moment" in a for a in alt.assumptions)
    assert alt.fingerprint() != polarized.fingerprint()


def test_neb_compatibility_holds_for_same_override(parent_cfg, gas_structure):
    polarized = spin_variants(
        parent_cfg, gas_structure, defect_structure_id="gas", defect_kind="re_he"
    )["polarized"]
    assert check_neb_magnetic_compatibility([polarized, polarized]).compatible
    alt = override_moments(polarized, {3: 0.4}, variant_label="v", assumption="trial")
    assert not check_neb_magnetic_compatibility([polarized, alt]).compatible


def test_upf_header_identity_not_filename(tmp_path):
    write_upf(tmp_path, "Re.fake.UPF", "Re", "15.0")
    header = parse_upf_header(tmp_path / "Re.fake.UPF")
    assert header.element == "Re" and header.z_valence == 15.0 and header.is_paw
    assert header.functional == "SLA PW PBX PBC"
    assert len(header.sha256) == 64


UPF_V1 = """<PP_INFO>
  Generated using Vanderbilt code, version 7 3 4
  Scalar-Relativistic Calculation
</PP_INFO>
<PP_HEADER>
   0                   Version Number
  Ta                   Element
   US                  Ultrasoft pseudopotential
    T                  Nonlinear Core Correction
 SLA  PW   PBE  PBE     PBE  Exchange-Correlation functional
   13.00000000000      Z valence
  -70.00000000000      Total energy
    0.00000    0.00000 Suggested cutoff for wfc and rho
    2                  Max angular momentum component
</PP_HEADER>
"""


def test_upf_v1_header(tmp_path):
    (tmp_path / "ta.UPF").write_text(UPF_V1)
    h = parse_upf_header(tmp_path / "ta.UPF")
    assert (h.upf_version, h.element, h.pseudo_type, h.z_valence) == (1, "Ta", "US", 13.0)
    assert h.core_correction and not h.is_paw and not h.has_so
    assert h.functional == "SLA PW PBE PBE" and h.relativistic == "scalar"
    assert h.wfc_cutoff_ry is None and h.rho_cutoff_ry is None


def test_upf_v2_relativistic_and_oncv_radius_not_a_cutoff(tmp_path):
    (tmp_path / "H.upf").write_text(
        '<UPF version="2.0.1">\n<PP_HEADER element="H" pseudo_type="NC" relativistic="scalar" '
        'is_paw="false" core_correction="false" functional="PBE" z_valence="1.0" '
        'rho_cutoff="1.165e1"/>\n</UPF>\n'
    )
    h = parse_upf_header(tmp_path / "H.upf")
    assert h.relativistic == "scalar" and h.upf_version == 2
    assert h.wfc_cutoff_ry is None and h.rho_cutoff_ry is None


def test_equivalent_pbe_spellings_are_consistent(tmp_path):
    write_upf(tmp_path, "W.UPF", "W", "14.0")
    (tmp_path / "Ta.UPF").write_text(UPF_V1)
    result = assess_pseudopotentials(["W", "Ta"], tmp_path, {"W": "W.UPF", "Ta": "Ta.UPF"})
    assert result.consistent_functional and result.functionals == ("PBE",)


def test_pseudo_filename_lying_is_a_mismatch(tmp_path):
    write_upf(tmp_path, "Os.UPF", "Re", "15.0")
    result = assess_pseudopotentials(["Os"], tmp_path, {"Os": "Os.UPF"})
    assert not result.ready
    assert result.records[0].status == "MISMATCH"
    assert any("does not match" in b for b in result.blockers)


def test_missing_pseudopotential_blocks(tmp_path):
    write_upf(tmp_path, "W.UPF", "W", "14.0")
    result = assess_pseudopotentials(["W", "He"], tmp_path, {"W": "W.UPF", "He": "He.UPF"})
    assert not result.ready
    assert {r.species: r.status for r in result.records} == {"W": "VERIFIED", "He": "MISSING"}


def test_verified_set_is_ready_but_numerically_unestablished(tmp_path):
    write_upf(tmp_path, "W.UPF", "W", "14.0")
    write_upf(tmp_path, "Re.UPF", "Re", "15.0")
    result = assess_pseudopotentials(["W", "Re"], tmp_path, {"W": "W.UPF", "Re": "Re.UPF"})
    assert result.ready and result.cutoff_hint_ry == (50.0, 400.0)
    assert result.numerical_compatibility == "NOT_ESTABLISHED"


def test_mixed_functionals_block(tmp_path):
    write_upf(tmp_path, "W.UPF", "W", "14.0")
    (tmp_path / "Re.UPF").write_text(
        UPF.format(el="Re", z="15.0", cc="true").replace("SLA  PW   PBX  PBC", "SLA PW PSX PSC")
    )
    result = assess_pseudopotentials(["W", "Re"], tmp_path, {"W": "W.UPF", "Re": "Re.UPF"})
    assert not result.consistent_functional and not result.ready


def test_spin_polarized_needs_core_correction(tmp_path):
    write_upf(tmp_path, "Cr.UPF", "Cr", "14.0", cc="false")
    result = assess_pseudopotentials(["Cr"], tmp_path, {"Cr": "Cr.UPF"}, spin_polarized=True)
    assert not result.ready


def test_starting_polarization_uses_verified_valence(tmp_path):
    write_upf(tmp_path, "Cr.UPF", "Cr", "14.0")
    record = assess_pseudopotentials(["Cr"], tmp_path, {"Cr": "Cr.UPF"}).records[0]
    assert starting_polarization(1.4, record, "7.3") == pytest.approx(0.1)
    missing = assess_pseudopotentials(["He"], tmp_path, {}).records[0]
    with pytest.raises(SpeciesSupportError):
        starting_polarization(1.0, missing, "7.3")


def test_mlip_unsupported_for_missing_elements():
    check = check_mlip_support("m", ["W", "Re", "He"], ["W", "Cr", "Re"])
    assert check.status == "MLIP_UNSUPPORTED"
    assert check.missing == ("He",) and check.new_species_missing == ("He",)
    assert not check.prediction_allowed and check.dft_only_path_allowed
    with pytest.raises(SpeciesSupportError, match="MLIP_UNSUPPORTED"):
        require_mlip_support(check)


def test_mlip_supported_and_undeclared():
    ok = check_mlip_support("m", ["W", "H"], {"W", "H", "He"})
    assert ok.status == "SUPPORTED"
    require_mlip_support(ok)
    undeclared = check_mlip_support("m", ["W"], None)
    assert undeclared.status == "COVERAGE_UNDECLARED" and not undeclared.prediction_allowed
