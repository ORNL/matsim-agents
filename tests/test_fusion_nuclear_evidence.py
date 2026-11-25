import pytest
from pydantic import ValidationError

from matsim_agents.execution.ledger import EvidenceLedger
from matsim_agents.execution.properties import PropertyStatus
from matsim_agents.fusion.nuclear_evidence import (
    ExternalNuclearEvidence,
    NuclearEvidenceError,
    NuclearFamily,
    NuclearQuantity,
    activation_properties,
    from_pka_spectrum,
    to_ledger_record,
)
from matsim_agents.irradiation.contracts import (
    RATE_BASIS_PER_ATOM_PER_S,
    IrradiationScenario,
    PKASourceProvenance,
    PKASpectrum,
    RecoilSpecies,
    SpeciesSpectrum,
)

H = "a" * 64


def _base(family, quantities, **kw):
    data = dict(
        family=family,
        external_application="FISPACT-II",
        application_version="5.0",
        nuclear_data_library="TENDL-2019",
        input_sha256={"in": H},
        output_sha256={"out": H},
        material_id="W640Ta128Cr64V96Hf32Mo64",
        irradiation_scenario_id="demo-fw",
        quantities=tuple(quantities),
        synthetic=True,
    )
    data.update(kw)
    return ExternalNuclearEvidence(**data)


def test_dpa_needs_convention_and_never_becomes_defects():
    q = NuclearQuantity(name="dpa", value=1.0, unit="dpa")
    with pytest.raises(ValidationError, match="dpa_convention"):
        _base(NuclearFamily.PKA_DPA, [q])
    ev = _base(NuclearFamily.PKA_DPA, [q], dpa_convention="NRT")
    assert ev.dpa_convention == "NRT"
    with pytest.raises(ValidationError, match="cascade model"):
        _base(
            NuclearFamily.PKA_DPA,
            [NuclearQuantity(name="frenkel_pairs", value=1.0, unit="1")],
        )


def test_family_quantities_are_not_interchangeable():
    with pytest.raises(ValidationError, match="not a pka_dpa quantity"):
        _base(
            NuclearFamily.PKA_DPA,
            [NuclearQuantity(name="shutdown_dose_rate", value=1.0, unit="Sv/h")],
        )


def test_transmutation_is_isotope_resolved():
    with pytest.raises(ValidationError, match="isotope-resolved"):
        _base(
            NuclearFamily.TRANSMUTATION,
            [NuclearQuantity(name="concentration_appm", value=10.0, unit="appm")],
            exposure_time_s=3e7,
            isotope_inventory={"Re187": 1e-3},
        )
    ev = _base(
        NuclearFamily.TRANSMUTATION,
        [
            NuclearQuantity(
                name="concentration_appm",
                value=10.0,
                unit="appm",
                isotope=RecoilSpecies(element="Re", mass_number=187),
            ),
            NuclearQuantity(
                name="concentration_appm",
                value=1.0,
                unit="appm",
                isotope=RecoilSpecies(element="H", mass_number=3),
            ),
            NuclearQuantity(
                name="concentration_appm",
                value=2.0,
                unit="appm",
                isotope=RecoilSpecies(element="He", mass_number=4),
            ),
        ],
        exposure_time_s=3e7,
        isotope_inventory={"Re187": 1e-5},
    )
    assert ev.hydrogen_isotopes() == {"T": 1.0}
    assert ev.helium_isotopes() == {"He-4": 2.0}


def test_activation_rules():
    dose = NuclearQuantity(name="shutdown_dose_rate", value=1e-3, unit="Sv/h", uncertainty=1e-4)
    times = dict(exposure_time_s=3e7, cooling_time_s=3e8)
    with pytest.raises(ValidationError, match="SPECTRA-PKA"):
        _base(NuclearFamily.ACTIVATION, [dose], external_application="SPECTRA-PKA", **times)
    with pytest.raises(ValidationError, match="cooling_time_s"):
        _base(NuclearFamily.ACTIVATION, [dose], dose_model="contact", exposure_time_s=3e7)
    with pytest.raises(ValidationError, match="dose_model"):
        _base(NuclearFamily.ACTIVATION, [dose], **times)
    ev = _base(NuclearFamily.ACTIVATION, [dose], dose_model="contact gamma", **times)
    props = activation_properties(ev, composition="W", evidence_id=ev.evidence_id())
    assert props[0].status == PropertyStatus.APPROXIMATED
    assert "cooling" in props[0].conditions.irradiation
    assert props[0].limitations == ["SYNTHETIC demonstration data"]


def test_activation_without_uncertainty_is_not_scored():
    q = NuclearQuantity(name="decay_heat", value=1.0, unit="W/kg")
    ev = _base(NuclearFamily.ACTIVATION, [q], exposure_time_s=1.0, cooling_time_s=1.0)
    (p,) = activation_properties(ev, composition="W", evidence_id="x")
    assert p.status == PropertyStatus.MISSING and p.value is None


def test_hash_format_and_synthetic_qualification():
    with pytest.raises(ValidationError, match="sha256"):
        _base(
            NuclearFamily.PKA_DPA,
            [NuclearQuantity(name="pka_rate", value=1.0, unit="1/(atom s)")],
            input_sha256={"in": "xyz"},
        )
    with pytest.raises(ValidationError, match="non-synthetic"):
        _base(
            NuclearFamily.PKA_DPA,
            [NuclearQuantity(name="pka_rate", value=1.0, unit="1/(atom s)")],
            qualification_status="scientifically_qualified",
            qualification_basis=("benchmark",),
        )


def test_ledger_record_and_pka_import(tmp_path):
    ev = _base(
        NuclearFamily.PKA_DPA,
        [NuclearQuantity(name="dpa", value=1.0, unit="dpa", uncertainty=0.1)],
        dpa_convention="NRT",
    )
    rec = to_ledger_record(ev)
    assert rec.kind == "external"
    assert "SYNTHETIC demonstration data" in rec.physical_assumptions
    ledger = EvidenceLedger(tmp_path / "l.jsonl")
    ledger.append(rec)
    assert len(ledger.evidence()) == 1

    scenario = IrradiationScenario(
        scenario_id="demo-fw",
        parent_alloy_id="W",
        composition={"W": 1.0},
        neutron_spectrum_identity="demo",
        flux_normalization="1e14 n/cm2/s",
    )
    spec = PKASpectrum(
        scenario=scenario,
        spectra=(
            SpeciesSpectrum(
                species=RecoilSpecies(element="W", mass_number=184),
                bin_low_mev=(0.0,),
                bin_high_mev=(1.0,),
                rates=(2.0,),
            ),
        ),
        rate_unit="1/(atom s)",
        rate_basis=RATE_BASIS_PER_ATOM_PER_S,
        provenance=PKASourceProvenance(
            source="SPECTRA-PKA",
            nuclear_data_identity="TENDL-2019",
            input_sha256={"in": H},
            output_sha256={"out": H},
        ),
    )
    pka = from_pka_spectrum(spec, material_id="W", application_version="1.0")
    assert pka.family == NuclearFamily.PKA_DPA
    assert pka.quantities[0].value == 2.0
    assert pka.check_scenario(scenario) == []
    with pytest.raises(NuclearEvidenceError):
        from_pka_spectrum(
            spec.model_copy(update={"rate_basis": "unspecified"}),
            material_id="W",
            application_version="1.0",
        )
