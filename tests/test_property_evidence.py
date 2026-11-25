"""MP1: property-evidence contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from matsim_agents.core.contracts import (
    OBSERVABLE_UNITS,
    PROPERTY_UNITS,
    EvidenceLevel,
    ObservableRequirement,
)
from matsim_agents.execution.ledger import (
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    MethodRecord,
)
from matsim_agents.science.properties.calculators import (
    PROPERTY_FAMILY,
    PropertyConditions,
    PropertyEvidence,
    PropertyQualification,
    PropertyStatus,
    convert_to_canonical,
    to_ledger_record,
)

C = {"temperature_K": 300.0}


def _prop(**kw):
    base = dict(
        structure_id="s1",
        composition="W640Ta128Cr64V96Hf32Mo64",
        property="bulk_modulus",
        status=PropertyStatus.CALCULATED,
        value=250.0,
        unit="GPa",
        evidence_level=EvidenceLevel.CONVERGED_DFT,
        converged=True,
        uncertainty=5.0,
        evidence_ids=["ev-1"],
        method="stress-strain",
        backend="qe",
        conditions=PropertyConditions(),
    )
    base.update(kw)
    return PropertyEvidence(**base)


def test_canonical_units_cover_every_family():
    assert PROPERTY_UNITS["thermal_conductivity_total"] == "W/(m K)"
    assert PROPERTY_UNITS["electrical_conductivity"] == "S/m"
    assert PROPERTY_UNITS["electrical_resistivity"] == "ohm m"
    assert PROPERTY_UNITS["elastic_stiffness_tensor"] == "GPa"
    assert {PROPERTY_FAMILY[p] for p in PROPERTY_UNITS} == {
        "thermal",
        "electrical",
        "mechanical",
        "structural_performance",
        "plasma_interaction",
        "phase_stability",
        "nuclear",
    }
    assert OBSERVABLE_UNITS["energy_above_hull"] == "eV/atom"


def test_wrong_unit_and_unknown_property_rejected():
    with pytest.raises(ValidationError, match="reported in 'GPa', not 'MPa'"):
        _prop(unit="MPa")
    with pytest.raises(ValidationError, match="unknown property"):
        _prop(property="strength_index")
    with pytest.raises(ValidationError, match="unknown observable"):
        ObservableRequirement(name="strength_index", unit="GPa")
    assert ObservableRequirement(name="bulk_modulus", unit="GPa").name == "bulk_modulus"


def test_unit_conversion():
    assert convert_to_canonical("bulk_modulus", 250000.0, "MPa") == pytest.approx(250.0)
    assert convert_to_canonical("electrical_resistivity", 5.6, "uohm cm") == pytest.approx(5.6e-8)
    with pytest.raises(ValueError, match="cannot convert"):
        convert_to_canonical("bulk_modulus", 1.0, "eV/A^3")


def test_tensor_dimensions_validated():
    tensor = [[float(i == j) * 300 for j in range(6)] for i in range(6)]
    ok = _prop(
        property="elastic_stiffness_tensor",
        value=tensor,
        uncertainty=[[1.0] * 6 for _ in range(6)],
    )
    assert ok.value[0][0] == 300
    with pytest.raises(ValidationError, match="6x6 matrix"):
        _prop(
            property="elastic_stiffness_tensor", value=[[1.0] * 3] * 3, uncertainty=[[1.0] * 3] * 3
        )
    with pytest.raises(ValidationError, match="one finite number"):
        _prop(value=float("nan"))


def test_temperature_required_for_transport():
    with pytest.raises(ValidationError, match="temperature_K"):
        _prop(
            property="thermal_conductivity_total",
            unit="W/(m K)",
            value=20.0,
            evidence_level=EvidenceLevel.EXPERIMENTAL,
            status=PropertyStatus.MEASURED,
        )
    p = _prop(
        property="thermal_conductivity_total",
        unit="W/(m K)",
        value=20.0,
        conditions=PropertyConditions(temperature_K=600.0),
    )
    assert p.family == "thermal" and p.conditions.temperature_K == 600.0


def test_missing_and_unsupported_have_no_value():
    miss = _prop(
        status=PropertyStatus.MISSING,
        value=None,
        uncertainty=None,
        evidence_ids=[],
        evidence_level=EvidenceLevel.HYPOTHESIS,
        reason="no stress-strain calculation",
    )
    assert not miss.is_scoreable()
    with pytest.raises(ValidationError, match="has no value"):
        _prop(status=PropertyStatus.MISSING, value=0.0, reason="x")
    with pytest.raises(ValidationError, match="must say what is missing"):
        _prop(status=PropertyStatus.UNSUPPORTED, value=None, uncertainty=None)


def test_uncertainty_and_evidence_ids_required_for_values():
    with pytest.raises(ValidationError, match="needs an uncertainty"):
        _prop(uncertainty=None)
    with pytest.raises(ValidationError, match="evidence_ids"):
        _prop(evidence_ids=[])
    with pytest.raises(ValidationError, match="negative"):
        _prop(uncertainty=-1.0)


def test_status_rules_prevent_overclaiming():
    with pytest.raises(ValidationError, match="needs DFT-level evidence"):
        _prop(evidence_level=EvidenceLevel.MLIP_PREDICTION)
    with pytest.raises(ValidationError, match="converged=True"):
        _prop(converged=False)
    with pytest.raises(ValidationError, match="cannot carry DFT"):
        _prop(status=PropertyStatus.SURROGATE_PREDICTED)
    s = _prop(
        status=PropertyStatus.SURROGATE_PREDICTED, evidence_level=EvidenceLevel.MLIP_PREDICTION
    )
    assert s.is_scoreable()
    with pytest.raises(ValidationError, match="cannot be validated or qualified"):
        _prop(
            status=PropertyStatus.SURROGATE_PREDICTED,
            evidence_level=EvidenceLevel.MLIP_PREDICTION,
            qualification=PropertyQualification.NUMERICALLY_VALIDATED,
        )
    with pytest.raises(ValidationError, match="document its approximation"):
        _prop(status=PropertyStatus.APPROXIMATED, evidence_level=EvidenceLevel.HYPOTHESIS)
    with pytest.raises(ValidationError, match="qualification_basis"):
        _prop(qualification=PropertyQualification.SCIENTIFICALLY_QUALIFIED)


def test_unconverged_is_not_scoreable_or_qualified():
    p = _prop(
        status=PropertyStatus.UNCONVERGED,
        converged=False,
        reason="strain response not linear",
        uncertainty=None,
        evidence_ids=[],
    )
    assert not p.is_scoreable()
    with pytest.raises(ValidationError, match="cannot be qualified"):
        _prop(
            status=PropertyStatus.UNCONVERGED,
            converged=False,
            reason="x",
            qualification=PropertyQualification.NUMERICALLY_VALIDATED,
        )


def test_future_properties_never_derive_from_elastic():
    kw = dict(
        property="yield_strength",
        unit="MPa",
        value=900.0,
        evidence_level=EvidenceLevel.EXPERIMENTAL,
        status=PropertyStatus.MEASURED,
        conditions=PropertyConditions(temperature_K=300.0),
    )
    assert _prop(**kw).family == "structural_performance"
    with pytest.raises(ValidationError, match="cannot be derived from elastic"):
        _prop(**kw, derived_from=["youngs_modulus"])


def test_degradation_needs_irradiation_state():
    kw = dict(
        property="thermal_conductivity_degradation",
        unit="1",
        value=0.8,
        conditions=PropertyConditions(temperature_K=300.0),
    )
    with pytest.raises(ValidationError, match="irradiation or defect"):
        _prop(**kw)


def test_serialization_roundtrip_and_stable_id():
    p = _prop(conditions=PropertyConditions(temperature_K=300.0, magnetic_state="NM"))
    again = PropertyEvidence.model_validate_json(p.model_dump_json())
    assert again == p and again.property_id() == p.property_id()


def test_ledger_mapping(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    parent = ledger.append(
        EvidenceRecord(
            kind="relaxation",
            subject="W",
            method=MethodRecord(backend="qe"),
            evidence_level=EvidenceLevel.CONVERGED_DFT,
            outcome=EvidenceOutcome.MEASURED,
        )
    )
    rec = ledger.append(to_ledger_record(_prop(evidence_ids=[parent.evidence_id])))
    assert rec.kind == "property" and rec.outcome == EvidenceOutcome.MEASURED
    assert rec.values[0].unit == "GPa" and rec.uncertainties == {"bulk_modulus": 5.0}
    missing = _prop(
        status=PropertyStatus.MISSING,
        value=None,
        uncertainty=None,
        evidence_ids=[],
        evidence_level=EvidenceLevel.HYPOTHESIS,
        reason="not computed",
    )
    miss_rec = to_ledger_record(missing)
    assert miss_rec.outcome == EvidenceOutcome.FAILED_TO_COMPUTE and not miss_rec.values
