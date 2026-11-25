"""Material readiness records (synthetic evidence only where evidence is needed)."""

import pytest

from matsim_agents.workflows.qualification_matrix import (
    QualificationError,
    QualificationMatrix,
    QualificationStatus,
    demote,
    promote,
    record_evidence,
)
from matsim_agents.workflows.radiation_materials import (
    ATOMIC_WEIGHT,
    MaterialError,
    MaterialSpec,
    assess_all,
    assess_material,
    default_material_specs,
    to_atomic_fractions,
    to_weight_fractions,
)


def test_wt_to_at_conversion_for_v_4cr_4ti():
    at = to_atomic_fractions({"V": None, "Cr": 4.0, "Ti": 4.0}, "wt")
    assert sum(at.values()) == pytest.approx(1.0)
    assert at["V"] == pytest.approx(0.9184, abs=2e-4)
    assert at["Cr"] == pytest.approx(0.0391, abs=2e-4)
    assert at["Ti"] == pytest.approx(0.0425, abs=2e-4)


def test_wt_at_round_trip():
    at = to_atomic_fractions({"Fe": None, "Cr": 9.0, "W": 1.0}, "wt")
    wt = to_weight_fractions(at)
    assert wt["Cr"] == pytest.approx(0.09) and wt["W"] == pytest.approx(0.01)


def test_at_basis_is_not_reweighted():
    at = to_atomic_fractions({"W": None, "Re": 5.0}, "at")
    assert at == {"W": pytest.approx(0.95), "Re": pytest.approx(0.05)}


@pytest.mark.parametrize(
    "comp,basis",
    [
        ({"V": 50.0, "Cr": 40.0}, "wt"),
        ({"V": 80.0, "Cr": 40.0}, "wt"),
        ({"V": None, "Cr": None}, "wt"),
        ({"V": None, "Cr": 100.0}, "wt"),
        ({"V": None, "Cr": -1.0}, "wt"),
        ({"V": None, "Xx": 4.0}, "wt"),
        ({"V": None, "Cr": 4.0}, "wt%"),
    ],
)
def test_bad_compositions_are_rejected(comp, basis):
    with pytest.raises(MaterialError):
        to_atomic_fractions(comp, basis)


def test_every_element_has_a_weight_and_default_specs_cover_the_rows():
    assert all(w > 0 for w in ATOMIC_WEIGHT.values())
    matrix = QualificationMatrix()
    rows = {r.row_id for r in matrix.rows if r.dimension == "material"}
    assert set(default_material_specs()) == rows


def test_default_readiness_is_never_qualified_and_names_missing_evidence():
    records = assess_all(QualificationMatrix())
    assert len(records) == 5
    for rec in records:
        assert not rec.qualified and rec.matrix_status == "not_assessed"
        assert set(rec.missing_criteria) == {
            "composition",
            "reference",
            "defect-energies",
            "barrier",
        }
        assert any("without measured evidence" in b for b in rec.blockers)
    by_id = {r.row_id: r for r in records}
    assert by_id["material-v-4cr-4ti"].composition_fixed
    assert by_id["material-sic-sic"].composition_fixed
    for rid in ("material-w-rich", "material-fecr-rafm", "material-ods"):
        assert not by_id[rid].composition_fixed and not by_id[rid].ready_for_calculation


def test_composition_without_source_is_not_fixed():
    spec = MaterialSpec("material-w-rich", "W-5Re", {"W": None, "Re": 5.0}, "at", None, ())
    rec = assess_material(QualificationMatrix(), spec)
    assert rec.composition_at is not None and not rec.composition_fixed
    assert "composition has no cited source" in rec.blockers
    blank = MaterialSpec("material-w-rich", "W-5Re", {"W": None, "Re": 5.0}, "at", "  ", ())
    assert not assess_material(QualificationMatrix(), blank).composition_fixed


def test_scope_limits_are_surfaced_for_ods_and_sic():
    by_id = {r.row_id: r for r in assess_all(QualificationMatrix())}
    assert any("oxide" in s for s in by_id["material-ods"].scope_limits)
    assert any("fibre" in s for s in by_id["material-sic-sic"].scope_limits)


def test_synthetic_evidence_does_not_qualify_and_status_follows_the_matrix(tmp_path):
    artifact = tmp_path / "synthetic.json"
    artifact.write_text("{}")
    matrix = QualificationMatrix()
    for crit in ("composition", "reference", "defect-energies", "barrier"):
        matrix = record_evidence(
            matrix,
            "material-v-4cr-4ti",
            crit,
            artifact,
            summary="synthetic fixture",
            measured=True,
            synthetic=True,
        )
    rec = {r.row_id: r for r in assess_all(matrix)}["material-v-4cr-4ti"]
    assert not rec.qualified and len(rec.missing_criteria) == 4
    with pytest.raises(QualificationError):
        promote(matrix, "material-v-4cr-4ti", QualificationStatus.QUALIFIED)


def test_qualified_flag_is_read_from_the_matrix_only(tmp_path):
    artifact = tmp_path / "measured.json"
    artifact.write_text("{}")
    matrix = QualificationMatrix()
    for crit in ("composition", "reference", "defect-energies", "barrier"):
        matrix = record_evidence(
            matrix, "material-v-4cr-4ti", crit, artifact, summary="fixture", measured=True
        )
    matrix = promote(matrix, "material-v-4cr-4ti", QualificationStatus.QUALIFIED)
    rec = {r.row_id: r for r in assess_all(matrix)}["material-v-4cr-4ti"]
    assert rec.qualified and rec.missing_criteria == ()
    matrix = demote(matrix, "material-v-4cr-4ti", "artifact distrusted")
    assert not {r.row_id: r for r in assess_all(matrix)}["material-v-4cr-4ti"].qualified
