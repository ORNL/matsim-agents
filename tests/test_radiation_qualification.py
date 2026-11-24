"""E1: qualification matrix. Artifacts are synthetic stand-in files."""

import pytest

from matsim_agents.workflows.radiation_qualification import (
    QualificationError,
    QualificationMatrix,
    QualificationStatus,
    demote,
    load_matrix,
    promote,
    record_evidence,
    save_matrix,
)

ROW = "dft-neb"


def _artifact(tmp_path, name="a.txt", text="x"):
    path = tmp_path / name
    path.write_text(text)
    return path


def _full(tmp_path, **kwargs):
    matrix = QualificationMatrix()
    for crit in matrix.row(ROW).criteria:
        matrix = record_evidence(
            matrix,
            ROW,
            crit.criterion_id,
            _artifact(tmp_path, f"{crit.criterion_id}.txt", crit.criterion_id),
            summary="s",
            **{"measured": True, **kwargs},
        )
    return matrix


def test_fresh_matrix_has_nothing_qualified():
    matrix = QualificationMatrix()
    counts = matrix.status_counts()
    assert counts["not_assessed"] == len(matrix.rows) == 10
    assert counts["qualified"] == counts["exploratory"] == 0
    ids = {r.row_id for r in matrix.rows}
    assert {"mlip-validation", "dft-neb", "kinetic-model", "scalability"} <= ids
    assert {"material-w-rich", "material-sic-sic", "material-v-4cr-4ti"} <= ids


def test_promotion_without_evidence_is_refused():
    matrix = QualificationMatrix()
    for target in (QualificationStatus.EXPLORATORY, QualificationStatus.QUALIFIED):
        with pytest.raises(QualificationError):
            promote(matrix, ROW, target)


def test_partial_evidence_gives_exploratory_but_not_qualified(tmp_path):
    matrix = record_evidence(
        QualificationMatrix(),
        ROW,
        "relaxed-endpoints",
        _artifact(tmp_path),
        summary="s",
        measured=True,
    )
    matrix = promote(matrix, ROW, QualificationStatus.EXPLORATORY)
    assert matrix.row(ROW).status is QualificationStatus.EXPLORATORY
    with pytest.raises(QualificationError, match="criteria not met: cutoff-convergence"):
        promote(matrix, ROW, QualificationStatus.QUALIFIED)


def test_all_criteria_met_allows_qualified(tmp_path):
    matrix = promote(_full(tmp_path), ROW, QualificationStatus.QUALIFIED)
    assert matrix.row(ROW).status is QualificationStatus.QUALIFIED
    assert matrix.status_counts()["qualified"] == 1


def test_synthetic_or_unmeasured_evidence_never_counts(tmp_path):
    for kwargs in ({"synthetic": True}, {"measured": False}):
        matrix = _full(tmp_path, **kwargs)
        with pytest.raises(QualificationError, match="no measured, non-synthetic"):
            promote(matrix, ROW, QualificationStatus.EXPLORATORY)


def test_a_changed_or_missing_artifact_blocks_promotion(tmp_path):
    art = _artifact(tmp_path)
    matrix = record_evidence(
        QualificationMatrix(), ROW, "relaxed-endpoints", art, summary="s", measured=True
    )
    art.write_text("changed")
    with pytest.raises(QualificationError, match="changed since"):
        promote(matrix, ROW, QualificationStatus.EXPLORATORY)
    art.unlink()
    with pytest.raises(QualificationError, match="is missing"):
        promote(matrix, ROW, QualificationStatus.EXPLORATORY)


def test_unknown_row_criterion_and_artifact_are_rejected(tmp_path):
    matrix = QualificationMatrix()
    with pytest.raises(QualificationError, match="no row"):
        matrix.row("nope")
    with pytest.raises(QualificationError, match="no criterion"):
        record_evidence(matrix, ROW, "nope", _artifact(tmp_path), summary="s", measured=True)
    with pytest.raises(QualificationError, match="does not exist"):
        record_evidence(
            matrix, ROW, "relaxed-endpoints", tmp_path / "none", summary="s", measured=True
        )


def test_promotion_cannot_repeat_or_lower_through_promote(tmp_path):
    matrix = promote(_full(tmp_path), ROW, QualificationStatus.QUALIFIED)
    with pytest.raises(QualificationError, match="already"):
        promote(matrix, ROW, QualificationStatus.EXPLORATORY)
    with pytest.raises(QualificationError, match="demote"):
        promote(matrix, ROW, QualificationStatus.NOT_QUALIFIED)


def test_demotion_keeps_the_reason_and_cannot_raise(tmp_path):
    matrix = promote(_full(tmp_path), ROW, QualificationStatus.QUALIFIED)
    matrix = demote(matrix, ROW, "cutoff re-test failed")
    row = matrix.row(ROW)
    assert row.status is QualificationStatus.NOT_QUALIFIED and row.notes == (
        "cutoff re-test failed",
    )
    with pytest.raises(QualificationError, match="reason"):
        demote(matrix, ROW, " ")
    with pytest.raises(QualificationError, match="cannot raise"):
        demote(matrix, ROW, "x", QualificationStatus.QUALIFIED)


def test_save_and_load_round_trip_and_missing_file_is_fresh(tmp_path):
    path = tmp_path / "q" / "matrix.json"
    assert load_matrix(path).status_counts()["qualified"] == 0
    matrix = promote(_full(tmp_path), ROW, QualificationStatus.QUALIFIED)
    save_matrix(matrix, path)
    assert load_matrix(path) == matrix


def test_matrices_are_immutable_values(tmp_path):
    base = QualificationMatrix()
    record_evidence(base, ROW, "relaxed-endpoints", _artifact(tmp_path), summary="s", measured=True)
    assert base.row(ROW).evidence == ()
