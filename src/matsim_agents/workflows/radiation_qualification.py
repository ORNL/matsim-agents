"""Qualification matrix for the radiation-damage capabilities.

Software completeness and scientific qualification are separate. A row starts at
``not_assessed`` and moves only through ``record_evidence`` and ``promote``:

* evidence must be measured, not synthetic, and point to an artifact whose SHA-256 matches the
  file on disk when it is promoted;
* ``exploratory`` needs at least one criterion met; ``qualified`` needs every criterion met;
* a row never moves up on its own, and ``demote`` is always allowed.

Every row ships with its criteria and with no evidence, so a fresh matrix has nothing
qualified.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

MATRIX_SCHEMA = "matsim-radiation-qualification/1"


class QualificationStatus(StrEnum):
    NOT_ASSESSED = "not_assessed"
    NOT_QUALIFIED = "not_qualified"
    EXPLORATORY = "exploratory"
    QUALIFIED = "qualified"


_RANK = {
    QualificationStatus.NOT_ASSESSED: 0,
    QualificationStatus.NOT_QUALIFIED: 0,
    QualificationStatus.EXPLORATORY: 1,
    QualificationStatus.QUALIFIED: 2,
}


class QualificationError(ValueError):
    """A promotion or evidence record violates the matrix rules."""


class Criterion(BaseModel):
    model_config = ConfigDict(frozen=True)

    criterion_id: str
    description: str


class EvidenceItem(BaseModel):
    """A measured result tied to a criterion and to an artifact on disk."""

    model_config = ConfigDict(frozen=True)

    criterion_id: str
    summary: str
    artifact: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measured: bool
    synthetic: bool = False


class Row(BaseModel):
    model_config = ConfigDict(frozen=True)

    row_id: str
    dimension: str
    title: str
    criteria: tuple[Criterion, ...]
    status: QualificationStatus = QualificationStatus.NOT_ASSESSED
    evidence: tuple[EvidenceItem, ...] = ()
    notes: tuple[str, ...] = ()

    def met(self) -> set[str]:
        return {e.criterion_id for e in self.evidence if e.measured and not e.synthetic}

    def missing(self) -> list[str]:
        met = self.met()
        return [c.criterion_id for c in self.criteria if c.criterion_id not in met]


def _row(row_id: str, dimension: str, title: str, *criteria: tuple[str, str]) -> Row:
    return Row(
        row_id=row_id,
        dimension=dimension,
        title=title,
        criteria=tuple(Criterion(criterion_id=c, description=d) for c, d in criteria),
    )


def _material(row_id: str, title: str) -> Row:
    return _row(
        row_id,
        "material",
        title,
        ("composition", "composition and phase fixed from a cited source"),
        ("reference", "reference energies from a converged calculation"),
        ("defect-energies", "defect formation energies with convergence checks"),
        ("barrier", "a converged migration barrier with relaxed endpoints"),
    )


def default_rows() -> tuple[Row, ...]:
    return (
        _row(
            "mlip-validation",
            "mlip",
            "MLIP validation against DFT",
            ("dft-reference", "a converged DFT reference set on held-out defect structures"),
            ("force-error", "force and energy errors reported with uncertainty"),
            ("barrier-error", "MLIP barrier error against DFT NEB on at least one hop"),
        ),
        _row(
            "dft-neb",
            "dft",
            "DFT/NEB qualification",
            ("relaxed-endpoints", "endpoints relaxed to the force threshold"),
            ("cutoff-convergence", "barrier converged with cutoff within tolerance"),
            ("kpoint-convergence", "barrier converged with k-points within tolerance"),
            ("climbing-image", "a climbing image with a single imaginary mode at the saddle"),
        ),
        _row(
            "kinetic-model",
            "kinetics",
            "Kinetic model qualification",
            ("analytical-benchmarks", "agreement with analytical kinetics within statistics"),
            ("attempt-frequency", "attempt frequency from a phonon calculation, not assumed"),
            ("network-completeness", "a network checked for missing channels"),
        ),
        _row(
            "scalability",
            "scale",
            "Scalability",
            ("size-scaling", "measured cost against system size on the target hardware"),
            ("memory", "peak memory measured at the largest supported size"),
        ),
        _row(
            "reproducibility",
            "reproducibility",
            "Reproducibility",
            ("bitwise-seeded", "identical seeded results across two independent runs"),
            ("environment-pinned", "environment and versions recorded with the result"),
        ),
        _material("material-w-rich", "W-rich alloys"),
        _material("material-fecr-rafm", "Fe-Cr RAFM steels"),
        _material("material-v-4cr-4ti", "V-4Cr-4Ti (wt%)"),
        _material("material-ods", "ODS steels"),
        _material("material-sic-sic", "SiC/SiC composites"),
    )


class QualificationMatrix(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = Field(default=MATRIX_SCHEMA, alias="schema")
    rows: tuple[Row, ...] = Field(default_factory=default_rows)

    def row(self, row_id: str) -> Row:
        for row in self.rows:
            if row.row_id == row_id:
                return row
        raise QualificationError(f"no row {row_id!r}")

    def _replace(self, row: Row) -> QualificationMatrix:
        return self.model_copy(
            update={"rows": tuple(row if r.row_id == row.row_id else r for r in self.rows)}
        )

    def status_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in QualificationStatus}
        for row in self.rows:
            counts[row.status.value] += 1
        return counts


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_evidence(
    matrix: QualificationMatrix,
    row_id: str,
    criterion_id: str,
    artifact: str | os.PathLike,
    *,
    summary: str,
    measured: bool,
    synthetic: bool = False,
) -> QualificationMatrix:
    """Attach an artifact to a criterion; the hash is taken from the file now."""
    row = matrix.row(row_id)
    if criterion_id not in {c.criterion_id for c in row.criteria}:
        raise QualificationError(f"{row_id} has no criterion {criterion_id!r}")
    path = Path(artifact)
    if not path.is_file():
        raise QualificationError(f"artifact {path} does not exist")
    item = EvidenceItem(
        criterion_id=criterion_id,
        summary=summary,
        artifact=str(path),
        artifact_sha256=sha256_file(path),
        measured=measured,
        synthetic=synthetic,
    )
    return matrix._replace(row.model_copy(update={"evidence": (*row.evidence, item)}))


def promote(
    matrix: QualificationMatrix, row_id: str, target: QualificationStatus
) -> QualificationMatrix:
    """Raise a row to ``exploratory`` or ``qualified`` if its evidence supports it."""
    row = matrix.row(row_id)
    if target in {QualificationStatus.NOT_ASSESSED, QualificationStatus.NOT_QUALIFIED}:
        raise QualificationError("use demote() to lower a status")
    if _RANK[target] <= _RANK[row.status]:
        raise QualificationError(f"{row_id} is already {row.status.value}")
    problems: list[str] = []
    for item in row.evidence:
        path = Path(item.artifact)
        if not path.is_file():
            problems.append(f"artifact {item.artifact} is missing")
        elif sha256_file(path) != item.artifact_sha256:
            problems.append(f"artifact {item.artifact} changed since it was recorded")
    if problems:
        raise QualificationError(f"{row_id}: " + "; ".join(problems))
    met = row.met()
    if target is QualificationStatus.EXPLORATORY and not met:
        raise QualificationError(f"{row_id}: no measured, non-synthetic evidence")
    if target is QualificationStatus.QUALIFIED and row.missing():
        raise QualificationError(f"{row_id}: criteria not met: {', '.join(row.missing())}")
    return matrix._replace(row.model_copy(update={"status": target}))


def demote(
    matrix: QualificationMatrix,
    row_id: str,
    reason: str,
    target: QualificationStatus = QualificationStatus.NOT_QUALIFIED,
) -> QualificationMatrix:
    """Lower a row's status. Always allowed; the reason is kept."""
    row = matrix.row(row_id)
    if not reason.strip():
        raise QualificationError("a demotion needs a reason")
    if _RANK[target] > _RANK[row.status]:
        raise QualificationError("demote() cannot raise a status")
    return matrix._replace(
        row.model_copy(update={"status": target, "notes": (*row.notes, reason.strip())})
    )


def save_matrix(matrix: QualificationMatrix, path: str | os.PathLike) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, staging = tempfile.mkstemp(prefix=".qual-", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(matrix.model_dump(mode="json", by_alias=True), stream, indent=2)
            stream.write("\n")
        os.replace(staging, target)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def load_matrix(path: str | os.PathLike) -> QualificationMatrix:
    """The saved matrix, or a fresh one if the file does not exist."""
    target = Path(path)
    if not target.exists():
        return QualificationMatrix()
    return QualificationMatrix.model_validate_json(target.read_text())


__all__ = [
    "MATRIX_SCHEMA",
    "Criterion",
    "EvidenceItem",
    "QualificationError",
    "QualificationMatrix",
    "QualificationStatus",
    "Row",
    "default_rows",
    "demote",
    "load_matrix",
    "promote",
    "record_evidence",
    "save_matrix",
    "sha256_file",
]
