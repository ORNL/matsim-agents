"""Append-only evidence ledger: numerical evidence, hypothesis lineage, decisions.

Every scientific statement made by a workflow or an agent is traceable to
specific numerical artifacts:

- :class:`EvidenceRecord` is one numerical result (or a failure to compute it)
  with its method, fidelity, convergence, units, and input and output hashes.
- :class:`HypothesisRecord` is one revision of a claim. A revision never
  rewrites an earlier one: it names its predecessor, so the full history of a
  claim is preserved.
- :class:`DecisionRecord` is a workflow action (for example deactivating a
  formula) and the evidence it rests on.

Record identifiers are content hashes that exclude the recording time, so
replaying a workflow with the same outputs yields the same identifiers, and
appending an existing record is a no-op. The ledger file is a JSON-lines hash
chain: editing, deleting, or reordering a line is detected when it is read.
Before a hypothesis or decision is appended, deterministic checks reject
citations of unknown evidence, failures cited as evidence about a material,
and claims whose evidence has insufficient fidelity.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.execution.contracts import DFT_EVIDENCE, EvidenceLevel, at_least
from matsim_agents.execution.locking import exclusive_lock

LEDGER_SCHEMA_VERSION = 1
_GENESIS = "0" * 64


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class EvidenceOutcome(StrEnum):
    """What a calculation established; a failure is not evidence about a material."""

    MEASURED = "measured"
    NOT_CONVERGED = "not_converged"
    FAILED_TO_COMPUTE = "failed_to_compute"


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"
    SUPERSEDED = "superseded"


ClaimType = Literal[
    "thermodynamic_stability", "energy_ranking", "structure", "property", "synthesis", "other"
]
EvidenceKind = Literal[
    "relaxation",
    "energy_ranking",
    "stability",
    "reference_energies",
    "characterization",
    "formula_run",
    "external",
]


class MethodRecord(BaseModel):
    """How a value was computed: comparable values share a method."""

    model_config = ConfigDict(frozen=True)

    backend: str
    backend_version: str | None = None
    model: str | None = None
    model_sha256: str | None = None
    method_signature: str | None = None
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Method parameters (functional, pseudopotentials, cutoffs, k-points, ...).",
    )
    software_versions: dict[str, str] = Field(default_factory=dict)


class LedgerQuantity(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    value: float | str
    unit: str
    structure: str | None = None


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True)

    recorded_at: str = Field(default_factory=_now, description="Not part of the identifier.")

    ID_FIELD: ClassVar[str] = ""
    ID_PREFIX: ClassVar[str] = ""
    # Fields added after records were first written: left out of the identifier when unset.
    OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def content_id(self) -> str:
        payload = self.model_dump(mode="json", exclude={self.ID_FIELD, "recorded_at"})
        for name in self.OPTIONAL_FIELDS:
            if payload.get(name) in (None, {}, []):
                payload.pop(name, None)
        return f"{self.ID_PREFIX}-{_sha256_text(_canonical(payload))[:24]}"

    @model_validator(mode="after")
    def _identify(self) -> _Record:
        expected = self.content_id()
        given = getattr(self, self.ID_FIELD)
        if not given:
            object.__setattr__(self, self.ID_FIELD, expected)
        elif given != expected:
            raise ValueError(
                f"{self.ID_FIELD} {given!r} does not match the record's content ({expected!r})"
            )
        return self


class CostRecord(BaseModel):
    """Resources spent (observed) or forecast (expected), split by category."""

    model_config = ConfigDict(frozen=True)

    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_seconds: float = 0.0
    mlip_relaxations: int = 0
    mlip_seconds: float = 0.0
    dft_calculations: int = 0
    dft_seconds: float = 0.0
    fine_tuning_seconds: float = 0.0
    failed_jobs: int = 0
    failed_seconds: float = 0.0

    def __add__(self, other: CostRecord) -> CostRecord:
        return CostRecord(
            **{name: getattr(self, name) + getattr(other, name) for name in CostRecord.model_fields}
        )


class EvidenceRecord(_Record):
    """One numerical result, or a failure to compute it, with its full provenance."""

    ID_FIELD: ClassVar[str] = "evidence_id"
    ID_PREFIX: ClassVar[str] = "ev"
    OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"cost"})

    record_type: Literal["evidence"] = "evidence"
    evidence_id: str = ""
    kind: EvidenceKind
    subject: str = Field(description="What the evidence is about, e.g. a formula.")
    campaign_id: str | None = None
    characterization_contract_id: str | None = None
    calculation_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    method: MethodRecord
    evidence_level: EvidenceLevel
    outcome: EvidenceOutcome
    values: list[LedgerQuantity] = Field(default_factory=list)
    numerical_convergence: dict[str, Any] = Field(default_factory=dict)
    reference_set_id: str | None = None
    reference_complete: bool | None = Field(
        None, description="Whether the hull's reference phases were complete (stability)."
    )
    input_sha256: dict[str, str] = Field(default_factory=dict)
    output_artifact_sha256: dict[str, str] = Field(default_factory=dict)
    physical_assumptions: list[str] = Field(default_factory=list)
    uncertainties: dict[str, float] = Field(default_factory=dict)
    failure_reason: str | None = None
    parent_record_ids: list[str] = Field(default_factory=list)
    cost: CostRecord | None = None

    @model_validator(mode="after")
    def _consistent_outcome(self) -> EvidenceRecord:
        if self.outcome == EvidenceOutcome.FAILED_TO_COMPUTE and self.values:
            raise ValueError("a failure to compute has no values")
        if self.outcome != EvidenceOutcome.MEASURED and not self.failure_reason:
            raise ValueError(f"{self.outcome.value} evidence needs a failure_reason")
        return self


class HypothesisRecord(_Record):
    """One immutable revision of a scientific claim."""

    ID_FIELD: ClassVar[str] = "revision_id"
    ID_PREFIX: ClassVar[str] = "hr"

    record_type: Literal["hypothesis"] = "hypothesis"
    revision_id: str = ""
    hypothesis_id: str = Field(min_length=1)
    prior_revision_id: str | None = None
    campaign_id: str | None = None
    claim_type: ClaimType = "other"
    scientific_claim: str = Field(min_length=1)
    falsifiable_prediction: str | None = None
    status: HypothesisStatus
    evidence_ids: list[str] = Field(default_factory=list)
    support_level: EvidenceLevel | None = Field(
        None,
        description="Lowest fidelity among the cited evidence: the claim rests on all of it.",
    )
    scientific_rationale: str = ""
    proposed_tests: list[str] = Field(default_factory=list)
    agent_model_id: str | None = None
    imported_from: str | None = Field(
        None,
        description="Set by explicit migrations of records that predate the ledger; such "
        "revisions may lack evidence identifiers.",
    )


class DecisionRecord(_Record):
    """A workflow action and the evidence it rests on."""

    ID_FIELD: ClassVar[str] = "decision_id"
    ID_PREFIX: ClassVar[str] = "dc"
    OPTIONAL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"policy_version", "thresholds", "expected_cost", "observed_cost"}
    )

    record_type: Literal["decision"] = "decision"
    decision_id: str = ""
    campaign_id: str | None = None
    iteration: int | None = None
    action: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    actor: str = Field(min_length=1, description="Who decided, e.g. 'debate:<run id>'.")
    imported_from: str | None = None
    policy_version: str | None = Field(None, description="Version of the rule that decided.")
    thresholds: dict[str, float] = Field(default_factory=dict)
    expected_cost: CostRecord | None = None
    observed_cost: CostRecord | None = None


LedgerRecord = EvidenceRecord | HypothesisRecord | DecisionRecord
_RECORD_TYPES = {
    "evidence": EvidenceRecord,
    "hypothesis": HypothesisRecord,
    "decision": DecisionRecord,
}


def record_id(record: LedgerRecord) -> str:
    if isinstance(record, EvidenceRecord):
        return record.evidence_id
    if isinstance(record, HypothesisRecord):
        return record.revision_id
    return record.decision_id


class LedgerIntegrityError(RuntimeError):
    """The ledger file was edited, truncated, or reordered after it was written."""


class LedgerValidationError(ValueError):
    """A hypothesis or decision fails the deterministic evidence checks."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


# Claims about these subjects need at least this evidence, of these kinds.
CLAIM_REQUIREMENTS: dict[str, tuple[EvidenceLevel, frozenset[str]]] = {
    "thermodynamic_stability": (EvidenceLevel.CONVERGED_DFT, frozenset({"stability"})),
    "energy_ranking": (
        EvidenceLevel.MLIP_RELAXATION,
        frozenset({"energy_ranking", "stability"}),
    ),
}


class EvidenceLedger:
    """Hash-chained JSON-lines ledger, read fully and verified when opened.

    A checkpoint beside the ledger (``<ledger>.head``) records the sequence
    number and hash of the last entry written; opening a ledger that no longer
    reaches it (a truncated suffix, an emptied or deleted file) is an integrity
    error. A final line without its newline is an append that crashed before
    returning; it is ignored and trimmed by the next append.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.head_path = self.path.with_name(self.path.name + ".head")
        self._records: dict[str, LedgerRecord] = {}
        self._order: list[str] = []
        self._heads: dict[str, str] = {}
        self._tip = _GENESIS
        self._written = 0
        self._unfinished = 0
        self._digest = hashlib.sha256(b"").hexdigest()
        self._head: tuple[int, str] | None = None
        self._load()

    # ------------------------------------------------------------------ io --
    def _load(self) -> None:
        # The checkpoint is read before the file and written after each append,
        # so it never runs ahead of the bytes read. The bytes and checkpoint seen
        # are kept, so ``append`` catches any writer that ran since.
        head = self._read_head()
        data = self.path.read_bytes() if self.path.is_file() else b""
        self._head = head
        self._written = len(data)
        self._digest = hashlib.sha256(data).hexdigest()
        complete = data[: data.rfind(b"\n") + 1]
        self._unfinished = len(data) - len(complete)
        tip = _GENESIS
        hashes: list[str] = []
        # Not splitlines(): U+0085, U+2028 and U+2029 are legal inside JSON strings.
        for number, line in enumerate(complete.decode("utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry["seq"] != len(self._order) or entry["prev_hash"] != tip:
                    raise LedgerIntegrityError(
                        f"{self.path}:{number}: entry out of sequence or chain broken"
                    )
                if _sha256_text(tip + _canonical(entry["record"])) != entry["entry_hash"]:
                    raise LedgerIntegrityError(f"{self.path}:{number}: entry hash mismatch")
                record = _RECORD_TYPES[entry["record"]["record_type"]].model_validate(
                    entry["record"]
                )
            except LedgerIntegrityError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise LedgerIntegrityError(f"{self.path}:{number}: {exc}") from exc
            self._index(record)
            tip = entry["entry_hash"]
            hashes.append(tip)
        self._tip = tip
        self._check_head(head, hashes)

    def _read_head(self) -> tuple[int, str] | None:
        if not self.head_path.is_file():
            return None
        try:
            head = json.loads(self.head_path.read_text(encoding="utf-8"))
            return int(head["seq"]), str(head["tip"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise LedgerIntegrityError(f"{self.head_path}: unreadable checkpoint: {exc}") from exc

    def _check_head(self, head: tuple[int, str] | None, hashes: list[str]) -> None:
        if head is None:
            return
        seq, tip = head
        # Entries past the checkpoint are allowed: a writer may stop between
        # appending and updating the checkpoint.
        if seq > len(hashes) or (seq > 0 and hashes[seq - 1] != tip):
            raise LedgerIntegrityError(
                f"{self.path} holds {len(hashes)} entries but its checkpoint records "
                f"{seq}; entries were truncated or replaced"
            )

    def _write_head(self) -> None:
        temporary = self.head_path.with_name(self.head_path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(_canonical({"seq": len(self._order), "tip": self._tip}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.head_path)

    def _index(self, record: LedgerRecord) -> None:
        identifier = record_id(record)
        self._records[identifier] = record
        self._order.append(identifier)
        if isinstance(record, HypothesisRecord):
            self._heads[record.hypothesis_id] = identifier

    def append(self, record: LedgerRecord) -> LedgerRecord:
        """Validate and append a record; an identical record already present is returned."""
        identifier = record_id(record)
        if identifier in self._records:
            return self._existing(identifier)
        if isinstance(record, HypothesisRecord):
            record = record.model_copy(
                update={"support_level": self.support_level(record.evidence_ids)}
            )
            record = HypothesisRecord.model_validate(record.model_dump(exclude={"revision_id"}))
            identifier = record.revision_id
            if identifier in self._records:
                return self._existing(identifier)
        problems = self.problems(record)
        if problems:
            raise LedgerValidationError(problems)
        payload = record.model_dump(mode="json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A lock file, not flock: flock is unsupported on NERSC CFS compute nodes.
        with exclusive_lock(f"{self.path}.lock", timeout=60, poll_interval=0.05):
            data = self._unchanged_bytes()
            size = len(data)
            if self._unfinished:
                size -= self._unfinished
                os.truncate(self.path, size)
                data = data[:size]
                self._unfinished = 0
            entry_hash = _sha256_text(self._tip + _canonical(payload))
            entry = {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "seq": len(self._order),
                "prev_hash": self._tip,
                "entry_hash": entry_hash,
                "record": payload,
            }
            line = (_canonical(entry) + "\n").encode("utf-8")
            with self.path.open("ab") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            self._tip = entry_hash
            self._index(record)
            self._written = size + len(line)
            self._digest = hashlib.sha256(data + line).hexdigest()
            self._write_head()
            self._head = (len(self._order), self._tip)
        return record

    def _unchanged_bytes(self) -> bytes:
        """The ledger's bytes, provided no writer ran since it was read; call under the lock."""
        # A size check alone misses same-size rewrites and truncation back to the loaded size.
        data = self.path.read_bytes() if self.path.exists() else b""
        if (
            len(data) != self._written
            or hashlib.sha256(data).hexdigest() != self._digest
            or self._read_head() != self._head
        ):
            raise LedgerIntegrityError(f"{self.path} changed since it was read; reopen the ledger")
        return data

    def _existing(self, identifier: str) -> LedgerRecord:
        """A record already present; entries recovered past the checkpoint get covered by it."""
        if self._head != (len(self._order), self._tip):
            with exclusive_lock(f"{self.path}.lock", timeout=60, poll_interval=0.05):
                self._unchanged_bytes()
                self._write_head()
                self._head = (len(self._order), self._tip)
        return self._records[identifier]

    # ---------------------------------------------------------- validation --
    def support_level(self, evidence_ids: list[str]) -> EvidenceLevel | None:
        levels = [
            self._records[i].evidence_level
            for i in evidence_ids
            if isinstance(self._records.get(i), EvidenceRecord)
        ]
        return min(levels, key=list(EvidenceLevel).index) if levels else None

    def problems(self, record: LedgerRecord) -> list[str]:
        """Deterministic reasons the record may not enter the ledger (empty when valid)."""
        if isinstance(record, EvidenceRecord):
            unknown = [i for i in record.parent_record_ids if i not in self._records]
            return [f"unknown parent records {unknown}"] if unknown else []
        cited = record.evidence_ids
        problems = []
        unknown = [i for i in cited if not isinstance(self._records.get(i), EvidenceRecord)]
        if unknown:
            problems.append(f"cites evidence that is not in the ledger: {unknown}")
        evidence = [r for i in cited if isinstance(r := self._records.get(i), EvidenceRecord)]
        if isinstance(record, DecisionRecord):
            if record.imported_from is None and not evidence:
                problems.append(f"decision {record.action} on {record.subject} cites no evidence")
            return problems
        head = self._heads.get(record.hypothesis_id)
        if record.prior_revision_id != head:
            problems.append(
                f"revision of {record.hypothesis_id} must follow its latest revision {head!r}, "
                f"not {record.prior_revision_id!r}"
            )
        if record.imported_from is not None:
            return problems
        if record.status in {HypothesisStatus.SUPPORTED, HypothesisStatus.CONTRADICTED}:
            if not cited:
                problems.append(f"a {record.status.value} claim must cite evidence")
            failed = [e.evidence_id for e in evidence if e.outcome != EvidenceOutcome.MEASURED]
            if failed:
                problems.append(
                    f"failed or unconverged calculations {failed} are not evidence for or "
                    "against a material"
                )
            requirement = CLAIM_REQUIREMENTS.get(record.claim_type)
            if requirement is not None and evidence:
                level, kinds = requirement
                weakest = self.support_level([e.evidence_id for e in evidence])
                if weakest is not None and not at_least(weakest, level):
                    problems.append(
                        f"a {record.claim_type} claim needs {level.value} evidence; the weakest "
                        f"cited is {weakest.value}"
                    )
                if not any(e.kind in kinds for e in evidence):
                    problems.append(
                        f"a {record.claim_type} claim needs evidence of kind {sorted(kinds)}"
                    )
                if record.claim_type == "thermodynamic_stability" and not any(
                    e.kind == "stability"
                    and e.evidence_level in DFT_EVIDENCE
                    and e.reference_complete
                    for e in evidence
                ):
                    problems.append(
                        "a thermodynamic_stability claim needs a DFT hull with complete "
                        "reference phases"
                    )
        return problems

    # --------------------------------------------------------------- views --
    def __contains__(self, identifier: object) -> bool:
        return identifier in self._records

    def __len__(self) -> int:
        return len(self._order)

    def get(self, identifier: str) -> LedgerRecord | None:
        return self._records.get(identifier)

    def records(self) -> Iterator[LedgerRecord]:
        return (self._records[i] for i in self._order)

    def evidence(self, *, subject: str | None = None) -> list[EvidenceRecord]:
        return [
            r
            for r in self.records()
            if isinstance(r, EvidenceRecord) and (subject is None or r.subject == subject)
        ]

    def decisions(self) -> list[DecisionRecord]:
        return [r for r in self.records() if isinstance(r, DecisionRecord)]

    def history(self, hypothesis_id: str) -> list[HypothesisRecord]:
        """Every revision of a hypothesis, oldest first."""
        return [
            r
            for r in self.records()
            if isinstance(r, HypothesisRecord) and r.hypothesis_id == hypothesis_id
        ]

    def current_hypotheses(self) -> dict[str, HypothesisRecord]:
        """The latest revision of every hypothesis."""
        return {h: self._records[r] for h, r in self._heads.items()}  # type: ignore[misc]

    def verify_artifacts(self) -> list[str]:
        """Evidence whose recorded input or output files are missing or changed."""
        problems = []
        for record in self.evidence():
            for path, digest in {**record.input_sha256, **record.output_artifact_sha256}.items():
                if not Path(path).is_file():
                    problems.append(f"{record.evidence_id}: {path} is missing")
                elif file_sha256(path) != digest:
                    problems.append(f"{record.evidence_id}: {path} changed since it was recorded")
        return problems

    def cost_summary(self) -> dict[str, Any]:
        """Observed cost: evidence costs plus decision costs (e.g. LLM review), by category."""
        total = CostRecord()
        for record in self.evidence():
            if record.cost is not None:
                total = total + record.cost
        for decision in self.decisions():
            if decision.observed_cost is not None:
                total = total + decision.observed_cost
        return total.model_dump()

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "records": len(self),
            "evidence": len(self.evidence()),
            "decisions": len(self.decisions()),
            "cost": self.cost_summary(),
            "hypotheses": {
                h: {
                    "revision_id": r.revision_id,
                    "revisions": len(self.history(h)),
                    "status": r.status.value,
                    "claim_type": r.claim_type,
                    "claim": r.scientific_claim,
                    "evidence_ids": r.evidence_ids,
                    "support_level": r.support_level.value if r.support_level else None,
                }
                for h, r in self.current_hypotheses().items()
            },
        }


@functools.cache
def _installed_versions(packages: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    import platform
    from importlib import metadata

    versions = {"python": platform.python_version()}
    for package in ("matsim-agents", *packages):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return tuple(versions.items())


def software_versions(*packages: str) -> dict[str, str]:
    """Installed versions of matsim-agents, Python, and the given packages."""
    return dict(_installed_versions(packages))


def hashes_of(paths: list[str | Path | None]) -> dict[str, str]:
    """SHA-256 of the given files that exist."""
    return {
        str(path): file_sha256(path) for path in paths if path is not None and Path(path).is_file()
    }


__all__ = [
    "CLAIM_REQUIREMENTS",
    "CostRecord",
    "DecisionRecord",
    "EvidenceKind",
    "EvidenceLedger",
    "EvidenceOutcome",
    "EvidenceRecord",
    "HypothesisRecord",
    "HypothesisStatus",
    "LedgerIntegrityError",
    "LedgerQuantity",
    "LedgerRecord",
    "LedgerValidationError",
    "MethodRecord",
    "file_sha256",
    "hashes_of",
    "record_id",
    "software_versions",
]
