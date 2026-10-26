"""Append-only evidence ledger: identifiers, integrity, validation, and lineage."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from matsim_agents.cli import app
from matsim_agents.execution.contracts import EvidenceLevel
from matsim_agents.execution.ledger import (
    DecisionRecord,
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    HypothesisRecord,
    HypothesisStatus,
    LedgerIntegrityError,
    LedgerQuantity,
    LedgerValidationError,
    MethodRecord,
    hashes_of,
)


def _evidence(level=EvidenceLevel.CONVERGED_DFT, kind="stability", **extra) -> EvidenceRecord:
    data = {
        "kind": kind,
        "subject": "SiC",
        "method": MethodRecord(backend="vasp", method_signature="pbe"),
        "evidence_level": level,
        "outcome": EvidenceOutcome.MEASURED,
        "reference_complete": True if kind == "stability" else None,
        "values": [LedgerQuantity(name="energy_above_hull", value=0.0, unit="eV/atom")],
    }
    return EvidenceRecord(**{**data, **extra})


def _claim(evidence_ids, **extra) -> HypothesisRecord:
    data = {
        "hypothesis_id": "hyp-sic",
        "claim_type": "thermodynamic_stability",
        "scientific_claim": "3C-SiC is on the Si-C hull",
        "status": HypothesisStatus.SUPPORTED,
        "evidence_ids": evidence_ids,
    }
    return HypothesisRecord(**{**data, **extra})


def test_identifiers_are_content_hashes_and_appends_are_idempotent(tmp_path):
    first = _evidence()
    again = _evidence()
    assert first.evidence_id == again.evidence_id
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    ledger.append(first)
    ledger.append(again)
    assert len(ledger) == 1
    with pytest.raises(ValueError, match="does not match the record's content"):
        EvidenceRecord.model_validate({**first.model_dump(), "subject": "Si"})


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029", "\r"])
def test_unicode_line_separators_in_text_round_trip(tmp_path, separator):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    evidence = ledger.append(_evidence())
    text = f"line one{separator}two"
    claim = ledger.append(_claim([evidence.evidence_id], scientific_claim=text))
    reopened = EvidenceLedger(path)
    assert len(reopened) == 2
    assert reopened.history("hyp-sic")[-1].scientific_claim == claim.scientific_claim


def test_a_failure_to_compute_carries_no_values_and_a_reason():
    with pytest.raises(ValueError, match="needs a failure_reason"):
        _evidence(outcome=EvidenceOutcome.FAILED_TO_COMPUTE, values=[])
    with pytest.raises(ValueError, match="has no values"):
        _evidence(outcome=EvidenceOutcome.FAILED_TO_COMPUTE, failure_reason="pw.x crashed")


@pytest.mark.parametrize("edit", ["value", "delete", "reorder"])
def test_editing_the_ledger_file_is_detected(tmp_path, edit):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    first = ledger.append(_evidence())
    ledger.append(_evidence(level=EvidenceLevel.MLIP_RELAXATION, kind="energy_ranking"))
    ledger.append(_claim([first.evidence_id]))
    lines = path.read_text().splitlines()
    if edit == "value":
        lines[0] = lines[0].replace('"value":0.0', '"value":0.5')
    elif edit == "delete":
        del lines[1]
    else:
        lines[0], lines[1] = lines[1], lines[0]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerIntegrityError):
        EvidenceLedger(path)


def test_concurrent_writers_cannot_fork_the_chain(tmp_path):
    path = tmp_path / "ledger.jsonl"
    one, two = EvidenceLedger(path), EvidenceLedger(path)
    one.append(_evidence())
    with pytest.raises(LedgerIntegrityError, match="changed since it was read"):
        two.append(_evidence(level=EvidenceLevel.MLIP_RELAXATION, kind="energy_ranking"))
    assert len(EvidenceLedger(path)) == 1


@pytest.mark.parametrize("truncate", ["suffix", "empty", "deleted"])
def test_truncating_the_ledger_is_detected(tmp_path, truncate):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    first = ledger.append(_evidence())
    ledger.append(_claim([first.evidence_id]))
    lines = path.read_text().splitlines(keepends=True)
    if truncate == "suffix":
        path.write_text(lines[0])
    elif truncate == "empty":
        path.write_text("")
    else:
        path.unlink()
    with pytest.raises(LedgerIntegrityError, match="truncated or replaced"):
        EvidenceLedger(path)


def test_entries_past_the_checkpoint_are_accepted(tmp_path):
    # A writer that stopped between appending and updating the checkpoint.
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    ledger.append(_evidence())
    checkpoint = (tmp_path / "ledger.jsonl.head").read_text()
    ledger.append(_evidence(level=EvidenceLevel.MLIP_RELAXATION, kind="energy_ranking"))
    (tmp_path / "ledger.jsonl.head").write_text(checkpoint)
    reopened = EvidenceLedger(path)
    assert len(reopened) == 2
    reopened.append(_evidence(subject="Si"))
    assert len(EvidenceLedger(path)) == 3


@pytest.mark.parametrize("kind", ["evidence", "hypothesis"])
def test_retrying_a_recovered_append_checkpoints_it(tmp_path, kind):
    # The writer stopped before the checkpoint; its retry must protect the entry.
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    first = ledger.append(_evidence())
    checkpoint = (tmp_path / "ledger.jsonl.head").read_text()
    record = _evidence(subject="Si") if kind == "evidence" else _claim([first.evidence_id])
    ledger.append(record)
    (tmp_path / "ledger.jsonl.head").write_text(checkpoint)

    reopened = EvidenceLedger(path)
    reopened.append(record)
    assert json.loads((tmp_path / "ledger.jsonl.head").read_text())["seq"] == 2
    path.write_text(path.read_text().splitlines(keepends=True)[0])
    with pytest.raises(LedgerIntegrityError, match="truncated or replaced"):
        EvidenceLedger(path)


def test_an_unfinished_final_line_is_ignored_and_trimmed(tmp_path):
    # An append that crashed mid-write, before its checkpoint update.
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    ledger.append(_evidence())
    complete = path.read_bytes()
    checkpoint = (tmp_path / "ledger.jsonl.head").read_text()
    ledger.append(_evidence(subject="Si"))
    second = path.read_bytes()[len(complete) :]
    path.write_bytes(complete + second[:-1])
    (tmp_path / "ledger.jsonl.head").write_text(checkpoint)
    reopened = EvidenceLedger(path)
    assert len(reopened) == 1
    reopened.append(_evidence(subject="C"))
    assert [r.subject for r in EvidenceLedger(path).evidence()] == ["SiC", "C"]


def test_an_unfinished_line_the_checkpoint_covers_is_refused(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    ledger.append(_evidence())
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(LedgerIntegrityError, match="truncated or replaced"):
        EvidenceLedger(path)


def test_a_stale_writer_cannot_overwrite_a_same_size_replacement(tmp_path):
    # A torn tail replaced by another writer's complete entry of the same length.
    path = tmp_path / "ledger.jsonl"
    EvidenceLedger(path).append(_evidence())
    complete = path.read_bytes()
    record = _evidence(subject="Si")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "ledger.jsonl").write_bytes(complete)
    EvidenceLedger(scratch / "ledger.jsonl").append(record)
    line = (scratch / "ledger.jsonl").read_bytes()[len(complete) :]
    path.write_bytes(complete + line[:-1] + b" ")
    stale = EvidenceLedger(path)
    EvidenceLedger(path).append(record)
    assert path.stat().st_size == stale._written
    with pytest.raises(LedgerIntegrityError, match="changed since it was read"):
        stale.append(_evidence(subject="C"))
    assert [r.subject for r in EvidenceLedger(path).evidence()] == ["SiC", "Si"]


def test_a_stale_writer_cannot_hide_a_truncation_back_to_its_size(tmp_path):
    path = tmp_path / "ledger.jsonl"
    EvidenceLedger(path).append(_evidence())
    original = path.read_bytes()
    stale = EvidenceLedger(path)
    EvidenceLedger(path).append(_evidence(subject="Si"))
    path.write_bytes(original)
    with pytest.raises(LedgerIntegrityError, match="changed since it was read"):
        stale.append(_evidence(subject="C"))
    with pytest.raises(LedgerIntegrityError, match="truncated or replaced"):
        EvidenceLedger(path)


def test_an_appending_ledger_stays_usable_and_in_step(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    for subject in ("Si", "C", "SiC"):
        ledger.append(_evidence(subject=subject))
    assert ledger._written == path.stat().st_size
    assert json.loads((tmp_path / "ledger.jsonl.head").read_text())["seq"] == 3
    assert [r.subject for r in EvidenceLedger(path).evidence()] == ["Si", "C", "SiC"]


def test_an_append_racing_the_read_is_detected(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    EvidenceLedger(path).append(_evidence())
    original = EvidenceLedger._check_head

    def race(self, head, hashes):
        # Another writer appends after this ledger read the file.
        if not getattr(race, "done", False):
            race.done = True
            EvidenceLedger(path).append(_evidence(subject="Si"))
        original(self, head, hashes)

    monkeypatch.setattr(EvidenceLedger, "_check_head", race)
    stale = EvidenceLedger(path)
    assert len(stale) == 1
    with pytest.raises(LedgerIntegrityError, match="changed since it was read"):
        stale.append(_evidence(subject="C"))


def test_characterization_evidence_does_not_support_an_energy_ranking(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    relaxed = ledger.append(_evidence(level=EvidenceLevel.MLIP_RELAXATION, kind="characterization"))
    with pytest.raises(LedgerValidationError, match="needs evidence of kind"):
        ledger.append(
            _claim(
                [relaxed.evidence_id],
                hypothesis_id="hyp-rank",
                claim_type="energy_ranking",
                scientific_claim="3C is below 2H",
            )
        )


def test_claims_must_cite_existing_measured_evidence(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(LedgerValidationError, match="not in the ledger"):
        ledger.append(_claim(["ev-missing"]))
    with pytest.raises(LedgerValidationError, match="must cite evidence"):
        ledger.append(_claim([]))
    failed = ledger.append(
        _evidence(
            outcome=EvidenceOutcome.FAILED_TO_COMPUTE,
            values=[],
            failure_reason="SCF did not converge",
        )
    )
    with pytest.raises(LedgerValidationError, match="not evidence for or against"):
        ledger.append(_claim([failed.evidence_id], status=HypothesisStatus.CONTRADICTED))
    ledger.append(_claim([], status=HypothesisStatus.PROPOSED))
    with pytest.raises(LedgerValidationError, match="cites no evidence"):
        ledger.append(DecisionRecord(action="deactivate_formula", subject="SiC", actor="review"))


def test_mlip_and_dft_evidence_are_not_merged_into_one_fidelity(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    dft = ledger.append(_evidence())
    mlip = ledger.append(_evidence(level=EvidenceLevel.MLIP_RELAXATION, kind="energy_ranking"))
    incomplete = ledger.append(_evidence(reference_complete=False, uncertainties={"x": 1.0}))
    with pytest.raises(LedgerValidationError, match="weakest cited is mlip_relaxation"):
        ledger.append(_claim([dft.evidence_id, mlip.evidence_id]))
    with pytest.raises(LedgerValidationError, match="complete reference phases"):
        ledger.append(_claim([incomplete.evidence_id]))
    ranking = ledger.append(
        _claim(
            [mlip.evidence_id],
            hypothesis_id="hyp-rank",
            claim_type="energy_ranking",
            scientific_claim="3C is below 2H",
        )
    )
    assert ranking.support_level == EvidenceLevel.MLIP_RELAXATION
    stable = ledger.append(_claim([dft.evidence_id]))
    assert stable.support_level == EvidenceLevel.CONVERGED_DFT


def test_a_claim_keeps_its_full_revision_history(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    dft = ledger.append(_evidence())
    proposed = ledger.append(_claim([], status=HypothesisStatus.PROPOSED))
    with pytest.raises(LedgerValidationError, match="must follow its latest revision"):
        ledger.append(_claim([dft.evidence_id]))
    supported = ledger.append(_claim([dft.evidence_id], prior_revision_id=proposed.revision_id))
    superseded = ledger.append(
        _claim(
            [dft.evidence_id],
            status=HypothesisStatus.SUPERSEDED,
            prior_revision_id=supported.revision_id,
        )
    )
    with pytest.raises(LedgerValidationError, match="must follow its latest revision"):
        ledger.append(
            _claim([], status=HypothesisStatus.UNRESOLVED, prior_revision_id=proposed.revision_id)
        )
    reopened = EvidenceLedger(path)
    assert [r.status for r in reopened.history("hyp-sic")] == [
        HypothesisStatus.PROPOSED,
        HypothesisStatus.SUPPORTED,
        HypothesisStatus.SUPERSEDED,
    ]
    assert reopened.current_hypotheses()["hyp-sic"].revision_id == superseded.revision_id


def test_changed_artifacts_are_detected(tmp_path):
    structure = tmp_path / "relaxed.vasp"
    structure.write_text("Si\n")
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    ledger.append(_evidence(output_artifact_sha256=hashes_of([structure])))
    assert ledger.verify_artifacts() == []
    structure.write_text("C\n")
    [problem] = ledger.verify_artifacts()
    assert "changed since it was recorded" in problem


def test_cli_verifies_and_shows_the_ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = EvidenceLedger(path)
    dft = ledger.append(_evidence())
    ledger.append(_claim([dft.evidence_id]))
    runner = CliRunner()
    result = runner.invoke(app, ["ledger", "verify", str(path)])
    assert result.exit_code == 0 and "hash chain intact" in result.output
    shown = runner.invoke(app, ["ledger", "show", str(path)])
    assert json.loads(shown.output)["hypotheses"]["hyp-sic"]["status"] == "supported"
    path.write_text(path.read_text().replace('"value":0.0', '"value":1.0'))
    result = runner.invoke(app, ["ledger", "verify", str(path)])
    assert result.exit_code == 1 and "tampered" in result.output


def test_the_ledger_sums_observed_cost_by_category(tmp_path):
    from matsim_agents.execution.ledger import CostRecord

    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    dft = ledger.append(_evidence(cost=CostRecord(dft_calculations=1, dft_seconds=30.0)))
    ledger.append(_evidence(kind="relaxation", subject="Si", cost=CostRecord(failed_jobs=1)))
    ledger.append(
        DecisionRecord(
            action="promote",
            subject="SiC",
            evidence_ids=[dft.evidence_id],
            actor="debate:run-1",
            policy_version="review-v1",
            thresholds={"energy_above_hull_eV_per_atom": 0.05},
            observed_cost=CostRecord(llm_prompt_tokens=100, llm_completion_tokens=20),
        )
    )
    cost = EvidenceLedger(tmp_path / "ledger.jsonl").summary()["cost"]
    assert cost["dft_calculations"] == 1 and cost["dft_seconds"] == 30.0
    assert cost["failed_jobs"] == 1
    assert cost["llm_prompt_tokens"] == 100 and cost["llm_completion_tokens"] == 20
    assert cost["nodes"] == 0 and cost["gpu_hours"] == 0.0


def test_node_and_gpu_costs_round_trip(tmp_path):
    from matsim_agents.execution.ledger import CostRecord

    placed = EvidenceRecord(
        kind="relaxation",
        subject="Si",
        method=MethodRecord(backend="qe", method_signature="pbe"),
        evidence_level=EvidenceLevel.LOW_FIDELITY_DFT,
        outcome=EvidenceOutcome.FAILED_TO_COMPUTE,
        failure_reason="x",
        cost=CostRecord(
            failed_jobs=1,
            failed_seconds=3600.0,
            nodes=2,
            node_hours=2.0,
            devices={"rocm-gcd": 16},
            device_hours={"rocm-gcd": 16.0},
            gpus=8.0,
            gpu_hours=8.0,
        ),
    )
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    ledger.append(placed)
    ledger.append(
        _evidence(
            kind="relaxation",
            subject="Si",
            cost=CostRecord(nodes=1, devices={"cuda-gpu": 4}, gpus=4.0),
        )
    )
    reread = EvidenceLedger(tmp_path / "ledger.jsonl")
    assert reread.get(placed.evidence_id).cost == placed.cost
    cost = reread.summary()["cost"]
    assert (cost["nodes"], cost["node_hours"], cost["gpus"], cost["gpu_hours"]) == (
        3,
        2.0,
        12.0,
        8.0,
    )
    assert cost["devices"] == {"rocm-gcd": 16, "cuda-gpu": 4}
    assert cost["device_hours"] == {"rocm-gcd": 16.0}
