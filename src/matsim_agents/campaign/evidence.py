"""Campaign evidence in the append-only ledger.

A campaign keeps its ledger next to its state
(``<output_dir>/evidence-ledger.jsonl``). Every formula run contributes
evidence records: the stability report when there is one, and otherwise the
failure to compute it. Reviews cite these identifiers, and their hypothesis
revisions and decisions are appended after deterministic validation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from matsim_agents.discovery.stability import RankingMode, StabilityReport
from matsim_agents.execution.contracts import DFT_EVIDENCE, EvidenceLevel, WorkflowStatus
from matsim_agents.execution.ledger import (
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    LedgerQuantity,
    MethodRecord,
    hashes_of,
    software_versions,
)

if TYPE_CHECKING:
    from matsim_agents.campaign.state import CampaignState, FormulaRunRecord

LEDGER_FILE = "evidence-ledger.jsonl"


def new_hypothesis_id(campaign_id: str, claim: str) -> str:
    digest = hashlib.sha256(f"{campaign_id}\n{claim.strip()}".encode()).hexdigest()
    return f"hyp-{digest[:16]}"


def _method(report: StabilityReport, record: FormulaRunRecord | None) -> MethodRecord:
    evidence = record.evidence if record is not None else {}
    software = software_versions("ase")
    if report.evidence_level in DFT_EVIDENCE:
        refinement = evidence.get("dft_refinement", {})
        return MethodRecord(
            backend=str(refinement.get("backend") or "dft"),
            method_signature=refinement.get("method_signature"),
            software_versions=software,
        )
    return MethodRecord(
        backend=str(evidence.get("mlip_backend") or "mlip"),
        model=evidence.get("model_identifier"),
        model_sha256=evidence.get("model_checkpoint_hash"),
        software_versions=software,
    )


def stability_evidence(
    campaign: CampaignState, report: StabilityReport, record: FormulaRunRecord | None = None
) -> EvidenceRecord:
    """The ledger record of a stability report (a hull or a same-composition ranking)."""
    hull = report.ranking_mode == RankingMode.CONVEX_HULL
    values: list[LedgerQuantity] = []
    for phase in report.ranking:
        quantities = [
            ("energy_per_atom", phase.energy_per_atom_eV),
            ("relative_energy_per_atom", phase.delta_e_above_min_eV_per_atom),
            ("formation_energy_per_atom", phase.formation_energy_eV_per_atom),
            ("energy_above_hull", phase.energy_above_hull_eV_per_atom),
        ]
        values += [
            LedgerQuantity(
                name=name, value=float(value), unit="eV/atom", structure=phase.structure_path
            )
            for name, value in quantities
            if value is not None
        ]
    ground = report.ground_state
    complete = None
    if hull and campaign.reference_energies is not None:
        complete = (
            report.reference_set_id == campaign.reference_energies.identifier
            and not campaign.reference_energies.audit_completeness(
                list(campaign.element_set)
            ).provisional
        )
    return EvidenceRecord(
        kind="stability" if hull else "energy_ranking",
        subject=report.formula,
        campaign_id=campaign.campaign_id,
        run_id=record.output_dir if record is not None else None,
        task_id=f"iteration-{record.iteration}" if record is not None else None,
        method=_method(report, record),
        evidence_level=report.evidence_level,
        outcome=EvidenceOutcome.MEASURED if ground.converged else EvidenceOutcome.NOT_CONVERGED,
        failure_reason=None if ground.converged else "the ground state did not converge",
        values=values,
        numerical_convergence={
            "ground_state_converged": ground.converged,
            "ground_state_max_force_eV_per_A": ground.final_max_force_eV_per_A,
            "degeneracy_tolerance_eV_per_atom": report.degeneracy_tolerance_eV_per_atom,
            "near_degenerate_structures": len(report.near_degenerate_structure_paths),
        },
        reference_set_id=report.reference_set_id,
        reference_complete=complete,
        input_sha256=hashes_of([phase.structure_path for phase in report.ranking]),
        output_artifact_sha256=hashes_of(
            [phase.optimized_structure_path for phase in report.ranking]
        ),
        physical_assumptions=(
            ["0 K energies of the declared reference phases"]
            if hull
            else ["same-composition energy ranking; not a thermodynamic-stability claim"]
        ),
    )


def formula_failure_evidence(campaign: CampaignState, record: FormulaRunRecord) -> EvidenceRecord:
    """A formula run that produced no stability report: a failure, not evidence."""
    return EvidenceRecord(
        kind="formula_run",
        subject=record.formula,
        campaign_id=campaign.campaign_id,
        run_id=record.output_dir,
        task_id=f"iteration-{record.iteration}",
        method=MethodRecord(
            backend=str(record.evidence.get("mlip_backend") or "campaign"),
            model=record.evidence.get("model_identifier"),
            software_versions=software_versions("ase"),
        ),
        evidence_level=_failure_level(record),
        outcome=EvidenceOutcome.FAILED_TO_COMPUTE,
        failure_reason=record.failure_reason
        or f"no stability report (outcome {record.outcome_class or record.status})",
        numerical_convergence={"candidate_counts": dict(record.candidate_counts)},
    )


def _failure_level(record: FormulaRunRecord) -> EvidenceLevel:
    return (
        EvidenceLevel.LOW_FIDELITY_DFT
        if record.n_dft_calculations
        else EvidenceLevel.MLIP_RELAXATION
    )


_REPORT_KINDS = {"stability", "energy_ranking"}


def _latest_report_evidence(ledger: EvidenceLedger, formula: str) -> EvidenceRecord | None:
    reports = [e for e in ledger.evidence(subject=formula) if e.kind in _REPORT_KINDS]
    return reports[-1] if reports else None


def _same_measurement(a: EvidenceRecord, b: EvidenceRecord) -> bool:
    fields = (
        "kind",
        "evidence_level",
        "outcome",
        "values",
        "reference_set_id",
        "reference_complete",
        "method",
        "input_sha256",
        "output_artifact_sha256",
    )
    return all(getattr(a, name) == getattr(b, name) for name in fields)


def sync_stability_evidence(ledger: EvidenceLedger, campaign: CampaignState) -> list[str]:
    """Append evidence for every stability report that changed since it was recorded.

    Hull recalibration changes earlier reports' formation and hull energies;
    the new values become a new record whose parent is the previous one.
    """
    appended = []
    for formula, report in sorted(campaign.stability_reports.items()):
        record = campaign.formula_runs.get(formula)
        latest = _latest_report_evidence(ledger, formula)
        candidate = stability_evidence(campaign, report, record)
        if latest is not None and _same_measurement(latest, candidate):
            continue
        if latest is not None:
            candidate = candidate.model_copy(update={"parent_record_ids": [latest.evidence_id]})
            candidate = EvidenceRecord.model_validate(candidate.model_dump(exclude={"evidence_id"}))
        appended.append(ledger.append(candidate).evidence_id)  # type: ignore[union-attr]
    return appended


def record_formula_evidence(
    ledger: EvidenceLedger, campaign: CampaignState, record: FormulaRunRecord
) -> list[str]:
    """Append the formula run's evidence and remember its identifiers on the record."""
    if record.status not in {WorkflowStatus.COMPLETE, WorkflowStatus.FAILED}:
        return []
    if record.status == WorkflowStatus.COMPLETE and record.formula in campaign.stability_reports:
        sync_stability_evidence(ledger, campaign)
        latest = _latest_report_evidence(ledger, record.formula)
        assert latest is not None
        identifiers = [latest.evidence_id]
    else:
        appended = ledger.append(formula_failure_evidence(campaign, record))
        identifiers = [appended.evidence_id]  # type: ignore[union-attr]
    record.evidence["ledger_evidence_ids"] = identifiers
    return identifiers


def evidence_for_review(ledger: EvidenceLedger, formula: str) -> list[dict[str, Any]]:
    """What a reviewer may cite for a formula, with fidelity and comparability metadata."""
    return [
        {
            "evidence_id": e.evidence_id,
            "kind": e.kind,
            "evidence_level": e.evidence_level.value,
            "outcome": e.outcome.value,
            "method": {
                "backend": e.method.backend,
                "model": e.method.model,
                "method_signature": e.method.method_signature,
            },
            "reference_set_id": e.reference_set_id,
            "reference_complete": e.reference_complete,
            "numerical_convergence": e.numerical_convergence,
            "values": [
                {"name": q.name, "value": q.value, "unit": q.unit, "structure": q.structure}
                for q in e.values
                if q.name in {"energy_above_hull", "formation_energy_per_atom"}
                or (q.name == "relative_energy_per_atom")
            ],
            "failure_reason": e.failure_reason,
        }
        for e in ledger.evidence(subject=formula)
    ]


def open_campaign_ledger(campaign: CampaignState, output_dir: str | Path) -> EvidenceLedger:
    """Open the campaign's ledger and attach it to the campaign."""
    ledger = EvidenceLedger(Path(output_dir) / LEDGER_FILE)
    campaign.attach_ledger(ledger)
    return ledger


__all__ = [
    "LEDGER_FILE",
    "evidence_for_review",
    "formula_failure_evidence",
    "new_hypothesis_id",
    "open_campaign_ledger",
    "record_formula_evidence",
    "stability_evidence",
    "sync_stability_evidence",
]
