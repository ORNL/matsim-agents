"""Running the outstanding steps of a contract's recipe."""

from __future__ import annotations

from matsim_agents.core.contracts import (
    CharacterizationReport,
    CharacterizationStatus,
)
from matsim_agents.execution.ledger import (
    EvidenceLedger,
)
from matsim_agents.science.properties.recipes import (
    PROPERTY_RECIPES,
)
from matsim_agents.workflows.characterization.assessment import _assess, _assess_properties
from matsim_agents.workflows.characterization.contract import CharacterizationContract
from matsim_agents.workflows.characterization.evidence import _assessment_evidence
from matsim_agents.workflows.characterization.plan import plan_recipe
from matsim_agents.workflows.characterization.preflight import _preflight
from matsim_agents.workflows.characterization.relax_steps import Relax, _relax_with_recovery
from matsim_agents.workflows.characterization.state import (
    LEDGER_FILE,
    REPORT_FILE,
    _Run,
    _verify_completed_artifacts,
)
from matsim_agents.workflows.relaxation import (
    run_relaxation,
)


def run_characterization(
    contract: CharacterizationContract,
    *,
    relax: Relax = run_relaxation,
    ledger: EvidenceLedger | None = None,
) -> CharacterizationReport:
    """Run the outstanding steps of the contract's recipe and assess the evidence.

    Relaxations stop at the contract's budget and, for DFT, at its approval; the
    report then says which evidence is missing. ``relax`` runs one relaxation
    (the provenance-tracked relaxation workflow by default). Every relaxation,
    failed or not, and every conclusive assessment is appended to ``ledger``
    (by default ``evidence-ledger.jsonl`` in the run directory); the report
    cites the ledger's evidence identifiers.
    """
    with _Run(contract) as run:
        if ledger is None:
            ledger = EvidenceLedger(run.directory / LEDGER_FILE)
        return _execute(contract, run, relax, ledger)


def _execute(
    contract: CharacterizationContract, run: _Run, relax: Relax, ledger: EvidenceLedger
) -> CharacterizationReport:
    problem = _preflight(contract)
    if problem is None and contract.recipe in PROPERTY_RECIPES:
        report = _assess_properties(contract)
        (run.directory / REPORT_FILE).write_text(report.model_dump_json(indent=2) + "\n")
        return report
    if problem is not None:
        status, missing = problem
        report = CharacterizationReport(
            contract_id=contract.contract_id,
            contract_sha256=contract.digest(),
            recipe=contract.recipe,
            status=status,
            missing=missing,
            limitations=["No calculation was run: the contract cannot be answered as given."],
            recommended_next_test=(
                "Compute the missing references with the contract's DFT method."
                if status == CharacterizationStatus.INCONCLUSIVE
                else "Correct the contract (structures, references, or DFT method)."
            ),
        )
        (run.directory / REPORT_FILE).write_text(report.model_dump_json(indent=2) + "\n")
        return report
    steps = plan_recipe(contract)
    _verify_completed_artifacts(run, steps)
    blockers: list[str] = []
    budget = {
        "mlip": contract.budget.max_mlip_relaxations,
        "dft": contract.budget.max_dft_calculations,
    }
    final = steps[-1] if steps[-1].operator != "relax" else None
    for step in steps:
        if step.operator != "relax":
            continue
        fidelity = step.fidelity or "mlip"
        recovering = run.pending(step.step_id) is not None
        if run.done(step.step_id) is not None and not recovering:
            continue
        if step.needs_dft_approval and not contract.dft_approved:
            blockers.append(f"{step.step_id}: DFT is required but not approved (dft_approved)")
            continue
        if budget[fidelity] is not None and run.attempts(fidelity) >= budget[fidelity]:
            blockers.append(
                f"{step.step_id}: beyond the {fidelity} relaxation budget "
                f"({run.attempts(fidelity)} attempted)"
            )
            continue
        if recovering and final is not None:
            run.forget(final.step_id)
        _relax_with_recovery(contract, step, run, relax, ledger, budget[fidelity])
    report_path = run.directory / REPORT_FILE
    if final is not None and run.done(final.step_id) is not None and report_path.is_file():
        return CharacterizationReport.model_validate_json(report_path.read_text())
    report = _assess(contract, steps, run, blockers)
    report_path.write_text(report.model_dump_json(indent=2) + "\n")
    inputs_done = all(run.done(step.step_id) for step in steps if step.operator == "relax")
    if report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED:
        ledger.append(_assessment_evidence(contract, report, run))
    if final is not None and inputs_done:
        # Assessed from complete inputs, whatever the answer: not repeated on resume.
        run.record(final, {"status": "done", "characterization_status": report.status.value})
    return report
