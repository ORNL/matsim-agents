"""Relaxation steps: one attempt, and its recovery under the recovery policy."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from matsim_agents.core.contracts import (
    EvidenceLevel,
)
from matsim_agents.core.hashing import sha256_file
from matsim_agents.execution.ledger import (
    CostRecord,
    DecisionRecord,
    EvidenceLedger,
)
from matsim_agents.execution.recovery import (
    DEFAULT_POLICY,
    FailureClass,
    adjusted_settings,
    can_retry,
    classify_failure,
)
from matsim_agents.workflows.characterization.contract import CharacterizationContract
from matsim_agents.workflows.characterization.evidence import _relaxation_evidence
from matsim_agents.workflows.characterization.plan import RecipeStep
from matsim_agents.workflows.characterization.state import _Run
from matsim_agents.workflows.relaxation import (
    DFTBackendConfig,
    RelaxationMode,
    ScientificRelaxationConfig,
    ScientificRelaxationResult,
)

Relax = Callable[[ScientificRelaxationConfig], ScientificRelaxationResult]


def _relaxation_config(
    contract: CharacterizationContract,
    step: RecipeStep,
    work_dir: Path,
    *,
    structure: str | None = None,
    dft: DFTBackendConfig | None = None,
) -> ScientificRelaxationConfig:
    return ScientificRelaxationConfig(
        mode=RelaxationMode.DFT if step.fidelity == "dft" else RelaxationMode.MLIP,
        structure_path=str(structure or step.structure),
        output_root=str(work_dir),
        geometry=contract.geometry,
        mlip=contract.methods.mlip,
        dft=(dft or contract.methods.dft) if step.fidelity == "dft" else None,
        approvals=contract.approvals,
        dft_approved=contract.dft_approved,
        max_steps=contract.tolerances.max_relaxation_steps,
        force_tolerance_eV_per_A=contract.tolerances.force_tolerance_eV_per_A,
    )


_ALLOCATION = ("nodes", "devices", "device_kind", "gpus")


def _relax_record(result: ScientificRelaxationResult) -> dict[str, Any]:
    """What a relaxation step contributes: a numerical outcome or an infrastructure failure."""
    stage = result.stages[-1] if result.stages else None
    scf_iterations = stage.scf_iterations_per_step if stage else None
    if result.status == "failed" or stage is None:
        return {
            "status": "failed",
            "failure_reason": result.failure_reason or "no stage result",
            "failure_type": result.failure_type,
            "return_code": stage.return_code if stage else None,
            "scf_iterations_per_step": scf_iterations,
            **({name: getattr(stage, name) for name in _ALLOCATION} if stage else {}),
        }
    diagnostics = {
        "return_code": stage.return_code,
        "scf_converged": stage.scf_converged,
        "scf_iterations_per_step": scf_iterations,
        **{name: getattr(stage, name) for name in _ALLOCATION},
    }
    if (
        stage.energy_eV is None
        or stage.max_force_eV_per_A is None
        or stage.optimized_structure_path is None
    ):
        # The code ran but produced no usable result (e.g. it exited before the first step).
        return {
            "status": "failed",
            "failure_reason": stage.failure_reason
            or "the relaxation produced no energy, force, or structure",
            "run_directory": result.run_directory,
            **diagnostics,
        }
    if not (math.isfinite(stage.energy_eV) and math.isfinite(stage.max_force_eV_per_A)):
        return {
            "status": "failed",
            "failure_reason": (
                f"non-finite relaxation output: energy {stage.energy_eV} eV, "
                f"max force {stage.max_force_eV_per_A} eV/A"
            ),
            "run_directory": result.run_directory,
            **diagnostics,
        }
    return {
        "status": "done",
        "run_id": result.run_id,
        "run_directory": result.run_directory,
        "evidence_level": stage.evidence_level.value,
        "converged": stage.converged,
        "energy_eV": stage.energy_eV,
        "max_force_eV_per_A": stage.max_force_eV_per_A,
        "steps": stage.steps,
        "input_structure_path": stage.input_structure_path,
        "optimized_structure_path": stage.optimized_structure_path,
        "optimized_structure_sha256": sha256_file(stage.optimized_structure_path),
        "failure_reason": stage.failure_reason,
        **diagnostics,
    }


def _attempt(
    contract: CharacterizationContract,
    step: RecipeStep,
    run: _Run,
    relax: Relax,
    ledger: EvidenceLedger,
    structure: str,
    dft: DFTBackendConfig | None,
) -> dict[str, Any]:
    fidelity = step.fidelity or "mlip"
    run.charge(fidelity)
    attempt = run.attempts(fidelity)
    work_dir = run.directory / step.step_id
    if attempt > 1:
        work_dir = work_dir / f"attempt-{attempt}"
    started = time.monotonic()

    def finished(outcome: dict[str, Any]) -> dict[str, Any]:
        outcome = {
            **outcome,
            "structure": step.structure,
            "attempt": attempt,
            "wall_seconds": time.monotonic() - started,
        }
        # An attempt that failed before reporting a stage still held its placement.
        if dft is not None and dft.launch is not None:
            for name, value in dft.launch.allocation().items():
                if outcome.get(name) is None:
                    outcome[name] = value
        if structure != step.structure:
            outcome["input_structure"] = structure
        if dft is not None and dft != contract.methods.dft:
            outcome["dft_settings"] = dft.settings
        return outcome

    try:
        outcome = _relax_record(
            relax(_relaxation_config(contract, step, work_dir, structure=structure, dft=dft))
        )
    except Exception as exc:  # noqa: BLE001 - a failed tool is not evidence
        outcome = {
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_type": type(exc).__name__,
        }
    except BaseException as exc:
        # The charged attempt's cost is recorded; the step and its saved retry are kept.
        interrupted = {
            "status": "failed",
            "failure_reason": f"interrupted by {type(exc).__name__}",
            "failure_type": type(exc).__name__,
        }
        ledger.append(_relaxation_evidence(contract, step, finished(interrupted)))
        raise
    outcome = finished(outcome)
    # QE's ionic convergence message can precede a failed final SCF.
    if outcome["status"] == "done" and outcome.get("scf_converged") is False:
        outcome["converged"] = False
        outcome["evidence_level"] = EvidenceLevel.LOW_FIDELITY_DFT.value
        outcome["failure_reason"] = outcome.get("failure_reason") or "SCF did not converge"
    # A run killed after its last ionic step (e.g. a timeout) can leave finite results.
    if outcome["status"] == "done" and outcome.get("return_code") not in (None, 0):
        outcome["converged"] = False
        outcome["evidence_level"] = EvidenceLevel.LOW_FIDELITY_DFT.value
        outcome["failure_reason"] = (
            outcome.get("failure_reason") or f"exited with return code {outcome['return_code']}"
        )
    tolerance = contract.tolerances.force_tolerance_eV_per_A
    if not (
        outcome["status"] == "done"
        and outcome["converged"]
        and float(outcome["max_force_eV_per_A"]) <= tolerance
    ):
        outcome["failure_class"] = classify_failure(
            outcome, max_steps=contract.tolerances.max_relaxation_steps
        ).value
    return outcome


def _recovery_decision(
    contract: CharacterizationContract,
    step: RecipeStep,
    *,
    attempt: int,
    evidence_id: str,
    failure: FailureClass,
    action: str,
    rationale: str,
    used: int,
    budget: int | None,
) -> DecisionRecord:
    rule = DEFAULT_POLICY.rule(failure)
    return DecisionRecord(
        campaign_id=contract.contract_id,
        iteration=attempt,
        action=f"{action}:{failure.value}",
        subject=step.step_id,
        evidence_ids=[evidence_id],
        rationale=rationale,
        actor=f"recovery-policy:{DEFAULT_POLICY.version}",
        policy_version=DEFAULT_POLICY.version,
        thresholds={
            "max_retries": float(rule.max_retries),
            "retries_used": float(used),
            **({"dft_budget": float(budget)} if budget is not None else {}),
        },
        expected_cost=CostRecord(dft_calculations=1) if action == "recover" else None,
    )


def _resume_recovery(
    contract: CharacterizationContract,
    step: RecipeStep,
    run: _Run,
    ledger: EvidenceLedger,
    pending: dict[str, Any],
    budget: int | None,
) -> None:
    """Record the decision to run a saved retry; a decision already recorded is not repeated."""
    decision = pending.get("decision")
    if decision is None:
        failure = FailureClass(pending["failure"])
        rule = DEFAULT_POLICY.rule(failure)
        used = pending["retries"][failure.value] - 1
        decision = _recovery_decision(
            contract,
            step,
            attempt=pending["attempt"],
            evidence_id=pending["evidence_id"],
            failure=failure,
            action="recover",
            rationale=f"retry {used + 1} of {rule.max_retries}, deferred by the budget: "
            f"{rule.action}",
            used=used,
            budget=budget,
        ).model_dump(mode="json", exclude={"decision_id", "recorded_at"})
        run.set_pending(step.step_id, {**pending, "decision": decision})
    ledger.append(DecisionRecord.model_validate(decision))


def _relax_with_recovery(
    contract: CharacterizationContract,
    step: RecipeStep,
    run: _Run,
    relax: Relax,
    ledger: EvidenceLedger,
    budget: int | None,
) -> None:
    """Relax one step; a failed DFT attempt is retried as the recovery policy directs.

    Every retry is saved before it runs and kept until an outcome without a further
    retry is saved, so an interrupted or budget-stopped retry resumes with its
    structure, settings, and counters.
    """
    structure = str(step.structure)
    dft = contract.methods.dft if step.fidelity == "dft" else None
    retries: dict[FailureClass, int] = {}
    if (pending := run.pending(step.step_id)) is not None:
        structure = pending["structure"]
        dft = DFTBackendConfig.model_validate(pending["dft"])
        retries = {FailureClass(k): v for k, v in pending["retries"].items()}
        _resume_recovery(contract, step, run, ledger, pending, budget)
    while True:
        outcome = _attempt(contract, step, run, relax, ledger, structure, dft)
        evidence = ledger.append(_relaxation_evidence(contract, step, outcome))
        evidence_id = evidence.evidence_id  # type: ignore[union-attr]
        recorded = {**outcome, "evidence_id": evidence_id}
        if dft is None or "failure_class" not in outcome:
            run.record_attempt(step, recorded, None)
            return
        failure = FailureClass(outcome["failure_class"])
        rule = DEFAULT_POLICY.rule(failure)
        used = retries.get(failure, 0)
        within_budget = budget is None or run.attempts("dft") < budget
        allowed = can_retry(failure, timeout_sec=dft.timeout_sec) and used < rule.max_retries
        retry = allowed and within_budget
        if retry:
            rationale = f"retry {used + 1} of {rule.max_retries}: {rule.action}"
        elif not can_retry(failure, timeout_sec=dft.timeout_sec) or rule.max_retries == 0:
            rationale = f"no recovery for {failure.value}"
        elif allowed and not within_budget:
            rationale = f"the DFT budget ({budget}) is spent; the retry is saved"
        else:
            rationale = f"{rule.max_retries} retries for {failure.value} exhausted"
        decision = _recovery_decision(
            contract,
            step,
            attempt=outcome["attempt"],
            evidence_id=evidence_id,
            failure=failure,
            action="recover" if retry else "defer" if allowed else "give_up",
            rationale=rationale,
            used=used,
            budget=budget,
        )
        checkpoint = None
        if allowed:
            retries[failure] = used + 1
            if failure == FailureClass.IONIC_STEP_LIMIT and outcome.get("optimized_structure_path"):
                structure = outcome["optimized_structure_path"]
            settings, timeout = adjusted_settings(
                failure, dft.backend, dft.settings, dft.timeout_sec
            )
            dft = dft.model_copy(update={"settings": settings, "timeout_sec": timeout})
            checkpoint = {
                "structure": structure,
                **(
                    {"structure_sha256": sha256_file(structure)}
                    if structure != step.structure
                    else {}
                ),
                "dft": dft.model_dump(mode="json"),
                "retries": {k.value: v for k, v in retries.items()},
                "attempt": outcome["attempt"],
                "evidence_id": evidence_id,
                "failure": failure.value,
                "decision": decision.model_dump(mode="json", exclude={"decision_id", "recorded_at"})
                if retry
                else None,
            }
        run.record_attempt(step, recorded, checkpoint)
        ledger.append(decision)
        if not retry:
            return
