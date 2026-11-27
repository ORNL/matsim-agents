"""Ledger records of relaxation attempts and of conclusive assessments."""

from __future__ import annotations

from typing import Any

from matsim_agents.core.contracts import (
    CharacterizationReport,
    EvidenceLevel,
)
from matsim_agents.execution.ledger import (
    CostRecord,
    EvidenceOutcome,
    EvidenceRecord,
    LedgerQuantity,
    MethodRecord,
    hashes_of,
    software_versions,
)
from matsim_agents.workflows.characterization.contract import CharacterizationContract, _references
from matsim_agents.workflows.characterization.plan import RecipeStep, plan_recipe
from matsim_agents.workflows.characterization.state import _Run


def _method_record(
    contract: CharacterizationContract, fidelity: str, dft_settings: dict | None = None
) -> MethodRecord:
    if fidelity == "dft":
        dft = contract.methods.dft
        assert dft is not None
        directories = {"pseudo_dir": dft.pseudo_dir, "potcar_dir": dft.potcar_dir}
        return MethodRecord(
            backend=dft.backend,
            method_signature=contract.methods.dft_method_signature,
            settings={
                **(dft.settings if dft_settings is None else dft_settings),
                **{k: v for k, v in directories.items() if v},
            },
            software_versions=software_versions("ase"),
        )
    mlip = contract.methods.mlip
    if mlip is None:
        raise ValueError("MLIP steps require methods.mlip")
    return MethodRecord(
        backend=mlip.backend,
        model=mlip.model_reference,
        settings=mlip.model_dump(mode="json"),
        software_versions=software_versions("ase", "fairchem-core", "mace-torch"),
    )


def _relaxation_evidence(
    contract: CharacterizationContract, step: RecipeStep, outcome: dict[str, Any]
) -> EvidenceRecord:
    """The ledger record of one relaxation attempt: a result or a failure to compute it."""
    fidelity = step.fidelity or "mlip"
    tolerance = contract.tolerances.force_tolerance_eV_per_A
    seconds = float(outcome.get("wall_seconds") or 0.0)
    failed = outcome["status"] != "done" or outcome.get("return_code") not in (None, 0)
    nodes, devices = int(outcome.get("nodes") or 0), int(outcome.get("devices") or 0)
    gpus, kind, hours = (
        float(outcome.get("gpus") or 0.0),
        outcome.get("device_kind"),
        seconds / 3600,
    )
    cost = (
        CostRecord(failed_jobs=1, failed_seconds=seconds)
        if failed
        else CostRecord(dft_calculations=1, dft_seconds=seconds)
        if fidelity == "dft"
        else CostRecord(mlip_relaxations=1, mlip_seconds=seconds)
    ) + CostRecord(
        nodes=nodes,
        node_hours=nodes * hours,
        devices={kind: devices} if kind and devices else {},
        device_hours={kind: devices * hours} if kind and devices else {},
        gpus=gpus,
        gpu_hours=gpus * hours,
    )
    diagnosis = {"failure_class": outcome["failure_class"]} if outcome.get("failure_class") else {}
    if outcome.get("failure_class"):
        diagnosis.update(
            {
                key: outcome[key]
                for key in ("return_code", "scf_converged", "scf_iterations_per_step")
                if outcome.get(key) is not None
            }
        )
    common: dict[str, Any] = {
        "kind": "relaxation",
        "subject": contract.formula,
        "characterization_contract_id": contract.contract_id,
        "calculation_id": step.step_id,
        "task_id": f"attempt-{outcome['attempt']}",
        "run_id": outcome.get("run_id"),
        "method": _method_record(contract, fidelity, outcome.get("dft_settings")),
        "input_sha256": hashes_of([outcome.get("input_structure") or step.structure]),
        "cost": cost,
        "physical_assumptions": [
            f"relax_atoms={contract.geometry.relax_atoms}, "
            f"relax_cell={contract.geometry.relax_cell}, "
            f"pressure_GPa={contract.geometry.pressure_GPa}"
        ],
    }
    if outcome["status"] != "done":
        return EvidenceRecord(
            **common,
            evidence_level=(
                EvidenceLevel.LOW_FIDELITY_DFT
                if fidelity == "dft"
                else EvidenceLevel.MLIP_RELAXATION
            ),
            outcome=EvidenceOutcome.FAILED_TO_COMPUTE,
            failure_reason=outcome["failure_reason"],
            numerical_convergence=diagnosis,
        )
    from ase.io import read

    force = float(outcome["max_force_eV_per_A"])
    converged = bool(outcome["converged"]) and force <= tolerance
    n_atoms = len(read(outcome["optimized_structure_path"]))
    return EvidenceRecord(
        **common,
        evidence_level=EvidenceLevel(outcome["evidence_level"]),
        outcome=EvidenceOutcome.MEASURED if converged else EvidenceOutcome.NOT_CONVERGED,
        failure_reason=None
        if converged
        else outcome.get("failure_reason") or f"residual force above {tolerance} eV/A",
        values=[
            LedgerQuantity(name="total_energy", value=float(outcome["energy_eV"]), unit="eV"),
            LedgerQuantity(
                name="energy_per_atom", value=float(outcome["energy_eV"]) / n_atoms, unit="eV/atom"
            ),
            LedgerQuantity(name="residual_max_force", value=force, unit="eV/A"),
        ],
        numerical_convergence={
            "converged": converged,
            "max_force_eV_per_A": force,
            "force_tolerance_eV_per_A": tolerance,
            "steps": outcome.get("steps"),
            "max_steps": contract.tolerances.max_relaxation_steps,
            **diagnosis,
        },
        output_artifact_sha256={
            outcome["optimized_structure_path"]: outcome["optimized_structure_sha256"]
        },
    )


_ASSESSMENT_KIND = {
    "relaxation": "characterization",
    "energy_ranking": "energy_ranking",
    "thermodynamic_stability": "stability",
}


def _assessment_evidence(
    contract: CharacterizationContract, report: CharacterizationReport, run: _Run
) -> EvidenceRecord:
    """The ledger record of a sufficiently characterized answer, citing its relaxations."""
    relaxed = [
        outcome
        for step in plan_recipe(contract)
        if step.operator == "relax" and (outcome := run.done(step.step_id)) is not None
    ]
    stability = contract.recipe == "thermodynamic_stability"
    return EvidenceRecord(
        kind=_ASSESSMENT_KIND[contract.recipe],
        subject=contract.formula,
        characterization_contract_id=contract.contract_id,
        calculation_id=contract.recipe,
        method=_method_record(contract, "dft" if contract.needs_dft() else "mlip"),
        evidence_level=min(
            (EvidenceLevel(o["evidence_level"]) for o in relaxed), key=list(EvidenceLevel).index
        ),
        outcome=EvidenceOutcome.MEASURED,
        values=[
            LedgerQuantity(name=v.name, value=v.value, unit=v.unit, structure=v.structure)
            for v in report.observables
            if v.unit != "structure"
        ],
        numerical_convergence={"characterization_status": report.status.value},
        reference_set_id=_references(contract).identifier if stability else None,
        reference_complete=True if stability else None,
        input_sha256=hashes_of([contract.reference_energies]) if stability else {},
        physical_assumptions=report.limitations,
        parent_record_ids=[o["evidence_id"] for o in relaxed],
    )
