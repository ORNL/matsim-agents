"""Assessing the evidence of completed steps into a characterization report."""

from __future__ import annotations

from matsim_agents.core.contracts import (
    DFT_EVIDENCE,
    CharacterizationReport,
    CharacterizationStatus,
    EvidenceItem,
    EvidenceLevel,
    ObservableValue,
    at_least,
)
from matsim_agents.core.relaxation import RelaxationResult
from matsim_agents.science.defects.defects import MAGNETISM_LIMITATION
from matsim_agents.science.properties.recipes import (
    BackendContext,
    assess_property_recipe,
    load_property_evidence,
)
from matsim_agents.science.thermodynamics.stability import (
    RankingMode,
    StabilityReport,
    score_stability,
)
from matsim_agents.workflows.characterization.contract import CharacterizationContract, _references
from matsim_agents.workflows.characterization.plan import RecipeStep
from matsim_agents.workflows.characterization.state import _Run


def _assess_properties(contract: CharacterizationContract) -> CharacterizationReport:
    """Reuse compatible property evidence; report what is missing otherwise."""
    levels = {r.name: r.minimum_evidence_level for r in contract.observables}
    return assess_property_recipe(
        contract_id=contract.contract_id,
        contract_sha256=contract.digest(),
        recipe_name=contract.recipe,
        structures=list(contract.structures),
        composition=contract.formula,
        conditions=contract.conditions,
        evidence=load_property_evidence(list(contract.property_evidence)),
        minimum_levels=levels,
        backend=BackendContext(
            mlip_supports_stress=contract.mlip_supports_stress,
            dft_backend=contract.methods.dft.backend if contract.methods.dft else None,
            executables=contract.transport_executables,
            calculation_cost_node_hours=contract.node_hours_per_calculation,
            remaining_node_hours=contract.budget.max_node_hours,
        ),
        approved=contract.dft_approved,
        recipe_properties=[r.name for r in contract.observables],
    )


def _evidence_item(contract: CharacterizationContract, step_id: str, outcome: dict) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=outcome["evidence_id"],
        kind="relaxation",
        evidence_level=EvidenceLevel(outcome["evidence_level"]),
        method=outcome.get("run_id", ""),
        artifact=outcome.get("optimized_structure_path"),
        artifact_sha256=outcome.get("optimized_structure_sha256"),
        converged=outcome["converged"],
    )


def _relaxation_results(relaxed: list[tuple[str, dict]]) -> list[RelaxationResult]:
    return [
        RelaxationResult(
            structure_path=outcome["structure"],
            optimized_structure_path=outcome["optimized_structure_path"],
            trajectory_path="",
            log_csv_path="",
            final_energy_eV=outcome["energy_eV"],
            final_max_force_eV_per_A=outcome["max_force_eV_per_A"],
            num_steps=outcome["steps"],
            converged=outcome["converged"],
        )
        for _step_id, outcome in relaxed
    ]


def _assess(
    contract: CharacterizationContract,
    steps: list[RecipeStep],
    run: _Run,
    blockers: list[str],
) -> CharacterizationReport:
    relax_steps = [step for step in steps if step.operator == "relax"]
    outcomes = {step.step_id: run.state["steps"].get(step.step_id) for step in relax_steps}
    relaxed = [
        (step_id, outcome)
        for step_id, outcome in outcomes.items()
        if outcome is not None and outcome["status"] == "done"
    ]
    evidence = [_evidence_item(contract, step_id, outcome) for step_id, outcome in relaxed]
    base = {
        "contract_id": contract.contract_id,
        "contract_sha256": contract.digest(),
        "recipe": contract.recipe,
        "evidence": evidence,
        "methods": {
            "relaxation": "dft" if contract.needs_dft() else "mlip",
            **(
                {"dft_method_signature": contract.methods.dft_method_signature}
                if contract.methods.dft_method_signature
                else {}
            ),
        },
    }
    failed = [
        f"{step_id}: {outcome['failure_reason']}"
        for step_id, outcome in outcomes.items()
        if outcome is not None and outcome["status"] == "failed"
    ]
    if failed:
        return CharacterizationReport(
            **base,
            status=CharacterizationStatus.FAILED_INFRASTRUCTURE,
            missing=failed + blockers,
            limitations=["A calculation could not run; this is not evidence about the material."],
            recommended_next_test="Fix the failed calculations and run the contract again "
            "(completed steps are kept).",
        )
    tolerance = contract.tolerances.force_tolerance_eV_per_A
    observables = []
    for _step_id, outcome in relaxed:
        level = EvidenceLevel(outcome["evidence_level"])
        cite = [outcome["evidence_id"]]
        structure = outcome["structure"]
        observables += [
            ObservableValue(
                name="relaxed_structure",
                value=str(outcome["optimized_structure_path"]),
                unit="structure",
                evidence_level=level,
                structure=structure,
                evidence_ids=cite,
            ),
            ObservableValue(
                name="residual_max_force",
                value=float(outcome["max_force_eV_per_A"]),
                unit="eV/A",
                evidence_level=level,
                structure=structure,
                evidence_ids=cite,
            ),
        ]
    unconverged = [
        step_id
        for step_id, outcome in relaxed
        if not outcome["converged"] or outcome["max_force_eV_per_A"] > tolerance
    ]
    missing = blockers + [
        f"{step_id}: not converged to {tolerance} eV/A (numerical; not evidence against "
        "the material)"
        for step_id in unconverged
    ]
    if missing:
        return CharacterizationReport(
            **base,
            status=CharacterizationStatus.INCONCLUSIVE,
            observables=observables,
            missing=missing,
            recommended_next_test=(
                "Approve the required DFT calculations"
                if any("not approved" in item for item in blockers)
                else "Complete the missing relaxations (more steps or budget) and run the "
                "contract again"
            ),
        )
    level = _level(relaxed)
    below = [
        f"{r.name} at {r.minimum_evidence_level.value} (the evidence is {level.value})"
        for r in contract.observables
        if not at_least(level, r.minimum_evidence_level)
    ]
    if below:
        return CharacterizationReport(
            **base,
            status=CharacterizationStatus.INCONCLUSIVE,
            observables=observables,
            missing=below,
            recommended_next_test="Relax the structures at the required fidelity.",
        )
    return {
        "relaxation": _assess_relaxation,
        "energy_ranking": _assess_ranking,
        "thermodynamic_stability": _assess_stability,
    }[contract.recipe](contract, relaxed, observables, base)


def _energy_observables(
    contract: CharacterizationContract, relaxed: list[tuple[str, dict]]
) -> list[ObservableValue]:
    from ase.io import read

    values = []
    for _step_id, outcome in relaxed:
        n_atoms = len(read(outcome["optimized_structure_path"]))
        values.append(
            ObservableValue(
                name="energy_per_atom",
                value=float(outcome["energy_eV"]) / n_atoms,
                unit="eV/atom",
                evidence_level=EvidenceLevel(outcome["evidence_level"]),
                structure=outcome["structure"],
                evidence_ids=[outcome["evidence_id"]],
            )
        )
    return values


def _level(relaxed: list[tuple[str, dict]]) -> EvidenceLevel:
    return min(
        (EvidenceLevel(outcome["evidence_level"]) for _step, outcome in relaxed),
        key=list(EvidenceLevel).index,
    )


def _assess_relaxation(contract, relaxed, observables, base) -> CharacterizationReport:
    level = _level(relaxed)
    return CharacterizationReport(
        **base,
        status=CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED,
        conclusion=f"All {len(relaxed)} structure(s) of {contract.formula} relaxed to residual "
        f"forces at most {contract.tolerances.force_tolerance_eV_per_A} eV/A ({level.value}).",
        observables=observables + _energy_observables(contract, relaxed),
        limitations=[
            "A converged relaxation is a local minimum of the method's energy; it says "
            "nothing about stability against other structures or compositions."
        ],
    )


def _assess_ranking(contract, relaxed, observables, base) -> CharacterizationReport:
    level = _level(relaxed)
    report = score_stability(
        contract.formula,
        _relaxation_results(relaxed),
        force_tol_eV_per_A=contract.tolerances.force_tolerance_eV_per_A,
        degeneracy_tol_eV_per_atom=contract.tolerances.degeneracy_tolerance_eV_per_atom,
        evidence_level=level,
    )
    observables = (
        observables
        + _energy_observables(contract, relaxed)
        + _ranking_values(contract, relaxed, report)
    )
    limitations = [
        "A same-composition ranking; it does not establish formation energy, convex-hull "
        "stability, or synthesizability."
    ]
    if report.near_degenerate_structure_paths:
        return CharacterizationReport(
            **base,
            status=CharacterizationStatus.INCONCLUSIVE,
            observables=observables,
            missing=[
                f"a ground state distinguishable within "
                f"{contract.tolerances.degeneracy_tolerance_eV_per_atom} eV/atom: "
                f"{len(report.near_degenerate_structure_paths)} structure(s) are as low"
            ],
            limitations=limitations,
            recommended_next_test="Rank the near-degenerate structures at higher fidelity.",
        )
    ground = report.ground_state
    gap = report.ranking[1].delta_e_above_min_eV_per_atom if len(report.ranking) > 1 else None
    return CharacterizationReport(
        **base,
        status=CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED,
        conclusion=f"Lowest-energy structure of {contract.formula} among "
        f"{len(report.ranking)} ranked: {ground.structure_path}"
        + (f", {gap:.4g} eV/atom below the next" if gap is not None else "")
        + f" ({level.value}).",
        observables=observables,
        limitations=limitations,
    )


def _ranking_values(
    contract: CharacterizationContract, relaxed: list[tuple[str, dict]], report: StabilityReport
) -> list[ObservableValue]:
    evidence_by_input = {outcome["structure"]: outcome["evidence_id"] for _s, outcome in relaxed}
    return [
        ObservableValue(
            name="relative_energy_per_atom",
            value=phase.delta_e_above_min_eV_per_atom,
            unit="eV/atom",
            evidence_level=report.evidence_level,
            structure=phase.structure_path,
            evidence_ids=[evidence_by_input[phase.structure_path]],
        )
        for phase in report.ranking
    ]


def _assess_stability(contract, relaxed, observables, base) -> CharacterizationReport:
    level = _level(relaxed)
    limitations = [
        "Stability is relative to the declared reference phases (the lowest among those "
        "tested, not a global ground-state search), at 0 K without zero-point or "
        "finite-temperature contributions.",
        MAGNETISM_LIMITATION,
    ]
    if level not in DFT_EVIDENCE or not at_least(level, EvidenceLevel.CONVERGED_DFT):
        return CharacterizationReport(
            **base,
            status=CharacterizationStatus.INCONCLUSIVE,
            observables=observables,
            missing=[f"converged DFT energies (the evidence is {level.value})"],
            limitations=limitations,
            recommended_next_test="Relax the structures with converged DFT.",
        )
    references = _references(contract)
    report = score_stability(
        contract.formula,
        _relaxation_results(relaxed),
        force_tol_eV_per_A=contract.tolerances.force_tolerance_eV_per_A,
        degeneracy_tol_eV_per_atom=contract.tolerances.degeneracy_tolerance_eV_per_atom,
        ranking_mode=RankingMode.CONVEX_HULL,
        reference_energies=references,
        method_signature=contract.methods.dft_method_signature,
        evidence_level=level,
    )
    ground = report.ground_state
    cite = [
        outcome["evidence_id"]
        for _step_id, outcome in relaxed
        if outcome["structure"] == ground.structure_path
    ]
    observables = (
        observables
        + _energy_observables(contract, relaxed)
        + [
            ObservableValue(
                name=name,
                value=float(value),
                unit="eV/atom",
                evidence_level=level,
                structure=ground.structure_path,
                evidence_ids=cite,
            )
            for name, value in (
                ("formation_energy_per_atom", ground.formation_energy_eV_per_atom),
                ("energy_above_hull", ground.energy_above_hull_eV_per_atom),
            )
        ]
    )
    e_hull = float(ground.energy_above_hull_eV_per_atom)
    on_hull = e_hull <= contract.tolerances.hull_tolerance_eV_per_atom + 1e-8
    decomposition = ", ".join(sorted(ground.decomposition))
    conclusion = (
        f"{contract.formula} ({ground.structure_path}) is on the convex hull of the declared "
        f"reference phases within {contract.tolerances.hull_tolerance_eV_per_atom} eV/atom "
        f"({e_hull:.4g} eV/atom above it; {level.value}, "
        f"{references.method_signature})."
        if on_hull
        else f"{contract.formula} ({ground.structure_path}) lies {e_hull:.4g} eV/atom above "
        f"the convex hull of the declared reference phases and decomposes to "
        f"{decomposition} ({level.value}, {references.method_signature})."
    )
    return CharacterizationReport(
        **base,
        status=CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED,
        conclusion=conclusion,
        observables=observables,
        limitations=limitations,
    )
