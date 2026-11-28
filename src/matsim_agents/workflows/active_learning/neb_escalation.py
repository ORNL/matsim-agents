"""Escalation of sampled NEB pathways to DFT and the kinetics they feed."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from matsim_agents.core.fileio import append_line
from matsim_agents.execution.placement import Placement
from matsim_agents.science.neb.contracts import NebResult, NebTransition
from matsim_agents.workflows.active_learning.config import (
    ALConfig,
    NebEscalationConfig,
    NebSamplingConfig,
)
from matsim_agents.workflows.active_learning.dft_attempts import create_attempt, update_attempt
from matsim_agents.workflows.active_learning.iteration_state import IterationState, _write_state

_DFT_NEB_RESERVATIONS = "dft_neb_reservations.jsonl"


def _completed_dft_neb_count(root: Path) -> int:
    """Count reserved full DFT NEB runs so the configured cap survives resume.

    Reservations live in a root-level ledger because ``_scan_resume`` deletes an
    interrupted iteration directory, and a reserved QE run may already have cost
    its allocation. Runs recorded in an iteration state but not in the ledger (from before
    the ledger) are added; ledger-backed runs are never counted twice.
    """
    reserved_by_iteration: dict[int, int] = {}
    ledger = root / _DFT_NEB_RESERVATIONS
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line:
                iteration = int(json.loads(line)["iteration"])
                reserved_by_iteration[iteration] = reserved_by_iteration.get(iteration, 0) + 1
    legacy = 0
    for path in root.glob("iteration_*/state.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        count = state.get("n_dft_neb_runs", 0)
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid n_dft_neb_runs in {path}")
        legacy += max(0, count - reserved_by_iteration.get(int(state.get("iteration", -1)), 0))
    return sum(reserved_by_iteration.values()) + legacy


def _reserve_dft_neb(root: Path, iteration: int, transition_id: str) -> None:
    entry = json.dumps({"iteration": iteration, "transition_id": transition_id}, sort_keys=True)
    append_line(root / _DFT_NEB_RESERVATIONS, entry)


def _apply_neb_escalation(
    policy: NebEscalationConfig | None,
    outcome,
    iteration_dir: Path,
    pathway_directory: Path,
    state: IterationState,
    completed_dft_neb: int,
    env: Mapping[str, str] | None = None,
    timeout_sec: float | None = None,
) -> tuple[dict[str, Any], bool]:
    result = outcome.result
    uncertainty = result.pathway_uncertainty_eV
    record: dict[str, Any] = {
        "transition_id": result.provenance.get("transition_id"),
        "mlip_result": result.model_dump(mode="json"),
        "decision": "not_configured",
    }
    if policy is None:
        return record, False

    acceptable = (
        result.state == "converged_unqualified"
        and uncertainty is not None
        and uncertainty <= (policy.max_pathway_uncertainty_eV or -1.0)
    )
    run_dft = policy.policy == "always_dft_neb" or (
        policy.policy == "uncertainty_gated" and not acceptable
    )
    if acceptable:
        reason = None
    elif result.state != "converged_unqualified":
        reason = f"mlip_state_{result.state}"
    elif uncertainty is None:
        reason = "uncertainty_unavailable"  # unavailable is never read as low
    else:
        reason = "uncertainty_above_limit"
    record["escalation"] = {
        "policy": policy.policy,
        "mlip_model_id": result.provenance.get("model_id"),
        "reason": "always_dft_neb" if policy.policy == "always_dft_neb" else reason,
    }
    if not run_dft:
        record["decision"] = "accepted_mlip" if acceptable else "not_accepted"
        return record, False

    if completed_dft_neb >= policy.max_dft_neb_runs:
        raise RuntimeError(
            "full DFT NEB escalation required, but escalation.max_dft_neb_runs is exhausted"
        )
    transition = outcome.transition
    if transition is None:
        raise ValueError("full DFT NEB escalation requires typed transition metadata")
    if policy.dft_neb is None:
        raise ValueError("full DFT NEB escalation has no QE NEB configuration")
    if any(len(image.constraints) for image in outcome.images):
        # The QE NEB input carries no ASE constraints; running it would relax atoms the
        # surrogate pathway held fixed. Refuse before spending any of the DFT budget.
        raise ValueError(
            f"transition {transition.transition_id!r} has constrained images; full DFT NEB "
            "escalation does not preserve ASE constraints"
        )

    attempt = create_attempt(
        iteration_dir.parent,
        iteration=state.iteration,
        kind="neb",
        identity=transition.transition_id,
    )
    update_attempt(attempt, backend="qe", source_pathway=str(pathway_directory))
    _reserve_dft_neb(iteration_dir.parent, state.iteration, transition.transition_id)
    state.n_dft_neb_runs += 1
    record["decision"] = "dispatching_dft_neb"
    record["dft_attempt_id"] = attempt.name
    record["dft_attempt_directory"] = str(attempt)
    state.neb_pathways.append(record)
    _write_state(iteration_dir, state)

    from matsim_agents.science.neb.endpoints import MigrationPair
    from matsim_agents.workflows.neb.workflow import run_neb_workflow

    # The optimized MLIP path and the endpoints it was actually built from (relaxed or
    # reordered), not the transition's generated endpoints or a fresh linear guess.
    initial = getattr(outcome, "initial", None) or outcome.images[0]
    final = getattr(outcome, "final", None) or outcome.images[-1]
    pair = MigrationPair(
        initial=initial.copy(),
        final=final.copy(),
        transition=transition,
        final_config=None,
    )
    start_images = tuple(outcome.images) if len(outcome.images) == policy.dft_neb.n_images else None
    try:
        dft_outcome = run_neb_workflow(
            pair,
            policy.dft_neb,
            attempt / "neb",
            dispatch=True,
            allow_unrelaxed_endpoints=policy.allow_unrelaxed_endpoints,
            timeout_sec=timeout_sec,
            env=env,
            images=start_images,
        )
    except Exception as exc:
        update_attempt(attempt, status="failed", error=str(exc))
        raise
    record["dft_start_path"] = (
        "mlip_optimized" if start_images is not None else "linear_regenerated"
    )
    if dft_outcome.plan.status != "planned" or dft_outcome.result is None:
        update_attempt(attempt, status="failed", error="; ".join(dft_outcome.plan.reasons))
        raise RuntimeError(
            "QE NEB escalation did not produce a result: " + "; ".join(dft_outcome.plan.reasons)
        )
    record["decision"] = "dft_neb_complete"
    record["dft_result"] = dft_outcome.result.model_dump(mode="json")
    record["dft_run_directory"] = dft_outcome.plan.directory
    if dft_outcome.result.state not in ("converged_unqualified", "qualified"):
        record["decision"] = "dft_neb_unconverged"
    update_attempt(
        attempt,
        status="converged" if record["decision"] == "dft_neb_complete" else "failed",
        result_state=dft_outcome.result.state,
        work_dir=dft_outcome.plan.directory,
    )
    return record, True


def _escalate_neb_pathways(
    cfg: ALConfig,
    pathways,
    it_dir: Path,
    root: Path,
    state: IterationState,
    completed_dft_neb: int,
    placement: Placement,
) -> int:
    """Apply the escalation policy and kinetics output to each sampled NEB pathway."""
    env = None
    if cfg.escalation is not None:
        # An empty launcher runs neb.x on this node, so it needs the local device mask and threads.
        local = cfg.escalation.dft_neb is not None and not cfg.escalation.dft_neb.launcher
        env = placement.pool("dft").launches[0].environment(local=local)
    for pathway in pathways:
        outcome = pathway.outcome
        pathway_record, dft_neb_dispatched = _apply_neb_escalation(
            cfg.escalation,
            outcome,
            it_dir,
            pathway.directory,
            state,
            completed_dft_neb,
            env=env,
        )
        if pathway.parameters.kinetics is not None:
            kinetics_result = None
            if pathway_record["decision"] == "accepted_mlip":
                kinetics_result = outcome.result
            elif pathway_record["decision"] == "dft_neb_complete":
                kinetics_result = NebResult.model_validate(pathway_record["dft_result"])
            if kinetics_result is not None:
                kinetics_artifact = _write_neb_kinetics(
                    pathway.parameters,
                    outcome.transition,
                    kinetics_result,
                    pathway.directory / str(pathway_record["transition_id"]),
                )
                pathway_record["kinetics_artifact"] = str(kinetics_artifact)
        if dft_neb_dispatched:
            completed_dft_neb += 1
            state.neb_pathways[-1] = pathway_record
        else:
            state.neb_pathways.append(pathway_record)
    return completed_dft_neb


def _write_neb_kinetics(
    parameters: NebSamplingConfig,
    transition: NebTransition | None,
    result,
    directory: Path,
) -> Path:
    """Attach an accepted barrier to the existing exploratory-kinetics contracts."""
    from matsim_agents.science.kinetics.contracts import NetworkProvenance
    from matsim_agents.science.kinetics.evidence import attach_neb_result
    from matsim_agents.science.kinetics.graph import TransitionNetwork
    from matsim_agents.science.kinetics.rates import event_rate
    from matsim_agents.science.kinetics.transitions import add_candidate_transition
    from matsim_agents.science.kinetics.validation import validate_network

    kinetics = parameters.kinetics
    if kinetics is None or transition is None:
        raise ValueError("NEB kinetics output requires typed transition and rate assumptions")
    network = TransitionNetwork(
        NetworkProvenance(
            network_id=f"al-{transition.transition_id}",
            parent_id=transition.initial.parent_id,
            parent_ordering_id=transition.initial.parent_ordering_id,
            notes=("NEB barrier attached by active learning",),
        )
    )
    events = add_candidate_transition(network, transition)
    declared = transition.calculation
    evidence_transition = transition
    if declared is not None and (
        declared.method_id != result.method_id or declared.n_images != result.n_images
    ):
        # The accepted result came from another calculation than the one declared (an MLIP
        # NEB, or a QE escalation of the same transition); describe the one that ran.
        from matsim_agents.science.neb.contracts import NebCalculationSpec

        evidence_transition = transition.model_copy(
            update={
                "calculation": NebCalculationSpec(
                    n_images=result.n_images,
                    engine=(
                        "mlip" if result.provenance.get("backend") == "mlip" else declared.engine
                    ),
                    method_id=result.method_id,
                    ci_scheme=declared.ci_scheme,
                )
            }
        )
    attach_neb_result(
        network,
        evidence_transition,
        result,
        attempt_frequency_hz=kinetics.attempt_frequency_hz,
        prefactor_note=kinetics.prefactor_note,
    )
    rates = [
        event_rate(event, kinetics.temperature_K, require="exploratory").model_dump(mode="json")
        for event in network.events
    ]
    report = validate_network(network, events[0].source_id)
    target = directory / "kinetics-evidence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "network_provenance": network.provenance.model_dump(mode="json"),
        "states": [state.model_dump(mode="json") for state in network.states],
        "events": [event.model_dump(mode="json") for event in network.events],
        "rates": rates,
        "report": report.model_dump(mode="json"),
        "neb_result": result.model_dump(mode="json"),
        "declared_calculation": (None if declared is None else declared.model_dump(mode="json")),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return target
