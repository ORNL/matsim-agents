"""Radiation-damage NEB workflow: prepare, optionally dispatch, ingest and record.

Sequence: generate validated images, write and verify the pathway, prepare the first
``neb.x`` stage, record the plan, optionally run the stage, parse the output and append an
evidence record. Everything is a dry run unless ``dispatch=True`` is passed explicitly.

This module composes the existing B1/B2 building blocks and adds no orchestration framework.
It never reports a ``qualified`` barrier: converged output is at most ``converged_unqualified``,
and the evidence level stays ``low_fidelity_dft`` because endpoint relaxation, image-count and
supercell sensitivity, magnetic-state sensitivity and reference comparison are not assessed here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.backends.dft.process import run_launcher
from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.backends.dft.qe_neb_parse import parse_neb_output
from matsim_agents.backends.dft.qe_neb_run import RECORD_SUFFIX, NebRunRecord
from matsim_agents.backends.dft.qe_neb_stages import (
    StageEntry,
    StageError,
    load_history,
    prepare_stage,
    run_stage,
)
from matsim_agents.core.contracts import EvidenceLevel
from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.core.hashing import sha256_file as file_sha256
from matsim_agents.execution.ledger import (
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    LedgerQuantity,
    MethodRecord,
    hashes_of,
    software_versions,
)
from matsim_agents.science.neb.contracts import (
    MLIP_ENDPOINT_PREFIX,
    NebCalculationSpec,
    NebResult,
)
from matsim_agents.science.neb.endpoints import MigrationPair
from matsim_agents.science.neb.images import (
    DEFAULT_MAX_STEP,
    DEFAULT_MIN_DISTANCE,
    generate_linear_images,
)
from matsim_agents.science.neb.serialization import METADATA_FILE, read_pathway, write_pathway
from matsim_agents.workflows.neb_mapping import to_neb_result

PLAN_FILE = "neb-workflow.json"
RESULT_FILE = "neb-result.json"
PLAN_SCHEMA = "matsim-neb-workflow/1"
RESULT_SCHEMA = "matsim-neb-result/1"
PATHWAY_DIR = "pathway"
QE_DIR = "qe"

# Always true here: the workflow does not perform these checks, so a barrier is never qualified.
NOT_ASSESSED = (
    "image-count sensitivity not tested",
    "supercell-size sensitivity not tested",
    "saddle point not refined beyond the NEB image set",
    "magnetic-state sensitivity not tested",
    "no comparison against a reference calculation",
)

PlanStatus = Literal["planned", "rejected"]


class NebWorkflowError(RuntimeError):
    """The workflow cannot continue; the message says why."""


class NebPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = Field(default=PLAN_SCHEMA, alias="schema")
    status: PlanStatus
    directory: str
    transition_id: str
    method_id: str
    n_images: int
    config_hash: str
    dry_run: bool = True
    pathway_id: str | None = None
    pathway_state: str | None = None
    input_path: str | None = None
    input_sha256: str | None = None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class NebWorkflowOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan: NebPlan
    stage: StageEntry | None = None
    result: NebResult | None = None
    evidence_id: str | None = None
    qualification_blockers: tuple[str, ...] = ()


def _write_json(path: Path, data: Mapping) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_plan(directory: Path) -> NebPlan:
    try:
        return NebPlan.model_validate_json((directory / PLAN_FILE).read_text())
    except FileNotFoundError as exc:
        raise NebWorkflowError(
            f"{directory} has no {PLAN_FILE}; prepare the workflow first"
        ) from exc
    except ValueError as exc:
        raise NebWorkflowError(f"{directory / PLAN_FILE} is not a valid plan: {exc}") from exc


def read_neb_plan(directory: str | os.PathLike) -> NebPlan:
    """The saved plan of a prepared workflow directory."""
    return _read_plan(Path(directory).resolve())


def _method_id(pair: MigrationPair, config: QENebConfig, requested: str | None) -> str:
    endpoint_methods = {e.method for e in (pair.transition.initial, pair.transition.final)}
    endpoint_methods.discard(None)
    if len(endpoint_methods) > 1:
        raise NebWorkflowError(f"endpoints use different methods: {sorted(endpoint_methods)}")
    declared = next(iter(endpoint_methods), None)
    if declared is not None and declared.startswith(MLIP_ENDPOINT_PREFIX):
        # Surrogate-relaxed endpoints start a QE NEB; the QE method is not the surrogate's.
        declared = None
    if requested and declared and requested != declared:
        raise NebWorkflowError(f"method_id {requested!r} differs from the endpoints' {declared!r}")
    return requested or declared or f"qe-neb-{config.config_hash()[:12]}"


def prepare_neb_workflow(
    pair: MigrationPair,
    config: QENebConfig,
    directory: str | os.PathLike,
    *,
    method_id: str | None = None,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    max_step: float = DEFAULT_MAX_STEP,
    images: Sequence[Atoms] | None = None,
) -> NebPlan:
    """Validate a migration pair and prepare (but never run) the first NEB stage.

    An invalid pathway is recorded as ``rejected`` with its reasons and writes no QE input.
    Raises ``NebWorkflowError`` for a misconfigured request or an existing workflow directory.

    ``images`` is an already-built starting path (both endpoints included, e.g. one an MLIP
    NEB optimized). It is validated like a generated path instead of being regenerated
    as a linear guess.
    """
    root = Path(directory).resolve()
    if (root / PLAN_FILE).exists():
        raise NebWorkflowError(f"{root} already holds a workflow; use a new directory")
    try:
        config.check_structure(pair.initial)
    except ValueError as exc:
        raise NebWorkflowError(str(exc)) from exc
    method = _method_id(pair, config, method_id)
    stage_config = QENebConfig.model_validate(
        {**config.model_dump(mode="json"), "workdir": str(root / QE_DIR)}
    )
    root.mkdir(parents=True, exist_ok=True)
    base = {
        "directory": str(root),
        "transition_id": pair.transition.transition_id,
        "method_id": method,
        "n_images": config.n_images,
        "config_hash": stage_config.config_hash(),
    }

    def reject(reasons: list[str], **extra) -> NebPlan:
        plan = NebPlan(status="rejected", reasons=tuple(reasons), **base, **extra)
        _write_json(root / PLAN_FILE, plan.model_dump(mode="json", by_alias=True))
        return plan

    if images is None:
        try:
            generated = generate_linear_images(
                pair.initial,
                pair.final,
                config.n_images,
                min_distance=min_distance,
                max_step=max_step,
            )
        except ValueError as exc:
            return reject([f"images: {exc}"])
        path_images = generated.images
        path_parameters = dict(generated.parameters)
    else:
        path_images = tuple(image.copy() for image in images)
        if len(path_images) != config.n_images:
            return reject([f"images: {len(path_images)} supplied, config has {config.n_images}"])
        if any(len(image.constraints) for image in path_images):
            # The QE positions card carries no fixed-coordinate flags.
            return reject(["images: constrained images are not supported by the QE NEB input"])
        path_parameters = {
            "method": "supplied",
            "n_images": len(path_images),
            "min_distance": min_distance,
            "max_step": max_step,
        }

    calculation = NebCalculationSpec(
        n_images=config.n_images,
        engine="quantum_espresso",
        method_id=method,
        ci_scheme=config.ci_scheme,
    )
    transition = pair.transition.model_copy(update={"calculation": calculation})
    pathway_dir = root / PATHWAY_DIR
    pathway_id = write_pathway(
        pathway_dir,
        transition,
        pair.initial,
        pair.final,
        path_images,
        parameters=path_parameters,
        min_distance=min_distance,
        max_step=max_step,
    )
    assessment = read_pathway(pathway_dir).assessment
    pathway = {"pathway_id": pathway_id, "pathway_state": assessment.state}
    if assessment.state == "rejected":
        return reject(list(assessment.reasons), **pathway)

    warnings = list(assessment.reasons)
    try:
        entry = prepare_stage(stage_config, path_images, kind="initial")
    except (StageError, ValueError) as exc:
        return reject([f"qe input: {exc}"], **pathway)
    plan = NebPlan(
        status="planned",
        input_path=entry.input_path,
        input_sha256=entry.input_sha256,
        warnings=tuple(warnings),
        **base,
        **pathway,
    )
    _write_json(root / PLAN_FILE, plan.model_dump(mode="json", by_alias=True))
    return plan


def _evidence(
    root: Path,
    plan: NebPlan,
    stage: StageEntry,
    result: NebResult,
    transition,
    log: Path,
    blockers: tuple[str, ...],
) -> EvidenceRecord:
    config = QENebConfig.model_validate(stage.config)
    measured = result.state == "converged_unqualified"
    if measured:
        outcome, reason = EvidenceOutcome.MEASURED, None
    elif result.state == "unconverged":
        outcome, reason = EvidenceOutcome.NOT_CONVERGED, "NEB path did not converge"
    else:
        outcome, reason = EvidenceOutcome.FAILED_TO_COMPUTE, f"NEB run {result.state}"
    if reason and result.warnings:
        reason += "; " + "; ".join(result.warnings)
    values = []
    if measured:
        for name, value in (
            ("forward_barrier", result.forward_barrier_eV),
            ("reverse_barrier", result.reverse_barrier_eV),
        ):
            if value is not None:
                values.append(
                    LedgerQuantity(name=name, value=value, unit="eV", structure=plan.transition_id)
                )
    pathway_dir = root / PATHWAY_DIR
    inputs = hashes_of(
        [
            pathway_dir / METADATA_FILE,
            pathway_dir / "initial.extxyz",
            pathway_dir / "final.extxyz",
            pathway_dir / "images.extxyz",
            stage.input_path,
        ]
    )
    outputs = hashes_of(
        [log, Path(stage.config["workdir"]) / f"{config.prefix}.dat", stage.record_path]
    )
    return EvidenceRecord(
        kind="migration_barrier",
        subject=plan.transition_id,
        calculation_id=f"{plan.pathway_id}:stage{stage.index}",
        run_id=str(root),
        method=MethodRecord(
            backend="quantum_espresso_neb",
            method_signature=plan.method_id,
            settings={
                "config_hash": stage.config_hash,
                "ecutwfc_ry": config.ecutwfc_ry,
                "ecutrho_ry": config.ecutrho_ry,
                "kpts": list(config.kpts),
                "n_images": config.n_images,
                "ci_scheme": config.ci_scheme,
                "pseudopotentials": dict(config.pseudopotentials),
            },
            software_versions=software_versions("ase"),
        ),
        evidence_level=EvidenceLevel.LOW_FIDELITY_DFT,
        outcome=outcome,
        values=values,
        numerical_convergence={
            "parser_outcome": result.provenance.get("parser_outcome", ""),
            "iterations_completed": stage.iterations_completed,
            "max_interior_error_eV_per_A": stage.max_interior_error_eV_per_A,
            "path_thr_ev_ang": config.path_thr_ev_ang,
            "climbing_image_used": result.climbing_image_used,
        },
        input_sha256=inputs,
        output_artifact_sha256=outputs,
        physical_assumptions=[
            f"endpoints {transition.initial.status}/{transition.final.status}",
            *blockers,
        ],
        failure_reason=reason,
    )


def ingest_neb_results(
    directory: str | os.PathLike,
    *,
    ledger: EvidenceLedger | None = None,
) -> NebWorkflowOutcome:
    """Parse the latest run stage, write ``neb-result.json`` and optionally record evidence.

    The pathway is re-read with its hashes verified. The result is at most
    ``converged_unqualified``; ``qualification_blockers`` lists what stands in the way.
    """
    root = Path(directory).resolve()
    plan = _read_plan(root)
    if plan.status != "planned":
        raise NebWorkflowError("the workflow was rejected; there is nothing to ingest")
    try:
        pathway = read_pathway(root / PATHWAY_DIR)
        history = load_history(root / QE_DIR)
    except (ValueError, StageError) as exc:
        raise NebWorkflowError(str(exc)) from exc
    stage = history.stages[-1] if history.stages else None
    if stage is None or stage.status != "run":
        raise NebWorkflowError("no stage has been run; nothing to ingest")
    if pathway.pathway_id != plan.pathway_id:
        raise NebWorkflowError("the pathway on disk is not the one that was planned")
    config = QENebConfig.model_validate(stage.config)
    record = (
        NebRunRecord.model_validate_json(Path(stage.record_path).read_text())
        if stage.record_path and Path(stage.record_path).is_file()
        else None
    )
    log = Path(stage.record_path.removesuffix(RECORD_SUFFIX)) if stage.record_path else None
    if log is None or not log.is_file():
        raise NebWorkflowError("the stage log is missing")
    parse = parse_neb_output(
        log,
        record=record,
        expected_images=config.n_images,
        path_thr_ev_ang=config.path_thr_ev_ang,
        dat_path=Path(stage.config["workdir"]) / f"{config.prefix}.dat",
    )
    qualified_endpoints = (
        pathway.assessment.state == "numerically_qualified"
        and pathway.transition.initial.status == "numerically_qualified"
        and pathway.transition.final.status == "numerically_qualified"
    )
    result = to_neb_result(
        parse,
        method_id=plan.method_id,
        n_images=config.n_images,
        endpoints_qualified=qualified_endpoints,
        provenance={
            "pathway_id": pathway.pathway_id,
            "transition_id": pathway.transition.transition_id,
            "config_hash": stage.config_hash,
            "input_sha256": stage.input_sha256,
            "stage_index": str(stage.index),
        },
    )
    blockers: list[str] = []
    if result.state != "converged_unqualified":
        blockers.append(f"NEB not converged (state {result.state})")
    if not qualified_endpoints:
        blockers.append(f"endpoints not numerically qualified ({pathway.assessment.state})")
    blockers.extend(NOT_ASSESSED)

    evidence_id = None
    if ledger is not None:
        record_ = _evidence(root, plan, stage, result, pathway.transition, log, tuple(blockers))
        evidence_id = ledger.append(record_).evidence_id
    _write_json(
        root / RESULT_FILE,
        {
            "schema": RESULT_SCHEMA,
            "transition_id": plan.transition_id,
            "pathway_id": plan.pathway_id,
            "stage_index": stage.index,
            "result": result.model_dump(mode="json"),
            "evidence_id": evidence_id,
            "qualification_blockers": blockers,
            "log_sha256": file_sha256(log),
        },
    )
    return NebWorkflowOutcome(
        plan=plan,
        stage=stage,
        result=result,
        evidence_id=evidence_id,
        qualification_blockers=tuple(blockers),
    )


def run_neb_workflow(
    pair: MigrationPair,
    config: QENebConfig,
    directory: str | os.PathLike,
    *,
    dispatch: bool = False,
    allow_unrelaxed_endpoints: bool = False,
    launcher=run_launcher,
    timeout_sec: float | None = None,
    env: Mapping[str, str] | None = None,
    ledger: EvidenceLedger | None = None,
    method_id: str | None = None,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    max_step: float = DEFAULT_MAX_STEP,
    images: Sequence[Atoms] | None = None,
) -> NebWorkflowOutcome:
    """Prepare the workflow and, only with ``dispatch=True``, run and ingest it.

    A dry run (the default) writes the pathway, the QE input and the plan, and runs nothing.
    Dispatch refuses endpoints that are not relaxed unless ``allow_unrelaxed_endpoints`` is set,
    in which case the result carries that as a qualification blocker.
    """
    plan = prepare_neb_workflow(
        pair,
        config,
        directory,
        method_id=method_id,
        min_distance=min_distance,
        max_step=max_step,
        images=images,
    )
    if plan.status != "planned" or not dispatch:
        return NebWorkflowOutcome(plan=plan)
    return dispatch_neb_workflow(
        plan,
        config,
        allow_unrelaxed_endpoints=allow_unrelaxed_endpoints,
        launcher=launcher,
        timeout_sec=timeout_sec,
        env=env,
        ledger=ledger,
    )


def dispatch_neb_workflow(
    plan: NebPlan,
    config: QENebConfig,
    *,
    allow_unrelaxed_endpoints: bool = False,
    launcher=run_launcher,
    timeout_sec: float | None = None,
    env: Mapping[str, str] | None = None,
    ledger: EvidenceLedger | None = None,
) -> NebWorkflowOutcome:
    """Run the first prepared stage of a ``planned`` workflow and ingest the output."""
    if plan.status != "planned":
        raise NebWorkflowError("the workflow was rejected; there is nothing to dispatch")
    stage_config = QENebConfig.model_validate(
        {**config.model_dump(mode="json"), "workdir": str(Path(plan.directory) / QE_DIR)}
    )
    if stage_config.config_hash() != plan.config_hash:
        raise NebWorkflowError("the configuration differs from the prepared plan")
    problems = stage_config.runtime_problems()
    if problems:
        raise NebWorkflowError("cannot dispatch: " + "; ".join(problems))
    if plan.pathway_state == "endpoint_relaxation_pending" and not allow_unrelaxed_endpoints:
        raise NebWorkflowError(
            "cannot dispatch: endpoints are generated, not relaxed; relax them or pass "
            "allow_unrelaxed_endpoints"
        )
    run_stage(Path(plan.directory) / QE_DIR, timeout_sec=timeout_sec, env=env, launcher=launcher)
    outcome = ingest_neb_results(plan.directory, ledger=ledger)
    return outcome.model_copy(update={"plan": plan.model_copy(update={"dry_run": False})})


__all__ = [
    "NOT_ASSESSED",
    "PLAN_FILE",
    "RESULT_FILE",
    "NebPlan",
    "NebWorkflowError",
    "NebWorkflowOutcome",
    "dispatch_neb_workflow",
    "ingest_neb_results",
    "prepare_neb_workflow",
    "read_neb_plan",
    "run_neb_workflow",
]
