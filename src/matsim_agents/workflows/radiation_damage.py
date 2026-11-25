"""End-to-end radiation-damage recovery workflow: alloy -> vacancy hop -> NEB -> kinetics.

The workflow composes the existing building blocks (alloy generation, defect and migration
endpoints, the NEB workflow, the transition network and CT-KMC) behind one checkpointed
state file. It adds no scheduler and no orchestration framework.

Guarantees:

* Everything is a dry run unless ``dispatch=True`` is passed explicitly.
* The state file ``radiation-damage-run.json`` records each stage and the configuration hash;
  ``resume=True`` refuses a different configuration and skips finished stages.
* Kinetics need NEB evidence. Without it the stage is ``blocked``; a missing barrier is never
  replaced by zero or a default.
* A barrier from this workflow is at most ``unqualified``, so kinetics run in the
  ``exploratory`` mode and the result is never labelled qualified.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from matsim_agents.backends.dft.process import run_launcher
from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.execution.ledger import EvidenceLedger
from matsim_agents.science.defects.point_defects import generate_vacancy
from matsim_agents.science.kinetics.contracts import NetworkProvenance
from matsim_agents.science.kinetics.ensemble import EnsembleResult, run_ensemble
from matsim_agents.science.kinetics.evidence import attach_neb_result
from matsim_agents.science.kinetics.graph import TransitionNetwork
from matsim_agents.science.kinetics.transitions import add_candidate_transition, state_from_endpoint
from matsim_agents.science.kinetics.validation import NetworkReport, validate_network
from matsim_agents.science.neb.contracts import NebResult
from matsim_agents.science.neb.endpoints import (
    MigrationPair,
    generate_vacancy_hop,
    vacancy_neighbors,
)
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy
from matsim_agents.workflows.neb import (
    RESULT_FILE,
    NebWorkflowError,
    dispatch_neb_workflow,
    prepare_neb_workflow,
    read_neb_plan,
)

STATE_FILE = "radiation-damage-run.json"
STATE_SCHEMA = "matsim-radiation-damage-run/1"
NEB_DIR = "neb"
STAGES = ("structure", "neb_prepare", "neb_run", "kinetics")

StageStatus = Literal["pending", "done", "blocked", "skipped", "failed"]


class RadiationDamageError(RuntimeError):
    """The run cannot continue; the message says why."""


class RadiationDamageConfig(BaseModel):
    """One vacancy-hop recovery study of a seeded BCC alloy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    counts: Mapping[str, StrictInt]
    repeats: tuple[int, int, int] = (2, 2, 2)
    lattice_parameter: float = Field(default=3.16, gt=0, allow_inf_nan=False)
    seed: StrictInt = Field(ge=0)
    vacancy_site: int = Field(default=0, ge=0)
    mover_index: int = Field(default=0, ge=0)
    qe: QENebConfig
    temperature_K: float = Field(gt=0, allow_inf_nan=False)
    n_trajectories: int = Field(default=200, ge=1)
    horizon_s: float = Field(gt=0, allow_inf_nan=False)
    max_events: int = Field(default=10_000, ge=1)
    attempt_frequency_hz: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    prefactor_note: str | None = None

    def config_hash(self) -> str:
        import hashlib

        text = json.dumps(self.model_dump(mode="json"), sort_keys=True, allow_nan=False)
        return hashlib.sha256(text.encode()).hexdigest()

    def alloy_spec(self) -> AlloySpec:
        return AlloySpec(
            counts=dict(self.counts),
            repeats=self.repeats,
            lattice_parameter=self.lattice_parameter,
            seed=self.seed,
        )


class StageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: StageStatus = "pending"
    detail: str = ""


class RunState(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: str = Field(default=STATE_SCHEMA, alias="schema")
    config_hash: str
    transition_id: str | None = None
    dry_run: bool = True
    stages: tuple[StageRecord, ...]

    def stage(self, name: str) -> StageRecord:
        return next(s for s in self.stages if s.name == name)

    def with_stage(self, name: str, status: StageStatus, detail: str = "") -> RunState:
        updated = tuple(
            StageRecord(name=name, status=status, detail=detail) if s.name == name else s
            for s in self.stages
        )
        return self.model_copy(update={"stages": updated})


class RadiationDamageRun(BaseModel):
    """What a run produced; ``qualification`` is always ``not_qualified``."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    directory: str
    state: RunState
    network_report: NetworkReport | None = None
    ensemble: EnsembleResult | None = None
    qualification: Literal["not_qualified"] = "not_qualified"
    blockers: tuple[str, ...] = ()


def _write_state(root: Path, state: RunState) -> None:
    handle, staging = tempfile.mkstemp(prefix=".rd-run-", dir=root)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(state.model_dump(mode="json", by_alias=True), stream, indent=2)
            stream.write("\n")
        os.replace(staging, root / STATE_FILE)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def read_state(directory: str | os.PathLike) -> RunState:
    """The saved state of a run directory."""
    path = Path(directory) / STATE_FILE
    try:
        return RunState.model_validate_json(path.read_text())
    except FileNotFoundError as exc:
        raise RadiationDamageError(f"{path} does not exist") from exc
    except ValueError as exc:
        raise RadiationDamageError(f"{path} is not a valid run state: {exc}") from exc


def build_pair(config: RadiationDamageConfig) -> MigrationPair:
    """The deterministic vacancy-hop endpoints for ``config``."""
    parent = generate_alloy(config.alloy_spec())
    parent_id = f"alloy-seed{config.seed}"
    try:
        initial, defect = generate_vacancy(parent, parent_id, config.vacancy_site)
        movers = vacancy_neighbors(initial, defect, config.lattice_parameter)
        if config.mover_index >= len(movers):
            raise RadiationDamageError(
                f"mover_index {config.mover_index} but the vacancy has {len(movers)} neighbours"
            )
        return generate_vacancy_hop(
            initial, defect, parent, movers[config.mover_index], config.lattice_parameter
        )
    except (ValueError, IndexError) as exc:
        raise RadiationDamageError(f"cannot build the hop: {exc}") from exc


def _network(config: RadiationDamageConfig, pair: MigrationPair) -> TransitionNetwork:
    provenance = NetworkProvenance(
        network_id=f"rd-{config.config_hash()[:12]}",
        parent_id=pair.transition.initial.parent_id,
        parent_ordering_id=pair.transition.initial.parent_ordering_id,
        notes=("vacancy hop from the radiation-damage workflow",),
    )
    network = TransitionNetwork(provenance)
    add_candidate_transition(network, pair.transition)
    return network


def run_radiation_damage(
    config: RadiationDamageConfig,
    directory: str | os.PathLike,
    *,
    dispatch: bool = False,
    resume: bool = False,
    allow_unrelaxed_endpoints: bool = False,
    launcher=run_launcher,
    ledger: EvidenceLedger | None = None,
) -> RadiationDamageRun:
    """Run, or resume, the staged workflow in ``directory``.

    ``dispatch=False`` (the default) prepares the pathway and QE input and runs nothing.
    """
    root = Path(directory).resolve()
    chash = config.config_hash()
    if (root / STATE_FILE).exists():
        if not resume:
            raise RadiationDamageError(f"{root} already holds a run; pass resume=True")
        state = read_state(root)
        if state.config_hash != chash:
            raise RadiationDamageError("the configuration differs from the saved run")
    else:
        if resume:
            raise RadiationDamageError(f"{root} holds no run to resume")
        root.mkdir(parents=True, exist_ok=True)
        state = RunState(config_hash=chash, stages=tuple(StageRecord(name=n) for n in STAGES))
    state = state.model_copy(update={"dry_run": not dispatch})

    def save(new: RunState) -> RunState:
        _write_state(root, new)
        return new

    pair = build_pair(config)
    state = save(
        state.model_copy(update={"transition_id": pair.transition.transition_id}).with_stage(
            "structure", "done", f"transition {pair.transition.transition_id}"
        )
    )
    neb_dir = root / NEB_DIR

    if state.stage("neb_prepare").status != "done":
        try:
            plan = prepare_neb_workflow(pair, config.qe, neb_dir)
        except NebWorkflowError as exc:
            state = save(state.with_stage("neb_prepare", "failed", str(exc)))
            raise RadiationDamageError(str(exc)) from exc
        status: StageStatus = "done" if plan.status == "planned" else "failed"
        state = save(state.with_stage("neb_prepare", status, "; ".join(plan.reasons)))
        if plan.status != "planned":
            return RadiationDamageRun(
                directory=str(root), state=state, blockers=("NEB pathway rejected", *plan.reasons)
            )
    plan = read_neb_plan(neb_dir)
    if (
        plan.config_hash
        != config.qe.model_copy(update={"workdir": str(neb_dir / "qe")}).config_hash()
    ):
        raise RadiationDamageError("the saved NEB plan does not match the configuration")

    result_path = neb_dir / RESULT_FILE
    if state.stage("neb_run").status != "done" or not result_path.is_file():
        if not dispatch:
            state = save(state.with_stage("neb_run", "skipped", "dry run; neb.x not launched"))
        else:
            try:
                dispatch_neb_workflow(
                    plan,
                    config.qe,
                    allow_unrelaxed_endpoints=allow_unrelaxed_endpoints,
                    launcher=launcher,
                    ledger=ledger,
                )
            except NebWorkflowError as exc:
                state = save(state.with_stage("neb_run", "failed", str(exc)))
                raise RadiationDamageError(str(exc)) from exc
            state = save(state.with_stage("neb_run", "done", RESULT_FILE))

    blockers: list[str] = []
    if not result_path.is_file():
        reason = "no NEB result: barrier evidence is missing, not zero"
        state = save(state.with_stage("kinetics", "blocked", reason))
        return RadiationDamageRun(directory=str(root), state=state, blockers=(reason,))
    if config.attempt_frequency_hz is None:
        reason = "no attempt frequency supplied; rates are not defined"
        state = save(state.with_stage("kinetics", "blocked", reason))
        return RadiationDamageRun(directory=str(root), state=state, blockers=(reason,))

    saved = json.loads(result_path.read_text())
    neb_result = NebResult.model_validate(saved["result"])
    blockers.extend(saved.get("qualification_blockers", ()))
    network = _network(config, pair)
    attach_neb_result(
        network,
        pair.transition,
        neb_result,
        attempt_frequency_hz=config.attempt_frequency_hz,
        prefactor_note=config.prefactor_note or "user-supplied attempt frequency",
    )
    source_id = state_from_endpoint(pair.transition.initial).state_id
    report = validate_network(network, source_id)
    if not report.ready_for_exploratory_kmc:
        reason = "network not ready for exploratory kinetics: " + "; ".join(report.issues)
        state = save(state.with_stage("kinetics", "blocked", reason))
        return RadiationDamageRun(
            directory=str(root),
            state=state,
            network_report=report,
            blockers=(*blockers, reason),
        )
    ensemble = run_ensemble(
        network,
        source_id,
        config.temperature_K,
        n_trajectories=config.n_trajectories,
        seed=config.seed,
        horizon_s=config.horizon_s,
        max_events=config.max_events,
        require="exploratory",
    )
    state = save(state.with_stage("kinetics", "done", "exploratory ensemble"))
    return RadiationDamageRun(
        directory=str(root),
        state=state,
        network_report=report,
        ensemble=ensemble,
        blockers=tuple(blockers),
    )


__all__ = [
    "NEB_DIR",
    "STAGES",
    "STATE_FILE",
    "RadiationDamageConfig",
    "RadiationDamageError",
    "RadiationDamageRun",
    "RunState",
    "StageRecord",
    "build_pair",
    "read_state",
    "run_radiation_damage",
]
