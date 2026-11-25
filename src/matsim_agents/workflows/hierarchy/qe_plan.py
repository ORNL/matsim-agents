"""Full-size QE calculation records, old/new compatibility checks and the dispatch gate.

Every new production calculation is the full parent cell of N sites: pristine N atoms, vacancy N-1,
interstitial N+1, Frenkel pair N, divacancy N-2. A cell is never shrunk to fit resources;
an unaffordable job is ``RESOURCE_BLOCKED`` and its full-size input is kept. Historical smaller
cells are prior evidence only, recorded with their size and method. Mixing functionals,
pseudopotentials or magnetic conventions is never silent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from matsim_agents.execution.budget import BudgetLedger

DEFECT_ATOM_DELTA = {
    "pristine": 0,
    "vacancy": -1,
    "interstitial": 1,
    "frenkel_pair": 0,
    "divacancy": -2,
}
DefectKind = Literal["pristine", "vacancy", "interstitial", "frenkel_pair", "divacancy"]
PlanStatus = Literal["READY", "RESOURCE_BLOCKED", "BUDGET_BLOCKED", "INPUT_BLOCKED"]
HARD_MISMATCH_FIELDS = ("functional", "pseudopotentials", "magnetic_convention")
SOFT_MISMATCH_FIELDS = ("ecutwfc_ry", "ecutrho_ry", "smearing", "degauss_ry", "kpts")


def expected_atoms(kind: str, parent_atoms: int) -> int:
    return parent_atoms + DEFECT_ATOM_DELTA[kind]


def hash_input(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class MethodSignature(BaseModel):
    """Everything that makes two DFT results comparable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    functional: str
    pseudopotentials: dict[str, str]
    ecutwfc_ry: float
    ecutrho_ry: float
    smearing: str
    degauss_ry: float
    kpts: tuple[int, int, int]
    magnetic_convention: str


class CompatibilityReport(BaseModel):
    compatible: bool
    hard_mismatches: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def check_compatibility(new: MethodSignature, prior: MethodSignature) -> CompatibilityReport:
    """Hard mismatch (functional, pseudopotentials, magnetic convention) forbids mixing."""
    hard, soft = [], []
    for name in HARD_MISMATCH_FIELDS:
        a, b = getattr(new, name), getattr(prior, name)
        if a != b:
            hard.append(f"{name}: new={a!r} prior={b!r}")
    for name in SOFT_MISMATCH_FIELDS:
        a, b = getattr(new, name), getattr(prior, name)
        if a != b:
            soft.append(f"{name}: new={a!r} prior={b!r}")
    return CompatibilityReport(compatible=not hard, hard_mismatches=hard, warnings=soft)


class PriorEvidence(BaseModel):
    """A historical (possibly smaller-cell) result: evidence for planning, never a substitute."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    n_atoms: StrictInt = Field(gt=0)
    method: MethodSignature
    role: Literal["prior_evidence"] = "prior_evidence"


class ResourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: StrictInt = Field(gt=0)
    walltime_hours: float = Field(gt=0.0)
    memory_gb_per_node: float = Field(gt=0.0)
    gpus_per_node: StrictInt = Field(8, ge=0)

    @property
    def node_hours(self) -> float:
        return self.nodes * self.walltime_hours


class ResourceLimits(BaseModel):
    """What the target queue can actually provide for one job."""

    model_config = ConfigDict(extra="forbid")

    max_nodes: StrictInt = Field(gt=0)
    max_walltime_hours: float = Field(gt=0.0)
    node_memory_gb: float = Field(gt=0.0)
    gpus_per_node: StrictInt = Field(8, ge=0)
    available: bool = True


class CalculationSpec(BaseModel):
    """Complete record of one proposed production calculation."""

    model_config = ConfigDict(extra="forbid")

    calc_id: str
    kind: DefectKind
    category: str
    composition: dict[str, StrictInt]
    supercell: tuple[StrictInt, StrictInt, StrictInt]
    parent_atoms: StrictInt = Field(gt=0)
    n_atoms: StrictInt = Field(gt=0)
    method: MethodSignature
    magnetic_treatment: str
    conv_thr_ry: float = Field(gt=0.0)
    input_hash: str | None = None
    input_path: str | None = None
    resources: ResourceRequest
    expected_information_gain: float = Field(ge=0.0)
    prior_evidence: list[PriorEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _full_size(self) -> CalculationSpec:
        want = expected_atoms(self.kind, self.parent_atoms)
        if self.n_atoms != want:
            raise ValueError(
                f"{self.calc_id}: {self.kind} must have {want} atoms (parent "
                f"{self.parent_atoms}); got {self.n_atoms}. Cells are never shrunk."
            )
        if sum(self.composition.values()) != self.n_atoms:
            raise ValueError(f"{self.calc_id}: composition does not sum to {self.n_atoms} atoms")
        return self

    @property
    def estimated_node_hours(self) -> float:
        return self.resources.node_hours

    def with_input(self, text: str, path: str | None = None) -> CalculationSpec:
        return self.model_copy(update={"input_hash": hash_input(text), "input_path": path})

    def record(self) -> dict[str, Any]:
        out = json.loads(self.model_dump_json())
        out["estimated_node_hours"] = self.estimated_node_hours
        return out


class DispatchDecision(BaseModel):
    calc_id: str
    status: PlanStatus
    reasons: list[str] = Field(default_factory=list)
    compatibility_warnings: list[str] = Field(default_factory=list)
    estimated_node_hours: float
    input_preserved: bool = True


def resource_problems(req: ResourceRequest, limits: ResourceLimits) -> list[str]:
    problems = []
    if not limits.available:
        problems.append("target resource unavailable")
    if req.nodes > limits.max_nodes:
        problems.append(f"needs {req.nodes} nodes, limit {limits.max_nodes}")
    if req.walltime_hours > limits.max_walltime_hours:
        problems.append(
            f"needs {req.walltime_hours} h walltime, limit {limits.max_walltime_hours} h"
        )
    if req.memory_gb_per_node > limits.node_memory_gb:
        problems.append(f"needs {req.memory_gb_per_node} GB/node, limit {limits.node_memory_gb} GB")
    if req.gpus_per_node > limits.gpus_per_node:
        problems.append(f"needs {req.gpus_per_node} GPUs/node, limit {limits.gpus_per_node}")
    return problems


def gate_dispatch(
    spec: CalculationSpec,
    limits: ResourceLimits,
    ledger: BudgetLedger,
    *,
    input_ready: bool = True,
    prior: Iterable[PriorEvidence] = (),
) -> DispatchDecision:
    """Decide READY / RESOURCE_BLOCKED / BUDGET_BLOCKED / INPUT_BLOCKED without shrinking."""
    reasons: list[str] = []
    warnings: list[str] = []
    for ev in [*spec.prior_evidence, *prior]:
        report = check_compatibility(spec.method, ev.method)
        warnings += [f"{ev.dataset} ({ev.n_atoms} atoms): {w}" for w in report.warnings]
        if not report.compatible:
            warnings += [f"{ev.dataset} excluded: {m}" for m in report.hard_mismatches]
    if not input_ready or spec.input_hash is None:
        return DispatchDecision(
            calc_id=spec.calc_id,
            status="INPUT_BLOCKED",
            reasons=["no verified full-size input"],
            compatibility_warnings=warnings,
            estimated_node_hours=spec.estimated_node_hours,
        )
    reasons = resource_problems(spec.resources, limits)
    if reasons:
        return DispatchDecision(
            calc_id=spec.calc_id,
            status="RESOURCE_BLOCKED",
            reasons=reasons,
            compatibility_warnings=warnings,
            estimated_node_hours=spec.estimated_node_hours,
        )
    if not ledger.can_dispatch(spec.category, spec.estimated_node_hours):  # type: ignore[arg-type]
        return DispatchDecision(
            calc_id=spec.calc_id,
            status="BUDGET_BLOCKED",
            reasons=[
                f"{spec.estimated_node_hours:.1f} node-hours exceeds available "
                f"{ledger.available(spec.category):.1f} in {spec.category}"  # type: ignore[arg-type]
            ],
            compatibility_warnings=warnings,
            estimated_node_hours=spec.estimated_node_hours,
        )
    return DispatchDecision(
        calc_id=spec.calc_id,
        status="READY",
        compatibility_warnings=warnings,
        estimated_node_hours=spec.estimated_node_hours,
    )


def apply_gate_to_manifest(
    entries: list[dict[str, Any]],
    request: ResourceRequest,
    limits: ResourceLimits,
) -> list[dict[str, Any]]:
    """Mark prepared manifest entries ``RESOURCE_BLOCKED`` when the queue cannot run them.

    Only the status changes: structures, inputs and hashes stay on disk at full size.
    """
    problems = resource_problems(request, limits)
    out = []
    for entry in entries:
        entry = dict(entry)
        entry["estimated_node_hours"] = request.node_hours
        if problems and entry.get("status") == "PREPARED":
            entry["status"] = "RESOURCE_BLOCKED"
            entry["resource_blockers"] = problems
        out.append(entry)
    return out
