"""Evidence-gated campaign stages (0-5) and the next-action decision.

A stage passes only on measured evidence (coverage, convergence, transfer verdicts, held-out
accuracy, defect/magnetic validation). Calculation counts never open a gate. Missing evidence is
reported as ``NOT_YET_ESTABLISHED`` and keeps the gate closed.

Stages: 0 reuse and inventory (no new DFT) | 1 lower-component foundations | 2 sparse
higher-component bridges | 3 six-component validation (reserve kept) | 4 defect-focused |
5 NEB.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.science.acquisition.acquisition import NOT_YET_ESTABLISHED

Decision = Literal[
    "reuse_inventory",
    "continue_lower",
    "add_bridges",
    "improve_magnetic",
    "add_defects",
    "promote_neb",
    "stop",
]
GateStatus = Literal["passed", "failed", "NOT_YET_ESTABLISHED"]
Validation = Literal["validated", "unsupported", "pending"]
STAGE_NAMES = {
    0: "reuse_and_inventory",
    1: "lower_component_foundations",
    2: "higher_component_bridges",
    3: "six_component_validation",
    4: "defect_focused",
    5: "neb",
}


class StageThresholds(BaseModel):
    """All gate thresholds are explicit configuration; none is a hidden constant."""

    model_config = ConfigDict(extra="forbid")

    min_pair_coverage: float = Field(0.8, ge=0.0, le=1.0)
    min_higher_coverage: float = Field(0.5, ge=0.0, le=1.0)
    min_converged_fraction: float = Field(0.9, ge=0.0, le=1.0)
    min_validated_transfers: int = Field(1, ge=1)
    max_heldout_mae: float | None = Field(None, gt=0.0)
    min_budget_fraction: float = Field(0.0, ge=0.0, le=1.0)


class CampaignEvidence(BaseModel):
    """Measured state of the campaign. ``None`` means not measured."""

    model_config = ConfigDict(extra="forbid")

    inventory_complete: bool = False
    compatibility_checked: bool = False
    n_reused_records: int = Field(0, ge=0)
    n_new_calculations: int = Field(0, ge=0)
    coverage_by_order: dict[int, float] = Field(default_factory=dict)
    converged_fraction: float | None = Field(None, ge=0.0, le=1.0)
    transfer_status: dict[str, str] = Field(default_factory=dict)
    heldout_mae: float | None = Field(None, ge=0.0)
    validation_reserve_intact: bool = True
    magnetic_status: Validation = "pending"
    defect_status: Validation = "pending"
    neb_endpoints_ready: bool = False
    neb_status: Validation = "pending"
    budget_remaining: float = Field(0.0, ge=0.0)
    budget_total: float = Field(1.0, gt=0.0)


class GateResult(BaseModel):
    stage: int
    name: str
    status: GateStatus
    detail: str


class StageReport(BaseModel):
    stage: int
    stage_name: str
    passed: bool
    gates: list[GateResult]


class CampaignPlan(BaseModel):
    completed_stages: list[int]
    current_stage: int | None
    decision: Decision
    reason: str
    stages: list[StageReport]
    gate_summary: dict[str, str]


def _gate(stage: int, name: str, ok: bool | None, detail: str) -> GateResult:
    status: GateStatus = NOT_YET_ESTABLISHED if ok is None else ("passed" if ok else "failed")  # type: ignore[assignment]
    return GateResult(stage=stage, name=name, status=status, detail=detail)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _lower_pair(ev: CampaignEvidence) -> float | None:
    return ev.coverage_by_order.get(2)


def _higher(ev: CampaignEvidence) -> float | None:
    return _mean([v for k, v in ev.coverage_by_order.items() if k >= 3])


def stage_gates(stage: int, ev: CampaignEvidence, thr: StageThresholds) -> list[GateResult]:
    if stage == 0:
        return [
            _gate(0, "inventory_complete", ev.inventory_complete, "existing data inventoried"),
            _gate(
                0,
                "compatibility_checked",
                ev.compatibility_checked,
                "functional, pseudopotential and magnetic conventions compared",
            ),
        ]
    if stage == 1:
        pair = _lower_pair(ev)
        conv = ev.converged_fraction
        return [
            _gate(
                1,
                "pair_coverage",
                None if pair is None else pair >= thr.min_pair_coverage,
                f"pair coverage {pair} vs {thr.min_pair_coverage}",
            ),
            _gate(
                1,
                "converged_fraction",
                None if conv is None else conv >= thr.min_converged_fraction,
                f"converged {conv} vs {thr.min_converged_fraction}",
            ),
        ]
    if stage == 2:
        higher = _higher(ev)
        validated = sum(1 for s in ev.transfer_status.values() if s == "validated")
        return [
            _gate(
                2,
                "higher_order_coverage",
                None if higher is None else higher >= thr.min_higher_coverage,
                f"higher-order coverage {higher} vs {thr.min_higher_coverage}",
            ),
            _gate(
                2,
                "validated_transfer",
                validated >= thr.min_validated_transfers if ev.transfer_status else None,
                f"{validated} validated transfer verdict(s), need {thr.min_validated_transfers}",
            ),
        ]
    if stage == 3:
        mae = ev.heldout_mae
        mae_ok = (
            None if mae is None else (thr.max_heldout_mae is None or mae <= thr.max_heldout_mae)
        )
        return [
            _gate(3, "heldout_accuracy", mae_ok, f"held-out MAE {mae} vs {thr.max_heldout_mae}"),
            _gate(
                3, "reserve_intact", ev.validation_reserve_intact, "validation reserve not drawn"
            ),
        ]
    if stage == 4:
        return [
            _gate(
                4,
                "defect_transfer",
                None if ev.defect_status == "pending" else ev.defect_status == "validated",
                f"defect transferability {ev.defect_status}",
            ),
            _gate(
                4,
                "magnetic_transfer",
                None if ev.magnetic_status == "pending" else ev.magnetic_status == "validated",
                f"magnetic transferability {ev.magnetic_status}",
            ),
        ]
    if stage == 5:
        return [
            _gate(5, "neb_endpoints_ready", ev.neb_endpoints_ready, "relaxed endpoints exist"),
            _gate(
                5,
                "neb_validated",
                None if ev.neb_status == "pending" else ev.neb_status == "validated",
                f"NEB barriers {ev.neb_status}",
            ),
        ]
    raise ValueError(f"unknown stage {stage}")


def evaluate_stages(ev: CampaignEvidence, thr: StageThresholds | None = None) -> list[StageReport]:
    thr = thr or StageThresholds()
    reports = []
    for stage in sorted(STAGE_NAMES):
        gates = stage_gates(stage, ev, thr)
        reports.append(
            StageReport(
                stage=stage,
                stage_name=STAGE_NAMES[stage],
                passed=all(g.status == "passed" for g in gates),
                gates=gates,
            )
        )
    return reports


def decide(ev: CampaignEvidence, thr: StageThresholds | None = None) -> CampaignPlan:
    """Choose the next action from evidence; stages are passed strictly in order."""
    thr = thr or StageThresholds()
    reports = evaluate_stages(ev, thr)
    done: list[int] = []
    for r in reports:
        if not r.passed:
            break
        done.append(r.stage)
    current = None if len(done) == len(reports) else len(done)
    summary = {f"{g.stage}:{g.name}": g.status for r in reports for g in r.gates}

    def plan(decision: Decision, reason: str) -> CampaignPlan:
        return CampaignPlan(
            completed_stages=done,
            current_stage=current,
            decision=decision,
            reason=reason,
            stages=reports,
            gate_summary=summary,
        )

    if current is None:
        return plan("stop", "all stage gates passed on evidence")
    if current == 0:
        return plan("reuse_inventory", "stage 0: inventory and compatibility before any new DFT")
    if ev.budget_remaining / ev.budget_total <= thr.min_budget_fraction:
        return plan("stop", "budget exhausted; no further dispatch without re-authorization")
    reasons = {
        1: ("continue_lower", "lower-component foundations lack coverage or convergence"),
        2: ("add_bridges", "higher-order interactions or transfer evidence are missing"),
        3: ("add_bridges", "six-component validation lacks accuracy evidence"),
    }
    if current in reasons:
        decision, reason = reasons[current]
        return plan(decision, reason)  # type: ignore[arg-type]
    if current == 4:
        if ev.magnetic_status != "validated":
            return plan("improve_magnetic", "magnetic transferability is not validated")
        return plan("add_defects", "defect transferability is not validated")
    if ev.neb_endpoints_ready and ev.neb_status == "pending":
        return plan("promote_neb", "defect gates passed and endpoints are ready for NEB")
    return plan("add_defects", "NEB needs relaxed defect endpoints first")
