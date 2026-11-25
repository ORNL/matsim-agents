"""Resumable planner for the hierarchical W-Ta-Cr-V-Hf-Mo transfer campaign (M12-M19 glue).

One :class:`CampaignWorkspace` directory holds the state (``state.json``) and the node-hour
ledger (``ledger.json``). Every step is deterministic, writes before it acts, and is safe to
re-run: a calculation that already has a ledger entry or a submission record is never
submitted again. A submission that was started but not confirmed becomes
``needs_manual_check`` and is never resubmitted automatically.

Nothing here claims real-alloy transferability. Verdicts come from held-out experiments on the
data supplied; absent data is ``NOT_YET_ESTABLISHED``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.execution.budget import (
    CATEGORIES,
    BudgetLedger,
    Category,
)
from matsim_agents.science.acquisition.acquisition import (
    NOT_YET_ESTABLISHED,
    CandidateAcquisitionInputs,
    CandidateSelection,
    expected_transfer_gain,
    select_candidate_batch,
)
from matsim_agents.workflows.candidate_registry import (
    CandidateRegistry,
    DatasetEntry,
    calculation_id,
    hierarchy_subspaces,
    ingest_dataset,
)
from matsim_agents.workflows.hierarchy.config import CampaignConfig
from matsim_agents.workflows.hierarchy.coverage import (
    CoverageReport,
    StructureDescriptor,
    cluster_key,
    coverage_report,
    required_interactions,
)
from matsim_agents.workflows.hierarchy.qe_plan import (
    CalculationSpec,
    DefectKind,
    DispatchDecision,
    check_compatibility,
    expected_atoms,
    gate_dispatch,
    hash_input,
)
from matsim_agents.workflows.hierarchy.stages import (
    CampaignEvidence,
    CampaignPlan,
    Validation,
    decide,
)
from matsim_agents.workflows.hierarchy.transfer import (
    TransferData,
    TransferExperiment,
    TransferVerdict,
    run_transfer_experiment,
    standard_experiments,
)

Submitter = Callable[[CalculationSpec], str]
JobStatus = Callable[[str], tuple[str, float | None]]
_TRANSFER_FIELDS_STR = (
    "system",
    "composition_id",
    "ordering_id",
    "environment",
    "sro_regime",
    "magnetic",
)


# --------------------------------------------------------------------------- data files
def save_transfer_data(data: TransferData, path: str | Path) -> None:
    """Write labelled rows as an ``.npz`` with no pickled objects."""
    arrays: dict[str, Any] = {
        "property": np.array(data.property),
        "ids": np.array(data.ids),
        "X": data.X,
        "y": data.y,
        "cost_node_hours": np.asarray(data.cost_node_hours, dtype=float),
    }
    for name in _TRANSFER_FIELDS_STR:
        arrays[name] = np.array(getattr(data, name))
    np.savez(Path(path), **arrays)


def load_transfer_data(path: str | Path) -> TransferData:
    with np.load(Path(path), allow_pickle=False) as z:
        return TransferData(
            property=str(z["property"]),
            ids=tuple(str(v) for v in z["ids"]),
            X=z["X"],
            y=z["y"],
            cost_node_hours=z["cost_node_hours"],
            **{n: tuple(str(v) for v in z[n]) for n in _TRANSFER_FIELDS_STR},
        )


def load_dataset_entries(path: str | Path) -> list[DatasetEntry]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of dataset entries")
    return [DatasetEntry(**item) for item in raw]


class CandidateProposal(BaseModel):
    """A proposed full-size production calculation and the evidence about it."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    structure_hash: str
    category: Category
    kind: DefectKind = "pristine"
    composition: dict[str, int]
    interactions: list[str] = Field(default_factory=list)
    region: str = "default"
    relevance: float = Field(0.0, ge=0.0, le=1.0)
    uncertainty: float = Field(0.0, ge=0.0, le=1.0)
    diversity: float = Field(0.0, ge=0.0, le=1.0)
    magnetic_relevance: float = Field(0.0, ge=0.0, le=1.0)
    defect_relevance: float = Field(0.0, ge=0.0, le=1.0)
    connectivity: float = Field(1.0, ge=0.0, le=1.0)
    expected_information_gain: float = Field(0.0, ge=0.0)
    magnetic_initial: dict[str, Any] | None = None
    input_path: str | None = None
    estimated_node_hours: float | None = Field(None, gt=0.0)

    @property
    def order(self) -> int:
        return len(self.composition)


def load_proposals(path: str | Path) -> list[CandidateProposal]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of candidate proposals")
    proposals = [CandidateProposal(**item) for item in raw]
    ids = [p.candidate_id for p in proposals]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: candidate ids must be unique")
    return proposals


# --------------------------------------------------------------------------- state
class SubmissionRecord(BaseModel):
    calc_id: str
    candidate_id: str
    category: str
    order: int
    interactions: list[str] = Field(default_factory=list)
    predicted_node_hours: float
    state: Literal["submitting", "submitted", "finished", "needs_manual_check"]
    job_id: str | None = None
    actual_node_hours: float | None = None
    outcome: str | None = None


class TransferRun(BaseModel):
    property: str
    experiment: str
    target_order: int
    verdicts: list[TransferVerdict]
    heldout_mae: float | None = None
    best_benefit: float | None = None


class CampaignState(BaseModel):
    campaign_id: str
    registry: CandidateRegistry = Field(default_factory=CandidateRegistry)
    dataset_compatibility: dict[str, dict[str, Any]] = Field(default_factory=dict)
    inventory_done: bool = False
    transfer: list[TransferRun] = Field(default_factory=list)
    submissions: dict[str, SubmissionRecord] = Field(default_factory=dict)
    log: list[str] = Field(default_factory=list)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


class CampaignWorkspace:
    """State directory; ``save`` is atomic so an interrupted run never leaves a torn file."""

    def __init__(self, root: str | Path, config: CampaignConfig):
        self.root = Path(root)
        self.config = config
        self.state_path = self.root / "state.json"
        self.ledger_path = self.root / "ledger.json"
        self.state, self.ledger = self._load()

    def _load(self) -> tuple[CampaignState, BudgetLedger]:
        cfg = self.config
        if self.state_path.exists():
            state = CampaignState.model_validate_json(self.state_path.read_text())
            if state.campaign_id != cfg.campaign.campaign_id:
                raise ValueError(
                    f"{self.root} belongs to campaign {state.campaign_id!r}, "
                    f"not {cfg.campaign.campaign_id!r}"
                )
        else:
            state = CampaignState(campaign_id=cfg.campaign.campaign_id)
        if self.ledger_path.exists():
            ledger = BudgetLedger.load(self.ledger_path)
        else:
            ledger = BudgetLedger(
                total_node_hours=cfg.budget.total_node_hours,
                validation_reserve_fraction=cfg.budget.validation_reserve_fraction,
                allocation=dict(cfg.budget.allocation),
            )
        return state, ledger

    def save(self) -> None:
        _atomic_write(self.state_path, self.state.model_dump_json(indent=2))
        self.ledger.save(self.ledger_path)


# --------------------------------------------------------------------------- inventory
def inventory(ws: CampaignWorkspace) -> list[dict[str, Any]]:
    """Ingest every configured dataset once, record method compatibility, never run DFT."""
    cfg, state = ws.config, ws.state
    rows = []
    for src in cfg.data.datasets:
        if src.name in state.registry.datasets:
            record = state.registry.datasets[src.name]
        else:
            record = ingest_dataset(
                state.registry,
                src.name,
                load_dataset_entries(src.path),
                target_elements=cfg.elements,
            )
        if src.method is None:
            compat = {"compatible": None, "hard_mismatches": [], "warnings": ["method unknown"]}
        else:
            rep = check_compatibility(cfg.qe.method, src.method)
            compat = {
                "compatible": rep.compatible,
                "hard_mismatches": list(rep.hard_mismatches),
                "warnings": list(rep.warnings),
            }
        state.dataset_compatibility[src.name] = compat
        rows.append({**record.model_dump(), "compatibility": compat})
    state.inventory_done = True
    ws.save()
    return rows


def _usable_datasets(ws: CampaignWorkspace) -> set[str]:
    """Datasets whose method is compatible or unknown-but-not-contradicted; mismatches excluded."""
    return {
        name
        for name, compat in ws.state.dataset_compatibility.items()
        if compat.get("compatible") is not False
    }


# --------------------------------------------------------------------------- coverage
def _descriptor(interactions: Iterable[str], composition: dict[str, int]) -> StructureDescriptor:
    total = sum(composition.values()) or 1
    return StructureDescriptor(
        composition={e: n / total for e, n in composition.items()},
        pair_frequency={},
        warren_cowley={},
        local_composition={},
        motifs={},
        distortion=0.0,
        sro_regime="random",
        interactions=sorted(set(interactions)),
    )


def _record_interactions(record: Any) -> tuple[list[str], str]:
    """Explicit interactions if the dataset gave them, else a composition-level assumption."""
    explicit = record.local_environments.get("interactions") if record.local_environments else None
    if explicit:
        return list(explicit), "explicit"
    return [cluster_key(record.element_set)], "composition_level"


def coverage(ws: CampaignWorkspace) -> dict[str, Any]:
    """Interaction coverage of the reused data plus converged new calculations."""
    usable = _usable_datasets(ws)
    descriptors: list[StructureDescriptor] = []
    bases: set[str] = set()
    for record in ws.state.registry.candidates.values():
        if record.source_dataset not in usable:
            continue
        keys, basis = _record_interactions(record)
        bases.add(basis)
        descriptors.append(_descriptor(keys, record.composition))
    for sub in ws.state.submissions.values():
        if sub.state == "finished" and sub.outcome == "converged":
            descriptors.append(_descriptor(sub.interactions, {}))
            bases.add("explicit")
    report: CoverageReport = coverage_report(descriptors, required_interactions(ws.config.elements))
    return {
        "basis": sorted(bases) or ["no_data"],
        "report": report,
        "subspace_records": ws.state.registry.target_coverage(
            hierarchy_subspaces(ws.config.elements)
        ),
    }


# --------------------------------------------------------------------------- transferability
def _experiments(cfg: CampaignConfig) -> list[TransferExperiment]:
    te = cfg.transfer_evaluation
    wanted = set(te.experiments)
    return [
        e.model_copy(
            update={
                "n_new": te.n_new,
                "seeds": te.seeds,
                "min_seeds": te.min_seeds,
                "min_eval_rows": te.min_eval_rows,
            }
        )
        for e in standard_experiments()
        if e.name in wanted
    ]


def transferability(
    ws: CampaignWorkspace, data: dict[str, TransferData] | None = None
) -> list[TransferRun]:
    """Held-out verdicts per property and experiment; no data means NOT_YET_ESTABLISHED."""
    cfg = ws.config
    te = cfg.transfer_evaluation
    if data is None:
        data = {p: load_transfer_data(path) for p, path in cfg.data.transfer_data.items()}
    properties = list(
        dict.fromkeys([*te.required_properties, te.magnetic_property, te.defect_property])
    )
    runs: list[TransferRun] = []
    for exp in _experiments(cfg):
        for prop in properties:
            if prop not in data:
                verdict = TransferVerdict(
                    property=prop,
                    experiment=exp.name,
                    n_new=0,
                    status="insufficient_evidence",
                    reason="no labelled data for this property",
                )
                runs.append(
                    TransferRun(
                        property=prop,
                        experiment=exp.name,
                        target_order=exp.target_order,
                        verdicts=[verdict],
                    )
                )
                continue
            result = run_transfer_experiment(data[prop], exp)
            top = max(exp.n_new)
            rows = [s for s in result.summary if s.strategy == "D" and s.n_new == top]
            benefits = [
                v.benefit_vs_target_only.mean
                for v in result.verdicts
                if v.status == "useful_transfer" and v.benefit_vs_target_only is not None
            ]
            runs.append(
                TransferRun(
                    property=prop,
                    experiment=exp.name,
                    target_order=exp.target_order,
                    verdicts=result.verdicts,
                    heldout_mae=rows[0].mae_mean if rows else None,
                    best_benefit=max(benefits) if benefits else None,
                )
            )
    ws.state.transfer = runs
    ws.save()
    return runs


def aggregate_status(runs: Iterable[TransferRun], prop: str) -> Validation:
    """validated only if every measured verdict is validated; any negative result is unsupported."""
    statuses = [v.registry_status for r in runs if r.property == prop for v in r.verdicts]
    measured = [s for s in statuses if s != NOT_YET_ESTABLISHED]
    if not measured:
        return "pending"
    if any(s == "unsupported" for s in measured):
        return "unsupported"
    return "validated"


def _gain_for_order(runs: list[TransferRun], prop: str, order: int) -> float | None:
    matching = [r for r in runs if r.property == prop and r.target_order == order]
    if not matching:
        return None
    scale = max((r.best_benefit or 0.0 for r in runs if r.property == prop), default=0.0)
    gains = []
    for run in matching:
        for v in run.verdicts:
            benefit = v.benefit_vs_target_only.mean if v.benefit_vs_target_only else None
            gains.append(expected_transfer_gain(v.registry_status, benefit, scale or 1.0))
    known = [g for g in gains if g is not None]
    return min(known) if known else None


# --------------------------------------------------------------------------- planning
class PlannedCalculation(BaseModel):
    candidate_id: str
    calc_id: str
    category: str
    order: int
    predicted_node_hours: float
    selected: bool
    reason: str
    status: str | None = None
    status_reasons: list[str] = Field(default_factory=list)


class DispatchPlan(BaseModel):
    acquisition: str
    selected: list[str]
    expected_node_hours: float
    ready_node_hours: float
    calculations: list[PlannedCalculation]
    missing_interactions_targeted: list[str]


def _calc_id(cfg: CampaignConfig, p: CandidateProposal) -> str:
    return calculation_id(
        p.structure_hash,
        "quantum_espresso",
        cfg.qe.method.model_dump(mode="json"),
        p.magnetic_initial,
    )


def build_spec(cfg: CampaignConfig, p: CandidateProposal, calc_id: str) -> CalculationSpec:
    resources = cfg.qe.resources
    if p.estimated_node_hours is not None:
        resources = resources.model_copy(
            update={"walltime_hours": p.estimated_node_hours / resources.nodes}
        )
    spec = CalculationSpec(
        calc_id=calc_id,
        kind=p.kind,
        category=p.category,
        composition=p.composition,
        supercell=cfg.target.supercell,
        parent_atoms=cfg.target.n_sites,
        n_atoms=sum(p.composition.values()),
        method=cfg.qe.method,
        magnetic_treatment=cfg.qe.magnetic_treatment,
        conv_thr_ry=cfg.qe.conv_thr_ry,
        resources=resources,
        expected_information_gain=p.expected_information_gain,
    )
    if cfg.qe.input_ready and p.input_path and Path(p.input_path).is_file():
        spec = spec.with_input(Path(p.input_path).read_text(), p.input_path)
    return spec


def _inputs(
    ws: CampaignWorkspace,
    proposals: list[CandidateProposal],
    runs: list[TransferRun],
) -> tuple[list[CandidateAcquisitionInputs], dict[str, CalculationSpec]]:
    cfg = ws.config
    missing = set(coverage(ws)["report"].missing)
    reserved = set(cfg.data.reserved_validation_ids)
    prop = cfg.transfer_evaluation.required_properties[0]
    items, specs = [], {}
    for p in proposals:
        calc_id = _calc_id(cfg, p)
        compatible, why, cost = True, "", 1.0
        try:
            spec = build_spec(cfg, p, calc_id)
            specs[p.candidate_id] = spec
            cost = max(spec.estimated_node_hours, 1e-9)
        except ValueError as exc:
            compatible, why = False, f"invalid full-size record: {exc}"
        if p.candidate_id in reserved or p.structure_hash in reserved or calc_id in reserved:
            compatible, why = False, "reserved validation example"
        items.append(
            CandidateAcquisitionInputs(
                candidate_id=p.candidate_id,
                calculation_id=calc_id,
                region=p.region,
                relevance=p.relevance,
                newly_covered=frozenset(p.interactions) & missing,
                uncertainty=p.uncertainty,
                transfer_gain=_gain_for_order(runs, prop, p.order),
                diversity=p.diversity,
                magnetic_relevance=p.magnetic_relevance,
                defect_relevance=p.defect_relevance,
                connectivity=p.connectivity,
                compatible=compatible,
                incompatibility_reason=why,
                cost_node_hours=cost,
            )
        )
    return items, specs


def baseline_selection(
    items: list[CandidateAcquisitionInputs],
    *,
    batch_size: int,
    max_batch_node_hours: float | None,
    existing: Iterable[str],
) -> CandidateSelection:
    """Deterministic baseline: cheapest first, then candidate id; no scientific scoring."""
    seen, picked, spent = set(existing), [], 0.0
    for item in sorted(items, key=lambda i: (i.cost_node_hours, i.candidate_id)):
        if not item.compatible or item.calculation_id in seen:
            continue
        if max_batch_node_hours is not None and spent + item.cost_node_hours > max_batch_node_hours:
            continue
        picked.append(item.candidate_id)
        seen.add(item.calculation_id)
        spent += item.cost_node_hours
        if len(picked) >= batch_size:
            break
    return CandidateSelection(selected=picked, total_node_hours=spent, scores=[])


def _existing_calc_ids(ws: CampaignWorkspace) -> set[str]:
    ids = {e.calc_id for e in ws.ledger.entries}
    ids |= set(ws.state.submissions)
    ids |= set(ws.state.registry.calculations)
    return ids


def plan(
    ws: CampaignWorkspace,
    proposals: list[CandidateProposal],
    *,
    acquisition: str | None = None,
) -> DispatchPlan:
    """Preview what would be dispatched and what it would cost; submits nothing."""
    cfg = ws.config
    acquisition = acquisition or cfg.sampling.acquisition
    items, specs = _inputs(ws, proposals, ws.state.transfer)
    cov = coverage(ws)["report"]
    existing = _existing_calc_ids(ws)
    policy = cfg.sampling.policy
    if acquisition == "policy":
        selection = select_candidate_batch(
            items,
            policy,
            existing_calculation_ids=existing,
            missing_interactions=cov.missing,
        )
    else:
        selection = baseline_selection(
            items,
            batch_size=policy.batch_size,
            max_batch_node_hours=policy.max_batch_node_hours,
            existing=existing,
        )
    by_score = selection.by_id()
    by_prop = {p.candidate_id: p for p in proposals}
    picked = set(selection.selected)
    calcs: list[PlannedCalculation] = []
    ready_hours = 0.0
    # Dispatch decisions are evaluated against a scratch copy so a preview never mutates the ledger.
    scratch = ws.ledger.model_copy(deep=True)
    for item in sorted(items, key=lambda i: i.candidate_id):
        spec = specs.get(item.candidate_id)
        sel = item.candidate_id in picked
        reason = (
            by_score[item.candidate_id].reason
            if item.candidate_id in by_score
            else ("deterministic baseline" if sel else "not selected")
        )
        decision: DispatchDecision | None = None
        if sel and spec is not None:
            decision = gate_dispatch(
                spec,
                cfg.qe.limits,
                scratch,
                input_ready=cfg.qe.input_ready and spec.input_hash is not None,
            )
            if decision.status == "READY":
                scratch.reserve(spec.calc_id, spec.category, spec.estimated_node_hours)
                ready_hours += spec.estimated_node_hours
        calcs.append(
            PlannedCalculation(
                candidate_id=item.candidate_id,
                calc_id=item.calculation_id,
                category=by_prop[item.candidate_id].category,
                order=by_prop[item.candidate_id].order,
                predicted_node_hours=item.cost_node_hours,
                selected=sel,
                reason=reason,
                status=decision.status if decision else None,
                status_reasons=decision.reasons if decision else [],
            )
        )
    targeted = sorted({k for pid in picked for k in by_prop[pid].interactions} & set(cov.missing))
    return DispatchPlan(
        acquisition=acquisition,
        selected=list(selection.selected),
        expected_node_hours=selection.total_node_hours,
        ready_node_hours=ready_hours,
        calculations=calcs,
        missing_interactions_targeted=targeted,
    )


def compare_acquisition(
    ws: CampaignWorkspace, proposals: list[CandidateProposal]
) -> dict[str, Any]:
    """Policy vs deterministic baseline on the same candidates and the same batch limits."""
    cfg = ws.config
    items, _ = _inputs(ws, proposals, ws.state.transfer)
    missing = set(coverage(ws)["report"].missing)
    by_prop = {p.candidate_id: p for p in proposals}
    existing = _existing_calc_ids(ws)
    policy = select_candidate_batch(
        items,
        cfg.sampling.policy,
        existing_calculation_ids=existing,
        missing_interactions=missing,
    )
    base = baseline_selection(
        items,
        batch_size=cfg.sampling.policy.batch_size,
        max_batch_node_hours=cfg.sampling.policy.max_batch_node_hours,
        existing=existing,
    )

    def covered(sel: CandidateSelection) -> int:
        return len({k for c in sel.selected for k in by_prop[c].interactions} & missing)

    return {
        "policy": {
            "covered": covered(policy),
            "node_hours": policy.total_node_hours,
            "selected": list(policy.selected),
        },
        "baseline": {
            "covered": covered(base),
            "node_hours": base.total_node_hours,
            "selected": list(base.selected),
        },
        "policy_beats_baseline": covered(policy) > covered(base),
    }


# --------------------------------------------------------------------------- dispatch
class DispatchOutcome(BaseModel):
    submitted: list[str] = Field(default_factory=list)
    skipped_duplicates: list[str] = Field(default_factory=list)
    blocked: dict[str, str] = Field(default_factory=dict)
    needs_manual_check: list[str] = Field(default_factory=list)
    stopped_because: str | None = None


def _target_label(cfg: CampaignConfig) -> str:
    t = cfg.target
    formula = "".join(f"{el}{n}" for el, n in t.composition.items())
    cell = "x".join(str(r) for r in t.supercell)
    return f"{formula} ({cell} {t.parent_lattice.upper()}, {t.n_sites} sites)"


def sbatch_submitter(cfg: CampaignConfig) -> Submitter:
    """Real submitter: ``sbatch --parsable -q <qos> -A <account>``, no shell parsing."""
    if not cfg.execution.account:
        raise ValueError("execution.account is required to submit with sbatch")

    def submit(spec: CalculationSpec) -> str:
        if spec.input_path is None:
            raise RuntimeError(f"{spec.calc_id}: no input to submit")
        workdir = Path(spec.input_path).parent
        proc = subprocess.run(
            [
                "sbatch",
                "--parsable",
                "-q",
                cfg.execution.qos,
                "-A",
                cfg.execution.account,
                f"--job-name={spec.calc_id[:40]}",
                cfg.execution.job_script,
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip().split(";")[0]

    return submit


def dispatch(
    ws: CampaignWorkspace,
    proposals: list[CandidateProposal],
    submitter: Submitter,
    *,
    approved_by: str | None = None,
) -> DispatchOutcome:
    """Submit a bounded batch; dry-run, a wrong approver or an exhausted allowance stops it."""
    cfg = ws.config
    ex = cfg.execution
    out = DispatchOutcome()
    if ex.dry_run:
        out.stopped_because = "execution.dry_run is true; nothing was submitted"
        return out
    if not approved_by or not approved_by.strip() or approved_by != ex.approved_by:
        out.stopped_because = "approved_by does not match execution.approved_by"
        return out
    for calc_id, sub in ws.state.submissions.items():
        if sub.state == "submitting":
            sub.state = "needs_manual_check"
            ws.state.log.append(f"{calc_id}: unconfirmed submission; not resubmitted")
        if sub.state == "needs_manual_check":
            out.needs_manual_check.append(calc_id)
    budget_left = (ex.approved_node_hours or 0.0) - sum(
        s.predicted_node_hours for s in ws.state.submissions.values()
    )
    preview = plan(ws, proposals)
    by_prop = {p.candidate_id: p for p in proposals}
    n = 0
    for calc in preview.calculations:
        if not calc.selected:
            continue
        if calc.calc_id in ws.state.submissions or any(
            e.calc_id == calc.calc_id for e in ws.ledger.entries
        ):
            out.skipped_duplicates.append(calc.calc_id)
            continue
        if calc.status != "READY":
            out.blocked[calc.calc_id] = f"{calc.status}: {'; '.join(calc.status_reasons)}"
            continue
        if n >= ex.max_jobs_per_dispatch:
            out.stopped_because = f"max_jobs_per_dispatch={ex.max_jobs_per_dispatch} reached"
            break
        if calc.predicted_node_hours > budget_left + 1e-9:
            out.stopped_because = "approved_node_hours exhausted"
            break
        proposal = by_prop[calc.candidate_id]
        spec = build_spec(cfg, proposal, calc.calc_id)
        ws.ledger.reserve(calc.calc_id, spec.category, spec.estimated_node_hours)
        record = SubmissionRecord(
            calc_id=calc.calc_id,
            candidate_id=calc.candidate_id,
            category=spec.category,
            order=proposal.order,
            interactions=proposal.interactions,
            predicted_node_hours=spec.estimated_node_hours,
            state="submitting",
        )
        ws.state.submissions[calc.calc_id] = record
        ws.save()  # written before the scheduler is contacted
        try:
            job_id = submitter(spec)
        except Exception as exc:  # noqa: BLE001 - any submitter failure must release the reservation
            record.state = "needs_manual_check"
            ws.state.log.append(f"{calc.calc_id}: submitter failed ({exc}); check the queue")
            out.needs_manual_check.append(calc.calc_id)
            ws.save()
            out.stopped_because = f"submitter failed for {calc.calc_id}"
            break
        ws.ledger.commit(calc.calc_id, job_id)
        record.state, record.job_id = "submitted", job_id
        budget_left -= spec.estimated_node_hours
        n += 1
        out.submitted.append(calc.calc_id)
        ws.save()
    ws.save()
    return out


def slurm_status(job_id: str) -> tuple[str, float | None]:
    """(state, node-hours) from ``sacct``; node-hours is ``None`` until the job has finished."""
    proc = subprocess.run(
        ["sacct", "-j", job_id, "-n", "-P", "-X", "-o", "State,ElapsedRaw,NNodes"],
        capture_output=True,
        text=True,
        check=True,
    )
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    if not line:
        return "UNKNOWN", None
    state, elapsed, nodes = line.split("|")
    state = state.split()[0]
    if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
        return state, None
    return state, int(elapsed) / 3600.0 * int(nodes)


def monitor(ws: CampaignWorkspace, status: JobStatus = slurm_status) -> dict[str, str]:
    """Poll submitted jobs once; finished jobs consume their actual node-hours (failures too)."""
    seen: dict[str, str] = {}
    for calc_id, sub in ws.state.submissions.items():
        if sub.state != "submitted" or sub.job_id is None:
            seen[calc_id] = sub.state
            continue
        state, hours = status(sub.job_id)
        if hours is None:
            seen[calc_id] = state
            continue
        outcome = "converged" if state == "COMPLETED" else "failed"
        ws.ledger.record_result(calc_id, hours, outcome)  # type: ignore[arg-type]
        sub.state, sub.actual_node_hours, sub.outcome = "finished", hours, outcome
        seen[calc_id] = f"finished:{outcome}"
    ws.save()
    return seen


# --------------------------------------------------------------------------- review
class CampaignReport(BaseModel):
    campaign_id: str
    target: str
    datasets_reused: list[str]
    subspaces_covered: list[str]
    target_coverage_fraction: float
    coverage_basis: list[str]
    missing_interactions: int
    missing_higher_order: int
    proposed_lower_order: int
    proposed_higher_order: int
    predicted_node_hours: float
    actual_node_hours: float
    transfer_improvement: float | None
    transfer_verdicts: dict[str, str]
    heldout_mae: float | None
    magnetic_transferability: str
    defect_transferability: str
    scientific_coverage: dict[str, str]
    remaining_allocation: float
    decision: str
    reason: str
    gate_summary: dict[str, str]
    notes: list[str]

    def table(self) -> list[tuple[str, str]]:
        def show(value: object) -> str:
            return NOT_YET_ESTABLISHED if value is None else str(value)

        return [
            ("campaign", self.campaign_id),
            ("target", self.target),
            ("datasets reused", ", ".join(self.datasets_reused) or "none"),
            ("subspaces covered", f"{len(self.subspaces_covered)}"),
            (
                "target coverage",
                f"{self.target_coverage_fraction:.3f} ({','.join(self.coverage_basis)})",
            ),
            (
                "missing interactions",
                f"{self.missing_interactions} ({self.missing_higher_order} higher-order)",
            ),
            ("proposed lower-order calcs", str(self.proposed_lower_order)),
            ("proposed higher-order calcs", str(self.proposed_higher_order)),
            ("predicted node-hours", f"{self.predicted_node_hours:.1f}"),
            ("actual node-hours", f"{self.actual_node_hours:.1f}"),
            ("transfer improvement", show(self.transfer_improvement)),
            (
                "transfer verdicts",
                ", ".join(f"{k}={v}" for k, v in self.transfer_verdicts.items())
                or NOT_YET_ESTABLISHED,
            ),
            ("held-out accuracy (MAE)", show(self.heldout_mae)),
            ("magnetic transferability", self.magnetic_transferability),
            ("defect transferability", self.defect_transferability),
            ("remaining allocation", f"{self.remaining_allocation:.1f}"),
            ("next decision", f"{self.decision}: {self.reason}"),
        ]


def evidence(ws: CampaignWorkspace) -> CampaignEvidence:
    cfg, state = ws.config, ws.state
    runs = state.transfer
    cov = coverage(ws)["report"]
    summary = ws.ledger.summary()
    finished = [s for s in state.submissions.values() if s.state == "finished"]
    primary = cfg.transfer_evaluation.required_properties[0]
    status = {
        f"{r.property}:{r.experiment}": v.registry_status
        for r in runs
        for v in r.verdicts
        if r.property == primary
    }
    top = max((r.target_order for r in runs if r.property == primary), default=None)
    held = [r.heldout_mae for r in runs if r.property == primary and r.target_order == top]
    heldout = held[0] if held and top == 6 else None
    return CampaignEvidence(
        inventory_complete=state.inventory_done,
        compatibility_checked=state.inventory_done
        and all(c.get("compatible") is not None for c in state.dataset_compatibility.values()),
        n_reused_records=len(state.registry.calculations),
        n_new_calculations=len(finished),
        coverage_by_order=dict(cov.fraction_by_order),
        converged_fraction=(
            sum(s.outcome == "converged" for s in finished) / len(finished) if finished else None
        ),
        transfer_status=status,
        heldout_mae=heldout,
        validation_reserve_intact=summary.validation_reserve_remaining
        >= summary.validation_reserve - 1e-9,
        magnetic_status=aggregate_status(runs, cfg.transfer_evaluation.magnetic_property),
        defect_status=aggregate_status(runs, cfg.transfer_evaluation.defect_property),
        budget_remaining=summary.remaining,
        budget_total=summary.total,
    )


def review(
    ws: CampaignWorkspace, proposals: list[CandidateProposal] | None = None
) -> tuple[CampaignReport, CampaignPlan]:
    cfg, state = ws.config, ws.state
    ev = evidence(ws)
    decision = decide(ev, cfg.transfer_evaluation.thresholds)
    cov = coverage(ws)
    report_cov: CoverageReport = cov["report"]
    summary = ws.ledger.summary()
    chosen = plan(ws, proposals).selected if proposals else []
    by_prop = {p.candidate_id: p for p in (proposals or [])}
    primary = cfg.transfer_evaluation.required_properties[0]
    benefits = [r.best_benefit for r in state.transfer if r.property == primary and r.best_benefit]
    runs = state.transfer
    magnetic = aggregate_status(runs, cfg.transfer_evaluation.magnetic_property)
    defect = aggregate_status(runs, cfg.transfer_evaluation.defect_property)
    notes = [
        "No claim of real-alloy transferability beyond the held-out verdicts listed here.",
    ]
    if "composition_level" in cov["basis"]:
        notes.append(
            "coverage uses composition-level interactions for entries without explicit ones"
        )
    if not state.transfer:
        notes.append("transferability has not been evaluated")
    return (
        CampaignReport(
            campaign_id=cfg.campaign.campaign_id,
            target=_target_label(cfg),
            datasets_reused=sorted(state.registry.datasets),
            subspaces_covered=sorted(
                {s for d in state.registry.datasets.values() for s in d.covered_subspaces}
            ),
            target_coverage_fraction=report_cov.fraction,
            coverage_basis=cov["basis"],
            missing_interactions=len(report_cov.missing),
            missing_higher_order=len(report_cov.missing_higher_order),
            proposed_lower_order=sum(1 for c in chosen if by_prop[c].order < 6),
            proposed_higher_order=sum(1 for c in chosen if by_prop[c].order >= 6),
            predicted_node_hours=summary.predicted,
            actual_node_hours=summary.actual,
            transfer_improvement=max(benefits) if benefits else None,
            transfer_verdicts={
                f"{r.property}:{r.experiment}": v.registry_status for r in runs for v in r.verdicts
            },
            heldout_mae=ev.heldout_mae,
            magnetic_transferability=magnetic,
            defect_transferability=defect,
            scientific_coverage={
                "transfer": "measured" if state.transfer else NOT_YET_ESTABLISHED,
                "magnetic": magnetic if magnetic != "pending" else NOT_YET_ESTABLISHED,
                "defect": defect if defect != "pending" else NOT_YET_ESTABLISHED,
                "target_accuracy": NOT_YET_ESTABLISHED if ev.heldout_mae is None else "measured",
            },
            remaining_allocation=summary.remaining,
            decision=decision.decision,
            reason=decision.reason,
            gate_summary=decision.gate_summary,
            notes=notes,
        ),
        decision,
    )


__all__ = [
    "CATEGORIES",
    "CampaignReport",
    "CampaignState",
    "CampaignWorkspace",
    "CandidateProposal",
    "DispatchOutcome",
    "DispatchPlan",
    "aggregate_status",
    "baseline_selection",
    "compare_acquisition",
    "coverage",
    "dispatch",
    "evidence",
    "expected_atoms",
    "hash_input",
    "inventory",
    "load_dataset_entries",
    "load_proposals",
    "load_transfer_data",
    "monitor",
    "plan",
    "review",
    "save_transfer_data",
    "sbatch_submitter",
    "slurm_status",
    "transferability",
]
