"""E3 adaptive-characterization controller: ingest DFT results, account, gate, prepare.

This module connects the existing pieces; it adds no new learning machinery.

* Qualification reuses ``scripts/verify_vasp_run.verify`` on raw VASP output.
* Evidence, decisions and hypotheses go through ``EvidenceLedger``; each strategy
  has its own ledger and is charged the full counterfactual cost of its requests.
* The fine-tuning gate is deterministic and never trains.  Passing every
  condition yields ``AUTHORIZATION_REQUIRED``; no DFT or training job is launched.
* Next-round preparation writes an unauthorized manifest draft and never submits.
"""
from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ase.io import read

from benchmarks.e3.run import (
    REPO,
    STRATEGIES,
    _structure_hash,
    load_protocol,
    sha256_bytes,
)
from benchmarks.e3.state import (
    E3Attempt,
    E3CalculationRecord,
    E3State,
    E3StrategyState,
    E3WorkflowState,
    GateOutcome,
)
from matsim_agents.execution.contracts import EvidenceLevel
from matsim_agents.execution.ledger import (
    CostRecord,
    DecisionRecord,
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    LedgerQuantity,
    MethodRecord,
)

CAMPAIGN = "e3"
ACTOR = "e3-controller"
POLICY_VERSION = "e3-controller-v1"
FREEZE_ACTION = "freeze_dft_request"
INGEST_ACTION = "ingest_dft_result"
VERIFIER = REPO / "scripts" / "verify_vasp_run.py"
FROZEN_ROUND1_LEDGERS = REPO / "runs" / "e3" / "round1-bcc-20261108"
DEFAULT_MANIFEST = REPO / "deployments" / "aurora" / "jobs" / "e3-round1-bcc" / "batch-manifest.json"
DEFAULT_REPLAY = REPO / "runs" / "e3" / "replay-20261108"
DEFAULT_REFERENCE_ROLES = REPO / "research" / "paper" / "e3" / "reference-roles-bcc-v1.json"
VASP_VERSION = "6.6.1"
ROUND_PREFIX = {"bcc_refractory": "BCC"}


def e3_id(candidate_id: str, round_number: int, material: str = "bcc_refractory") -> str:
    """Round-scoped E3 calculation ID; round 1 matches ``stage_round1.e3_id``."""
    return f"E3-R{round_number}-{ROUND_PREFIX[material]}-" + candidate_id.rsplit("-", 1)[1]


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_vasp_run", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify


def file_sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


# --------------------------------------------------------------- run record --
_LINE = re.compile(r"^\[e4-ab (?P<ts>[^\]]+)\] (?P<body>.*)$")
_KV = re.compile(r"(\w+)=(\S+)")


@dataclass
class StageResult:
    """What the run record says about a calculation's Normal stage."""

    status: str  # "pending" | "converged" | "failed"
    reason: str | None = None
    tag: str | None = None
    attempt_dir: Path | None = None
    final: str | None = None
    pbs_job_id: str | None = None
    elapsed_seconds: float | None = None
    node_hours: float | None = None
    nodes: int | None = None
    attempts: list[E3Attempt] = field(default_factory=list)


def parse_run_record(path: Path) -> list[tuple[str, dict[str, str]]]:
    events = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        body = match["body"]
        head = body.split(" ", 1)[0]
        kind = head if "=" not in head else head.split("=", 1)[0]
        events.append((kind, dict(_KV.findall(body))))
    return events


def locate_normal_stage(case_dir: Path, repeat: str, variant: str = "alcf") -> StageResult:
    """Identify the converged PREC=Normal attempt; Low-stage output is never a label."""
    repeat_dir = Path(case_dir) / variant / repeat
    log = repeat_dir / "run-record.log"
    if not log.is_file():
        return StageResult("pending", reason="run has not started (no run-record.log)")
    events = parse_run_record(log)
    attempts: dict[str, E3Attempt] = {}
    starts, ends, exits, allocation, header = [], {}, [], None, {}
    for kind, kv in events:
        if kind == "job":
            header = kv
        elif kind == "start" and "tag" in kv:
            starts.append(kv)
            attempts[kv["tag"]] = E3Attempt(
                tag=kv["tag"], prec=kv.get("prec"), attempt_dir=Path(kv.get("dir", "")).name or None
            )
        elif kind == "end" and "tag" in kv:
            ends[kv["tag"]] = kv
            if kv["tag"] in attempts:
                attempts[kv["tag"]].status = kv.get("status")
                attempts[kv["tag"]].rc = int(kv["rc"]) if kv.get("rc", "").lstrip("-").isdigit() else None
        elif kind == "allocation_accounting":
            allocation = kv
        elif kind == "EXIT":
            exits.append(kv)
    result = StageResult("pending", attempts=list(attempts.values()))
    result.pbs_job_id = (allocation or {}).get("pbs_job_id") or header.get("job")
    if allocation:
        result.elapsed_seconds = float(allocation["elapsed_seconds"])
        result.node_hours = float(allocation["elapsed_node_hours"])
        result.nodes = int(allocation["allocated_nodes"])
    if not exits:
        result.reason = "job has not exited"
        return result
    final_exit = exits[-1]
    result.final = final_exit.get("final")
    if result.node_hours is None:
        seconds = sum(float(e.get("seconds", 0)) for e in ends.values())
        result.elapsed_seconds, result.nodes = seconds, 1
        result.node_hours = seconds / 3600.0
    normal = [s for s in starts if s.get("prec", "").lower() == "normal"]
    if not normal:
        result.status, result.reason = "failed", "job exited without a PREC=Normal stage"
        return result
    tag = normal[-1]["tag"]
    result.tag = tag
    result.attempt_dir = repeat_dir / "attempts" / Path(normal[-1].get("dir", "")).name
    end = ends.get(tag)
    if end is None or end.get("rc") != "0" or end.get("status") != "converged":
        result.status = "failed"
        result.reason = f"Normal stage {tag} did not end converged: {end}"
        return result
    if final_exit.get("status") != "0" or result.final != f"converged:{tag}":
        result.status = "failed"
        result.reason = f"job exit {final_exit.get('status')} final={result.final}, expected converged:{tag}"
        return result
    snapshot = result.attempt_dir / "snapshot-tag"
    if snapshot.is_file() and snapshot.read_text().strip() != tag:
        result.status = "failed"
        result.reason = f"{snapshot} names {snapshot.read_text().strip()}, not {tag}"
        return result
    result.status = "converged"
    return result


# ------------------------------------------------------------ qualification --
def _incar_text(value: Any) -> str:
    if isinstance(value, bool):
        return "T" if value else "F"
    return str(value)


def _same_logical(actual: str | None, expected: bool) -> bool:
    if actual is None:
        return False
    norm = actual.strip().strip(".").upper()
    return norm in ({"T", "TRUE"} if expected else {"F", "FALSE"})


def qualify(attempt_dir: Path, material_protocol: dict[str, Any], paw: list[str], verify=None) -> dict:
    """Verify a Normal-stage attempt against the frozen method signature."""
    verify = verify or load_verifier()
    incar = material_protocol["method"]["incar"]
    numeric = {k: _incar_text(v) for k, v in incar.items() if not isinstance(v, bool)}
    max_force = abs(float(incar["EDIFFG"])) if float(incar["EDIFFG"]) < 0 else None
    result = verify(Path(attempt_dir), "relax", max_force, None, numeric, paw)
    tags = (result.get("method") or {}).get("incar") or {}
    for key, want in incar.items():
        if isinstance(want, bool):
            ok = bool(tags) and _same_logical(tags.get(key), want)
            result["checks"][f"incar_{key}"] = {"passed": ok, "expected": _incar_text(want), "actual": tags.get(key)}
    result["qualified"] = all(c["passed"] for c in result["checks"].values())
    return result


# ------------------------------------------------------------------ context --
@dataclass
class Context:
    material: str
    protocol: dict[str, Any]
    protocol_sha: str
    material_protocol: dict[str, Any]
    pool: dict[str, Any]  # candidate_id -> Atoms
    withheld_ids: set[str]
    withheld_groups: set[str]
    validation_groups: list[str]
    test_groups: list[str]


def load_context(material: str, replay_dir: Path, reference_roles: Path | None, protocol_path: Path | None = None) -> Context:
    protocol, protocol_sha = load_protocol(protocol_path) if protocol_path else load_protocol()
    source = Path(replay_dir) / material
    pool = {a.info["frame_id"]: a for a in read(source / "acquisition.extxyz", ":")}
    withheld_ids: set[str] = set()
    withheld_groups: set[str] = set()
    split = source / "split"
    for role in ("validation", "test"):
        path = split / f"{role}.extxyz"
        if path.is_file():
            for atoms in read(path, ":"):
                withheld_ids.add(atoms.info.get("candidate_id") or atoms.info.get("frame_id"))
                withheld_groups.add(atoms.info.get("e3_group"))
    manifest = split / "split-manifest.json"
    if manifest.is_file():
        for frame in json.loads(manifest.read_text())["frames"]:
            if frame["role"] in ("validation", "test"):
                withheld_groups.add(frame["group_id"])
    if reference_roles and Path(reference_roles).is_file():
        for item in json.loads(Path(reference_roles).read_text())["assignments"]:
            withheld_ids.add(item["calculation_id"])
    prepare = json.loads((source / "prepare.json").read_text())
    roles = prepare["references"]["roles"]
    withheld_ids.discard(None)
    withheld_groups.discard(None)
    return Context(
        material=material,
        protocol=protocol,
        protocol_sha=protocol_sha,
        material_protocol=protocol["materials"][material],
        pool=pool,
        withheld_ids=withheld_ids,
        withheld_groups=withheld_groups,
        validation_groups=list(roles.get("validation", [])),
        test_groups=list(roles.get("test", [])),
    )


def ledger_path(ledger_dir: Path, strategy: str) -> Path:
    return Path(ledger_dir) / strategy / "ledger.jsonl"


def guard_case(case: dict[str, Any], ctx: Context, ledgers: dict[str, EvidenceLedger]) -> list[str]:
    """Leakage and provenance checks; any reason rejects the case before ingestion."""
    e3 = case["e3"]
    cid = e3["candidate_id"]
    reasons = []
    if cid in ctx.withheld_ids:
        reasons.append(f"{cid} is a withheld validation/test/reference structure")
    atoms = ctx.pool.get(cid)
    if atoms is None:
        reasons.append(f"{cid} is not in the acquisition pool")
    else:
        if atoms.info.get("e3_group") in ctx.withheld_groups:
            reasons.append(f"{cid} group {atoms.info.get('e3_group')} is a validation/test group")
        if _structure_hash(atoms) != e3["structure_sha256"]:
            reasons.append(f"{cid} pool structure hash differs from the frozen request")
    poscar = Path(case["case_dir"]) / "POSCAR"
    if not poscar.is_file():
        reasons.append(f"{poscar} is missing")
    else:
        digest = file_sha256(poscar)
        if digest != case["input_hashes"]["POSCAR"] or digest != case.get("source_poscar_sha256", digest):
            reasons.append(f"{poscar} differs from the staged input hash")
    # protocol_id names the Low->Normal staging workflow; the DFT method itself is
    # enforced on the Normal INCAR by qualify(). Here only the staged EDIFFG is bound.
    frozen_ediffg = float(ctx.material_protocol["method"]["incar"]["EDIFFG"])
    if case.get("ediffg_eV_per_A") is None or float(case["ediffg_eV_per_A"]) != frozen_ediffg:
        reasons.append(f"case EDIFFG {case.get('ediffg_eV_per_A')} is not the frozen {frozen_ediffg}")
    for strategy in e3["requested_by"]:
        ledger = ledgers.get(strategy)
        decision_id = e3["decision_ids"].get(strategy)
        record = ledger.get(decision_id) if ledger is not None and decision_id else None
        if not (
            isinstance(record, DecisionRecord)
            and record.action == FREEZE_ACTION
            and record.subject == cid
        ):
            reasons.append(f"{strategy} ledger has no {FREEZE_ACTION} {decision_id} for {cid}")
    return reasons


# ---------------------------------------------------------------- ingestion --
_VALUE_UNITS = {
    "energy_sigma0_per_atom_eV": "eV/atom",
    "final_energy_sigma0_eV": "eV",
    "max_force_eV_per_A": "eV/A",
    "max_abs_principal_stress_kbar": "kbar",
}


def _cost(stage: StageResult, failed: bool) -> CostRecord:
    seconds = stage.elapsed_seconds or 0.0
    return CostRecord(
        dft_calculations=1,
        dft_seconds=seconds,
        failed_jobs=1 if failed else 0,
        failed_seconds=seconds if failed else 0.0,
        nodes=stage.nodes or 0,
        node_hours=stage.node_hours or 0.0,
    )


def build_evidence(
    calc: E3CalculationRecord, case: dict[str, Any], stage: StageResult, verdict: dict | None, ctx: Context
) -> EvidenceRecord:
    method = MethodRecord(
        backend="vasp",
        backend_version=VASP_VERSION,
        method_signature=ctx.material_protocol["method_signature"],
        settings=(verdict or {}).get("method", {}).get("incar", {}),
        software_versions={"executable_sha256": case.get("executable_sha256", "")} if case.get("executable_sha256") else {},
    )
    assumptions = [
        f"acquisition group {calc.group}",
        "PREC=Normal stage only; PREC=Low frames are not E3 labels",
        "physical calculation shared across strategies is charged in full to each requester",
    ]
    common = dict(
        kind="relaxation",
        subject=calc.candidate_id,
        campaign_id=CAMPAIGN,
        calculation_id=calc.calculation_id,
        run_id=stage.pbs_job_id,
        task_id=stage.tag,
        method=method,
        evidence_level=EvidenceLevel.CONVERGED_DFT,
        physical_assumptions=assumptions,
    )
    group = LedgerQuantity(name="e3_group", value=calc.group or "unknown", unit="label")
    if verdict is None or stage.status != "converged" or "files_present" in verdict["checks"]:
        reason = stage.reason or "Normal-stage output files missing"
        if verdict is not None and "files_present" in verdict["checks"]:
            reason = f"missing {verdict['checks']['files_present'].get('missing')}"
        return EvidenceRecord(
            **common, outcome=EvidenceOutcome.FAILED_TO_COMPUTE, failure_reason=reason, cost=_cost(stage, True)
        )
    d = Path(verdict["directory"])
    provenance = verdict["provenance_sha256"]
    inputs = {str(d / n): provenance[n] for n in ("INCAR", "KPOINTS", "POSCAR", "POTCAR") if provenance.get(n)}
    outputs = {str(d / n): provenance[n] for n in ("CONTCAR", "OUTCAR", "vasprun.xml") if provenance.get(n)}
    values = [group] + [
        LedgerQuantity(name=k, value=float(verdict["values"][k]), unit=u, structure=calc.structure_sha256)
        for k, u in _VALUE_UNITS.items()
        if verdict["values"].get(k) is not None
    ]
    failed = sorted(k for k, c in verdict["checks"].items() if not c["passed"])
    return EvidenceRecord(
        **common,
        outcome=EvidenceOutcome.MEASURED if verdict["qualified"] else EvidenceOutcome.NOT_CONVERGED,
        failure_reason=None if verdict["qualified"] else f"failed checks: {failed}",
        values=values,
        numerical_convergence=verdict["checks"],
        input_sha256=inputs,
        output_artifact_sha256=outputs,
        cost=_cost(stage, False),
    )


def ingest_case(
    case: dict[str, Any],
    ctx: Context,
    ledgers: dict[str, EvidenceLedger],
    state: E3WorkflowState,
    *,
    variant: str = "alcf",
    verify=None,
) -> E3CalculationRecord:
    e3 = case["e3"]
    cid = e3["candidate_id"]
    round_record = state.round(int(e3["round"]))
    atoms = ctx.pool.get(cid)
    calc = round_record.calculations.get(case["calculation_id"]) or E3CalculationRecord(
        candidate_id=cid,
        calculation_id=case["calculation_id"],
        round=int(e3["round"]),
        requested_by=list(e3["requested_by"]),
        decision_ids=dict(e3["decision_ids"]),
        structure_sha256=e3["structure_sha256"],
        method_signature=ctx.material_protocol["method_signature"],
        group=atoms.info.get("e3_group") if atoms is not None else None,
    )
    round_record.calculations[calc.calculation_id] = calc
    for strategy in calc.requested_by:
        round_record.selections.setdefault(strategy, [])
        if cid not in round_record.selections[strategy]:
            round_record.selections[strategy].append(cid)
    if calc.state in (E3State.EVIDENCE_INGESTED, E3State.FAILED) and set(calc.evidence_ids) >= set(calc.requested_by):
        return calc
    reasons = guard_case(case, ctx, ledgers)
    if reasons:
        calc.state, calc.rejection_reasons = E3State.REJECTED, reasons
        return calc
    stage = locate_normal_stage(Path(case["case_dir"]), case.get("repeat", "repeat-01"), variant)
    calc.attempts, calc.pbs_job_id, calc.final_status = stage.attempts, stage.pbs_job_id, stage.final
    calc.elapsed_seconds, calc.node_hours, calc.nodes = stage.elapsed_seconds, stage.node_hours, stage.nodes
    if stage.status == "pending":
        calc.state = E3State.AWAITING_DFT
        calc.rejection_reasons = [stage.reason] if stage.reason else []
        return calc
    verdict = None
    if stage.status == "converged":
        calc.normal_attempt_dir = str(stage.attempt_dir)
        paw = [c["label"] for c in case["paw_components"]]
        verdict = qualify(stage.attempt_dir, ctx.material_protocol, paw, verify)
        calc.qualified = bool(verdict["qualified"])
        calc.values = {k: verdict.get("values", {}).get(k) for k in _VALUE_UNITS}
    else:
        calc.qualified = False
    evidence = build_evidence(calc, case, stage, verdict, ctx)
    calc.rejection_reasons = [evidence.failure_reason] if evidence.failure_reason else []
    for strategy in calc.requested_by:
        ledger = ledgers[strategy]
        stored = ledger.append(evidence)
        decision = ledger.append(
            DecisionRecord(
                campaign_id=CAMPAIGN,
                iteration=calc.round,
                action=INGEST_ACTION,
                subject=cid,
                evidence_ids=[stored.evidence_id],
                rationale=(
                    f"Result of {strategy} request {calc.decision_ids[strategy]} "
                    f"({calc.calculation_id}); outcome {stored.outcome.value}."
                ),
                actor=ACTOR,
                policy_version=POLICY_VERSION,
                # the DFT cost is carried by the evidence record; cost_summary() adds both
            )
        )
        calc.evidence_ids[strategy] = stored.evidence_id
        calc.ingest_decision_ids[strategy] = decision.decision_id
    calc.state = (
        E3State.EVIDENCE_INGESTED if evidence.outcome != EvidenceOutcome.FAILED_TO_COMPUTE else E3State.FAILED
    )
    return calc


# --------------------------------------------------------------- accounting --
def account(state: E3WorkflowState, ctx: Context, ledger_dir: Path) -> None:
    """Logical attempts, counterfactual charges and unique physical cost."""
    budget = ctx.protocol["budget"]
    for record in state.rounds:
        seen: dict[tuple[str, str], float] = {}
        for calc in record.calculations.values():
            if calc.state == E3State.REJECTED:
                continue
            key = (calc.structure_sha256, calc.method_signature)
            if calc.node_hours is not None and calc.state != E3State.AWAITING_DFT:
                seen[key] = calc.node_hours
        record.unique_physical_calculations = len(
            {(c.structure_sha256, c.method_signature) for c in record.calculations.values() if c.state != E3State.REJECTED}
        )
        record.unique_physical_node_hours = round(sum(seen.values()), 6)
    totals = {s: 0 for s in STRATEGIES}
    for record in state.rounds:
        for calc in record.calculations.values():
            if calc.state == E3State.REJECTED:
                continue
            for strategy in calc.requested_by:
                totals[strategy] += 1
    current = state.round(state.current_round)
    for strategy in STRATEGIES:
        path = ledger_path(ledger_dir, strategy)
        st = current.strategies.get(strategy) or E3StrategyState(strategy=strategy, ledger_path=str(path))
        st.logical_attempts = totals[strategy]
        st.remaining_attempts = max(0, int(budget["max_attempts_per_strategy"]) - totals[strategy])
        if path.is_file():
            cost = EvidenceLedger(path).cost_summary()
            st.counterfactual_node_hours = round(cost["node_hours"], 6)
            st.counterfactual_dft_calculations = int(cost["dft_calculations"])
            st.llm_prompt_tokens = int(cost["llm_prompt_tokens"])
            st.llm_completion_tokens = int(cost["llm_completion_tokens"])
            st.llm_seconds = float(cost["llm_seconds"])
        mine = [c for c in current.calculations.values() if strategy in c.requested_by and c.state != E3State.REJECTED]
        if any(c.state == E3State.AWAITING_DFT for c in mine):
            st.state = E3State.AWAITING_DFT
        elif mine:
            st.state = E3State.EVIDENCE_INGESTED
        current.strategies[strategy] = st


# -------------------------------------------------------- fine-tuning gate --
def _latest_metrics(ledger: EvidenceLedger, material: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for record in ledger.evidence():
        if (
            record.kind == "characterization"
            and record.subject.startswith(f"{material}:")
            and record.outcome == EvidenceOutcome.MEASURED
            and record.evidence_level == EvidenceLevel.MLIP_PREDICTION
        ):
            values = {v.name: v.value for v in record.values if isinstance(v.value, (int, float))}
            if "force_component_mae_eV_A" in values:
                metrics[record.subject.split(":", 1)[1]] = values
    return metrics


def _elements(group: str) -> set[str]:
    label = group.split(":", 1)[-1].split("/", 1)[0]
    return set(re.findall(r"[A-Z][a-z]?", label))


def fine_tuning_gate(ledger: EvidenceLedger, ctx: Context) -> tuple[GateOutcome, list[str]]:
    """Protocol ``conditional_fine_tuning`` checks, in order.  Never trains."""
    splits = ctx.protocol["splits"]
    n_val, n_test = len(ctx.validation_groups), len(ctx.test_groups)
    if n_val < splits["min_validation_groups"] or n_test < splits["min_test_groups"]:
        return GateOutcome.INSUFFICIENT_VALIDATION_EVIDENCE, [
            f"{n_val} validation and {n_test} test reference groups qualified; protocol needs "
            f">= {splits['min_validation_groups']} and >= {splits['min_test_groups']}"
        ]
    criteria = ctx.material_protocol["accuracy_criteria"]
    deficits = []
    for model, values in sorted(_latest_metrics(ledger, ctx.material).items()):
        for name, limit in criteria.items():
            if isinstance(limit, (int, float)) and name in values and values[name] > limit:
                deficits.append(f"{model} {name}={values[name]:.4g} > {limit}")
    if not deficits:
        return GateOutcome.NO_VALIDATION_DEFICIT, ["no zero-shot model fails a frozen validation criterion"]
    groups = set()
    for record in ledger.evidence():
        if record.kind == "relaxation" and record.outcome == EvidenceOutcome.MEASURED:
            for value in record.values:
                if value.name == "e3_group" and value.value not in ctx.withheld_groups:
                    groups.add(str(value.value))
    if len(groups) < 2:
        return GateOutcome.INSUFFICIENT_QUALIFIED_DATA, deficits + [
            f"{len(groups)} independent qualified acquired groups; need >= 2"
        ]
    acquired = set().union(*(_elements(g) for g in groups))
    missing = sorted(set().union(*(_elements(g) for g in ctx.validation_groups)) - acquired)
    if missing:
        return GateOutcome.INSUFFICIENT_COVERAGE, deficits + [f"insufficient coverage: no acquired label contains {missing}"]
    return GateOutcome.AUTHORIZATION_REQUIRED, deficits + [
        f"{len(groups)} qualified groups cover the validation composition systems; fine-tuning needs explicit authorization"
    ]


# ------------------------------------------------------------ orchestration --
def open_ledgers(ledger_dir: Path) -> dict[str, EvidenceLedger]:
    ledgers = {}
    for strategy in STRATEGIES:
        path = ledger_path(ledger_dir, strategy)
        if not path.is_file():
            raise FileNotFoundError(f"{path}: copy the strategy's round ledger here before ingesting")
        ledgers[strategy] = EvidenceLedger(path)
    return ledgers


def ingest_round(
    *,
    material: str,
    round_number: int,
    manifest_path: Path,
    ledger_dir: Path,
    replay_dir: Path,
    state_path: Path,
    reference_roles: Path | None = DEFAULT_REFERENCE_ROLES,
    protocol_path: Path | None = None,
    variant: str = "alcf",
    verify=None,
) -> E3WorkflowState:
    ledger_dir = Path(ledger_dir).resolve()
    if FROZEN_ROUND1_LEDGERS.resolve() in (ledger_dir, *ledger_dir.parents):
        raise ValueError(f"{ledger_dir} is the frozen Round 1 record; ingest into a copy")
    ctx = load_context(material, replay_dir, reference_roles, protocol_path)
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    if Path(state_path).is_file():
        state = E3WorkflowState.load(state_path)
        if state.protocol_sha256 != ctx.protocol_sha:
            raise ValueError("protocol changed since the state was created")
    else:
        state = E3WorkflowState(
            material=material,
            method_signature=ctx.material_protocol["method_signature"],
            protocol_sha256=ctx.protocol_sha,
        )
    state.batch_manifest_sha256 = sha256_bytes(manifest_bytes)
    state.current_round = max(state.current_round, round_number)
    ledgers = open_ledgers(ledger_dir)
    cases = [c for c in manifest["cases"] if int(c["e3"]["round"]) == round_number]
    for case in cases:
        if "executable_sha256" not in case and manifest.get("executable_sha256"):
            case = {**case, "executable_sha256": manifest["executable_sha256"]}
        ingest_case(case, ctx, ledgers, state, variant=variant, verify=verify)
    account(state, ctx, ledger_dir)
    record = state.round(round_number)
    for strategy, st in record.strategies.items():
        outcome, reasons = fine_tuning_gate(EvidenceLedger(ledger_path(ledger_dir, strategy)), ctx)
        st.gate_outcome, st.gate_reasons = outcome, reasons
    pending = any(c.state == E3State.AWAITING_DFT for c in record.calculations.values())
    record.state = E3State.AWAITING_DFT if pending else E3State.ADAPTATION_ASSESSED
    state.save(state_path)
    return state


# ---------------------------------------------------------- next round prep --
def prepare_next_round(
    state: E3WorkflowState,
    selections: dict[str, list[str]],
    ctx: Context,
    out: Path,
) -> dict[str, Any]:
    """Validate next-round selections and write an unauthorized manifest draft."""
    previous = state.round(state.current_round)
    if any(c.state == E3State.AWAITING_DFT for c in previous.calculations.values()):
        raise ValueError(f"round {state.current_round} still has calculations awaiting DFT")
    number = state.current_round + 1
    budget = ctx.protocol["budget"]
    if number > int(budget["rounds"]):
        raise ValueError(f"protocol allows {budget['rounds']} rounds")
    acquired = state.acquired_structures()
    problems, requests = [], {}
    for strategy, chosen in selections.items():
        if strategy not in STRATEGIES:
            problems.append(f"unknown strategy {strategy}")
            continue
        st = previous.strategies.get(strategy)
        remaining = st.remaining_attempts if st else int(budget["max_attempts_per_strategy"])
        if len(chosen) > int(budget["max_attempts_per_round"]):
            problems.append(f"{strategy} selected {len(chosen)} > {budget['max_attempts_per_round']} per round")
        if len(chosen) > remaining:
            problems.append(f"{strategy} selected {len(chosen)} with {remaining} attempts remaining")
        if len(set(chosen)) != len(chosen):
            problems.append(f"{strategy} selected a structure twice")
        for cid in chosen:
            atoms = ctx.pool.get(cid)
            if atoms is None:
                problems.append(f"{cid} is not in the acquisition pool")
                continue
            if cid in ctx.withheld_ids or atoms.info.get("e3_group") in ctx.withheld_groups:
                problems.append(f"{cid} is withheld from acquisition")
            digest = _structure_hash(atoms)
            if digest in acquired:
                problems.append(f"{cid} was already acquired")
            entry = requests.setdefault(
                cid,
                {
                    "calculation_id": e3_id(cid, number, ctx.material),
                    "candidate_id": cid,
                    "group": atoms.info.get("e3_group"),
                    "structure_sha256": digest,
                    "method_signature": ctx.material_protocol["method_signature"],
                    "requested_by": [],
                },
            )
            entry["requested_by"].append(strategy)
    if problems:
        raise ValueError("; ".join(problems))
    manifest = {
        "schema_version": 1,
        "batch_id": f"E3-round{number}-{ctx.material}",
        "round": number,
        "protocol_sha256": state.protocol_sha256,
        "execution_authorized": False,
        "authorization": None,
        "note": "draft; staging, qsub and any DFT require explicit authorization",
        "requests": sorted(requests.values(), key=lambda r: r["calculation_id"]),
        "unique_physical_calculations": len(requests),
        "logical_attempts": {s: len(v) for s, v in selections.items()},
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    record = state.round(number)
    record.selections = {s: list(v) for s, v in selections.items()}
    previous.next_round_manifest = str(out)
    previous.state = E3State.NEXT_ROUND_READY
    return manifest
