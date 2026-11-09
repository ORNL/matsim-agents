"""E3 adaptive workflow: ingestion, leakage guard, accounting, gate, reassessment, next round."""

import json
import shutil
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write

from benchmarks.e3 import controller, reassessment
from benchmarks.e3.run import LLM, RULE, STRATEGIES, _structure_hash, load_protocol, sha256_bytes
from benchmarks.e3.state import E3State, E3WorkflowState, GateOutcome
from matsim_agents.execution.contracts import EvidenceLevel
from matsim_agents.execution.ledger import (
    DecisionRecord,
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    HypothesisStatus,
    LedgerQuantity,
    MethodRecord,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests/data/dft_outputs/vasp/vc_relax_5steps"
MATERIAL = "bcc_refractory"
NORMAL_INCAR = """PREC = Normal
ENCUT = 350.0
EDIFF = 1.0E-6
EDIFFG = -0.01
ISPIN = 1
ISMEAR = 1
SIGMA = 0.2
LREAL = F
ISYM = 0
IBRION = 2
ISIF = 3
NELM = 170
"""
GROUPS = {
    "E4-BCC-0001": "Nb-Ta-Hf-Ti/c1",
    "E4-BCC-0002": "Nb-V-Hf-Ti/c2",
    "E4-BCC-0003": "Ta-Hf-Zr-Ti/c3",
    "E4-BCC-0004": "Nb-Ta-V-Zr/c4",
    "E4-BCC-0005": "Nb-Ta-V-Hf/c5",
    "E4-BCC-0099": "Nb-Ta-V-Ti/c9",  # a validation group: must never be acquired
}


def _atoms(i):
    return Atoms("Nb2", positions=[[0, 0, 0], [1.6 + 0.01 * i, 0, 0]], cell=[5, 5, 5], pbc=True)


def _seed_ledger(path: Path, requests: dict[str, str]) -> dict[str, str]:
    ledger = EvidenceLedger(path)
    seed = ledger.append(
        EvidenceRecord(
            kind="characterization",
            subject=f"{MATERIAL}:mace-mp-0-small",
            campaign_id="e3",
            method=MethodRecord(backend="mace", model="mace-mp-0-small"),
            evidence_level=EvidenceLevel.MLIP_PREDICTION,
            outcome=EvidenceOutcome.MEASURED,
            values=[
                LedgerQuantity(name="force_component_mae_eV_A", value=0.1567, unit="eV/A"),
                LedgerQuantity(name="relative_energy_mae_meV_atom", value=0.0404, unit="meV/atom"),
            ],
        )
    )
    ids = {}
    for cid, actor in requests.items():
        decision = ledger.append(
            DecisionRecord(
                campaign_id="e3",
                iteration=1,
                action="freeze_dft_request",
                subject=cid,
                evidence_ids=[seed.evidence_id],
                actor=actor,
            )
        )
        ids[cid] = decision.decision_id
    return ids


def _run_log(
    repeat: Path, cid: str, normal: bool = True, normal_ok: bool = True, exited: bool = True
):
    root = repeat / "attempts"
    lines = [
        f"job=900.pbs host=x calculation_id={cid} variant=alcf repeat=repeat-01 root={repeat}",
        "stage=Low completed_ionic_steps=0 maximum_ionic_steps=240",
        f"start tag=0 prec=Low from=- dir={root}/attempt-01",
        "end tag=0 rc=0 ionic_steps=5 max_force_eV_per_A=0.008 seconds=1800 status=converged",
    ]
    if normal:
        lines += [
            "stage=Normal completed_ionic_steps=0 maximum_ionic_steps=240",
            f"start tag=N0 prec=Normal from=0 dir={root}/attempt-02",
            f"end tag=N0 rc=0 ionic_steps=5 max_force_eV_per_A=0.009 seconds=1800 "
            f"status={'converged' if normal_ok else 'checkpoint:time'}",
        ]
    if exited:
        final = "converged:N0" if normal and normal_ok else "converged:0"
        lines += [
            f"allocation_accounting pbs_job_id=900.pbs calculation_id={cid} allocated_nodes=1 "
            "requested_nodes=1 walltime_limit_seconds=86400 requested_node_hours=24.000000 "
            "elapsed_seconds=3600 elapsed_node_hours=1.000000",
            f"EXIT job=900.pbs calculation_id={cid} status=0 final={final}",
        ]
    (repeat / "run-record.log").write_text(
        "".join(f"[e4-ab 2026-11-09T00:00:00+00:00] {line}\n" for line in lines)
    )


def _attempt(repeat: Path, name: str, prec: str, tag: str):
    d = repeat / "attempts" / name
    d.mkdir(parents=True)
    for f in ("OUTCAR", "OSZICAR", "CONTCAR", "POSCAR", "vasprun.xml"):
        shutil.copy(FIXTURE / f, d / f)
    (d / "INCAR").write_text(NORMAL_INCAR.replace("PREC = Normal", f"PREC = {prec}"))
    (d / "KPOINTS").write_text("Gamma\n0\nGamma\n2 2 2\n")
    (d / "snapshot-tag").write_text(tag + "\n")


@pytest.fixture
def campaign(tmp_path):
    """A miniature E3 round 1: 3 cases (rule-only, llm-only, shared)."""
    frames = []
    for i, (cid, group) in enumerate(GROUPS.items()):
        atoms = _atoms(i)
        atoms.info.update(frame_id=cid, e3_group=group)
        frames.append(atoms)
    source = tmp_path / "replay" / MATERIAL
    (source / "split").mkdir(parents=True)
    write(source / "acquisition.extxyz", [a for a in frames if a.info["frame_id"] != "E4-BCC-0099"])
    val = _atoms(50)
    val.info.update(candidate_id="E4-BCC-0099", e3_group="Nb-Ta-V-Ti/c9")
    write(source / "split" / "validation.extxyz", [val])
    test = _atoms(60)
    test.info.update(candidate_id="E4-BCC-0070", e3_group="Nb-Ta-V-Hf/c18")
    write(source / "split" / "test.extxyz", [test])
    (source / "split" / "split-manifest.json").write_text(
        json.dumps(
            {"frames": [{"config_id": "x", "role": "validation", "group_id": "Nb-Ta-V-Ti/c9"}]}
        )
    )
    (source / "prepare.json").write_text(
        json.dumps(
            {
                "references": {
                    "roles": {"validation": ["ab-pilot:Nb22Ta22V21Hf21Zr21Ti21"], "test": []}
                }
            }
        )
    )
    roles = tmp_path / "roles.json"
    roles.write_text(
        json.dumps({"assignments": [{"calculation_id": "E4-BCC-0561", "e3_role": "validation"}]})
    )

    ledgers = tmp_path / "ledgers"
    decisions = {
        RULE: _seed_ledger(
            ledgers / RULE / "ledger.jsonl", {"E4-BCC-0001": RULE, "E4-BCC-0003": RULE}
        ),
        LLM: _seed_ledger(ledgers / LLM / "ledger.jsonl", {"E4-BCC-0002": LLM, "E4-BCC-0003": LLM}),
    }
    pool = {a.info["frame_id"]: a for a in frames}
    cases = []
    for cid, requested in (
        ("E4-BCC-0001", [RULE]),
        ("E4-BCC-0002", [LLM]),
        ("E4-BCC-0003", [LLM, RULE]),
    ):
        calc = controller.e3_id(cid, 1)
        case_dir = tmp_path / "inputs" / calc
        repeat = case_dir / "alcf" / "repeat-01"
        repeat.mkdir(parents=True)
        shutil.copy(FIXTURE / "POSCAR", case_dir / "POSCAR")
        poscar_sha = sha256_bytes((case_dir / "POSCAR").read_bytes())
        _attempt(repeat, "attempt-01", "Low", "0")
        _attempt(repeat, "attempt-02", "Normal", "N0")
        _run_log(repeat, calc)
        cases.append(
            {
                "calculation_id": calc,
                "case_dir": str(case_dir),
                "repeat": "repeat-01",
                "protocol_id": "bcc-pbe-low-normal-v1-approved-ediffg-0.01",
                "ediffg_eV_per_A": "-0.01",
                "source_poscar_sha256": poscar_sha,
                "input_hashes": {"POSCAR": poscar_sha},
                "paw_components": [{"label": "C"}, {"label": "Si"}],
                "e3": {
                    "candidate_id": cid,
                    "requested_by": requested,
                    "round": 1,
                    "decision_ids": {s: decisions[s][cid] for s in requested},
                    "structure_sha256": _structure_hash(pool[cid]),
                },
            }
        )
    manifest = tmp_path / "batch-manifest.json"
    manifest.write_text(json.dumps({"executable_sha256": "abc", "cases": cases}))

    class Campaign:
        pass

    c = Campaign()
    c.root, c.manifest, c.ledgers, c.replay, c.roles = (
        tmp_path,
        manifest,
        ledgers,
        tmp_path / "replay",
        roles,
    )
    c.state = tmp_path / "state.json"
    c.cases = cases
    c.pool = pool

    def rewrite(mutator):
        data = json.loads(manifest.read_text())
        mutator(data["cases"])
        manifest.write_text(json.dumps(data))

    c.rewrite = rewrite

    def ingest():
        return controller.ingest_round(
            material=MATERIAL,
            round_number=1,
            manifest_path=manifest,
            ledger_dir=ledgers,
            replay_dir=c.replay,
            state_path=c.state,
            reference_roles=roles,
        )

    c.ingest = ingest
    c.context = lambda: controller.load_context(MATERIAL, c.replay, roles)
    return c


def _ledger(c, strategy):
    return EvidenceLedger(c.ledgers / strategy / "ledger.jsonl")


# ------------------------------------------------------------------ parsing --
def test_normal_stage_located_and_low_stage_never_labelled(campaign):
    case_dir = Path(campaign.cases[0]["case_dir"])
    stage = controller.locate_normal_stage(case_dir, "repeat-01")
    assert stage.status == "converged" and stage.tag == "N0"
    assert stage.attempt_dir.name == "attempt-02"
    assert stage.node_hours == 1.0 and stage.nodes == 1 and stage.pbs_job_id == "900.pbs"
    assert [a.prec for a in stage.attempts] == ["Low", "Normal"]

    repeat = case_dir / "alcf" / "repeat-01"
    _run_log(repeat, "x", normal=False)
    stage = controller.locate_normal_stage(case_dir, "repeat-01")
    assert stage.status == "failed" and "PREC=Normal" in stage.reason

    _run_log(repeat, "x", normal_ok=False)
    assert controller.locate_normal_stage(case_dir, "repeat-01").status == "failed"

    _run_log(repeat, "x", exited=False)
    assert controller.locate_normal_stage(case_dir, "repeat-01").status == "pending"


def test_qualification_rejects_wrong_signature(campaign):
    material = load_protocol()[0]["materials"][MATERIAL]
    attempt = Path(campaign.cases[0]["case_dir"]) / "alcf/repeat-01/attempts"
    assert controller.qualify(attempt / "attempt-02", material, ["C", "Si"])["qualified"]
    low = controller.qualify(attempt / "attempt-01", material, ["C", "Si"])
    assert not low["qualified"] and not low["checks"]["incar_PREC"]["passed"]
    wrong_paw = controller.qualify(attempt / "attempt-02", material, ["Nb_pv", "Si"])
    assert not wrong_paw["checks"]["paw_datasets"]["passed"]
    (attempt / "attempt-02" / "INCAR").write_text(NORMAL_INCAR.replace("LREAL = F", "LREAL = Auto"))
    assert not controller.qualify(attempt / "attempt-02", material, ["C", "Si"])["checks"][
        "incar_LREAL"
    ]["passed"]


# ---------------------------------------------------------------- ingestion --
def test_ingest_writes_per_strategy_evidence_and_accounts_shared_cost(campaign):
    state = campaign.ingest()
    record = state.round(1)
    assert {c.state for c in record.calculations.values()} == {E3State.EVIDENCE_INGESTED}
    shared = record.calculations["E3-R1-BCC-0003"]
    assert set(shared.evidence_ids) == {LLM, RULE}
    assert (
        shared.evidence_ids[LLM] == shared.evidence_ids[RULE]
    )  # identical content, one physical run

    for strategy in STRATEGIES:
        ledger = _ledger(campaign, strategy)
        relax = [e for e in ledger.evidence() if e.kind == "relaxation"]
        assert len(relax) == 2
        assert all(e.outcome == EvidenceOutcome.MEASURED for e in relax)
        assert all(e.evidence_level == EvidenceLevel.CONVERGED_DFT for e in relax)
        assert all(e.calculation_id.startswith("E3-R1-BCC-") for e in relax)
        ingests = [d for d in ledger.decisions() if d.action == controller.INGEST_ACTION]
        assert len(ingests) == 2 and all(d.actor == controller.ACTOR for d in ingests)
        assert not ledger.verify_artifacts()
        st = record.strategies[strategy]
        assert st.logical_attempts == 2 and st.remaining_attempts == 6
        assert st.counterfactual_node_hours == 2.0 and st.counterfactual_dft_calculations == 2

    # three physical calculations at 1 node-hour each; the shared one counted once
    assert record.unique_physical_calculations == 3
    assert record.unique_physical_node_hours == 3.0
    assert (
        E3WorkflowState.load(campaign.state).round(1).calculations.keys()
        == record.calculations.keys()
    )


def test_reingest_is_idempotent(campaign):
    campaign.ingest()
    sizes = {s: len(_ledger(campaign, s)) for s in STRATEGIES}
    state = campaign.ingest()
    assert {s: len(_ledger(campaign, s)) for s in STRATEGIES} == sizes
    assert state.round(1).strategies[RULE].counterfactual_node_hours == 2.0


def test_frozen_round1_ledgers_are_never_written(campaign):
    with pytest.raises(ValueError, match="frozen"):
        controller.ingest_round(
            material=MATERIAL,
            round_number=1,
            manifest_path=campaign.manifest,
            ledger_dir=controller.FROZEN_ROUND1_LEDGERS / MATERIAL,
            replay_dir=campaign.replay,
            state_path=campaign.state,
        )


def test_pending_job_leaves_no_record_and_blocks_next_round(campaign):
    repeat = Path(campaign.cases[1]["case_dir"]) / "alcf/repeat-01"
    _run_log(repeat, "x", exited=False)
    state = campaign.ingest()
    calc = state.round(1).calculations["E3-R1-BCC-0002"]
    assert calc.state == E3State.AWAITING_DFT and not calc.evidence_ids
    assert state.round(1).strategies[LLM].state == E3State.AWAITING_DFT
    assert state.round(1).state == E3State.AWAITING_DFT
    assert [e for e in _ledger(campaign, LLM).evidence() if e.subject == "E4-BCC-0002"] == []
    with pytest.raises(ValueError, match="awaiting DFT"):
        controller.prepare_next_round(
            state, {RULE: ["E4-BCC-0004"]}, campaign.context(), campaign.root / "r2.json"
        )


def test_failed_normal_stage_is_recorded_without_values(campaign):
    repeat = Path(campaign.cases[0]["case_dir"]) / "alcf/repeat-01"
    _run_log(repeat, "x", normal=False)
    state = campaign.ingest()
    calc = state.round(1).calculations["E3-R1-BCC-0001"]
    assert calc.state == E3State.FAILED
    evidence = _ledger(campaign, RULE).get(calc.evidence_ids[RULE])
    assert evidence.outcome == EvidenceOutcome.FAILED_TO_COMPUTE and not evidence.values
    assert evidence.cost.failed_jobs == 1 and evidence.cost.node_hours == 1.0
    assert state.round(1).strategies[RULE].logical_attempts == 2  # failures are charged


def test_unqualified_normal_stage_is_not_converged(campaign):
    attempt = Path(campaign.cases[0]["case_dir"]) / "alcf/repeat-01/attempts/attempt-02"
    (attempt / "INCAR").write_text(NORMAL_INCAR.replace("ENCUT = 350.0", "ENCUT = 520.0"))
    state = campaign.ingest()
    calc = state.round(1).calculations["E3-R1-BCC-0001"]
    evidence = _ledger(campaign, RULE).get(calc.evidence_ids[RULE])
    assert (
        evidence.outcome == EvidenceOutcome.NOT_CONVERGED
        and "incar_ENCUT" in evidence.failure_reason
    )


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda cs: cs[0]["e3"].update(structure_sha256="0" * 64), "pool structure hash"),
        (lambda cs: cs[0]["input_hashes"].update(POSCAR="0" * 64), "staged input hash"),
        (lambda cs: cs[0]["e3"]["decision_ids"].update({RULE: "dc-unknown"}), "freeze_dft_request"),
        (lambda cs: cs[0]["e3"].update(candidate_id="E4-BCC-0561"), "withheld"),
        (lambda cs: cs[0].update(ediffg_eV_per_A="-0.05"), "is not the frozen"),
    ],
)
def test_guard_rejects_leakage_and_provenance_mismatch(campaign, mutate, reason):
    campaign.rewrite(mutate)
    state = campaign.ingest()
    calc = state.round(1).calculations["E3-R1-BCC-0001"]
    assert calc.state == E3State.REJECTED
    assert any(reason in r for r in calc.rejection_reasons)
    assert not calc.evidence_ids
    assert state.round(1).strategies[RULE].logical_attempts == 1


# -------------------------------------------------------------------- gate --
def test_gate_reports_insufficient_validation_and_never_trains(campaign):
    state = campaign.ingest()
    for st in state.round(1).strategies.values():
        assert st.gate_outcome == GateOutcome.INSUFFICIENT_VALIDATION_EVIDENCE
        assert "1 validation and 0 test" in st.gate_reasons[0]


def _sufficient_roles(campaign):
    prepare = campaign.replay / MATERIAL / "prepare.json"
    prepare.write_text(
        json.dumps(
            {
                "references": {
                    "roles": {
                        "validation": ["Nb-Ta-Hf-Ti/v1", "Nb-V-Zr-Ti/v2"],
                        "test": ["Nb-Ta-V-Hf/t1"],
                    }
                }
            }
        )
    )


def test_gate_passes_to_authorization_required_with_coverage(campaign):
    _sufficient_roles(campaign)
    state = campaign.ingest()
    # RULE acquired Nb-Ta-Hf-Ti and Ta-Hf-Zr-Ti: two groups, but no V
    assert state.round(1).strategies[RULE].gate_outcome == GateOutcome.INSUFFICIENT_COVERAGE
    assert "insufficient coverage" in state.round(1).strategies[RULE].gate_reasons[-1]
    # LLM acquired Nb-V-Hf-Ti and Ta-Hf-Zr-Ti: covers every validation element
    assert state.round(1).strategies[LLM].gate_outcome == GateOutcome.AUTHORIZATION_REQUIRED


def test_gate_needs_two_qualified_groups(campaign):
    _sufficient_roles(campaign)
    _run_log(Path(campaign.cases[0]["case_dir"]) / "alcf/repeat-01", "x", normal=False)
    state = campaign.ingest()
    assert state.round(1).strategies[RULE].gate_outcome == GateOutcome.INSUFFICIENT_QUALIFIED_DATA


# ----------------------------------------------------------- reassessment --
class FakeChat:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        class Response:
            content = json.dumps(self.reply) if isinstance(self.reply, dict) else self.reply
            usage_metadata = {"input_tokens": 100, "output_tokens": 20}

        return Response()


def test_deterministic_reassessment_is_unresolved_with_insufficient_validation(campaign):
    state = campaign.ingest()
    results = reassessment.reassess_round(state, campaign.context(), campaign.ledgers)
    assert {r.status for r in results.values()} == {"unresolved"}
    ledger = _ledger(campaign, RULE)
    current = ledger.current_hypotheses()["e3:bcc_refractory:transfer"]
    assert current.status == HypothesisStatus.UNRESOLVED
    assert "Insufficient validation evidence" in current.scientific_rationale
    assert state.round(1).strategies[RULE].state == E3State.SCIENTIFICALLY_UNRESOLVED
    # a later reassessment chains onto the head; it never forks
    reassessment.deterministic_reassessment(ledger, campaign.context(), "x")
    history = EvidenceLedger(ledger.path).history("e3:bcc_refractory:transfer")
    assert len(history) == 2 and history[1].prior_revision_id == history[0].revision_id


def _llm_reply(campaign, **override):
    ledger = _ledger(campaign, LLM)
    dft = [e.evidence_id for e in ledger.evidence() if e.kind == "relaxation"]
    reply = {
        "status": "unresolved",
        "scientific_claim": "Transfer cannot be decided with one validation group.",
        "rationale": "Acquired labels are training data; validation needs more groups.",
        "cited_evidence_ids": dft,
        "proposed_tests": ["qualify a second validation group"],
    }
    reply.update(override)
    return reply


def test_llm_reassessment_accepted_and_costed(campaign):
    state = campaign.ingest()
    chat = FakeChat(_llm_reply(campaign))
    results = reassessment.reassess_round(
        state, campaign.context(), campaign.ledgers, chat=chat, llm_actor="llm:test"
    )
    assert results[LLM].accepted and results[LLM].status == "unresolved"
    assert results[LLM].cost.llm_prompt_tokens == 100
    assert state.round(1).strategies[LLM].llm_prompt_tokens == 100
    assert "Evidence:" in chat.prompts[0] and "E4-BCC-0099" not in chat.prompts[0]


@pytest.mark.parametrize(
    "override, problem",
    [
        ({"cited_evidence_ids": ["ev-not-mine"]}, "not in this strategy's ledger"),
        ({"rationale": "Force MAE is 0.0123 eV/A."}, "numbers not in the prompt"),
        ({"status": "certain"}, "invalid status"),
    ],
)
def test_llm_reassessment_rejected_not_repaired(campaign, override, problem):
    state = campaign.ingest()
    chat = FakeChat(_llm_reply(campaign, **override))
    results = reassessment.reassess_round(state, campaign.context(), campaign.ledgers, chat=chat)
    assert not results[LLM].accepted and any(problem in p for p in results[LLM].problems)
    ledger = _ledger(campaign, LLM)
    assert ledger.current_hypotheses() == {}
    rejected = [d for d in ledger.decisions() if d.action == reassessment.REJECT_ACTION]
    assert len(rejected) == 1 and rejected[0].observed_cost.llm_prompt_tokens == 100


def test_llm_cannot_support_with_unconverged_evidence(campaign):
    attempt = Path(campaign.cases[1]["case_dir"]) / "alcf/repeat-01/attempts/attempt-02"
    (attempt / "INCAR").write_text(NORMAL_INCAR.replace("ENCUT = 350.0", "ENCUT = 520.0"))
    state = campaign.ingest()
    chat = FakeChat(_llm_reply(campaign, status="supported"))
    results = reassessment.reassess_round(state, campaign.context(), campaign.ledgers, chat=chat)
    assert not results[LLM].accepted
    assert any("failed or unconverged" in p for p in results[LLM].problems)


def test_unparseable_llm_reply_rejected(campaign):
    state = campaign.ingest()
    results = reassessment.reassess_round(
        state, campaign.context(), campaign.ledgers, chat=FakeChat("no json")
    )
    assert not results[LLM].accepted and "unparseable" in results[LLM].problems[0]


# ------------------------------------------------------------- next round --
def test_next_round_manifest_is_unauthorized_and_budgeted(campaign):
    state = campaign.ingest()
    ctx = campaign.context()
    out = campaign.root / "round2.json"
    manifest = controller.prepare_next_round(
        state, {RULE: ["E4-BCC-0004"], LLM: ["E4-BCC-0004", "E4-BCC-0005"]}, ctx, out
    )
    assert manifest["execution_authorized"] is False and manifest["authorization"] is None
    assert [r["calculation_id"] for r in manifest["requests"]] == [
        "E3-R2-BCC-0004",
        "E3-R2-BCC-0005",
    ]
    assert manifest["requests"][0]["requested_by"] == [RULE, LLM]
    assert manifest["unique_physical_calculations"] == 2
    assert json.loads(out.read_text()) == manifest
    assert state.round(1).state == E3State.NEXT_ROUND_READY
    assert state.round(1).calculations  # round history preserved
    assert state.round(2).selections[LLM] == ["E4-BCC-0004", "E4-BCC-0005"]


@pytest.mark.parametrize(
    "selection, problem",
    [
        ({RULE: ["E4-BCC-0001"]}, "already acquired"),
        ({RULE: ["E4-BCC-0099"]}, "not in the acquisition pool"),
        ({RULE: ["E4-BCC-0004", "E4-BCC-0005", "E4-BCC-0002"]}, "per round"),
        ({RULE: ["E4-BCC-0004", "E4-BCC-0004"]}, "twice"),
    ],
)
def test_next_round_rejects_invalid_selection(campaign, selection, problem):
    state = campaign.ingest()
    with pytest.raises(ValueError, match=problem):
        controller.prepare_next_round(
            state, selection, campaign.context(), campaign.root / "r2.json"
        )


def test_next_round_respects_total_strategy_budget(campaign):
    state = campaign.ingest()
    state.round(1).strategies[RULE].remaining_attempts = 1
    with pytest.raises(ValueError, match="1 attempts remaining"):
        controller.prepare_next_round(
            state,
            {RULE: ["E4-BCC-0004", "E4-BCC-0005"]},
            campaign.context(),
            campaign.root / "r2.json",
        )
