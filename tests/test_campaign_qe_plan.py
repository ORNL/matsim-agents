from __future__ import annotations

import pytest
from pydantic import ValidationError

from matsim_agents.execution.budget import BudgetLedger
from matsim_agents.workflows.hierarchy.qe_plan import (
    CalculationSpec,
    MethodSignature,
    PriorEvidence,
    ResourceLimits,
    ResourceRequest,
    apply_gate_to_manifest,
    check_compatibility,
    expected_atoms,
    gate_dispatch,
    hash_input,
)

PARENT = 1024
COMP = {"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}
PSEUDOS = {
    "W": "W.upf",
    "Ta": "Ta.upf",
    "Cr": "Cr.upf",
    "V": "V.upf",
    "Hf": "Hf.upf",
    "Mo": "Mo.upf",
}


def method(**kw) -> MethodSignature:
    base = dict(
        functional="PBE",
        pseudopotentials=PSEUDOS,
        ecutwfc_ry=45.0,
        ecutrho_ry=360.0,
        smearing="mv",
        degauss_ry=0.02,
        kpts=(1, 1, 1),
        magnetic_convention="collinear-initial-moments",
    )
    base.update(kw)
    return MethodSignature(**base)


def spec(kind="pristine", n_atoms=None, nodes=64, hours=12.0, **kw) -> CalculationSpec:
    n = n_atoms if n_atoms is not None else expected_atoms(kind, PARENT)
    comp = dict(COMP)
    comp["W"] += n - PARENT
    base = dict(
        calc_id=f"{kind}-1",
        kind=kind,
        category="defect",
        composition=comp,
        supercell=(8, 8, 8),
        parent_atoms=PARENT,
        n_atoms=n,
        method=method(),
        magnetic_treatment="nonmagnetic",
        conv_thr_ry=1e-6,
        resources=ResourceRequest(nodes=nodes, walltime_hours=hours, memory_gb_per_node=400),
        expected_information_gain=0.4,
    )
    base.update(kw)
    return CalculationSpec(**base).with_input("&control ...", "/tmp/pw.in")


LIMITS = ResourceLimits(max_nodes=9408, max_walltime_hours=24.0, node_memory_gb=512)
DEBUG = ResourceLimits(max_nodes=8, max_walltime_hours=2.0, node_memory_gb=512)


def ledger() -> BudgetLedger:
    return BudgetLedger(total_node_hours=10_000.0, allocation={"defect": 5000.0})


@pytest.mark.parametrize(
    "kind,atoms",
    [
        ("pristine", 1024),
        ("vacancy", 1023),
        ("interstitial", 1025),
        ("frenkel_pair", 1024),
        ("divacancy", 1022),
    ],
)
def test_defect_cells_have_the_exact_full_size_atom_count(kind, atoms):
    assert expected_atoms(kind, PARENT) == atoms
    assert spec(kind).n_atoms == atoms


@pytest.mark.parametrize("kind,wrong", [("pristine", 128), ("vacancy", 1024), ("divacancy", 1023)])
def test_shrunken_or_wrong_cells_are_rejected(kind, wrong):
    with pytest.raises(ValidationError, match="never shrunk"):
        spec(kind, n_atoms=wrong)


def test_composition_must_match_atom_count():
    with pytest.raises(ValidationError, match="composition"):
        spec(composition={"W": 10})


def test_record_holds_every_required_field():
    rec = spec().record()
    for key in (
        "composition",
        "supercell",
        "method",
        "magnetic_treatment",
        "conv_thr_ry",
        "input_hash",
        "estimated_node_hours",
        "expected_information_gain",
    ):
        assert rec[key] not in (None, {})
    for key in ("functional", "pseudopotentials", "ecutwfc_ry", "ecutrho_ry", "smearing", "kpts"):
        assert key in rec["method"]
    assert rec["input_hash"] == hash_input("&control ...") and rec["estimated_node_hours"] == 768


def test_compatibility_blocks_hard_mismatches_and_flags_soft_ones():
    assert check_compatibility(method(), method()).compatible
    pbe_sol = check_compatibility(method(), method(functional="PBEsol"))
    assert not pbe_sol.compatible and "functional" in pbe_sol.hard_mismatches[0]
    other_pp = check_compatibility(method(), method(pseudopotentials={**PSEUDOS, "W": "W_pbe.upf"}))
    assert not other_pp.compatible
    mag = check_compatibility(method(), method(magnetic_convention="noncollinear"))
    assert not mag.compatible
    soft = check_compatibility(method(), method(ecutwfc_ry=60.0, kpts=(2, 2, 2)))
    assert soft.compatible and len(soft.warnings) == 2


def test_incompatible_prior_data_is_reported_never_silently_mixed():
    old = PriorEvidence(dataset="hist-128", n_atoms=128, method=method(functional="LDA"))
    decision = gate_dispatch(spec(), LIMITS, ledger(), prior=[old])
    assert decision.status == "READY"
    assert any(
        "hist-128 excluded" in w and "functional" in w for w in decision.compatibility_warnings
    )
    smaller = PriorEvidence(dataset="hist-54", n_atoms=54, method=method(ecutwfc_ry=40.0))
    warn = gate_dispatch(spec(), LIMITS, ledger(), prior=[smaller]).compatibility_warnings
    assert any("54 atoms" in w for w in warn)


def test_ready_when_resources_and_budget_allow():
    d = gate_dispatch(spec(), LIMITS, ledger())
    assert d.status == "READY" and d.estimated_node_hours == 768 and d.input_preserved


def test_resource_blocked_keeps_the_full_size_input_and_never_shrinks():
    s = spec(nodes=64, hours=12.0)
    d = gate_dispatch(s, DEBUG, ledger())
    assert d.status == "RESOURCE_BLOCKED" and d.input_preserved
    assert any("nodes" in r for r in d.reasons) and any("walltime" in r for r in d.reasons)
    assert s.n_atoms == 1024 and s.input_hash is not None


@pytest.mark.parametrize(
    "limits,needle",
    [
        (ResourceLimits(max_nodes=64, max_walltime_hours=24, node_memory_gb=100), "GB/node"),
        (
            ResourceLimits(
                max_nodes=64, max_walltime_hours=24, node_memory_gb=512, gpus_per_node=4
            ),
            "GPUs",
        ),
        (
            ResourceLimits(
                max_nodes=64, max_walltime_hours=24, node_memory_gb=512, available=False
            ),
            "unavailable",
        ),
    ],
)
def test_each_resource_limit_blocks(limits, needle):
    d = gate_dispatch(spec(), limits, ledger())
    assert d.status == "RESOURCE_BLOCKED" and any(needle in r for r in d.reasons)


def test_budget_blocked_when_cost_exceeds_uncommitted_budget():
    small = BudgetLedger(total_node_hours=1000.0, allocation={"defect": 500.0})
    d = gate_dispatch(spec(), LIMITS, small)
    assert d.status == "BUDGET_BLOCKED" and "768" in d.reasons[0]


def test_missing_input_blocks_before_resources():
    bare = spec().model_copy(update={"input_hash": None})
    assert gate_dispatch(bare, LIMITS, ledger()).status == "INPUT_BLOCKED"
    assert gate_dispatch(spec(), LIMITS, ledger(), input_ready=False).status == "INPUT_BLOCKED"


def test_manifest_gate_marks_resource_blocked_without_touching_inputs():
    entries = [
        {"calc_id": "a", "status": "PREPARED", "qe_atoms": 1024, "input_sha256": "x"},
        {"calc_id": "b", "status": "INPUT_BLOCKED: missing pseudopotentials W", "qe_atoms": 1023},
    ]
    req = ResourceRequest(nodes=64, walltime_hours=12.0, memory_gb_per_node=400)
    out = apply_gate_to_manifest(entries, req, DEBUG)
    assert out[0]["status"] == "RESOURCE_BLOCKED" and out[0]["qe_atoms"] == 1024
    assert out[0]["input_sha256"] == "x" and out[0]["estimated_node_hours"] == 768
    assert out[1]["status"].startswith("INPUT_BLOCKED")
    assert entries[0]["status"] == "PREPARED"
    ok = apply_gate_to_manifest(entries, req, LIMITS)
    assert ok[0]["status"] == "PREPARED" and "resource_blockers" not in ok[0]
