"""Unit tests for run-path UQ gate handoff logic and audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from matsim_agents.graph import uq_gate_node
from matsim_agents.state import MatSimState, RelaxationResult


def _result(base_path: str, top_weight: float | None) -> RelaxationResult:
    return RelaxationResult(
        structure_path=base_path,
        optimized_structure_path=base_path,
        trajectory_path=str(Path(base_path).with_suffix(".traj")),
        log_csv_path=str(Path(base_path).with_suffix(".csv")),
        final_energy_eV=-1.0,
        final_max_force_eV_per_A=0.01,
        num_steps=10,
        converged=True,
        top_branch=0,
        top_branch_weight=top_weight,
    )


def test_uq_gate_policy_off_returns_empty(si_vasp: str):
    state = MatSimState(results=[_result(si_vasp, 0.2)])
    out = uq_gate_node(state, config={"configurable": {"trigger_al_handoff": False}})
    assert out == {}


def test_uq_gate_min_relaxations_not_met_writes_audit(tmp_path: Path, si_vasp: str):
    audit = tmp_path / "handoff.jsonl"
    state = MatSimState(objective="test", results=[_result(si_vasp, 0.2)])
    cfg = {
        "configurable": {
            "trigger_al_handoff": True,
            "uq_min_relaxations_for_handoff": 2,
            "al_handoff_audit_path": str(audit),
        }
    }

    out = uq_gate_node(state, config=cfg)
    assert "handoff_events" in out
    assert "only 1 relaxation" in out["handoff_events"][0]

    rec = json.loads(audit.read_text(encoding="utf-8").strip())
    assert rec["decision"]["action"] == "not_triggered"


def test_uq_gate_no_weights_not_triggered(tmp_path: Path, si_vasp: str):
    audit = tmp_path / "handoff.jsonl"
    state = MatSimState(objective="test", results=[_result(si_vasp, None)])
    cfg = {
        "configurable": {
            "trigger_al_handoff": True,
            "uq_min_relaxations_for_handoff": 1,
            "al_handoff_audit_path": str(audit),
        }
    }

    out = uq_gate_node(state, config=cfg)
    assert "no top_branch_weight" in out["handoff_events"][0]
    rec = json.loads(audit.read_text(encoding="utf-8").strip())
    assert rec["decision"]["action"] == "not_triggered"


def test_uq_gate_triggered_missing_config(tmp_path: Path, si_vasp: str):
    audit = tmp_path / "handoff.jsonl"
    state = MatSimState(objective="test", results=[_result(si_vasp, 0.1)])
    cfg = {
        "configurable": {
            "trigger_al_handoff": True,
            "uq_top_weight_threshold": 0.6,
            "uq_min_unreliable_fraction": 0.25,
            "uq_min_relaxations_for_handoff": 1,
            "al_handoff_audit_path": str(audit),
        }
    }

    out = uq_gate_node(state, config=cfg)
    assert "config not set" in out["handoff_events"][0]
    rec = json.loads(audit.read_text(encoding="utf-8").strip())
    assert rec["decision"]["action"] == "triggered_but_missing_config"


def test_uq_gate_triggered_dry_run_with_mocked_al(tmp_path: Path, si_vasp: str, monkeypatch):
    class _DummyALConfig:
        def __init__(self):
            self.md = SimpleNamespace(
                seed_source=SimpleNamespace(kind="paths", compositions=[], paths=[], prompt="x")
            )
            self.loop = SimpleNamespace(out_dir=tmp_path / "al-out")

    monkeypatch.setattr(
        "matsim_agents.active_learning.ALConfig.from_yaml",
        lambda _path: _DummyALConfig(),
    )
    monkeypatch.setattr(
        "matsim_agents.active_learning.loop.run_active_learning",
        lambda _cfg: None,
    )

    audit = tmp_path / "handoff.jsonl"
    state = MatSimState(objective="test", results=[_result(si_vasp, 0.1)])
    cfg = {
        "configurable": {
            "trigger_al_handoff": True,
            "active_learning_config": "dummy.yaml",
            "active_learning_dry_run": True,
            "uq_top_weight_threshold": 0.6,
            "uq_min_unreliable_fraction": 0.25,
            "uq_min_relaxations_for_handoff": 1,
            "al_handoff_audit_path": str(audit),
            "output_dir": str(tmp_path),
        }
    }

    out = uq_gate_node(state, config=cfg)
    assert "DRY-RUN triggered" in out["handoff_events"][0]
    rec = json.loads(audit.read_text(encoding="utf-8").strip())
    assert rec["decision"]["action"] == "triggered_dry_run"
