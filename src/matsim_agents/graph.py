"""Assemble the LangGraph workflow.

The graph implements a Plan -> Execute (loop) -> UQ gate -> Analyze pipeline:

    planner ─► executor ──┐
                  ▲       │
                  └───────┤  while pending_tasks
                          ▼
                        uq_gate ─► analyst ─► END
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from matsim_agents.agents import analyst_node, executor_node, planner_node
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.state import MatSimState


def _route_after_executor(state: MatSimState) -> Literal["executor", "uq_gate"]:
    if state.pending_tasks and state.iteration < state.max_iterations:
        return "executor"
    return "uq_gate"


def _infer_formula_from_latest_result(state: MatSimState) -> str | None:
    if not state.results:
        return None
    latest = state.results[-1]
    try:
        from ase.io import read as ase_read

        atoms = ase_read(latest.optimized_structure_path)
        raw = atoms.get_chemical_formula(mode="hill", empirical=True)
        comp = parse_composition(raw)
        return comp.formula if comp is not None else raw
    except Exception:  # pragma: no cover - depends on runtime files/backends
        return None


def _default_handoff_audit_path(output_dir: str) -> Path:
    return Path(output_dir) / "discovery" / "al_handoff_events.jsonl"


def _append_handoff_audit_record(
    *, cfg: dict, state: MatSimState, action: str, message: str
) -> None:
    output_dir = cfg.get("output_dir") or "./outputs"
    custom_path = cfg.get("al_handoff_audit_path")
    audit_path = Path(custom_path) if custom_path else _default_handoff_audit_path(output_dir)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    top_weights = [
        float(r.top_branch_weight) for r in state.results if r.top_branch_weight is not None
    ]
    mean_top = (sum(top_weights) / len(top_weights)) if top_weights else None
    threshold = float(cfg.get("uq_top_weight_threshold", 0.6))
    frac_unrel = None
    if top_weights:
        frac_unrel = sum(1 for w in top_weights if w < threshold) / len(top_weights)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "objective": state.objective,
        "composition": _infer_formula_from_latest_result(state),
        "n_relaxations": len(state.results),
        "uq": {
            "top_weights": top_weights,
            "mean_top_weight": mean_top,
            "unreliable_fraction": frac_unrel,
            "top_weight_threshold": threshold,
            "min_unreliable_fraction_threshold": float(cfg.get("uq_min_unreliable_fraction", 0.25)),
            "min_relaxations_for_handoff": int(cfg.get("uq_min_relaxations_for_handoff", 1)),
        },
        "decision": {
            "action": action,
            "message": message,
            "active_learning_config": cfg.get("active_learning_config"),
            "active_learning_dry_run": bool(cfg.get("active_learning_dry_run", True)),
        },
    }
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def uq_gate_node(state: MatSimState, config: RunnableConfig = None) -> dict:
    """Optional UQ gate after relaxations that can hand off to active learning."""
    cfg = (config or {}).get("configurable", {}) if config else {}
    if not bool(cfg.get("trigger_al_handoff", False)):
        return {}
    if not state.results:
        return {}

    top_weights = [
        float(r.top_branch_weight) for r in state.results if r.top_branch_weight is not None
    ]
    min_relax = int(cfg.get("uq_min_relaxations_for_handoff", 1))
    if len(state.results) < min_relax:
        msg = (
            f"AL handoff not triggered: only {len(state.results)} relaxation(s), "
            f"need >= {min_relax}."
        )
        _append_handoff_audit_record(cfg=cfg, state=state, action="not_triggered", message=msg)
        return {"handoff_events": [msg]}
    if not top_weights:
        msg = "AL handoff not triggered: no top_branch_weight UQ signal available."
        _append_handoff_audit_record(cfg=cfg, state=state, action="not_triggered", message=msg)
        return {"handoff_events": [msg]}

    thr = float(cfg.get("uq_top_weight_threshold", 0.6))
    min_frac = float(cfg.get("uq_min_unreliable_fraction", 0.25))
    mean_top = sum(top_weights) / len(top_weights)
    frac_unrel = sum(1 for w in top_weights if w < thr) / len(top_weights)
    should = mean_top < thr or frac_unrel >= min_frac
    reason = (
        f"mean_top_weight={mean_top:.3f}, unreliable_fraction={frac_unrel:.3f}, "
        f"thresholds: top<{thr:.3f} or frac>={min_frac:.3f}"
    )

    if not should:
        msg = f"AL handoff not triggered: {reason}."
        _append_handoff_audit_record(cfg=cfg, state=state, action="not_triggered", message=msg)
        return {"handoff_events": [msg]}

    al_cfg_path = cfg.get("active_learning_config")
    if not al_cfg_path:
        msg = (
            "AL handoff requested by UQ gate but skipped: active learning config not set "
            "(pass --al-config)."
        )
        _append_handoff_audit_record(
            cfg=cfg, state=state, action="triggered_but_missing_config", message=msg
        )
        return {"handoff_events": [msg]}

    composition = _infer_formula_from_latest_result(state)
    if not composition:
        msg = (
            "AL handoff requested but skipped: could not infer composition "
            "from optimized structure."
        )
        _append_handoff_audit_record(
            cfg=cfg, state=state, action="triggered_but_missing_composition", message=msg
        )
        return {"handoff_events": [msg]}

    from matsim_agents.active_learning import ALConfig
    from matsim_agents.active_learning.loop import run_active_learning

    al_cfg = ALConfig.from_yaml(al_cfg_path)
    al_cfg.md.seed_source.kind = "compositions"
    al_cfg.md.seed_source.compositions = [composition]
    al_cfg.md.seed_source.paths = []
    al_cfg.md.seed_source.prompt = None
    out_root = Path(cfg.get("output_dir") or "./outputs")
    al_cfg.loop.out_dir = out_root / "run_handoff" / "al_handoff" / composition.replace("/", "_")

    dry = bool(cfg.get("active_learning_dry_run", True))
    if dry:
        msg = (
            f"AL handoff DRY-RUN triggered ({reason}). Would run AL for {composition} "
            f"with out_dir={al_cfg.loop.out_dir}."
        )
        _append_handoff_audit_record(cfg=cfg, state=state, action="triggered_dry_run", message=msg)
        return {"handoff_events": [msg]}

    run_active_learning(al_cfg)
    msg = (
        f"AL handoff executed ({reason}). Active learning launched for {composition}; "
        f"out_dir={al_cfg.loop.out_dir}."
    )
    _append_handoff_audit_record(cfg=cfg, state=state, action="triggered_run", message=msg)
    return {"handoff_events": [msg]}


def build_graph(checkpointer=None):
    """Compile and return the matsim-agents LangGraph workflow."""
    graph = StateGraph(MatSimState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("uq_gate", uq_gate_node)
    graph.add_node("analyst", analyst_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor", _route_after_executor, {"executor": "executor", "uq_gate": "uq_gate"}
    )
    graph.add_edge("uq_gate", "analyst")
    graph.add_edge("analyst", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
