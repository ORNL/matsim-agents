"""LangGraph supervisor for discovery -> UQ -> optional active-learning handoff.

This module provides a higher-level agentic orchestration layer that keeps the
heavy numerical kernels deterministic (relaxation, DFT, AL loop), while using
LangGraph for control-flow decisions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

from langgraph.graph import END, StateGraph

from matsim_agents.active_learning import ALConfig
from matsim_agents.active_learning.loop import run_active_learning
from matsim_agents.discovery import (
    Composition,
    CompositionExplorationResult,
    explore_composition,
    parse_composition,
)


class SupervisorConfig(BaseModel):
    """Inputs and policy knobs for the supervisor graph."""

    composition: str
    mlp_backend: str = "hydragnn"
    logdir: str | None = None
    mlp_checkpoint: str | None = None
    output_dir: str = "./outputs"
    checkpoint: str | None = None
    mlp_device: str = "cuda"
    uma_model_name: str = "uma-s-1p1"
    uma_task: str = "omat"
    precision: str | None = None
    mlp_precision: str | None = None
    optimizer: str = "FIRE"
    maxiter: int = 200
    maxstep: float = 1e-2
    fmax: float = 0.02
    relative_increase_threshold: float = 0.05
    n_random: int = 50
    random_seed: int = 0

    trigger_active_learning_on_high_uq: bool = True
    active_learning_config: str | None = None
    active_learning_dry_run: bool = True
    uq_top_weight_threshold: float = 0.6
    uq_min_unreliable_fraction: float = 0.25
    uq_min_relaxations_for_handoff: int = 3
    al_handoff_audit_path: str | None = None


class SupervisorState(TypedDict, total=False):
    cfg: SupervisorConfig
    parsed_composition: Composition
    exploration: CompositionExplorationResult
    uq_mean_top_weight: float
    uq_unreliable_fraction: float
    uq_n_with_weights: int
    should_handoff: bool
    handoff_reason: str
    handoff_message: str
    summary: str
    error: str


def _default_handoff_audit_path(cfg: SupervisorConfig) -> Path:
    return Path(cfg.output_dir) / "discovery" / "al_handoff_events.jsonl"


def _append_handoff_audit_record(state: SupervisorState, action: str, message: str | None) -> None:
    cfg = state["cfg"]
    audit_path = (
        Path(cfg.al_handoff_audit_path)
        if cfg.al_handoff_audit_path
        else _default_handoff_audit_path(cfg)
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    exploration = state.get("exploration")
    n_relaxations = len(exploration.relaxations) if exploration is not None else 0

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "composition": cfg.composition,
        "n_relaxations": n_relaxations,
        "uq": {
            "mean_top_weight": state.get("uq_mean_top_weight"),
            "unreliable_fraction": state.get("uq_unreliable_fraction"),
            "n_with_weights": state.get("uq_n_with_weights", 0),
            "top_weight_threshold": cfg.uq_top_weight_threshold,
            "min_unreliable_fraction_threshold": cfg.uq_min_unreliable_fraction,
            "min_relaxations_for_handoff": cfg.uq_min_relaxations_for_handoff,
        },
        "decision": {
            "should_handoff": state.get("should_handoff", False),
            "reason": state.get("handoff_reason", ""),
            "action": action,
            "message": message,
            "active_learning_config": cfg.active_learning_config,
            "active_learning_dry_run": cfg.active_learning_dry_run,
        },
    }
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _prepare_node(state: SupervisorState) -> dict:
    cfg = state["cfg"]
    parsed = parse_composition(cfg.composition)
    if parsed is None:
        return {"error": f"Could not parse composition: {cfg.composition!r}"}
    return {"parsed_composition": parsed}


def _route_after_prepare(state: SupervisorState) -> str:
    return "summarize" if state.get("error") else "explore"


def _explore_node(state: SupervisorState) -> dict:
    cfg = state["cfg"]
    comp = state["parsed_composition"]
    out_dir = str(Path(cfg.output_dir) / "discovery")
    exploration = explore_composition(
        comp,
        mlp_backend=cfg.mlp_backend,
        logdir=cfg.logdir,
        mlp_checkpoint=cfg.mlp_checkpoint,
        uma_model_name=cfg.uma_model_name,
        uma_task=cfg.uma_task,
        checkpoint=cfg.checkpoint,
        output_dir=out_dir,
        mlp_device=cfg.mlp_device,
        precision=cfg.precision,
        mlp_precision=cfg.mlp_precision,
        optimizer=cfg.optimizer,
        maxiter=cfg.maxiter,
        maxstep=cfg.maxstep,
        fmax=cfg.fmax,
        relative_increase_threshold=cfg.relative_increase_threshold,
        n_random=cfg.n_random,
        random_seed=cfg.random_seed,
    )
    return {"exploration": exploration}


def _evaluate_uq_node(state: SupervisorState) -> dict:
    cfg = state["cfg"]
    exploration = state["exploration"]
    n_relax = len(exploration.relaxations)
    if n_relax < cfg.uq_min_relaxations_for_handoff:
        return {
            "uq_mean_top_weight": float("nan"),
            "uq_unreliable_fraction": 0.0,
            "uq_n_with_weights": 0,
            "should_handoff": False,
            "handoff_reason": (
                f"handoff skipped: only {n_relax} relaxation(s), "
                f"need >= {cfg.uq_min_relaxations_for_handoff}"
            ),
        }

    weights = [
        float(r.top_branch_weight)
        for r in exploration.relaxations
        if r.top_branch_weight is not None
    ]
    if not weights:
        return {
            "uq_mean_top_weight": float("nan"),
            "uq_unreliable_fraction": 0.0,
            "uq_n_with_weights": 0,
            "should_handoff": False,
            "handoff_reason": "handoff skipped: no branch-weight UQ available from relaxations",
        }

    n_unreliable = sum(1 for w in weights if w < cfg.uq_top_weight_threshold)
    mean_top = sum(weights) / len(weights)
    frac_unreliable = n_unreliable / len(weights)

    should = cfg.trigger_active_learning_on_high_uq and (
        mean_top < cfg.uq_top_weight_threshold
        or frac_unreliable >= cfg.uq_min_unreliable_fraction
    )
    reason = (
        f"mean_top_weight={mean_top:.3f}, unreliable_fraction={frac_unreliable:.3f}, "
        f"thresholds: top<{cfg.uq_top_weight_threshold:.3f} or "
        f"frac>={cfg.uq_min_unreliable_fraction:.3f}"
    )
    return {
        "uq_mean_top_weight": mean_top,
        "uq_unreliable_fraction": frac_unreliable,
        "uq_n_with_weights": len(weights),
        "should_handoff": should,
        "handoff_reason": reason,
    }


def _route_after_uq(state: SupervisorState) -> str:
    return "handoff" if state.get("should_handoff", False) else "summarize"


def _handoff_node(state: SupervisorState) -> dict:
    cfg = state["cfg"]
    exploration = state["exploration"]

    if not cfg.active_learning_config:
        msg = (
            "AL handoff requested but skipped: active_learning_config is not set. "
            "Set --al-config for supervisor execution."
        )
        _append_handoff_audit_record(state, action="triggered_but_missing_config", message=msg)
        return {"handoff_message": msg}

    al_cfg = ALConfig.from_yaml(cfg.active_learning_config)
    al_cfg.md.seed_source.kind = "compositions"
    al_cfg.md.seed_source.compositions = [exploration.composition.formula]
    al_cfg.md.seed_source.paths = []
    al_cfg.md.seed_source.prompt = None

    safe_formula = exploration.composition.formula.replace("/", "_")
    al_cfg.loop.out_dir = Path(cfg.output_dir) / "discovery" / "al_handoff" / safe_formula

    if cfg.active_learning_dry_run:
        msg = (
            "AL handoff DRY-RUN: would run active learning for "
            f"{exploration.composition.formula} with seed_source.kind='compositions', "
            f"compositions=[{exploration.composition.formula}], out_dir={al_cfg.loop.out_dir}."
        )
        _append_handoff_audit_record(state, action="triggered_dry_run", message=msg)
        return {"handoff_message": msg}

    run_active_learning(al_cfg)
    msg = (
        "AL handoff completed: active learning started from supervisor for "
        f"{exploration.composition.formula}. out_dir={al_cfg.loop.out_dir}."
    )
    _append_handoff_audit_record(state, action="triggered_run", message=msg)
    return {"handoff_message": msg}


def _summarize_node(state: SupervisorState) -> dict:
    if state.get("error"):
        return {"summary": f"Supervisor failed early: {state['error']}"}

    exploration = state.get("exploration")
    lines: list[str] = []
    if exploration is not None:
        lines.append(
            f"Explored {exploration.composition.formula}: "
            f"{len(exploration.relaxations)} relaxation(s), "
            f"{len(exploration.failures)} failure(s)."
        )
        if exploration.stability is not None:
            lines.append(exploration.stability.summary)

    if state.get("handoff_reason"):
        lines.append("UQ decision: " + state["handoff_reason"])

    if state.get("should_handoff") and state.get("handoff_message"):
        lines.append(state["handoff_message"])
    elif not state.get("should_handoff") and state.get("handoff_reason"):
        _append_handoff_audit_record(state, action="not_triggered", message=None)

    if not lines:
        lines.append("Supervisor completed with no actionable output.")
    return {"summary": "\n".join(lines)}


def build_supervisor_graph():
    """Compile and return the supervisor LangGraph workflow."""
    graph = StateGraph(SupervisorState)
    graph.add_node("prepare", _prepare_node)
    graph.add_node("explore", _explore_node)
    graph.add_node("evaluate_uq", _evaluate_uq_node)
    graph.add_node("handoff", _handoff_node)
    graph.add_node("summarize", _summarize_node)

    graph.set_entry_point("prepare")
    graph.add_conditional_edges(
        "prepare",
        _route_after_prepare,
        {"explore": "explore", "summarize": "summarize"},
    )
    graph.add_edge("explore", "evaluate_uq")
    graph.add_conditional_edges(
        "evaluate_uq",
        _route_after_uq,
        {"handoff": "handoff", "summarize": "summarize"},
    )
    graph.add_edge("handoff", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


def run_supervisor(cfg: SupervisorConfig) -> SupervisorState:
    """Run one supervisor pass for a target composition."""
    graph = build_supervisor_graph()
    final = graph.invoke({"cfg": cfg})
    return final
