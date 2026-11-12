"""Round-level execution of strategy-selected E3-MLIP generation actions."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from matsim_agents.backends.mlip.pool import MLIPWorkerPool
from matsim_agents.discovery.seeds import SeedGenerationAction
from matsim_agents.discovery.wrapper import CompositionExplorationResult, explore_composition

E3_MLIP_MODEL = "medium-omat-0"
E3_MLIP_CHECKPOINT_RELATIVE_PATH = Path("mace/maceomat0mediummodel")
E3_MLIP_CHECKPOINT_SHA256 = "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a"
E3_STRATEGIES = ("rule_based_adaptation", "llm_guided_adaptation")
_INITIAL_EVIDENCE_ID = "campaign-design"


def _verified_checkpoint() -> tuple[Path, str]:
    artifact_dir = os.environ.get("MACE_ARTIFACT_DIR")
    if not artifact_dir:
        raise RuntimeError("MACE_ARTIFACT_DIR must identify the frozen E3-MLIP model cache")
    checkpoint_path = Path(artifact_dir) / E3_MLIP_CHECKPOINT_RELATIVE_PATH
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"frozen E3-MLIP checkpoint is missing: {checkpoint_path}")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != E3_MLIP_CHECKPOINT_SHA256:
        raise ValueError(
            "frozen E3-MLIP checkpoint hash mismatch: "
            f"expected {E3_MLIP_CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    return checkpoint_path.resolve(), actual_sha256


def _verified_checkpoint_path() -> Path:
    return _verified_checkpoint()[0]


def _verified_checkpoint_sha256() -> str:
    return _verified_checkpoint()[1]


def load_action_history(
    output_dir: str | Path,
    strategy: Literal["rule_based_adaptation", "llm_guided_adaptation"],
    before_round_index: int,
) -> list[dict[str, Any]]:
    """Load only completed action summaries from earlier strategy-local rounds."""
    if before_round_index < 1:
        raise ValueError("before_round_index must be positive")
    strategy_dir = Path(output_dir) / strategy
    history = []
    for path in sorted(strategy_dir.glob("round-*/round-manifest.json")):
        match = re.fullmatch(r"round-(\d+)", path.parent.name)
        if match is None:
            raise ValueError(f"invalid E3-MLIP round directory: {path.parent.name!r}")
        if int(match.group(1)) >= before_round_index:
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("strategy") != strategy:
            raise ValueError(f"{path} contains history for another strategy")
        if manifest.get("status") not in {"complete", "incomplete"}:
            continue
        history.extend(
            action
            for action in manifest.get("actions", [])
            if action.get("status") == "complete" and action.get("evidence_id")
        )
    return history


def _action_space_by_id(
    action_space: list[SeedGenerationAction],
) -> dict[str, SeedGenerationAction]:
    actions = {action.action_id: action for action in action_space}
    if not actions:
        raise ValueError("the allowed action space must not be empty")
    if len(actions) != len(action_space):
        raise ValueError("allowed action IDs must be unique")
    return actions


def _coverage_dimensions(action: SeedGenerationAction) -> set[str]:
    dimensions = {
        f"composition:{action.formula}",
        f"size:{action.formula}:{action.formula_units}",
    }
    if action.max_aflow_candidates:
        dimensions.add("generator:aflow")
    if action.pyxtal_count:
        dimensions.add("generator:pyxtal")
    dimensions.update(f"prototype:{item}" for item in action.prototype_ids)
    dimensions.update(f"space-group:{item}" for item in action.space_groups)
    if not action.prototype_ids and action.max_aflow_candidates:
        dimensions.add(f"prototype-family:{action.formula}")
    if not action.space_groups and action.pyxtal_count:
        dimensions.add(f"space-group-family:{action.formula}")
    return dimensions


def _smoothed_convergence_rate(
    action: SeedGenerationAction, history: list[dict[str, Any]]
) -> float:
    relevant = [
        record
        for record in history
        if record["formula"] == action.formula and record["formula_units"] == action.formula_units
    ]
    attempted = sum(int(record["attempted"]) for record in relevant)
    converged = sum(int(record["converged"]) for record in relevant)
    return (converged + 1) / (attempted + 2) if attempted else 0.5


def select_rule_based_actions(
    action_space: list[SeedGenerationAction],
    history: list[dict[str, Any]],
    n_actions: int,
) -> list[SeedGenerationAction]:
    """Select least-covered formula, size, prototype, symmetry and source directions."""
    if n_actions < 1:
        raise ValueError("n_actions must be positive")
    allowed = _action_space_by_id(action_space)
    prior = {item["action_id"] for item in history}
    eligible = [action for action in action_space if action.action_id not in prior]
    if n_actions > len(eligible):
        raise ValueError(
            f"requested {n_actions} actions but only {len(eligible)} unused actions remain"
        )

    covered: dict[str, int] = {}
    for record in history:
        action = allowed.get(record["action_id"])
        if action is None:
            action = SeedGenerationAction.model_validate(record["action"])
        for dimension in _coverage_dimensions(action):
            covered[dimension] = covered.get(dimension, 0) + 1

    selected = []
    while len(selected) < n_actions:
        ranked = sorted(
            eligible,
            key=lambda action: (
                -(
                    sum(
                        1 / (1 + covered.get(dimension, 0))
                        for dimension in _coverage_dimensions(action)
                    )
                    + 0.25 * _smoothed_convergence_rate(action, history)
                ),
                action.action_id,
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        eligible.remove(chosen)
        for dimension in _coverage_dimensions(chosen):
            covered[dimension] = covered.get(dimension, 0) + 1
    return selected


def _llm_action_prompt(
    action_space: list[SeedGenerationAction],
    history: list[dict[str, Any]],
    n_actions: int,
) -> str:
    evidence = []
    for record in history:
        backend_errors = int(record.get("backend_errors", record.get("failed", 0)))
        nonconverged = int(
            record.get(
                "nonconverged",
                max(0, int(record["attempted"]) - int(record["converged"]) - backend_errors),
            )
        )
        evidence.append(
            {
                "evidence_id": record["evidence_id"],
                "action_id": record["action_id"],
                "formula": record["formula"],
                "formula_units": record["formula_units"],
                "generated": record["generated"],
                "attempted": record["attempted"],
                "converged": record["converged"],
                "failed": record.get("failed", backend_errors + nonconverged),
                "backend_errors": backend_errors,
                "nonconverged": nonconverged,
                "unrelaxed": record["unrelaxed"],
                "outcome_class": record["outcome_class"],
            }
        )
    return (
        "Select the next E3-MLIP structure-generation actions. Use only the allowed "
        "action IDs. Consider coverage gaps, optimization outcomes, and generator "
        "yield. Do not claim DFT evidence or that the selected strategy is superior. "
        "Return JSON only with keys selected_action_ids, cited_evidence_ids, rationale. "
        f"Select exactly {n_actions} actions; do not repeat a prior action ID.\n"
        f"Initial campaign-design evidence: {_INITIAL_EVIDENCE_ID} (frozen class/size "
        "quotas, shared MACE-OMAT-0 medium model, and matched optimizer settings).\n"
        f"Prior strategy-local MLIP evidence: {json.dumps(evidence)}\n"
        "Allowed actions: "
        f"{json.dumps([action.model_dump(mode='json') for action in action_space])}\n"
    )


def select_llm_guided_actions(
    chat: Any,
    action_space: list[SeedGenerationAction],
    history: list[dict[str, Any]],
    n_actions: int,
) -> tuple[list[SeedGenerationAction], list[str], str]:
    """Ask the LLM once per round and validate its proposal against the allow-list."""
    if n_actions < 1:
        raise ValueError("n_actions must be positive")
    allowed = _action_space_by_id(action_space)
    if n_actions > len(allowed):
        raise ValueError("requested actions exceed the allowed action space")
    prior = {record["action_id"] for record in history}
    prompt = _llm_action_prompt(action_space, history, n_actions)
    reply_object = chat.invoke(prompt)
    reply = reply_object.content if hasattr(reply_object, "content") else str(reply_object)
    match = re.search(r"\{.*\}", reply, re.S)
    if match is None:
        raise ValueError("LLM reply contains no JSON object")
    proposal = json.loads(match.group(0))
    required_fields = {"selected_action_ids", "cited_evidence_ids", "rationale"}
    if not isinstance(proposal, dict) or set(proposal) != required_fields:
        raise ValueError("LLM proposal must contain exactly the required response fields")
    selected_ids = proposal["selected_action_ids"]
    cited_ids = proposal["cited_evidence_ids"]
    rationale = proposal["rationale"]
    if (
        not isinstance(selected_ids, list)
        or len(selected_ids) != n_actions
        or not all(isinstance(item, str) for item in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
    ):
        raise ValueError("LLM must select the requested number of distinct action IDs")
    if not set(selected_ids) <= set(allowed):
        raise ValueError("LLM selected an action outside the allow-listed action space")
    if set(selected_ids) & prior:
        raise ValueError("LLM selected an action already used in this strategy's history")
    valid_evidence = {_INITIAL_EVIDENCE_ID} | {record["evidence_id"] for record in history}
    if (
        not isinstance(cited_ids, list)
        or not cited_ids
        or not all(isinstance(item, str) for item in cited_ids)
        or not set(cited_ids) <= valid_evidence
    ):
        raise ValueError("LLM must cite valid initial or strategy-local evidence IDs")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("LLM rationale must be a non-empty string")
    return [allowed[action_id] for action_id in selected_ids], cited_ids, rationale


def run_generation_round(
    strategy: Literal["rule_based_adaptation", "llm_guided_adaptation"],
    round_index: int,
    actions: list[SeedGenerationAction] | None,
    output_dir: str | Path,
    *,
    action_space: list[SeedGenerationAction] | None = None,
    n_actions: int | None = None,
    chat: Any = None,
    mlip_pool: MLIPWorkerPool | None = None,
    mlp_device: str = "cuda",
    optimizer: str = "FIRE",
    maxiter: int = 500,
    maxstep: float = 0.05,
    fmax: float = 0.02,
    relax_cell: bool = False,
) -> list[CompositionExplorationResult]:
    """Select or accept a strategy's actions, then execute identical MACE settings.

    If ``actions`` is None, select from the allow-listed action space with the
    strategy-specific policy. No DFT references, labels, or structures are read.
    """
    if round_index < 1:
        raise ValueError("round_index must be positive")
    if strategy not in E3_STRATEGIES:
        raise ValueError(f"unsupported E3-MLIP strategy: {strategy!r}")
    if actions is None:
        if action_space is None or n_actions is None:
            raise ValueError("action_space and n_actions are required when actions are selected")
        history = load_action_history(output_dir, strategy, round_index)
        if strategy == "rule_based_adaptation":
            actions = select_rule_based_actions(action_space, history, n_actions)
            selection_record = {
                "policy": "coverage score + 0.25 * smoothed convergence rate; "
                "tie-break by action_id",
                "cited_evidence_ids": [
                    _INITIAL_EVIDENCE_ID,
                    *[record["evidence_id"] for record in history],
                ],
                "rationale": "Selected actions greedily to maximize under-covered "
                "composition, size, generator, prototype, and space-group dimensions, "
                "with a secondary convergence-rate term from prior same-composition/size "
                "MLIP outcomes.",
            }
        else:
            if chat is None:
                raise ValueError("LLM-guided action selection requires a chat model")
            actions, cited_ids, rationale = select_llm_guided_actions(
                chat, action_space, history, n_actions
            )
            selection_record = {
                "policy": "llm_guided_adaptation",
                "model_id": getattr(chat, "model_id", None),
                "cited_evidence_ids": cited_ids,
                "rationale": rationale,
            }
    elif action_space is not None or n_actions is not None:
        raise ValueError("pass either explicit actions or action_space/n_actions, not both")
    else:
        selection_record = {
            "policy": "provided_actions",
            "cited_evidence_ids": [],
            "rationale": None,
        }
    if not actions:
        raise ValueError("a generation round must contain at least one action")
    action_ids = [action.action_id for action in actions]
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("generation action IDs must be unique within a round")
    if maxiter < 1 or maxstep <= 0 or fmax <= 0:
        raise ValueError("maxiter, maxstep, and fmax must be positive")
    checkpoint_path, checkpoint_sha256 = _verified_checkpoint()

    round_dir = Path(output_dir) / strategy / f"round-{round_index:02d}"
    round_dir.mkdir(parents=True, exist_ok=False)
    results: list[CompositionExplorationResult] = []
    action_records: list[dict[str, object]] = []
    manifest_path = round_dir / "round-manifest.json"

    def write_manifest(status: str) -> None:
        manifest_path.write_text(
            json.dumps(
                {
                    "strategy": strategy,
                    "round": round_index,
                    "status": status,
                    "model": "MACE-OMAT-0 medium",
                    "mace_model": E3_MLIP_MODEL,
                    "mace_model_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "optimizer": optimizer,
                    "maxiter": maxiter,
                    "maxstep_A": maxstep,
                    "fmax_eV_A": fmax,
                    "relax_cell": relax_cell,
                    "selection": selection_record,
                    "actions": action_records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    write_manifest("running")
    try:
        for action in actions:
            action_record = {
                "evidence_id": None,
                "action_id": action.action_id,
                "model": "MACE-OMAT-0 medium",
                "checkpoint_sha256": checkpoint_sha256,
                "action": action.model_dump(mode="json"),
                "formula": action.formula,
                "formula_units": action.formula_units,
                "aflow_candidate_cap": action.max_aflow_candidates,
                "pyxtal_requested": action.pyxtal_count,
                "status": "running",
            }
            action_records.append(action_record)
            write_manifest("running")
            try:
                result = explore_composition(
                    action.formula,
                    output_dir=str(round_dir),
                    mlip_backend="mace",
                    mace_family="checkpoint",
                    mace_model=str(checkpoint_path),
                    mlp_device=mlp_device,
                    optimizer=optimizer,
                    maxiter=maxiter,
                    maxstep=maxstep,
                    fmax=fmax,
                    relax_cell=relax_cell,
                    generation_action=action,
                    mlip_pool=mlip_pool,
                )
                result_path = round_dir / "action-results" / f"{action.action_id}.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                temp_result_path = result_path.with_suffix(".json.tmp")
                result_payload = result.model_dump(mode="json")
                result_payload["model"] = "MACE-OMAT-0 medium"
                result_payload["mace_model"] = E3_MLIP_MODEL
                result_payload["mace_model_path"] = str(checkpoint_path)
                result_payload["checkpoint_sha256"] = checkpoint_sha256
                result_payload["optimizer_protocol"] = {
                    "optimizer": optimizer,
                    "maxiter": maxiter,
                    "maxstep_A": maxstep,
                    "fmax_eV_A": fmax,
                    "relax_cell": relax_cell,
                }
                result_payload["generation_action"] = action.model_dump(mode="json")
                temp_result_path.write_text(
                    json.dumps(result_payload, indent=2) + "\n", encoding="utf-8"
                )
                temp_result_path.replace(result_path)
            except Exception as exc:
                action_record.update(
                    status="failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                write_manifest("running")
                raise
            results.append(result)
            action_record.update(
                status="complete",
                result_path=str(result_path.relative_to(round_dir)),
                generated=len(result.phase_candidates),
                attempted=result.candidate_counts["attempted"],
                converged=result.candidate_counts["converged"],
                failed=result.candidate_counts["failed"],
                backend_errors=result.candidate_counts["backend_errors"],
                nonconverged=result.candidate_counts["nonconverged"],
                unrelaxed=len(result.unrelaxed_candidate_paths),
                outcome_class=result.outcome_class,
            )
            evidence_payload = {
                key: value for key, value in action_record.items() if key != "evidence_id"
            }
            action_record["evidence_id"] = hashlib.sha256(
                json.dumps(evidence_payload, sort_keys=True).encode()
            ).hexdigest()[:16]
            write_manifest("running")
    finally:
        if action_records and action_records[-1]["status"] == "running":
            action_records[-1].update(
                status="failed",
                error_type="InterruptedError",
                error="round execution ended before this action completed",
            )
        write_manifest("complete" if len(results) == len(actions) else "incomplete")

    return results
