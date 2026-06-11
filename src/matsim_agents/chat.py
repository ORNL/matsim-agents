"""Interactive REPL for hypothesis-driven materials discovery.

A user chats with an LLM (default Qwen 2.5 via Ollama). After every
assistant response we inspect the conversation for newly proposed
chemical compositions; when one is detected the user is asked whether to
launch a substantial atomistic exploration via
:func:`matsim_agents.discovery.explore_composition`.

The REPL is intentionally synchronous and self-contained so it can be
invoked from notebooks, scripts, or the ``matsim-agents chat`` CLI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from matsim_agents.discovery import (
    Composition,
    CompositionExplorationResult,
    explore_composition,
    extract_compositions,
)
from matsim_agents.llm import get_chat_model
from matsim_agents.tools.relaxation import RelaxStructureInput, _run as _run_relaxation

DEFAULT_SYSTEM_PROMPT = """You are a materials-discovery research partner.
Your role is to help the user generate, critique, and refine hypotheses for
new functional materials (battery cathodes, photovoltaics, catalysts,
2D materials, ...).

Guidelines:
* Be concrete: propose specific chemical compositions (e.g. Li2MnO3,
  Cs2AgBiBr6) and the property targets that motivate them.
* Justify each proposal with physics/chemistry reasoning: ionic radii,
  oxidation states, expected band gap, magnetic ordering, etc.
* When you propose a new composition, write the formula clearly so it can
  be picked up by the system. The system will offer the user the option
  to run a HydraGNN-driven atomistic exploration of crystal phases for
  that composition to test chemical and dynamical stability claims.
* Cite established materials when comparing.
"""


@dataclass
class DiscoveryChatConfig:
    logdir: str
    mlp_checkpoint: str
    output_dir: str
    checkpoint: str | None = None
    mlp_device: str = "cuda"
    precision: str | None = None
    mlp_precision: str | None = None
    optimizer: str = "FIRE"
    maxiter: int = 200
    maxstep: float = 1e-2
    fmax: float = 0.02
    relative_increase_threshold: float = 0.05
    n_random: int = 50  # pyXtal random structures (per composition); 0 disables.
    random_seed: int = 0
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:14b"
    llm_base_url: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    auto_confirm: bool = False  # if True, skip the interactive y/N prompt
    trigger_active_learning_on_high_uq: bool = True
    active_learning_config: str | None = None
    active_learning_dry_run: bool = True
    uq_top_weight_threshold: float = 0.6
    uq_min_unreliable_fraction: float = 0.25
    uq_min_relaxations_for_handoff: int = 3
    al_handoff_audit_path: str | None = None


@dataclass
class DiscoveryChatSession:
    """Mutable state of a single chat session."""

    config: DiscoveryChatConfig
    messages: list[BaseMessage] = field(default_factory=list)
    seen_compositions: set[str] = field(default_factory=set)
    explorations: list[CompositionExplorationResult] = field(default_factory=list)


def _print(msg: str) -> None:
    try:
        from rich import print as rprint  # type: ignore

        rprint(msg)
    except Exception:  # pragma: no cover
        print(msg)


def _confirm(prompt: str, *, auto: bool) -> bool:
    if auto:
        _print(f"[bold yellow]auto-confirm:[/bold yellow] {prompt} -> yes")
        return True
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


def _kickoff_exploration(
    composition: Composition,
    cfg: DiscoveryChatConfig,
) -> CompositionExplorationResult:
    """Run the auxiliary atomistic exploration with live progress prints."""
    _print(f"\n[bold cyan]>>> Exploring composition {composition.formula}[/bold cyan]")
    out_dir = os.path.join(cfg.output_dir, "discovery")

    def _tag(cand) -> str:
        # Short label for live progress: prototype AFLOW id or pyxtal_sgNNN.
        if cand.prototype_id:
            return cand.prototype_id[:24]
        if cand.source == "random" and cand.space_group is not None:
            return f"pyxtal_sg{int(cand.space_group):03d}"
        return "seed"

    def _on_start(cand):
        _print(f"  [dim]starting[/dim] {_tag(cand):<26} {cand.structure_path}")

    def _on_done(cand, result):
        novel = " [yellow](novel)[/yellow]" if cand.needs_dft_verification else ""
        _print(
            f"  [green]done[/green]    {_tag(cand):<26} "
            f"E={result.final_energy_eV:.4f} eV  "
            f"|F|max={result.final_max_force_eV_per_A:.4f} eV/\u00c5  "
            f"steps={result.num_steps}{novel}"
        )

    result = explore_composition(
        composition,
        logdir=cfg.logdir,
        mlp_checkpoint=cfg.mlp_checkpoint,
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
        on_phase_start=_on_start,
        on_phase_done=_on_done,
    )

    if result.stability is not None:
        _print(f"\n[bold green]Stability report for {composition.formula}:[/bold green]")
        _print(result.stability.summary)
    if result.failures:
        _print("[bold red]Failures:[/bold red] " + "; ".join(result.failures))
    return result


def _summarize_for_llm(exploration: CompositionExplorationResult) -> str:
    """Compact JSON-ish summary fed back into the conversation."""
    lines = [f"Atomistic exploration completed for {exploration.composition.formula}."]
    if exploration.stability is not None:
        lines.append(exploration.stability.summary)
    else:
        lines.append("No relaxations succeeded.")
    if exploration.failures:
        lines.append("Failures: " + "; ".join(exploration.failures))
    return "\n".join(lines)


def _extract_relax_command(user_text: str) -> str | None:
    """Parse a chat command for single-structure relaxation.

    Supported form:
      - /relax <structure_path>
    """
    txt = user_text.strip()
    prefix = "/relax "
    if txt.lower().startswith(prefix):
        candidate = txt[len(prefix):].strip()
        return candidate or None
    return None


def _run_single_structure_relaxation(structure_path: str, cfg: DiscoveryChatConfig) -> str:
    """Run one HydraGNN relaxation and return a compact textual summary."""
    out_dir = Path(cfg.output_dir) / "single_relax"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _run_relaxation(
        RelaxStructureInput(
            structure_path=structure_path,
            logdir=cfg.logdir,
            mlp_checkpoint=cfg.mlp_checkpoint,
            checkpoint=cfg.checkpoint,
            output_dir=str(out_dir),
            mlp_device=cfg.mlp_device,
            precision=cfg.precision,
            mlp_precision=cfg.mlp_precision,
            optimizer=cfg.optimizer,
            maxiter=cfg.maxiter,
        )
    )
    return (
        f"Single-structure relaxation completed for {structure_path}. "
        f"E={result.final_energy_eV:.4f} eV, "
        f"|F|max={result.final_max_force_eV_per_A:.4f} eV/A, "
        f"steps={result.num_steps}, converged={result.converged}. "
        f"Optimized structure: {result.optimized_structure_path}."
    )


def _uq_handoff_metrics(exploration: CompositionExplorationResult, cfg: DiscoveryChatConfig) -> tuple[float, float, int]:
    """Return (mean_top_weight, unreliable_fraction, n_with_weights)."""
    weights = [
        float(r.top_branch_weight)
        for r in exploration.relaxations
        if r.top_branch_weight is not None
    ]
    if not weights:
        return float("nan"), 0.0, 0
    n_unreliable = sum(1 for w in weights if w < cfg.uq_top_weight_threshold)
    mean_top = sum(weights) / len(weights)
    frac_unreliable = n_unreliable / len(weights)
    return mean_top, frac_unreliable, len(weights)


def _should_handoff_to_active_learning(
    exploration: CompositionExplorationResult,
    cfg: DiscoveryChatConfig,
) -> tuple[bool, str]:
    n_relax = len(exploration.relaxations)
    if n_relax < cfg.uq_min_relaxations_for_handoff:
        return (
            False,
            f"handoff skipped: only {n_relax} relaxation(s), "
            f"need >= {cfg.uq_min_relaxations_for_handoff}",
        )

    mean_top, frac_unreliable, n_with_weights = _uq_handoff_metrics(exploration, cfg)
    if n_with_weights == 0:
        return False, "handoff skipped: no branch-weight UQ available from relaxations"

    should = (
        mean_top < cfg.uq_top_weight_threshold
        or frac_unreliable >= cfg.uq_min_unreliable_fraction
    )
    reason = (
        f"mean_top_weight={mean_top:.3f}, unreliable_fraction={frac_unreliable:.3f}, "
        f"thresholds: top<{cfg.uq_top_weight_threshold:.3f} or "
        f"frac>={cfg.uq_min_unreliable_fraction:.3f}"
    )
    return should, reason


def _default_handoff_audit_path(cfg: DiscoveryChatConfig) -> Path:
    return Path(cfg.output_dir) / "discovery" / "al_handoff_events.jsonl"


def _audit_handoff_event(
    *,
    cfg: DiscoveryChatConfig,
    exploration: CompositionExplorationResult,
    should_handoff: bool,
    handoff_reason: str,
    handoff_action: str,
    handoff_message: str | None,
) -> None:
    """Append a structured AL-handoff audit record as one JSON line."""
    mean_top, frac_unreliable, n_with_weights = _uq_handoff_metrics(exploration, cfg)
    weights = [
        float(r.top_branch_weight)
        for r in exploration.relaxations
        if r.top_branch_weight is not None
    ]
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "composition": exploration.composition.formula,
        "n_relaxations": len(exploration.relaxations),
        "n_relaxations_with_uq": n_with_weights,
        "uq": {
            "mean_top_weight": None if n_with_weights == 0 else mean_top,
            "unreliable_fraction": None if n_with_weights == 0 else frac_unreliable,
            "top_weight_threshold": cfg.uq_top_weight_threshold,
            "min_unreliable_fraction_threshold": cfg.uq_min_unreliable_fraction,
            "min_relaxations_for_handoff": cfg.uq_min_relaxations_for_handoff,
            "top_weights": weights,
        },
        "decision": {
            "should_handoff": should_handoff,
            "reason": handoff_reason,
            "action": handoff_action,
            "message": handoff_message,
            "active_learning_config": cfg.active_learning_config,
            "active_learning_dry_run": cfg.active_learning_dry_run,
        },
    }

    audit_path = (
        Path(cfg.al_handoff_audit_path)
        if cfg.al_handoff_audit_path
        else _default_handoff_audit_path(cfg)
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _handoff_to_active_learning(
    exploration: CompositionExplorationResult,
    cfg: DiscoveryChatConfig,
) -> str:
    if not cfg.active_learning_config:
        return (
            "AL handoff requested but skipped: active_learning_config is not set. "
            "Set --al-config (or DiscoveryChatConfig.active_learning_config)."
        )

    from matsim_agents.active_learning import ALConfig
    from matsim_agents.active_learning.loop import run_active_learning

    al_cfg = ALConfig.from_yaml(cfg.active_learning_config)
    al_cfg.md.seed_source.kind = "compositions"
    al_cfg.md.seed_source.compositions = [exploration.composition.formula]
    al_cfg.md.seed_source.paths = []
    al_cfg.md.seed_source.prompt = None

    safe_formula = exploration.composition.formula.replace("/", "_")
    al_cfg.loop.out_dir = (
        Path(cfg.output_dir) / "discovery" / "al_handoff" / safe_formula
    )

    if cfg.active_learning_dry_run:
        return (
            "AL handoff DRY-RUN: would run active learning for "
            f"{exploration.composition.formula} with seed_source.kind='compositions', "
            f"compositions=[{exploration.composition.formula}], out_dir={al_cfg.loop.out_dir}."
        )

    run_active_learning(al_cfg)
    return (
        "AL handoff completed: active learning started from discovery for "
        f"{exploration.composition.formula}. out_dir={al_cfg.loop.out_dir}."
    )


def chat_once(
    session: DiscoveryChatSession,
    user_text: str,
    *,
    on_assistant: Callable[[str], None] | None = None,
) -> str:
    """Send one user turn, get the assistant reply, and run discovery hooks."""
    cfg = session.config
    if not session.messages:
        session.messages.append(SystemMessage(content=cfg.system_prompt))

    # Optional direct operation inside discovery: single-structure relaxation.
    relax_path = _extract_relax_command(user_text)
    if relax_path is not None:
        session.messages.append(HumanMessage(content=user_text))
        try:
            assistant_text = _run_single_structure_relaxation(relax_path, cfg)
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            assistant_text = (
                f"Single-structure relaxation failed for {relax_path}: {exc!s}. "
                "Check path/model configuration and retry."
            )
        session.messages.append(AIMessage(content=assistant_text))
        if on_assistant is not None:
            on_assistant(assistant_text)
        return assistant_text

    session.messages.append(HumanMessage(content=user_text))

    llm = get_chat_model(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
    )
    response = llm.invoke(session.messages)
    assistant_text = (
        response.content if isinstance(response.content, str) else str(response.content)
    )
    session.messages.append(AIMessage(content=assistant_text))
    if on_assistant is not None:
        on_assistant(assistant_text)

    # Discovery hook: scan both user and assistant text for new compositions.
    for blob in (user_text, assistant_text):
        for comp in extract_compositions(blob):
            if comp.formula in session.seen_compositions:
                continue
            session.seen_compositions.add(comp.formula)
            if _confirm(
                f"\nProposed composition detected: {comp.formula}. "
                "Run HydraGNN-based phase exploration?",
                auto=cfg.auto_confirm,
            ):
                exploration = _kickoff_exploration(comp, cfg)
                session.explorations.append(exploration)
                if cfg.trigger_active_learning_on_high_uq:
                    should_handoff, handoff_reason = _should_handoff_to_active_learning(
                        exploration,
                        cfg,
                    )
                    if should_handoff:
                        handoff_msg = _handoff_to_active_learning(exploration, cfg)
                        handoff_action = (
                            "triggered_dry_run" if cfg.active_learning_dry_run else "triggered_run"
                        )
                        _audit_handoff_event(
                            cfg=cfg,
                            exploration=exploration,
                            should_handoff=should_handoff,
                            handoff_reason=handoff_reason,
                            handoff_action=handoff_action,
                            handoff_message=handoff_msg,
                        )
                        _print(f"\n[bold magenta]AL handoff[/bold magenta] {handoff_reason}")
                        _print(f"[bold magenta]AL handoff[/bold magenta] {handoff_msg}")
                        session.messages.append(
                            HumanMessage(content="[active_learning] " + handoff_msg)
                        )
                    else:
                        _audit_handoff_event(
                            cfg=cfg,
                            exploration=exploration,
                            should_handoff=should_handoff,
                            handoff_reason=handoff_reason,
                            handoff_action="not_triggered",
                            handoff_message=None,
                        )
                        _print(f"\n[dim]AL handoff not triggered:[/dim] {handoff_reason}")
                # Feed the result back into the conversation so the LLM can
                # incorporate it into subsequent reasoning. We use a
                # HumanMessage (not SystemMessage) because some chat
                # templates — notably Mistral's — reject a `system` turn
                # appearing after an `assistant` turn (vLLM raises HTTP
                # 400 "Unexpected role 'system' after role 'assistant'").
                # A user-role injection is portable across providers.
                session.messages.append(
                    HumanMessage(content="[discovery] " + _summarize_for_llm(exploration))
                )

    return assistant_text


def run_chat(config: DiscoveryChatConfig) -> DiscoveryChatSession:
    """Blocking REPL. Type ``exit`` / ``quit`` / Ctrl-D to leave."""
    session = DiscoveryChatSession(config=config)
    _print(
        f"[bold]matsim-agents discovery chat[/bold]  "
        f"(provider={config.llm_provider}, model={config.llm_model})\n"
        "Type 'exit' to quit. Propose compositions like 'Li2MnO3' to trigger exploration.\n"
    )
    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\n[dim]bye[/dim]")
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", ":q"}:
            break
        chat_once(session, user_text, on_assistant=lambda t: _print(f"\nassistant> {t}"))
    return session
