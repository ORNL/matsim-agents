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

import os
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
    supercell: tuple[int, int, int] | None = None
    min_atoms: int = 32
    include_2d: bool = False
    num_layers: int = 1
    vacuum: float = 15.0
    interlayer: float | None = None
    n_orderings: int = 1
    lattice_scales: tuple[float, ...] | None = None
    ordering_seed: int = 0
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:14b"
    llm_base_url: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    auto_confirm: bool = False  # if True, skip the interactive y/N prompt


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

    def _on_start(cand):
        _print(f"  [dim]starting[/dim] {cand.phase:<12} {cand.structure_path}")

    def _on_done(cand, result):
        _print(
            f"  [green]done[/green]    {cand.phase:<12} "
            f"E={result.final_energy_eV:.4f} eV  "
            f"|F|max={result.final_max_force_eV_per_A:.4f} eV/Å  "
            f"steps={result.num_steps}"
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
        supercell=cfg.supercell,
        min_atoms=cfg.min_atoms,
        include_2d=cfg.include_2d,
        num_layers=cfg.num_layers,
        vacuum=cfg.vacuum,
        interlayer=cfg.interlayer,
        n_orderings=cfg.n_orderings,
        lattice_scales=cfg.lattice_scales,
        ordering_seed=cfg.ordering_seed,
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

    session.messages.append(HumanMessage(content=user_text))

    llm = get_chat_model(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
    )
    response = llm.invoke(session.messages)
    assistant_text = response.content if isinstance(response.content, str) else str(response.content)
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
                # Feed the result back into the conversation so the LLM can
                # incorporate it into subsequent reasoning.
                session.messages.append(
                    SystemMessage(content="[discovery] " + _summarize_for_llm(exploration))
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
