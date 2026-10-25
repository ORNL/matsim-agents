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
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

from matsim_agents.backends.llm.provider import get_chat_model
from matsim_agents.backends.llm.replies import ModelReply, invoke_model
from matsim_agents.backends.llm.servers import LLMServices, api_key_from_env
from matsim_agents.backends.llm.transcript import Contribution, Transcript
from matsim_agents.backends.mlip.hydragnn_energy import has_declared_hydragnn_checkpoint
from matsim_agents.backends.mlip.pool import MLIPWorkerPool, release_device_memory
from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, relax_with_mlip
from matsim_agents.discovery import (
    Composition,
    CompositionExplorationResult,
    explore_composition,
    extract_compositions,
)
from matsim_agents.execution.placement import Placement, ResourcePlan
from matsim_agents.execution.run_directory import make_run_id

ATOMISTIC_EXPLORATION_GUIDANCE = """The system will offer the user the option
    to run a surrogate-driven atomistic exploration of crystal phases for
    that composition to test chemical and dynamical stability claims."""

LLM_ONLY_MODE_INSTRUCTION = """Session capability restriction: this is LLM-only mode.
Atomistic exploration, structure relaxation, and active-learning tools are
unavailable in this session. This restriction overrides any tool instructions
above. Do not offer to run those tools or imply that calculations were performed.
Discuss hypotheses and suggest external calculations instead; the user must
restart chat with --with-mlip to enable atomistic tools."""

DEFAULT_SYSTEM_PROMPT = f"""You are a materials-discovery research partner.
Your role is to help the user generate, critique, and refine hypotheses for
new functional materials (battery cathodes, photovoltaics, catalysts,
2D materials, ...).

Guidelines:
* Be concrete: propose specific chemical compositions (e.g. Li2MnO3,
  Cs2AgBiBr6) and the property targets that motivate them.
* Justify each proposal with physics/chemistry reasoning: ionic radii,
  oxidation states, expected band gap, magnetic ordering, etc.
* When you propose a new composition, write the formula clearly so it can
    be picked up by the system. {ATOMISTIC_EXPLORATION_GUIDANCE}
* Cite established materials when comparing.
"""


@dataclass
class DiscoveryChatConfig:
    llm_only: bool = False
    mlip_backend: str = "hydragnn"
    logdir: str | None = None
    hydragnn_branch_mlp_checkpoint: str | None = None
    hydragnn_inference_head: str | int | None = None
    output_dir: str = "./outputs"
    checkpoint: str | None = None
    mlp_device: str = "cuda"
    uma_model_name: str = "uma-s-1p1"
    uma_task: str = "omat"
    # Resource plan file: its llm pool serves the models named by llm_server and
    # critic_llm_servers, and its mlip pool relaxes candidates concurrently.
    resources: str | None = None
    llm_server: str | None = None
    critic_llm_servers: list[str] = field(default_factory=list)
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
    llm_api_key_env: str | None = None  # environment variable holding the proposer's key
    enable_hypothesis_debate: bool = False
    debate_rounds: int = 1
    critic_llm_provider: str | None = None
    critic_llm_model: str | None = None
    critic_llm_base_url: str | None = None
    critic_panel_models: list[str] = field(default_factory=list)
    critic_panel_providers: list[str] = field(default_factory=list)
    critic_panel_base_urls: list[str] = field(default_factory=list)
    critic_panel_api_key_envs: list[str | None] = field(default_factory=list)
    critic_cross_critique: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    auto_confirm: bool = False  # if True, skip the interactive y/N prompt
    trigger_active_learning_on_high_uq: bool = True
    active_learning_config: str | None = None
    active_learning_dry_run: bool = True
    uq_top_weight_threshold: float = 0.6
    uq_min_unreliable_fraction: float = 0.25
    uq_min_relaxations_for_handoff: int = 3
    al_handoff_audit_path: str | None = None

    def __post_init__(self) -> None:
        """Enforce a single, consistent surrogate-backend choice.

        The user selects exactly one ``mlip_backend`` and must supply the
        inputs that backend needs; the inputs for the other backend are
        ignored. This mirrors the validation in
        :class:`matsim_agents.backends.mlip.relaxation.RelaxStructureInput` so the
        error surfaces at configuration time rather than at the first
        relaxation.
        """
        if self.llm_only:
            self.trigger_active_learning_on_high_uq = False
            return
        if self.mlip_backend not in {"hydragnn", "uma"}:
            raise ValueError(
                f"mlip_backend must be 'hydragnn' or 'uma', got {self.mlip_backend!r}."
            )
        if self.mlip_backend == "hydragnn":
            if not self.logdir:
                raise ValueError(
                    "mlip_backend='hydragnn' requires 'logdir' "
                    "(HydraGNN logdir containing config.json + checkpoint)."
                )
            if (
                not self.hydragnn_branch_mlp_checkpoint
                and self.hydragnn_inference_head is None
                and not has_declared_hydragnn_checkpoint(self.logdir)
            ):
                raise ValueError(
                    "mlip_backend='hydragnn' requires 'hydragnn_branch_mlp_checkpoint' "
                    "(BranchWeightMLP .pt checkpoint)."
                )


@dataclass
class DiscoveryChatSession:
    """Mutable state of a single chat session."""

    config: DiscoveryChatConfig
    messages: list[BaseMessage] = field(default_factory=list)
    seen_compositions: set[str] = field(default_factory=set)
    explorations: list[CompositionExplorationResult] = field(default_factory=list)
    mlip_pool: MLIPWorkerPool | None = None
    llm_services: LLMServices | None = None
    placement: Placement | None = None
    # <output_dir>/chat/<session id>/: every message of the session, as it happens.
    transcript: Transcript | None = None

    def _placement(self) -> Placement | None:
        if self.placement is None and self.config.resources:
            plan = ResourcePlan.from_yaml(self.config.resources)
            self.placement = plan.place(self.config.output_dir)
        return self.placement

    def open(self) -> None:
        """Start the plan's LLM servers and point the proposer and critics at them."""
        placement = self._placement()
        if placement is None or "llm" not in placement.pools:
            if self.config.llm_server or self.config.critic_llm_servers:
                raise ValueError(
                    "--llm-server and --critic-llm-servers need a plan with an llm pool"
                )
            return
        self.llm_services = LLMServices.start(
            placement, log_dir=Path(self.config.output_dir) / "llm-servers"
        )
        proposer = self.llm_services.endpoint(self.config.llm_server)
        critics = [self.llm_services.endpoint(name) for name in self.config.critic_llm_servers]
        self.config = replace(
            self.config,
            llm_provider=proposer.provider,
            llm_model=proposer.model,
            llm_base_url=proposer.base_url,
            llm_api_key_env=proposer.api_key_env,
            critic_panel_models=[item.model for item in critics] or self.config.critic_panel_models,
            critic_panel_providers=[item.provider for item in critics]
            or self.config.critic_panel_providers,
            critic_panel_base_urls=[item.base_url or "" for item in critics]
            or self.config.critic_panel_base_urls,
            critic_panel_api_key_envs=[item.api_key_env for item in critics]
            or self.config.critic_panel_api_key_envs,
        )

    def record(
        self,
        kind: str,
        speaker: str,
        text: str = "",
        *,
        reply: ModelReply | None = None,
        provider: str | None = None,
        model: str | None = None,
        prompt: list[BaseMessage] | None = None,
    ) -> None:
        """Append one message (a person's turn, a model reply, or a tool result)."""
        if self.transcript is None:
            self._open_transcript(Path(self.config.output_dir) / "chat" / make_run_id())
        assert self.transcript is not None
        fields = {
            "id": f"{self.transcript.count + 1:04d}-{kind}",
            "kind": kind,
            "speaker": speaker,
            "provider": provider,
            "model": model,
        }
        contribution = (
            Contribution.from_reply(reply, **fields)
            if reply is not None
            else Contribution(text=text, **fields)
        )
        self.transcript.add(contribution, prompt=prompt)

    def _open_transcript(self, directory: Path) -> None:
        cfg = self.config
        critics = ", ".join(cfg.critic_panel_models) or cfg.critic_llm_model or "none"
        self.transcript = Transcript(
            directory,
            title="Discovery chat",
            preamble=(
                f"Proposer: {cfg.llm_provider} `{cfg.llm_model}`. "
                f"Hypothesis debate: {'on' if cfg.enable_hypothesis_debate else 'off'} "
                f"(critics: {critics}). LLM-only: {'yes' if cfg.llm_only else 'no'}."
            ),
        )

    def save(self) -> Path | None:
        """Atomically write ``session.json`` beside the transcript, so ``--resume`` can continue."""
        if self.transcript is None:
            return None
        directory = self.transcript.directory
        state = {
            "session_id": directory.name,
            "config": asdict(self.config),
            "messages": messages_to_dict(self.messages),
            "seen_compositions": sorted(self.seen_compositions),
            "explorations": [item.model_dump(mode="json") for item in self.explorations],
        }
        destination = directory / "session.json"
        fd, temporary = tempfile.mkstemp(prefix=".session.json.", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(state, stream, indent=2, default=str)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    @classmethod
    def resume(cls, config: DiscoveryChatConfig, session: str | Path) -> DiscoveryChatSession:
        """Continue a saved session: a directory, or a session id under ``<output_dir>/chat/``.

        The conversation and discovery state come from ``session.json``; the
        models and tools come from ``config``.
        """
        directory = Path(session)
        if not (directory / "session.json").is_file():
            directory = Path(config.output_dir) / "chat" / str(session)
        path = directory / "session.json"
        if not path.is_file():
            raise FileNotFoundError(f"no saved chat session at {path}")
        state = json.loads(path.read_text(encoding="utf-8"))
        restored = cls(
            config=config,
            messages=messages_from_dict(state["messages"]),
            seen_compositions=set(state["seen_compositions"]),
            explorations=[
                CompositionExplorationResult.model_validate(item) for item in state["explorations"]
            ],
        )
        restored._open_transcript(directory)
        return restored

    def relaxation_pool(self) -> MLIPWorkerPool:
        """Workers for the plan's mlip pool, or in-process relaxations reusing the model."""
        if self.mlip_pool is None:
            self.mlip_pool = MLIPWorkerPool.from_placement(self._placement())
        return self.mlip_pool

    def release_mlip_models(self) -> None:
        """Free exploration models before active learning uses the same devices."""
        if self.mlip_pool is not None and not self.config.active_learning_dry_run:
            self.mlip_pool.close()
            release_device_memory()

    def close(self) -> None:
        if self.mlip_pool is not None:
            self.mlip_pool.close()
            self.mlip_pool = None
        if self.llm_services is not None:
            self.llm_services.close()
            self.llm_services = None


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
    *,
    mlip_pool: MLIPWorkerPool | None = None,
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
        mlip_backend=cfg.mlip_backend,
        logdir=cfg.logdir,
        hydragnn_branch_mlp_checkpoint=cfg.hydragnn_branch_mlp_checkpoint,
        hydragnn_inference_head=cfg.hydragnn_inference_head,
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
        on_phase_start=_on_start,
        on_phase_done=_on_done,
        mlip_pool=mlip_pool,
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
        candidate = txt[len(prefix) :].strip()
        return candidate or None
    return None


def _is_clear_command(user_text: str) -> bool:
    """Return True if the user typed the conversation-reset command.

    Supported form:
      - /clear
    """
    return user_text.strip().lower() == "/clear"


def _extract_al_command(user_text: str) -> tuple[bool, str | None]:
    """Parse a chat command that triggers an active-learning handoff.

    Supported forms:
      - /al                (use the most recently discussed composition)
      - /al <composition>  (e.g. ``/al Li2MnO3``)

    Returns ``(is_al_command, composition_or_None)``.
    """
    txt = user_text.strip()
    low = txt.lower()
    if low == "/al":
        return True, None
    prefix = "/al "
    if low.startswith(prefix):
        arg = txt[len(prefix) :].strip()
        return True, (arg or None)
    return False, None


def _run_single_structure_relaxation(
    structure_path: str,
    cfg: DiscoveryChatConfig,
    *,
    mlip_pool: MLIPWorkerPool | None = None,
) -> str:
    """Run one MLIP relaxation and return a compact textual summary."""
    out_dir = Path(cfg.output_dir) / "single_relax"
    out_dir.mkdir(parents=True, exist_ok=True)
    relax = mlip_pool.relax_one if mlip_pool is not None else relax_with_mlip
    result = relax(
        RelaxStructureInput(
            structure_path=structure_path,
            mlip_backend=cfg.mlip_backend,
            logdir=cfg.logdir,
            hydragnn_branch_mlp_checkpoint=cfg.hydragnn_branch_mlp_checkpoint,
            checkpoint=cfg.checkpoint,
            uma_model_name=cfg.uma_model_name,
            uma_task=cfg.uma_task,
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


def _uq_handoff_metrics(
    exploration: CompositionExplorationResult, cfg: DiscoveryChatConfig
) -> tuple[float, float, int]:
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
        mean_top < cfg.uq_top_weight_threshold or frac_unreliable >= cfg.uq_min_unreliable_fraction
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
        "timestamp_utc": datetime.now(UTC).isoformat(),
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
    return _run_active_learning_for_formula(exploration.composition.formula, cfg)


def _run_active_learning_for_formula(
    formula: str,
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
    al_cfg.md.seed_source.compositions = [formula]
    al_cfg.md.seed_source.paths = []
    al_cfg.md.seed_source.prompt = None

    safe_formula = formula.replace("/", "_")
    al_cfg.loop.out_dir = Path(cfg.output_dir) / "discovery" / "al_handoff" / safe_formula

    if cfg.active_learning_dry_run:
        return (
            "AL handoff DRY-RUN: would run active learning for "
            f"{formula} with seed_source.kind='compositions', "
            f"compositions=[{formula}], out_dir={al_cfg.loop.out_dir}."
        )

    run_active_learning(al_cfg)
    return (
        "AL handoff completed: active learning started from discovery for "
        f"{formula}. out_dir={al_cfg.loop.out_dir}."
    )


def _debate_hypothesis_response(
    *,
    cfg: DiscoveryChatConfig,
    user_text: str,
    draft_response: str,
    record: Callable[..., None],
) -> str:
    """Run optional multi-LLM critique/revision rounds for hypothesis quality.

    ``record`` (:meth:`DiscoveryChatSession.record`) receives every critique and
    revision.
    """
    if not cfg.enable_hypothesis_debate:
        return draft_response

    rounds = max(1, int(cfg.debate_rounds))
    current = draft_response

    # Critic selection:
    # - panel mode: critic_panel_models (optionally paired with providers/base URLs)
    # - single critic mode: critic_llm_* fields
    # - fallback: same model as primary
    panel_models = [m for m in cfg.critic_panel_models if m]
    if panel_models:
        critic_specs: list[tuple[str, str, str, str | None, str | None]] = []
        for i, model_name in enumerate(panel_models):
            provider = (
                cfg.critic_panel_providers[i]
                if i < len(cfg.critic_panel_providers)
                else (cfg.critic_llm_provider or cfg.llm_provider)
            )
            base_url = (
                cfg.critic_panel_base_urls[i]
                if i < len(cfg.critic_panel_base_urls)
                else cfg.critic_llm_base_url
            )
            key_env = (
                cfg.critic_panel_api_key_envs[i] if i < len(cfg.critic_panel_api_key_envs) else None
            )
            critic_specs.append((f"critic_{i + 1}", provider, model_name, base_url, key_env))
    else:
        critic_specs = [
            (
                "critic_1",
                cfg.critic_llm_provider or cfg.llm_provider,
                cfg.critic_llm_model or cfg.llm_model,
                cfg.critic_llm_base_url,
                None,
            )
        ]

    critics = [
        (
            label,
            get_chat_model(
                provider=provider,
                model=model_name,
                base_url=base_url,
                api_key=api_key_from_env(key_env, owner=label),
            ),
        )
        for (label, provider, model_name, base_url, key_env) in critic_specs
    ]
    critic_models = {label: (provider, model) for label, provider, model, _, _ in critic_specs}

    def ask(kind: str, speaker: str, llm: object, messages: list[BaseMessage]) -> str:
        reply = invoke_model(llm, messages)
        provider, model = critic_models.get(speaker, (cfg.llm_provider, cfg.llm_model))
        record(kind, speaker, reply=reply, provider=provider, model=model, prompt=messages)
        return reply.text

    primary = get_chat_model(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        api_key=api_key_from_env(cfg.llm_api_key_env, owner="proposer"),
    )

    for _ in range(rounds):
        critiques: list[tuple[str, str]] = []
        for label, critic in critics:
            critique_text = ask(
                "critique",
                label,
                critic,
                [
                    SystemMessage(
                        content=(
                            "You are a skeptical scientific reviewer. Critique the assistant's "
                            "materials hypothesis response. Identify weak assumptions, missing "
                            "alternatives, and checks that would falsify the claim. Be concise."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"User query:\n{user_text}\n\n"
                            f"Assistant draft:\n{current}\n\n"
                            "Return: (1) key weaknesses, (2) concrete corrections, "
                            "(3) one strongest counter-hypothesis."
                        )
                    ),
                ],
            )
            critiques.append((label, critique_text))

        # Optional critic-to-critic challenge phase.
        if cfg.critic_cross_critique and len(critiques) > 1:
            refined: list[tuple[str, str]] = []
            for label, critic in critics:
                own = next((txt for lbl, txt in critiques if lbl == label), "")
                others = "\n\n".join([f"[{lbl}] {txt}" for lbl, txt in critiques if lbl != label])
                cross_text = ask(
                    "cross_critique",
                    label,
                    critic,
                    [
                        SystemMessage(
                            content=(
                                "You are in a scientific review panel. Reassess your critique "
                                "after seeing other critics. Keep what is strongest, remove weak "
                                "or redundant points, and produce one improved critique."
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"User query:\n{user_text}\n\n"
                                f"Assistant draft:\n{current}\n\n"
                                f"Your current critique ({label}):\n{own}\n\n"
                                f"Other critics:\n{others}\n\n"
                                "Return your final improved critique only."
                            )
                        ),
                    ],
                )
                refined.append((label, cross_text))
            critiques = refined

        critique_text = "\n\n".join([f"[{label}] {txt}" for label, txt in critiques])

        current = ask(
            "revision",
            "proposer",
            primary,
            [
                SystemMessage(
                    content=(
                        "Revise the assistant draft using all critic feedback. Keep the "
                        "response scientifically grounded, specific, and actionable."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User query:\n{user_text}\n\n"
                        f"Current draft:\n{current}\n\n"
                        f"Critique:\n{critique_text}\n\n"
                        "Produce the improved final response only."
                    )
                ),
            ],
        )

    return current


def chat_once(
    session: DiscoveryChatSession,
    user_text: str,
    *,
    on_assistant: Callable[[str], None] | None = None,
) -> str:
    """Send one user turn, get the assistant reply, and run discovery hooks."""
    try:
        return _chat_turn(session, user_text, on_assistant=on_assistant)
    finally:
        session.save()


def _chat_turn(
    session: DiscoveryChatSession,
    user_text: str,
    *,
    on_assistant: Callable[[str], None] | None = None,
) -> str:
    cfg = session.config
    system_prompt = cfg.system_prompt
    if cfg.llm_only:
        if system_prompt == DEFAULT_SYSTEM_PROMPT:
            system_prompt = system_prompt.replace(ATOMISTIC_EXPLORATION_GUIDANCE, "")
        system_prompt = f"{system_prompt.rstrip()}\n\n{LLM_ONLY_MODE_INSTRUCTION}"
    if not session.messages:
        session.messages.append(SystemMessage(content=system_prompt))
    elif isinstance(session.messages[0], SystemMessage):
        # A resumed session may have switched between LLM-only and --with-mlip.
        session.messages[0] = SystemMessage(content=system_prompt)
    session.record("user", "user", user_text)

    # Direct command: /clear resets the conversation and discovery state.
    if _is_clear_command(user_text):
        session.messages = [SystemMessage(content=system_prompt)]
        session.seen_compositions.clear()
        session.explorations.clear()
        assistant_text = "Conversation history and discovery state cleared."
        session.record("tool", "matsim-agents", assistant_text)
        if on_assistant is not None:
            on_assistant(assistant_text)
        return assistant_text

    # Direct command: /al [composition] triggers an active-learning handoff.
    is_al, al_formula = _extract_al_command(user_text)
    if is_al:
        session.messages.append(HumanMessage(content=user_text))
        if cfg.llm_only:
            assistant_text = (
                "Active learning is unavailable in LLM-only mode. "
                "Restart chat with --with-mlip to enable atomistic tools."
            )
        else:
            if al_formula is None:
                if session.explorations:
                    al_formula = session.explorations[-1].composition.formula
                elif session.seen_compositions:
                    al_formula = next(iter(session.seen_compositions))
            if al_formula is None:
                assistant_text = (
                    "Active learning command received but no composition was given and "
                    "none has been discussed yet. Use '/al <composition>' "
                    "(e.g. '/al Li2MnO3')."
                )
            else:
                try:
                    session.release_mlip_models()
                    assistant_text = _run_active_learning_for_formula(al_formula, cfg)
                except Exception as exc:  # pragma: no cover - depends on runtime environment
                    assistant_text = (
                        f"Active learning failed for {al_formula}: {exc!s}. "
                        "Check active_learning_config and retry."
                    )
        session.messages.append(AIMessage(content=assistant_text))
        session.record("tool", "matsim-agents", assistant_text)
        if on_assistant is not None:
            on_assistant(assistant_text)
        return assistant_text

    # Optional direct operation inside discovery: single-structure relaxation.
    relax_path = _extract_relax_command(user_text)
    if relax_path is not None:
        session.messages.append(HumanMessage(content=user_text))
        if cfg.llm_only:
            assistant_text = (
                "Structure relaxation is unavailable in LLM-only mode. "
                "Restart chat with --with-mlip to enable atomistic tools."
            )
        else:
            try:
                assistant_text = _run_single_structure_relaxation(
                    relax_path, cfg, mlip_pool=session.relaxation_pool()
                )
            except Exception as exc:  # pragma: no cover - depends on runtime environment
                assistant_text = (
                    f"Single-structure relaxation failed for {relax_path}: {exc!s}. "
                    "Check path/model configuration and retry."
                )
        session.messages.append(AIMessage(content=assistant_text))
        session.record("tool", "matsim-agents", assistant_text)
        if on_assistant is not None:
            on_assistant(assistant_text)
        return assistant_text

    session.messages.append(HumanMessage(content=user_text))

    llm = get_chat_model(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        api_key=api_key_from_env(cfg.llm_api_key_env, owner="proposer"),
    )
    reply = invoke_model(llm, session.messages)
    session.record(
        "draft" if cfg.enable_hypothesis_debate else "assistant",
        "proposer",
        reply=reply,
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        prompt=session.messages,
    )
    assistant_text = _debate_hypothesis_response(
        cfg=cfg,
        user_text=user_text,
        draft_response=reply.text,
        record=session.record,
    )
    session.messages.append(AIMessage(content=assistant_text))
    if on_assistant is not None:
        on_assistant(assistant_text)

    if cfg.llm_only:
        return assistant_text

    # Discovery hook: scan both user and assistant text for new compositions.
    for blob in (user_text, assistant_text):
        for comp in extract_compositions(blob):
            if comp.formula in session.seen_compositions:
                continue
            session.seen_compositions.add(comp.formula)
            if _confirm(
                f"\nProposed composition detected: {comp.formula}. "
                "Run surrogate-based phase exploration?",
                auto=cfg.auto_confirm,
            ):
                try:
                    exploration = _kickoff_exploration(
                        comp, cfg, mlip_pool=session.relaxation_pool()
                    )
                except BaseException:
                    # Not explored: a resumed session offers it again.
                    session.seen_compositions.discard(comp.formula)
                    raise
                session.explorations.append(exploration)
                if cfg.trigger_active_learning_on_high_uq:
                    should_handoff, handoff_reason = _should_handoff_to_active_learning(
                        exploration,
                        cfg,
                    )
                    if should_handoff:
                        session.release_mlip_models()
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
                        session.record("tool", "active_learning", handoff_msg)
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
                summary = _summarize_for_llm(exploration)
                session.messages.append(HumanMessage(content="[discovery] " + summary))
                session.record("tool", "discovery", summary)

    return assistant_text


def run_chat(
    config: DiscoveryChatConfig, *, resume: str | Path | None = None
) -> DiscoveryChatSession:
    """Blocking REPL. Type ``exit`` / ``quit`` / Ctrl-D to leave."""
    session = (
        DiscoveryChatSession(config=config)
        if resume is None
        else DiscoveryChatSession.resume(config, resume)
    )
    try:
        session.open()
        config = session.config
        if session.transcript is not None:
            _print(
                f"[dim]Resumed {session.transcript.directory.name}: "
                f"{len(session.messages)} message(s), "
                f"{len(session.explorations)} exploration(s).[/dim]"
            )
        _print(
            f"[bold]matsim-agents discovery chat[/bold]  "
            f"(provider={config.llm_provider}, model={config.llm_model}, "
            f"debate={'on' if config.enable_hypothesis_debate else 'off'})\n"
            "Type 'exit' to quit. Propose compositions like 'Li2MnO3' to trigger exploration.\n"
            "Commands: /relax <structure_path>, /al [composition], /clear.\n"
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
    finally:
        session.close()
        if session.transcript is not None:
            _print(f"[dim]Transcript: {session.transcript.markdown_path}[/dim]")
            _print(
                f"[dim]Resume with: matsim-agents chat --resume "
                f"{session.transcript.directory.name}[/dim]"
            )
    return session
