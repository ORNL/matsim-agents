"""Resolve initial MD seed structures from one of three sources.

Sources, in order of precedence:

1. **paths**         — explicit list of ASE-readable structure files.
2. **compositions**  — list of chemical formulas (e.g. ``["Cs2AgBiBr6"]``);
                       expanded into seeds via
                       :func:`matsim_agents.discovery.generate_seeds` (AFLOW
                       prototypes + optional pyXtal random search).
3. **prompt**        — natural-language prompt sent to an LLM; the LLM is
                       asked to return a JSON list of formulas which is
                       then expanded as in (2).

In all cases the final output is a list of ``Path`` objects pointing at
``.vasp`` (or other ASE-supported) seed files on disk. The MD sampler in
:mod:`active_learning.candidates` consumes them unchanged.

Why seed structures (and not relaxed ones)?
-------------------------------------------
The MD sampler immediately heats the structure (Langevin / NVT-Berendsen),
so we deliberately do *not* relax the proposed prototypes first — that
would waste a HydraGNN+ASE relaxation per seed and bias the sampler towards
already-equilibrated configurations. The very first AL iteration is the
natural place to discover that the surrogate disagrees with VASP/QE on
out-of-equilibrium frames.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matsim_agents.active_learning.config import LLMSeedConfig, SeedSourceConfig

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LLM prompting                                                               #
# --------------------------------------------------------------------------- #


_SEED_SYSTEM_PROMPT = """You are a materials-discovery research assistant.

The user describes a chemistry / property target. Your job is to propose a
diverse set of distinct chemical compositions that are scientifically
plausible candidates for the target. The compositions will be turned into
crystal-prototype seed structures and then used as starting points for an
active-learning loop that combines a graph-neural-network surrogate with
DFT (VASP or Quantum ESPRESSO).

REQUIREMENTS:
* Output ONLY a single JSON object with one field, "compositions", whose
  value is a list of reduced chemical formulas (Hill order). Example:
  {"compositions": ["LiCoO2", "LiNi0.8Co0.1Mn0.1O2", "LiFePO4"]}
* Use plain element symbols (no spaces, no LaTeX). Do not invent elements.
* Stoichiometries must be integers (avoid fractional formulas; use the
  closest integer formula instead).
* Aim for the number of compositions the user requests; if unspecified,
  return between 3 and 8.
* Prefer compositions whose stoichiometry matches a known crystal
  prototype in the AFLOW encyclopedia (elements, binary 1:1 / 1:2,
  ternary 1:1:3 perovskite and 1:2:4 spinel, quaternary 1:1:2:6 double
  perovskite, …). Exotic stoichiometries with no prototype match will
  produce no seeds unless ``n_random > 0`` is configured (which then
  requires the optional ``pyxtal`` dependency).
* No prose, no commentary outside the JSON. Do not wrap the JSON in
  Markdown code fences.
"""


def _ask_llm_for_compositions(
    prompt: str,
    llm_cfg: LLMSeedConfig,
    n_target: int,
) -> list[str]:
    """Send ``prompt`` to the LLM and parse a JSON list of formulas back."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from matsim_agents.backends.llm.provider import get_chat_model

    user_msg = f"Target: {prompt.strip()}\nReturn up to {n_target} distinct compositions."
    llm = get_chat_model(
        provider=llm_cfg.provider,
        model=llm_cfg.model,
        base_url=llm_cfg.base_url,
        temperature=llm_cfg.temperature,
    )
    response = llm.invoke(
        [
            SystemMessage(content=_SEED_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    )
    text = response.content if isinstance(response.content, str) else str(response.content)
    return _extract_formula_list(text)


def _extract_formula_list(text: str) -> list[str]:
    """Best-effort: parse a JSON object with a ``compositions`` field."""
    text = text.strip()
    # Strip Markdown code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Greedy extract the first {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        log.warning("LLM seed response did not contain a JSON object: %r", text[:200])
        return []
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        log.warning("Could not parse LLM seed JSON (%s): %r", exc, text[:200])
        return []
    raw = payload.get("compositions") or payload.get("formulas") or []
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


# --------------------------------------------------------------------------- #
# Composition → seed files                                                    #
# --------------------------------------------------------------------------- #


def _compositions_to_seed_files(
    formulas: list[str],
    out_dir: Path,
    *,
    max_phases_per_composition: int,
    n_random: int,
    random_seed: int,
) -> list[Path]:
    """Expand each formula into seed files via :func:`generate_seeds`.

    Per composition we keep ``max_phases_per_composition`` prototype-derived
    seeds plus all ``n_random`` pyXtal random-search seeds (the latter are
    flagged ``needs_dft_verification`` upstream).
    """
    from matsim_agents.discovery import generate_seeds, parse_composition

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_paths: list[Path] = []

    for formula in formulas:
        comp = parse_composition(formula)
        if comp is None:
            log.warning("Skipping unparseable formula: %r", formula)
            continue
        phase_dir = out_dir / comp.formula
        candidates = generate_seeds(
            comp,
            str(phase_dir),
            n_random=n_random,
            random_seed=random_seed,
        )
        if not candidates:
            log.warning("No seeds built for %s", comp.formula)
            continue
        proto = [c for c in candidates if c.source == "prototype"]
        rand = [c for c in candidates if c.source == "random"]
        chosen = proto[: max(1, max_phases_per_composition)] + rand
        for c in chosen:
            seed_paths.append(Path(c.structure_path))
        log.info(
            "%s: kept %d/%d prototype + %d random seeds",
            comp.formula,
            min(len(proto), max(1, max_phases_per_composition)),
            len(proto),
            len(rand),
        )

    return seed_paths


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def resolve_seed_structures(
    cfg: SeedSourceConfig,
    out_dir: Path,
) -> list[Path]:
    """Resolve a :class:`SeedSourceConfig` to a concrete list of seed files.

    Parameters
    ----------
    cfg
        The seed-source configuration block.
    out_dir
        Directory under which generated prototype structures are written
        (one subdirectory per formula). Ignored when ``cfg.kind == 'paths'``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.kind == "paths":
        if not cfg.paths:
            raise ValueError("seed_source.kind='paths' requires a non-empty paths list.")
        missing = [p for p in cfg.paths if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"Seed structures not found: {missing}")
        return [Path(p) for p in cfg.paths]

    if cfg.kind == "compositions":
        if not cfg.compositions:
            raise ValueError(
                "seed_source.kind='compositions' requires a non-empty compositions list."
            )
        return _compositions_to_seed_files(
            list(cfg.compositions),
            out_dir,
            max_phases_per_composition=cfg.max_phases_per_composition,
            n_random=cfg.n_random,
            random_seed=cfg.random_seed,
        )

    if cfg.kind == "prompt":
        if not cfg.prompt or not cfg.llm:
            raise ValueError("seed_source.kind='prompt' requires both 'prompt' and 'llm' fields.")
        n_target = max(1, cfg.max_compositions)
        log.info(
            "Asking LLM (%s/%s) for up to %d compositions for prompt: %r",
            cfg.llm.provider,
            cfg.llm.model,
            n_target,
            cfg.prompt[:120],
        )
        formulas = _ask_llm_for_compositions(cfg.prompt, cfg.llm, n_target)
        if not formulas:
            raise RuntimeError(
                "LLM did not return any usable compositions. "
                "Inspect the model output (rerun with --log-level DEBUG) "
                "or fall back to seed_source.kind='compositions'."
            )
        formulas = formulas[:n_target]
        log.info("LLM proposed %d compositions: %s", len(formulas), formulas)
        # Persist the LLM proposal so the run is reproducible.
        (out_dir / "llm_proposed_compositions.json").write_text(
            json.dumps({"prompt": cfg.prompt, "compositions": formulas}, indent=2)
        )
        return _compositions_to_seed_files(
            formulas,
            out_dir,
            max_phases_per_composition=cfg.max_phases_per_composition,
            n_random=cfg.n_random,
            random_seed=cfg.random_seed,
        )

    raise ValueError(f"Unknown seed_source.kind: {cfg.kind!r}")
