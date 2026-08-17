#!/usr/bin/env python
"""Demonstrate LLM hypothesis + multi-LLM debate, and how the atomistic MLIP
influences the hypothesis BEFORE vs AFTER active-learning fine-tuning.

This driver exercises the real matsim-agents debate machinery
(:func:`matsim_agents.chat._debate_hypothesis_response`) with a genuine panel of
*different* models drawn from the local model zoo (each served on its own
vLLM endpoint). It does not re-implement the debate prompts.

Pipeline
--------
1. Read the measured MLIP behaviour for one paper case from the completed
   fine-tune-eval artifacts (``eval/iter0.json`` = zero-shot, ``eval/iter<N>.json``
   = active-learning fine-tuned): held-out force MAE (eV/A) and reference-shifted
   energy MAE (meV/atom) for each backend. This is the *evidence* the agent has
   about how trustworthy the surrogate is, and it is the only thing that differs
   between the two conditions.
2. A proposer LLM drafts a scientific hypothesis about the case, given the
   evidence for one condition (pre- or post-fine-tuning).
3. A panel of critic LLMs (different model families) debates and revises the
   hypothesis via the framework's multi-round critique/cross-critique loop.
4. Repeat for both conditions and write the two transcripts plus a side-by-side
   comparison, so the shift attributable to fine-tuning the MLIP is explicit.

The models, endpoints and case are fully configurable so the same driver runs a
cheap smoke test (one small model as both proposer and critic) or the full
multi-family panel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from matsim_agents.chat import DiscoveryChatConfig, _debate_hypothesis_response
from matsim_agents.llm import get_chat_model

# Backends surfaced as evidence, mapped to (zero-shot variant dir, fine-tuned
# variant dir, human label). Both dirs share the same iter0 zero-shot baseline;
# the fine-tuned number is the last iteration of the fine-tuned variant.
_EVIDENCE_BACKENDS = [
    ("hydragnn", "hydragnn-unfrozen", "HydraGNN (full fine-tune)"),
    ("uma", "uma", "UMA (full fine-tune)"),
    ("mace-large", "mace-large", "MACE-MP-large (fine-tune)"),
]


def _read_eval(eval_dir: Path, endpoint: str) -> dict | None:
    """Return the iter0 (zero-shot) or last-iter (fine-tuned) eval JSON."""
    if not eval_dir.is_dir():
        return None
    iters = sorted(
        (int(p.stem[4:]), p)
        for p in eval_dir.glob("iter*.json")
        if p.stem[4:].isdigit()
    )
    if not iters:
        return None
    path = iters[0][1] if endpoint == "before" else iters[-1][1]
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _fmt_metrics(d: dict | None) -> str | None:
    if not d:
        return None
    fmae = d.get("force_mae_eV_per_A")
    emae = d.get("energy_mae_eV_per_atom_shifted")
    if fmae is None and emae is None:
        return None
    parts = []
    if fmae is not None:
        parts.append(f"force MAE {float(fmae):.3f} eV/A")
    if emae is not None:
        parts.append(f"energy MAE {float(emae) * 1000:.0f} meV/atom")
    return ", ".join(parts)


def build_evidence(runs_root: Path, case: str) -> tuple[str, str, dict]:
    """Return (evidence_pre, evidence_post, raw) strings for the case."""
    raw: dict = {"case": case, "before": {}, "after": {}}
    pre_lines, post_lines = [], []
    for zs_dir, ft_dir, label in _EVIDENCE_BACKENDS:
        before = _fmt_metrics(_read_eval(runs_root / zs_dir / case / "eval", "before"))
        after = _fmt_metrics(_read_eval(runs_root / ft_dir / case / "eval", "after"))
        if before:
            pre_lines.append(f"- {label}: {before}")
            raw["before"][label] = before
        if after:
            post_lines.append(f"- {label}: {after}")
            raw["after"][label] = after
    pre = (
        "Surrogate-accuracy evidence on held-out frames for this material "
        "(pretrained models, BEFORE any active-learning fine-tuning):\n"
        + "\n".join(pre_lines)
    )
    post = (
        "Surrogate-accuracy evidence on held-out frames for this material "
        "(AFTER active-learning fine-tuning on loop-collected data):\n"
        + "\n".join(post_lines)
    )
    return pre, post, raw


# Per-case material definitions (element set, stoichiometries, candidate
# crystallographic structures / competing polymorphs). Drawn from the paper AL
# fixtures in examples/paper_cases/*.yaml.
MATERIAL_REGISTRY: dict[str, dict] = {
    "lifepo4-al-001": {
        "pretty": "LiFePO4 olivine cathode",
        "elements": "Li, Fe, P, O",
        "stoichiometry": (
            "the LiFePO4 (1:1:1:4) composition, including partially delithiated "
            "Li_xFePO4 (0 <= x <= 1)"
        ),
        "structures": (
            "the olivine (Pnma) polymorph versus the competing maricite polymorph "
            "and decomposition into Li3PO4 + iron phosphates/oxides"
        ),
    },
    "cantor-fcc-al-001": {
        "pretty": "Cantor high-entropy alloy",
        "elements": "Cr, Mn, Fe, Co, Ni",
        "stoichiometry": (
            "the equimolar CrMnFeCoNi composition and off-equimolar variants "
            "(e.g. Cr2MnFe2CoNi, CrMnFe2CoNi2, CrMnFeCoNi2, CrMn2FeCoNi)"
        ),
        "structures": (
            "a single-phase FCC solid solution versus competing HCP and BCC "
            "orderings / phase separation"
        ),
    },
    "hea-bcc-al-001": {
        "pretty": "refractory BCC high-entropy alloy",
        "elements": "Nb, Ta, V, Hf, Zr, Ti",
        "stoichiometry": (
            "the equimolar NbTaVHfZrTi sextet and its five-element sub-lattice "
            "variants (NbTaVHfZr, NbTaVHfTi, NbTaVZrTi, NbTaHfZrTi, NbVHfZrTi, "
            "TaVHfZrTi)"
        ),
        "structures": (
            "a single-phase BCC (Im-3m) solid solution versus competing FCC "
            "ordering / multi-phase decomposition"
        ),
    },
    "phosphorene-2d-al-001": {
        "pretty": "2D phosphorene monolayer",
        "elements": "P",
        "stoichiometry": "the elemental single-layer phosphorus sheet",
        "structures": (
            "the puckered black-phosphorene (Pmna) allotrope versus the buckled "
            "blue-phosphorene (P-3m1) allotrope"
        ),
    },
    "zn-formate-mof-uma-al-001": {
        "pretty": "zinc-formate metal-organic framework",
        "elements": "Zn, C, O, H",
        "stoichiometry": "the Zn(HCOO)2 framework composition",
        "structures": (
            "the alpha (Pna2_1) framework polymorph versus the chiral beta "
            "(P2_12_12_1) polymorph"
        ),
    },
}


def material_spec(case: str, pretty_override: str | None = None) -> dict:
    spec = dict(
        MATERIAL_REGISTRY.get(
            case,
            {
                "pretty": pretty_override or case,
                "elements": "the specified elements",
                "stoichiometry": "the specified composition",
                "structures": "the candidate crystallographic structures",
            },
        )
    )
    if pretty_override:
        spec["pretty"] = pretty_override
    return spec


_HYPOTHESIS_TASK = (
    "Materials-characterization question for the {pretty} (element set: "
    "{elements}).\n"
    "Formulate a concrete, falsifiable hypothesis that identifies which "
    "crystallographic structure is simultaneously CHEMICALLY stable "
    "(thermodynamically: formation energy on or below the convex hull, with no "
    "driving force to decompose into competing phases) and DYNAMICALLY stable (no "
    "imaginary phonon modes; all real vibrational frequencies) for {stoichiometry}. "
    "The candidate structures to discriminate among are: {structures}.\n"
    "You assess stability with a machine-learning interatomic potential (MLIP) "
    "surrogate that computes relative formation energies, relaxes candidate cells, "
    "and screens vibrational (dynamical) stability. Because competing polymorphs "
    "for such systems are typically separated by only tens of meV/atom, the "
    "surrogate's energy error must be well below that spacing to resolve the stable "
    "structure, and its force error must be small enough to trust phonon signs. "
    "State the numerical stability criteria that would FALSIFY the hypothesis (a "
    "formation-energy / energy-above-hull tolerance in meV/atom and a residual-"
    "force / phonon tolerance in eV/A), decide whether the surrogate is accurate "
    "enough to assert the stable structure directly, or whether ambiguous "
    "structures must be escalated to first-principles DFT and routed back into the "
    "active-learning loop for new labels. Ground the confidence of your stability "
    "verdict in the surrogate-accuracy evidence below.\n\n{evidence}"
)

_PROPOSER_SYSTEM = (
    "You are a computational-materials research partner specializing in crystal-"
    "structure prediction and phase stability (thermodynamic convex-hull and "
    "phonon/dynamical stability analysis). Given a set of elements, candidate "
    "stoichiometries and crystallographic structures, plus quantitative evidence "
    "about a surrogate potential's accuracy, produce a concise, falsifiable "
    "hypothesis about which structure is both chemically and dynamically stable, "
    "with explicit numerical falsification criteria and a clear recommended next "
    "action. Be specific and numerical; do not hedge."
)


def draft_hypothesis(cfg: DiscoveryChatConfig, user_text: str) -> str:
    proposer = get_chat_model(
        provider=cfg.llm_provider, model=cfg.llm_model, base_url=cfg.llm_base_url
    )
    rsp = proposer.invoke(
        [SystemMessage(content=_PROPOSER_SYSTEM), HumanMessage(content=user_text)]
    )
    return rsp.content if isinstance(rsp.content, str) else str(rsp.content)


def run_condition(cfg: DiscoveryChatConfig, spec: dict, evidence: str) -> dict:
    user_text = _HYPOTHESIS_TASK.format(
        pretty=spec["pretty"],
        elements=spec["elements"],
        stoichiometry=spec["stoichiometry"],
        structures=spec["structures"],
        evidence=evidence,
    )
    draft = draft_hypothesis(cfg, user_text)
    final = _debate_hypothesis_response(
        cfg=cfg, user_text=user_text, draft_response=draft
    )
    return {"user_text": user_text, "draft": draft, "final": final}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="cantor-fcc-al-001", choices=list(MATERIAL_REGISTRY))
    p.add_argument("--case-pretty", default=None, help="Override the display name.")
    p.add_argument(
        "--runs-root",
        default=os.environ.get(
            "MATSIM_RUNS_ROOT",
            "/global/cfs/projectdirs/m5216/mlupopa/runs/finetune-eval",
        ),
    )
    p.add_argument("--output-dir", required=True)
    # Proposer.
    p.add_argument("--proposer-model", required=True)
    p.add_argument("--proposer-base-url", required=True)
    # Critic panel (repeatable, paired by position).
    p.add_argument("--critic-model", action="append", default=[])
    p.add_argument("--critic-base-url", action="append", default=[])
    p.add_argument("--provider", default="vllm")
    p.add_argument("--debate-rounds", type=int, default=2)
    p.add_argument("--no-cross-critique", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = DiscoveryChatConfig(
        mlip_backend="uma",  # debate-only session; no relaxation is run here.
        llm_provider=args.provider,
        llm_model=args.proposer_model,
        llm_base_url=args.proposer_base_url,
        enable_hypothesis_debate=True,
        debate_rounds=args.debate_rounds,
        critic_panel_models=list(args.critic_model),
        critic_panel_providers=[args.provider] * len(args.critic_model),
        critic_panel_base_urls=list(args.critic_base_url),
        critic_cross_critique=not args.no_cross_critique,
    )

    spec = material_spec(args.case, args.case_pretty)
    pre, post, raw = build_evidence(Path(args.runs_root), args.case)
    print(f"[evidence] case={args.case} ({spec['pretty']})", flush=True)
    print(pre, flush=True)
    print(post, flush=True)

    print("\n[debate] condition = BEFORE fine-tuning ...", flush=True)
    before = run_condition(cfg, spec, pre)
    print("\n[debate] condition = AFTER fine-tuning ...", flush=True)
    after = run_condition(cfg, spec, post)

    result = {
        "case": args.case,
        "case_pretty": spec["pretty"],
        "material": spec,
        "proposer": args.proposer_model,
        "critics": list(args.critic_model),
        "debate_rounds": args.debate_rounds,
        "cross_critique": not args.no_cross_critique,
        "evidence": raw,
        "before_fine_tuning": before,
        "after_fine_tuning": after,
    }
    (out / "hypothesis_debate_beforeafter.json").write_text(
        json.dumps(result, indent=2)
    )
    (out / "before_final.md").write_text(before["final"])
    (out / "after_final.md").write_text(after["final"])
    print(f"\n[done] wrote transcripts to {out}", flush=True)


if __name__ == "__main__":
    main()
