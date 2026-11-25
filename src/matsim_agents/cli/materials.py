"""Fusion plasma-facing materials scoring: ``matsim-agents materials ...`` (F10).

Reads evidence, scores, and reports. It runs no calculation and changes no evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

materials_app = typer.Typer(
    add_completion=False,
    help="Eight-objective fusion PFC scoring of existing evidence (runs no calculation).",
)
app.add_typer(materials_app, name="materials")

BUNDLE = typer.Argument(
    None,
    exists=True,
    dir_okay=False,
    help="YAML/JSON bundle: scenario, candidates, optional constraints and benchmarks.",
)
POLICY = typer.Option(None, "--policy", help="Policy YAML; default is the built-in policy.")
BENCH = typer.Option(None, "--benchmarks", help="Benchmark YAML (overrides the bundle's).")
SYNTH = typer.Option(False, "--synthetic", help="Use the SYNTHETIC A/B/C demonstration bundle.")
JSON = typer.Option(False, "--json", help="Print JSON instead of text.")


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error[/red]: {message}")
    return typer.Exit(code=1)


def _read(path: Path) -> object:
    import yaml

    return yaml.safe_load(path.read_text())


def _load(bundle: Path | None, policy: Path | None, benchmarks: Path | None, synthetic: bool):
    from pydantic import ValidationError

    from matsim_agents.fusion import synthetic as syn
    from matsim_agents.fusion.policy import builtin_policy, load_policy
    from matsim_agents.fusion.scenario import EvaluationScenario, FusionEvidence
    from matsim_agents.fusion.scoring import BenchmarkSet, load_benchmarks
    from matsim_agents.science.acquisition.selection import Constraint

    if (bundle is None) == (not synthetic):
        raise _fail("give a bundle file or --synthetic, not both or neither")
    try:
        pol = load_policy(policy) if policy else builtin_policy()
        if synthetic:
            scen, cands, cons, bench = syn.SCENARIO, syn.candidates(), [], syn.BENCHMARKS
        else:
            data = _read(bundle)  # type: ignore[arg-type]
            if not isinstance(data, dict):
                raise ValueError("the bundle must be a mapping")
            scen = EvaluationScenario.model_validate(data["scenario"])
            raw = data["candidates"]
            if not isinstance(raw, dict) or not raw:
                raise ValueError("candidates must be a non-empty mapping")
            cands = {
                str(cid): [FusionEvidence.model_validate(r) for r in recs]
                for cid, recs in raw.items()
            }
            cons = [Constraint.model_validate(c) for c in data.get("constraints") or []]
            bench = (
                BenchmarkSet.model_validate(data["benchmarks"]) if "benchmarks" in data else None
            )
        if benchmarks:
            bench = load_benchmarks(benchmarks)
        if bench is None:
            raise ValueError("no benchmarks: put them in the bundle or pass --benchmarks")
    except (ValidationError, KeyError, TypeError, ValueError, OSError) as exc:
        raise _fail(str(exc)) from None
    return pol, bench, scen, cands, cons


def _score(bundle, policy, benchmarks, synthetic):
    from matsim_agents.fusion.scoring import ScoringError, score_candidates

    pol, bench, scen, cands, cons = _load(bundle, policy, benchmarks, synthetic)
    try:
        return pol, score_candidates(pol, bench, scen, cands, constraints=cons)
    except (ScoringError, ValueError) as exc:
        raise _fail(str(exc)) from None


def _emit(payload: dict, text: str, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(text)


@materials_app.command("validate")
def validate(
    bundle: Path | None = BUNDLE,
    policy: Path | None = POLICY,
    benchmarks: Path | None = BENCH,
    synthetic: bool = SYNTH,
    as_json: bool = JSON,
) -> None:
    """Check that the policy, benchmarks, scenario and evidence parse and agree."""
    pol, bench, scen, cands, cons = _load(bundle, policy, benchmarks, synthetic)
    payload = {
        "policy_version": pol.policy_version,
        "policy_digest": pol.digest(),
        "benchmark_version": bench.benchmark_version,
        "synthetic_benchmarks": bench.synthetic,
        "scenario_id": scen.scenario_id,
        "candidates": {k: len(v) for k, v in sorted(cands.items())},
        "constraints": len(cons),
    }
    _emit(payload, "valid: " + ", ".join(f"{k}={v}" for k, v in payload.items()), as_json)


@materials_app.command("score")
def score(
    bundle: Path | None = BUNDLE,
    policy: Path | None = POLICY,
    benchmarks: Path | None = BENCH,
    synthetic: bool = SYNTH,
    out: Path | None = typer.Option(None, "--out", help="Write the scoring result JSON here."),
    as_json: bool = JSON,
) -> None:
    """Score candidates; incomplete candidates are reported, never zero-filled."""
    from matsim_agents.fusion.scoring import render_report

    _, result = _score(bundle, policy, benchmarks, synthetic)
    payload = result.model_dump(mode="json") | {"digest": result.digest()}
    if out:
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    _emit(payload, render_report(result), as_json)


@materials_app.command("report")
def report(
    result_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Score JSON."),
) -> None:
    """Render a saved scoring result."""
    from pydantic import ValidationError

    from matsim_agents.fusion.scoring import ScoringResult, render_report

    try:
        data = json.loads(result_path.read_text())
        data.pop("digest", None)
        result = ScoringResult.model_validate(data)
    except (ValidationError, ValueError, OSError) as exc:
        raise _fail(f"{result_path}: {exc}") from None
    typer.echo(render_report(result))


@materials_app.command("pareto")
def pareto(
    bundle: Path | None = BUNDLE,
    policy: Path | None = POLICY,
    benchmarks: Path | None = BENCH,
    synthetic: bool = SYNTH,
    as_json: bool = JSON,
) -> None:
    """Pareto front and pairwise dominance among complete candidates."""
    _, r = _score(bundle, policy, benchmarks, synthetic)
    payload = {
        "pareto_front": list(r.pareto_front),
        "pareto_authoritative": r.pareto_authoritative,
        "pareto_objectives": list(r.pareto_objectives),
        "dominance": [d.model_dump(mode="json") for d in r.dominance],
    }
    lines = [
        f"Pareto front: {', '.join(r.pareto_front) or 'none'}"
        + ("" if r.pareto_authoritative else " (not authoritative)"),
        *(f"{d.winner} > {d.loser}: {d.dominance.value}" for d in r.dominance),
    ]
    _emit(payload, "\n".join(lines), as_json)


@materials_app.command("sensitivity")
def sensitivity(
    bundle: Path | None = BUNDLE,
    policy: Path | None = POLICY,
    benchmarks: Path | None = BENCH,
    synthetic: bool = SYNTH,
    as_json: bool = JSON,
) -> None:
    """Ranking changes when each weight is scaled."""
    _, r = _score(bundle, policy, benchmarks, synthetic)
    payload = {
        "ranking": list(r.ranking),
        "sensitivity": [s.model_dump(mode="json") for s in r.sensitivity],
    }
    lines = [f"baseline ranking: {', '.join(r.ranking) or 'none'}"] + [
        f"{s.objective} x{s.factor:g}: top {s.top_candidate}, rank changes {s.rank_changes}"
        for s in r.sensitivity
    ]
    _emit(payload, "\n".join(lines), as_json)


@materials_app.command("packet-export")
def packet_export(
    bundle: Path | None = BUNDLE,
    policy: Path | None = POLICY,
    benchmarks: Path | None = BENCH,
    synthetic: bool = SYNTH,
    out: Path = typer.Option(..., "--out", help="Write the evidence packet JSON here."),
) -> None:
    """Export the evidence packet an LLM may read to revise a hypothesis."""
    from matsim_agents.fusion.revision import build_packet

    pol, r = _score(bundle, policy, benchmarks, synthetic)
    packet = build_packet(r, pol)
    out.write_text(packet.canonical_text())
    typer.echo(f"wrote {out} (scores {r.digest()})")


@materials_app.command("import")
def import_nuclear(
    evidence: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="External nuclear evidence JSON/YAML."
    ),
    ledger: Path | None = typer.Option(None, "--ledger", help="Evidence ledger to append to."),
    campaign_id: str | None = typer.Option(None, "--campaign-id"),
    as_json: bool = JSON,
) -> None:
    """Validate external nuclear evidence and optionally append it to a ledger."""
    from pydantic import ValidationError

    from matsim_agents.execution.ledger import EvidenceLedger, record_id
    from matsim_agents.fusion.nuclear_evidence import (
        ExternalNuclearEvidence,
        to_ledger_record,
    )

    try:
        ev = ExternalNuclearEvidence.model_validate(_read(evidence))
    except (ValidationError, ValueError, OSError) as exc:
        raise _fail(f"{evidence}: {exc}") from None
    payload: dict = {
        "evidence_id": ev.evidence_id(),
        "family": ev.family.value,
        "conditions": ev.condition_label(),
        "qualification": ev.qualification_status.value,
        "synthetic": ev.synthetic,
    }
    if ledger:
        rec = EvidenceLedger(ledger).append(to_ledger_record(ev, campaign_id=campaign_id))
        payload["ledger_id"] = record_id(rec)
    _emit(payload, "imported: " + ", ".join(f"{k}={v}" for k, v in payload.items()), as_json)
