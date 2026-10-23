"""Benchmark subcommand group: ``matsim-agents benchmark ...``."""

from __future__ import annotations

from pathlib import Path

import typer

from matsim_agents.cli._app import app, console

benchmark_app = typer.Typer(
    add_completion=False,
    help="Grade the workflow against questions with known answers.",
)
app.add_typer(benchmark_app, name="benchmark")

# Source checkouts ship the suite next to src/; installed wheels do not.
DEFAULT_SUITE = Path(__file__).resolve().parents[3] / "benchmarks" / "known_answers"


@benchmark_app.command("known-answers")
def known_answers(
    suite: Path = typer.Argument(
        DEFAULT_SUITE,
        file_okay=False,
        help="Suite directory with cases.yaml (default: the repository's Si/SiC suite).",
    ),
    tier: str = typer.Option(
        "replay",
        help="replay: recorded pw.x outputs, no DFT runs; live: run pw.x for every case.",
    ),
    output: Path = typer.Option(
        Path("known-answer-runs"),
        "--output",
        "-o",
        help="Each invocation runs in a new subdirectory of this directory.",
    ),
    ledger: Path | None = typer.Option(
        None, help="Evidence ledger for evidence and verdicts (default: in --output)."
    ),
    case: list[str] = typer.Option(None, "--case", help="Run only these case IDs."),
):
    """Run the known-answer cases and grade them; exit 1 when any case fails."""

    from matsim_agents.campaign.known_answers import run_known_answers
    from matsim_agents.execution.ledger import EvidenceLedger

    if tier not in {"replay", "live"}:
        raise typer.BadParameter("tier must be 'replay' or 'live'", param_hint="--tier")
    if not (suite / "cases.yaml").is_file():
        raise typer.BadParameter(f"{suite} has no cases.yaml", param_hint="SUITE")
    result = run_known_answers(
        suite,
        tier=tier,  # type: ignore[arg-type]
        output_dir=output,
        ledger=EvidenceLedger(ledger) if ledger else None,
        cases=case or None,
    )
    for verdict in result.verdicts:
        mark = "[green]PASS[/green]" if verdict.passed else "[red]FAIL[/red]"
        console.print(f"{mark} {verdict.case_id} ({verdict.status.value})")
        for reason in verdict.reasons:
            console.print(f"    {reason}")
    passed = sum(verdict.passed for verdict in result.verdicts)
    console.print(f"{passed}/{len(result.verdicts)} known-answer cases passed ({result.tier}).")
    console.print(f"Runs: {result.run_dir}")
    if not result.passed:
        raise typer.Exit(code=1)


DEFAULT_AGENT_SUITE = DEFAULT_SUITE.parent / "agent_eval"


@benchmark_app.command("agent-eval")
def agent_eval(
    config: Path = typer.Argument(
        ...,
        dir_okay=False,
        exists=True,
        help="YAML with participants, arms, rounds, completion_tokens_per_case, decoding, "
        "random_seeds.",
    ),
    suite: Path = typer.Option(
        DEFAULT_AGENT_SUITE,
        file_okay=False,
        help="Suite directory with cases.yaml (default: the repository's Si/SiC claims).",
    ),
    output: Path = typer.Option(
        Path("agent-eval-runs"),
        "--output",
        "-o",
        help="Each invocation runs in a new subdirectory of this directory.",
    ),
    case: list[str] = typer.Option(None, "--case", help="Run only these case IDs."),
):
    """Ask each arm about each claim, grade the answers, and compare the arms."""

    import yaml

    from matsim_agents.campaign.agent_eval import AgentEvalConfig, run_agent_eval

    if not (suite / "cases.yaml").is_file():
        raise typer.BadParameter(f"{suite} has no cases.yaml", param_hint="--suite")
    with open(config, encoding="utf-8") as stream:
        cfg = AgentEvalConfig.model_validate(yaml.safe_load(stream))
    result = run_agent_eval(suite, cfg, output_dir=output, cases=case or None)
    for summary in result.summaries:
        metrics = ", ".join(f"{name} {value:.2f}" for name, value in summary.metrics.items())
        cost = summary.llm_cost
        console.print(
            f"{summary.arm.value}: {metrics}; {cost['calls']:.0f} calls, "
            f"{cost['completion_tokens']:.0f} completion tokens, {cost['latency_s']:.1f} s; "
            f"{summary.numerical_cost['dft_requests']} DFT requests"
        )
    for c in result.comparisons:
        low, high = c.confidence_interval
        console.print(
            f"{c.treatment.value} - {c.control.value} {c.metric}: {c.mean_difference:+.3f} "
            f"[{low:+.3f}, {high:+.3f}], p={c.two_sided_p_value:.3f} ({c.n_pairs} pairs)"
        )
    console.print(f"Runs: {result.run_dir}")
