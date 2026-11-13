import fcntl
import json
from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from typer.testing import CliRunner

from matsim_agents.backends.mlip.relaxation import _RelaxCalculator
from matsim_agents.cli import app
from matsim_agents.discovery.seeds import PhaseCandidate
from matsim_agents.workflows import (
    ExplorationConfig,
    resume_exploration,
    run_exploration,
    summarize_exploration,
)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import matsim_agents.workflows.exploration_rounds as workflow

    checkpoint = tmp_path / "model.model"
    checkpoint.write_text("test-only, not deserialized")
    cfg = ExplorationConfig(
        formula="Cu",
        checkpoint=checkpoint,
        output_dir=tmp_path / "runs",
        iterations=3,
        max_relaxations=2,
        generation={"max_aflow_candidates": 2, "pyxtal_count": 0, "random_seed": 10},
        relaxation={"maxiter": 2},
    )
    calls = {"loads": 0, "seeds": [], "relax": []}
    monkeypatch.setattr(
        "matsim_agents.workflows.exploration.importlib.util.find_spec", lambda name: object()
    )

    def generate(action, output_dir):
        calls["seeds"].append(action.random_seed)
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        candidates = []
        for i in range(2):
            source = path / f"seed-{i}.extxyz"
            write(source, bulk("Cu", "fcc", a=3.6))
            candidates.append(
                PhaseCandidate(
                    formula="Cu",
                    structure_path=str(source),
                    prototype_id="fcc",
                )
            )
        return candidates

    def calculator(args):
        calls["loads"] += 1
        return _RelaxCalculator(EMT(), 0, 0, None)

    original = workflow.relax_with_mlip

    def relax(args, **kwargs):
        calls["relax"].append(args.structure_path)
        return original(args, **kwargs)

    monkeypatch.setattr(workflow, "generate_seed_action", generate)
    monkeypatch.setattr(workflow, "relax_with_mlip", relax)
    monkeypatch.setattr("matsim_agents.backends.mlip.relaxation._relaxation_calculator", calculator)
    return cfg, calls, workflow


def test_three_rounds_reuse_one_model_and_rank_cumulatively(setup):
    cfg, calls, _ = setup
    result = run_exploration(cfg)
    assert result.status == "completed"
    assert result.rounds_completed == result.rounds_requested == 3
    assert result.counts["attempted"] == result.counts["converged"] == 6
    assert calls["seeds"] == [10, 11, 12]
    assert calls["loads"] == result.calculator_loads == 1
    ranking = json.loads((Path(result.run_directory) / "cumulative-ranking.json").read_text())
    assert len(ranking["ranking"]) == 6
    assert summarize_exploration(result.run_directory) == result
    before = len(calls["relax"])
    assert resume_exploration(result.run_directory) == result
    assert len(calls["relax"]) == before
    assert calls["loads"] == 1


def test_interrupt_resume_reuses_completed_attempts_and_generation(setup, monkeypatch):
    cfg, calls, workflow = setup
    original = workflow.relax_with_mlip
    count = 0

    def interrupt(args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt("simulated interruption")
        return original(args, **kwargs)

    monkeypatch.setattr(workflow, "relax_with_mlip", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_exploration(cfg)
    directory = next(cfg.output_dir.iterdir())
    partial = summarize_exploration(directory)
    assert partial.status == "failed"
    assert partial.counts["attempted"] == 1
    assert partial.outcome == "incomplete"
    first = calls["relax"][0]
    monkeypatch.setattr(workflow, "relax_with_mlip", original)
    resumed = resume_exploration(directory)
    assert resumed.status == "completed"
    assert resumed.counts["attempted"] == 6
    assert calls["seeds"] == [10, 11, 12]
    assert calls["relax"].count(first) == 1
    assert resumed.calculator_loads == calls["loads"] == 2


def test_backend_failures_consume_budget_and_are_not_retried(setup, monkeypatch):
    cfg, calls, workflow = setup
    original = workflow.relax_with_mlip
    failed = []

    def fail(args, **kwargs):
        if args.structure_path.endswith("seed-0.extxyz"):
            failed.append(args.structure_path)
            raise RuntimeError("backend unavailable")
        return original(args, **kwargs)

    monkeypatch.setattr(workflow, "relax_with_mlip", fail)
    result = run_exploration(cfg)
    assert result.status == "failed"
    assert result.counts["attempted"] == 6
    assert result.counts["backend_errors"] == 3
    resume_exploration(result.run_directory)
    assert len(failed) == 3
    assert calls["loads"] == 1


@pytest.mark.parametrize("change", ["checkpoint", "candidate", "result", "compatibility"])
def test_resume_rejects_changed_inputs_or_runtime(setup, change):
    cfg, _, _ = setup
    summary = run_exploration(cfg)
    directory = Path(summary.run_directory)
    state = json.loads((directory / "rounds-state.json").read_text())
    if change == "checkpoint":
        cfg.checkpoint.write_text("changed")
    elif change == "candidate":
        Path(state["rounds"][0]["candidates"][0]["structure_path"]).write_text("changed")
    elif change == "result":
        Path(state["rounds"][0]["attempts"][0]["result"]["log_csv_path"]).unlink()
    else:
        state["compatibility"]["ase"] = "different"
        (directory / "rounds-state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Resume|resume"):
        resume_exploration(directory)


def test_resume_refuses_active_writer(setup):
    cfg, _, _ = setup
    result = run_exploration(cfg)
    with (Path(result.run_directory) / ".exploration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another process"):
            resume_exploration(result.run_directory)


def test_resume_cli_and_single_batch_compatibility(setup):
    cfg, _, _ = setup
    result = run_exploration(cfg)
    output = CliRunner().invoke(app, ["explore", "resume", result.run_directory])
    assert output.exit_code == 0, output.output
    assert "Rounds: 3/3" in output.output
    legacy = cfg.output_dir / "not-checkpointed"
    legacy.mkdir()
    (legacy / "provenance.json").write_text("{}")
    output = CliRunner().invoke(app, ["explore", "resume", str(legacy)])
    assert output.exit_code == 1
    assert "Cannot resume" in output.output
    assert "iterations > 1" in output.output


def test_per_round_budget(setup):
    cfg, calls, _ = setup
    cfg.max_relaxations = 1
    result = run_exploration(cfg)
    assert result.counts["generated"] == 6
    assert result.counts["attempted"] == 3
    assert calls["loads"] == 1


@pytest.mark.parametrize("interrupt_at", [1, 4])
def test_ranking_interruption_recovers_without_repeating_attempts(setup, monkeypatch, interrupt_at):
    cfg, calls, workflow = setup
    original = workflow._write_ranking

    ranking_calls = 0

    def interrupt(*args):
        nonlocal ranking_calls
        ranking_calls += 1
        if ranking_calls == interrupt_at:
            raise KeyboardInterrupt("ranking interrupted")
        return original(*args)

    monkeypatch.setattr(workflow, "_write_ranking", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_exploration(cfg)
    directory = next(cfg.output_dir.iterdir())
    assert summarize_exploration(directory).rounds_completed == min(interrupt_at, 3)
    first_round_paths = list(calls["relax"])
    monkeypatch.setattr(workflow, "_write_ranking", original)
    result = resume_exploration(directory)
    assert result.status == "completed"
    assert result.counts["attempted"] == 6
    for path in first_round_paths:
        assert calls["relax"].count(path) == 1


def test_nonconverged_attempts_are_never_retried_or_ranked(setup, monkeypatch):
    cfg, calls, workflow = setup
    original = workflow.relax_with_mlip

    def unconverged(args, **kwargs):
        result = original(args, **kwargs)
        return result.model_copy(
            update={
                "converged": False,
                "termination_reason": "optimization_time_limit",
            }
        )

    monkeypatch.setattr(workflow, "relax_with_mlip", unconverged)
    summary = run_exploration(cfg)
    assert summary.status == "completed"
    assert summary.counts["nonconverged"] == 6
    assert summary.termination_reasons == {"optimization_time_limit": 6}
    ranking_path = Path(summary.run_directory) / "cumulative-ranking.json"
    assert json.loads(ranking_path.read_text()) is None
    resume_exploration(summary.run_directory)
    assert len(calls["relax"]) == 6


def test_progress_includes_directory_and_completed_attempts(setup):
    cfg, _, _ = setup
    events = []
    result = run_exploration(cfg, on_progress=events.append)
    assert events[0] == f"Run directory: {result.run_directory}"
    assert len([message for message in events if ", attempt " in message]) == 6


@pytest.mark.parametrize("change", ["configuration", "attempts", "complete", "artifact-coverage"])
def test_resume_rejects_inconsistent_checkpoint(setup, change):
    cfg, _, _ = setup
    result = run_exploration(cfg)
    path = Path(result.run_directory) / "rounds-state.json"
    state = json.loads(path.read_text())
    if change == "configuration":
        state["config"]["relaxation"]["maxiter"] += 1
    elif change == "attempts":
        state["rounds"][0]["attempts"][0]["candidate"] = 1
    elif change == "complete":
        state["rounds"].pop()
    else:
        state["artifacts"] = {}
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        resume_exploration(result.run_directory)
