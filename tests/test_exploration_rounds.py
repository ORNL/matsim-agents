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
from matsim_agents.execution.locking import exclusive_lock
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
        "matsim_agents.workflows.exploration_config.importlib.util.find_spec", lambda name: object()
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
    with (
        exclusive_lock(Path(result.run_directory) / ".exploration.lock", timeout=0),
        pytest.raises(ValueError, match="another process"),
    ):
        resume_exploration(result.run_directory)


@pytest.mark.parametrize("iterations", [1, 3])
def test_exploration_runs_and_resumes_when_cfs_rejects_flock(setup, monkeypatch, iterations):
    cfg, calls, _ = setup
    cfg = cfg.model_copy(update={"iterations": iterations})

    def unavailable(*args, **kwargs):
        raise OSError(524, "CFS does not support flock")

    monkeypatch.setattr(fcntl, "flock", unavailable)
    result = run_exploration(cfg)
    assert result.rounds_completed == iterations
    assert calls["seeds"] == [10 + index for index in range(iterations)]
    assert result.counts["attempted"] == 2 * iterations
    assert not (Path(result.run_directory) / ".exploration.lock").exists()
    assert resume_exploration(result.run_directory) == result
    assert not (Path(result.run_directory) / ".exploration.lock").exists()


def test_exploration_lock_released_after_interruption(setup):
    cfg, _, _ = setup

    def stop(message):
        raise KeyboardInterrupt("before generation")

    with pytest.raises(KeyboardInterrupt, match="before generation"):
        run_exploration(cfg, on_progress=stop)
    directory = next(cfg.output_dir.iterdir())
    assert not (directory / ".exploration.lock").exists()
    assert resume_exploration(directory).rounds_completed == 3


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
    assert "Unsupported exploration run format" in output.output


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


@pytest.fixture
def diverse_setup(setup, monkeypatch):
    cfg, calls, workflow = setup
    cfg.selection = "diverse"

    def generate(action, output_dir):
        calls["seeds"].append(action.random_seed)
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        candidates = []
        for index, kind in enumerate(("fcc", "bcc", "sc")):
            source = path / f"seed-{index}.extxyz"
            write(source, bulk("Cu", kind, a=3.6))
            candidates.append(
                PhaseCandidate(
                    formula="Cu",
                    structure_path=str(source),
                    prototype_id=kind,
                )
            )
        return candidates

    monkeypatch.setattr(workflow, "generate_seed_action", generate)
    return cfg, calls, workflow


def test_diverse_rounds_defer_then_explore_fresh_candidates(diverse_setup):
    cfg, calls, _ = diverse_setup
    result = run_exploration(cfg)
    assert result.status == "completed"
    assert result.counts["generated"] == 9
    assert result.counts["selected"] == result.counts["attempted"] == 3
    assert result.counts["duplicate"] == 5
    assert result.counts["deferred"] == 1
    assert calls["loads"] == 1
    state = json.loads((Path(result.run_directory) / "rounds-state.json").read_text())
    assert [round_["selection"]["selected"] for round_ in state["rounds"]] == [[0, 2], [1], []]
    assert len(state["rounds"][1]["selection"]["references"]) == 4
    assert summarize_exploration(result.run_directory) == result
    before = len(calls["relax"])
    assert resume_exploration(result.run_directory) == result
    assert len(calls["relax"]) == before
    output = CliRunner().invoke(app, ["explore", "summary", result.run_directory])
    assert output.exit_code == 0
    assert "duplicate: 5" in output.output


def test_diverse_resume_uses_saved_nonsequential_selection(diverse_setup, monkeypatch):
    cfg, calls, workflow = diverse_setup

    def interrupt(message):
        if message.startswith("Round 1, attempt 1:"):
            raise KeyboardInterrupt("after checkpoint")

    with pytest.raises(KeyboardInterrupt):
        run_exploration(cfg, on_progress=interrupt)
    directory = next(cfg.output_dir.iterdir())
    state = json.loads((directory / "rounds-state.json").read_text())
    assert state["rounds"][0]["selection"]["selected"] == [0, 2]
    original = workflow.select_diverse_candidates
    selections = []

    def select(*args, **kwargs):
        selections.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow, "select_diverse_candidates", select)
    result = resume_exploration(directory)
    assert result.counts["attempted"] == 3
    assert len(selections) == 2  # only the outstanding rounds
    assert calls["seeds"] == [10, 11, 12]
    assert calls["relax"][0].endswith("round-0001/seeds/seed-0.extxyz")
    assert calls["relax"][1].endswith("round-0001/seeds/seed-2.extxyz")
    assert result.calculator_loads == 2


def test_diverse_failed_attempt_is_not_selected_again(setup, monkeypatch):
    cfg, calls, workflow = setup
    cfg.selection = "diverse"

    def fail(*args, **kwargs):
        calls["relax"].append("failed")
        raise RuntimeError("expected backend failure")

    monkeypatch.setattr(workflow, "relax_with_mlip", fail)
    result = run_exploration(cfg)
    assert result.status == "failed"
    assert result.counts["backend_errors"] == result.counts["attempted"] == 1
    assert result.counts["duplicate"] == 5
    resume_exploration(result.run_directory)
    assert calls["relax"] == ["failed"]


def test_diverse_does_not_use_nonconverged_endpoints(setup, monkeypatch):
    cfg, _, workflow = setup
    cfg.selection = "diverse"
    original = workflow.relax_with_mlip

    def nonconverged(args, **kwargs):
        return original(args, **kwargs).model_copy(update={"converged": False})

    monkeypatch.setattr(workflow, "relax_with_mlip", nonconverged)
    result = run_exploration(cfg)
    state = json.loads((Path(result.run_directory) / "rounds-state.json").read_text())
    references = state["rounds"][1]["selection"]["references"]
    assert references == [state["rounds"][0]["candidates"][0]["structure_path"]]
    assert result.counts["nonconverged"] == 1


@pytest.mark.parametrize(
    "change",
    ["missing", "out-of-range", "budget", "decisions", "attempts", "references", "duplicate"],
)
def test_diverse_journal_integrity(diverse_setup, change):
    cfg, _, _ = diverse_setup
    result = run_exploration(cfg)
    path = Path(result.run_directory) / "rounds-state.json"
    state = json.loads(path.read_text())
    selection = state["rounds"][0]["selection"]
    if change == "missing":
        state["rounds"][0]["selection"] = None
    elif change == "out-of-range":
        selection["selected"] = [0, 3]
    elif change == "budget":
        selection["selected"] = [0, 1, 2]
        selection["decisions"][1]["disposition"] = "selected"
    elif change == "decisions":
        selection["decisions"].pop()
    elif change == "attempts":
        state["rounds"][0]["attempts"][1]["candidate"] = 1
    elif change == "references":
        state["rounds"][1]["selection"]["references"] = []
    else:
        state["rounds"][1]["selection"]["decisions"][0]["duplicate_of"] = "/missing/structure"
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        resume_exploration(result.run_directory)


def test_diverse_supports_one_checkpointed_round(diverse_setup):
    cfg, calls, _ = diverse_setup
    cfg.iterations = 1
    result = run_exploration(cfg)
    assert result.rounds_completed == result.rounds_requested == 1
    assert result.counts["attempted"] == 2
    assert result.counts["deferred"] == 1
    assert resume_exploration(result.run_directory) == result
    assert len(calls["relax"]) == 2


@pytest.mark.parametrize("version", [None, 1, 3])
def test_other_checkpoint_versions_are_rejected_without_migration(setup, version):
    cfg, calls, _ = setup
    result = run_exploration(cfg)
    directory = Path(result.run_directory)
    path = directory / "rounds-state.json"
    state = json.loads(path.read_text())
    if version is None:
        state.pop("schema_version")
    else:
        state["schema_version"] = version
    path.write_text(json.dumps(state))
    before = {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    for action in (summarize_exploration, resume_exploration):
        with pytest.raises(ValueError, match="requires version 2"):
            action(directory)
    assert len(calls["relax"]) == 6
    assert before == {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_one_and_many_rounds_share_first_round_behavior(setup):
    cfg, _, _ = setup
    one = run_exploration(cfg.model_copy(update={"iterations": 1}))
    many = run_exploration(cfg)
    one_state = json.loads((Path(one.run_directory) / "rounds-state.json").read_text())
    many_state = json.loads((Path(many.run_directory) / "rounds-state.json").read_text())
    one_round = one_state["rounds"][0]
    many_round = many_state["rounds"][0]
    for first, second in zip(one_round["attempts"], many_round["attempts"], strict=True):
        assert first["candidate"] == second["candidate"]
        for field in (
            "final_energy_eV",
            "final_max_force_eV_per_A",
            "converged",
            "termination_reason",
        ):
            assert first["result"][field] == second["result"][field]
    for directory in (Path(one.run_directory), Path(many.run_directory)):
        assert all(
            (directory / filename).is_file()
            for filename in (
                "request.json",
                "resolved_config.json",
                "provenance.json",
                "rounds-state.json",
                "cumulative-ranking.json",
                "exploration-summary.json",
            )
        )


def test_converged_endpoint_changes_next_round_selection(diverse_setup, monkeypatch):
    cfg, calls, workflow = diverse_setup
    cfg.max_relaxations = 1
    original = workflow.relax_with_mlip

    def relaxed_to_bcc(args, **kwargs):
        result = original(args, **kwargs)
        if args.structure_path.endswith("seed-0.extxyz"):
            write(result.optimized_structure_path, bulk("Cu", "bcc", a=3.6))
        return result

    monkeypatch.setattr(workflow, "relax_with_mlip", relaxed_to_bcc)
    result = run_exploration(cfg)
    state = json.loads((Path(result.run_directory) / "rounds-state.json").read_text())
    second = state["rounds"][1]["selection"]
    assert second["selected"] == [2]
    assert second["decisions"][1]["disposition"] == "duplicate"
    assert (
        second["decisions"][1]["duplicate_of"]
        == (state["rounds"][0]["attempts"][0]["result"]["optimized_structure_path"])
    )
    assert len(calls["relax"]) == 2


def test_deferred_candidate_is_covered_by_resume_hashes(diverse_setup):
    cfg, _, _ = diverse_setup
    result = run_exploration(cfg)
    directory = Path(result.run_directory)
    state = json.loads((directory / "rounds-state.json").read_text())
    deferred = state["rounds"][0]["candidates"][1]["structure_path"]
    Path(deferred).write_text("changed deferred input")
    with pytest.raises(ValueError, match="missing or changed artifact"):
        resume_exploration(directory)


def test_invalid_candidate_selection_records_incomplete_error(setup, monkeypatch):
    cfg, _, workflow = setup
    cfg.selection = "diverse"
    original = workflow.generate_seed_action

    def invalid(*args, **kwargs):
        candidates = original(*args, **kwargs)
        atoms = bulk("Al", "fcc", a=4.0)
        write(candidates[0].structure_path, atoms)
        return candidates

    monkeypatch.setattr(workflow, "generate_seed_action", invalid)
    with pytest.raises(ValueError, match="composition"):
        run_exploration(cfg)
    summary = summarize_exploration(next(cfg.output_dir.iterdir()))
    assert summary.status == "failed"
    assert summary.outcome == "incomplete"
    assert summary.counts["attempted"] == 0
    assert any("composition" in error for error in summary.errors)


@pytest.mark.parametrize("change", ["delete", "modify", "missing-hash"])
def test_completed_resume_requires_final_ranking_integrity(setup, change):
    cfg, calls, _ = setup
    result = run_exploration(cfg)
    directory = Path(result.run_directory)
    ranking = directory / "cumulative-ranking.json"
    journal = directory / "rounds-state.json"
    state = json.loads(journal.read_text())
    assert str(ranking) in state["artifacts"]
    if change == "delete":
        ranking.unlink()
    elif change == "modify":
        ranking.write_text("{}")
    else:
        del state["artifacts"][str(ranking)]
        journal.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Resume refused"):
        resume_exploration(directory)
    assert len(calls["relax"]) == 6


def test_intermediate_ranking_is_untracked_and_rebuilt(setup, monkeypatch):
    cfg, calls, workflow = setup
    original = workflow._write_ranking

    def interrupt(run, state):
        original(run, state)
        raise KeyboardInterrupt("after intermediate ranking")

    monkeypatch.setattr(workflow, "_write_ranking", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_exploration(cfg)
    directory = next(cfg.output_dir.iterdir())
    ranking = directory / "cumulative-ranking.json"
    state = json.loads((directory / "rounds-state.json").read_text())
    assert not state["complete"]
    assert str(ranking) not in state["artifacts"]
    ranking.write_text("invalid intermediate ranking")
    monkeypatch.setattr(workflow, "_write_ranking", original)
    result = resume_exploration(directory)
    assert result.status == "completed"
    assert len(calls["relax"]) == 6
    assert len(json.loads(ranking.read_text())["ranking"]) == 6


def test_repeated_minima_do_not_produce_polymorph_or_stability_claims(setup):
    cfg, _, _ = setup
    result = run_exploration(cfg)
    ranking = json.loads((Path(result.run_directory) / "cumulative-ranking.json").read_text())
    assert len(ranking["ranking"]) == result.counts["converged"] == 6
    assert "chemically_stable_proxy" not in ranking
    assert "near_degenerate_structure_paths" not in ranking
    assert "degeneracy_tolerance_eV_per_atom" not in ranking
    assert "degeneracy_reference_structure_path" not in ranking
    assert "Ground-state assignment is uncertain" not in ranking["summary"]
    assert "Chemical-stability proxy:" not in ranking["summary"]
    assert "Equivalent relaxed structures may appear more than once" in ranking["summary"]
    assert ranking["ground_state"]["energy_per_atom_eV"] == min(
        phase["energy_per_atom_eV"] for phase in ranking["ranking"]
    )


@pytest.mark.parametrize("stage", ["generation", "attempt", "final-ranking"])
def test_interrupted_hashing_only_publishes_last_committed_checkpoint(setup, monkeypatch, stage):
    cfg, calls, workflow = setup
    original = workflow._remember
    partial_paths = []

    def interrupt(state, filenames):
        is_target = (
            (stage == "generation" and "/seeds/" in filenames[0])
            or (stage == "attempt" and "/attempt-" in filenames[0])
            or (stage == "final-ranking" and filenames[0].endswith("cumulative-ranking.json"))
        )
        if is_target and not partial_paths:
            original(state, filenames[:1])
            partial_paths.append(str(Path(filenames[0]).resolve()))
            raise KeyboardInterrupt("interrupted during output hashing")
        original(state, filenames)

    monkeypatch.setattr(workflow, "_remember", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_exploration(cfg)
    directory = next(cfg.output_dir.iterdir())
    state = json.loads((directory / "rounds-state.json").read_text())
    assert partial_paths[0] not in state["artifacts"]
    assert "KeyboardInterrupt" in state["error"]
    assert not state["complete"]
    if stage == "generation":
        assert state["rounds"] == []
    elif stage == "attempt":
        assert state["rounds"][0]["attempts"] == []
    else:
        assert all(round_["complete"] for round_ in state["rounds"])
    Path(partial_paths[0]).write_text("uncommitted output may be replaced")
    result = resume_exploration(directory)
    assert result.status == "completed"
    assert result.counts["attempted"] == 6
    assert len(calls["relax"]) == (7 if stage == "attempt" else 6)
    assert resume_exploration(directory) == result


def test_interrupted_journal_write_does_not_publish_uncommitted_attempt(setup, monkeypatch):
    cfg, _, workflow = setup
    original = workflow.ScientificRunDirectory.write_json
    interrupted = False

    def write_json(run, name, payload):
        nonlocal interrupted
        if (
            name == "rounds-state.json"
            and isinstance(payload, workflow.RoundState)
            and payload.rounds
            and payload.rounds[0].attempts
            and not interrupted
        ):
            interrupted = True
            raise KeyboardInterrupt("before atomic journal commit")
        return original(run, name, payload)

    monkeypatch.setattr(workflow.ScientificRunDirectory, "write_json", write_json)
    with pytest.raises(KeyboardInterrupt):
        run_exploration(cfg)
    directory = next(cfg.output_dir.iterdir())
    state = json.loads((directory / "rounds-state.json").read_text())
    assert state["rounds"][0]["attempts"] == []
    assert all("/attempt-" not in path for path in state["artifacts"])
    assert resume_exploration(directory).counts["attempted"] == 6
