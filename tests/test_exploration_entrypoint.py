from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from typer.testing import CliRunner

from matsim_agents.backends.mlip.config import HydraGNNConfig
from matsim_agents.backends.mlip.relaxation import RelaxCalculator
from matsim_agents.cli import app
from matsim_agents.discovery.seed_structures import PhaseCandidate
from matsim_agents.execution.contracts import EvidenceLevel, ProvenanceRecord
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.workflows import (
    ExplorationConfig,
    check_exploration,
    load_exploration_config,
    resume_exploration,
    run_exploration,
    summarize_exploration,
)
from matsim_agents.workflows.exploration_reporting import (
    ExplorationSummary,
    write_exploration_report,
)


@pytest.fixture
def config_path(tmp_path):
    checkpoint = tmp_path / "model.model"
    checkpoint.write_bytes(b"test checkpoint, never deserialized")
    path = tmp_path / "explore.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "formula": "Cu",
                "mlip": {
                    "backend": "mace",
                    "family": "checkpoint",
                    "model": "model.model",
                    "device": "cpu",
                    "precision": "fp64",
                },
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "output_dir": "runs",
                "generation": {"pyxtal_count": 0, "max_aflow_candidates": 3},
                "relaxation": {"maxiter": 2},
            }
        )
    )
    return path


@pytest.fixture
def available_dependencies(monkeypatch):
    monkeypatch.setattr(
        "matsim_agents.workflows.exploration_config.importlib.util.find_spec", lambda name: object()
    )


@pytest.fixture
def emt_workflow(monkeypatch, available_dependencies):
    calls = {"loads": 0, "actions": []}

    def generate(action, output_dir):
        calls["actions"].append(action)
        path = Path(output_dir)
        path.mkdir(parents=True)
        candidates = []
        for index in range(action.max_aflow_candidates):
            source = path / f"candidate-{index}.extxyz"
            write(source, bulk("Cu", "fcc", a=3.6))
            candidates.append(
                PhaseCandidate(
                    formula="Cu", structure_path=str(source), prototype_id=f"phase-{index}"
                )
            )
        return candidates

    def calculator(args):
        calls["loads"] += 1
        return RelaxCalculator(EMT(), 0, 0, None)

    monkeypatch.setattr("matsim_agents.workflows.exploration_rounds.generate_seed_action", generate)
    monkeypatch.setattr("matsim_agents.backends.mlip.relaxation._relaxation_calculator", calculator)
    return calls


def test_paths_resolve_relative_to_config_not_cwd(config_path, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = load_exploration_config(config_path)
    assert cfg.mlip.model == str(config_path.parent / "model.model")
    assert cfg.output_dir == config_path.parent / "runs"


def _backend_mlip_block(backend: str, root: Path) -> dict:
    if backend == "hydragnn":
        logdir = root / "hydragnn-logdir"
        logdir.mkdir()
        (logdir / "config.json").write_text("{}")
        (logdir / "best_model.pk").write_bytes(b"test weights, never deserialized")
        mlp = root / "branch.pt"
        mlp.write_bytes(b"test branch weights, never deserialized")
        return {
            "backend": "hydragnn",
            "logdir": "hydragnn-logdir",
            "checkpoint": "best_model.pk",
            "mlp_checkpoint": "branch.pt",
        }
    if backend == "uma":
        return {"backend": "uma", "model_name": "uma-s-1p1", "device": "cpu"}
    return {
        "backend": "mace",
        "family": "mace_mp",
        "model": "medium-omat-0",
        "device": "cpu",
    }


@pytest.mark.parametrize("backend", ["hydragnn", "uma", "mace"])
def test_every_mlip_backend_runs_the_same_exploration(tmp_path, emt_workflow, monkeypatch, backend):
    import matsim_agents.workflows.exploration_rounds as workflow

    requests = []
    original = workflow.relax_with_mlip

    def relax(args, **kwargs):
        requests.append(args)
        return original(args, **kwargs)

    monkeypatch.setattr(workflow, "relax_with_mlip", relax)
    path = tmp_path / "explore.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "formula": "Cu",
                "mlip": _backend_mlip_block(backend, tmp_path),
                "output_dir": "runs",
                "generation": {"pyxtal_count": 0, "max_aflow_candidates": 2},
                "relaxation": {"maxiter": 2},
            }
        )
    )
    cfg = load_exploration_config(path)
    check = check_exploration(cfg)
    assert check.ready, check.errors
    assert check.backend == backend
    assert check.model_identifier.startswith(f"{backend}:")
    summary = run_exploration(cfg)
    assert summary.status == "completed"
    assert {request.mlip.backend for request in requests} == {backend}
    provenance = json.loads((Path(summary.run_directory) / "provenance.json").read_text())
    assert provenance["backend"] == backend
    assert provenance["model_identifier"] == check.model_identifier
    if backend == "hydragnn":
        logdir = tmp_path / "hydragnn-logdir"
        assert requests[0].mlip.logdir == logdir
        assert requests[0].mlip.mlp_checkpoint == tmp_path / "branch.pt"
        assert set(check.model_files) == {
            str(logdir / "config.json"),
            str(logdir / "best_model.pk"),
            str(tmp_path / "branch.pt"),
        }
    elif backend == "uma":
        assert check.model_files == {}
        assert requests[0].mlip.model_name == "uma-s-1p1"
    else:
        assert requests[0].mlip.model == "medium-omat-0"


def test_exploration_requires_an_mlip_block(config_path):
    raw = yaml.safe_load(config_path.read_text())
    raw.pop("mlip")
    with pytest.raises(ValueError, match="mlip"):
        ExplorationConfig.model_validate(raw)


def test_hash_pin_requires_a_local_model_file(tmp_path, available_dependencies):
    cfg = ExplorationConfig(
        formula="Cu",
        mlip={"backend": "uma", "model_name": "uma-s-1p1"},
        checkpoint_sha256="0" * 64,
    )
    result = check_exploration(cfg)
    assert not result.ready
    assert any("requires a local model file" in error for error in result.errors)


@pytest.mark.parametrize(
    "outcome,counts,next_step",
    [
        ("incomplete", {"converged": 0, "nonconverged": 0}, "Resume outstanding attempts"),
        (
            "relaxation_non_convergence",
            {"converged": 0, "nonconverged": 2},
            "force/stress histories",
        ),
        ("usable_minimum", {"converged": 1, "nonconverged": 0}, "Review the ranked structures"),
    ],
)
def test_human_report_distinguishes_outcome_and_recovery(outcome, counts, next_step, tmp_path):
    run = ScientificRunDirectory.create(
        tmp_path,
        workflow="explore",
        request={},
        resolved_config={},
        provenance=ProvenanceRecord(workflow="explore", evidence_level=EvidenceLevel.HYPOTHESIS),
    )
    write_exploration_report(
        run,
        ExplorationSummary(
            run_directory=str(run.path),
            status="failed" if outcome == "incomplete" else "completed",
            outcome=outcome,
            counts=counts,
            termination_reasons={"step_limit": 2},
            errors=["backend failure"],
            rounds_completed=1,
            rounds_requested=3,
        ),
    )
    report = (run.path / "report.md").read_text()
    assert next_step in report
    assert "step_limit: 2" in report
    assert "backend failure" in report
    assert "1/3" in report
    assert "do not establish thermodynamic stability" in report
    assert "No LLM, DFT, or scheduler job is launched" in report


def test_report_write_failure_preserves_original_interruption(
    config_path, available_dependencies, monkeypatch, caplog
):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("requested stop")

    def report_failure(*args, **kwargs):
        raise OSError("report storage unavailable")

    monkeypatch.setattr("matsim_agents.workflows.exploration_rounds._generate_round", interrupt)
    monkeypatch.setattr(
        "matsim_agents.workflows.exploration_rounds.write_exploration_report", report_failure
    )
    with pytest.raises(KeyboardInterrupt, match="requested stop"):
        run_exploration(load_exploration_config(config_path))
    assert "preserving original failure" in caplog.text
    assert "report storage unavailable" in caplog.text


def test_report_write_failure_after_execution_is_not_silenced(
    config_path, emt_workflow, monkeypatch
):
    def report_failure(*args, **kwargs):
        raise OSError("report storage unavailable")

    monkeypatch.setattr(
        "matsim_agents.workflows.exploration_rounds.write_exploration_report", report_failure
    )
    with pytest.raises(OSError, match="report storage unavailable"):
        run_exploration(load_exploration_config(config_path))


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown_option": True},
        {"formula": "not a formula"},
        {"generation": {"pyxtal_count": 0, "max_aflow_candidates": 0}},
        {"generation": {"space_groups": [231]}},
        {"generation": {"unknown_option": 1}},
        {"relaxation": {"fmax": 0}},
        {"relaxation": {"fmax": 0.1}},
        {"relaxation": {"maxiter": 0}},
        {"relaxation": {"max_optimization_seconds": float("inf")}},
        {"relaxation": {"cell_mask": [True] * 6}},
        {"relaxation": {"unknown_option": 1}},
        {"max_relaxations": 0},
    ],
)
def test_rejects_bad_configs(config_path, changes):
    raw = yaml.safe_load(config_path.read_text())
    with pytest.raises(ValueError):
        ExplorationConfig.model_validate({**raw, **changes})


def test_preflight_has_no_scientific_side_effects(config_path, available_dependencies):
    result = check_exploration(load_exploration_config(config_path))
    assert result.ready
    assert (
        result.checkpoint_sha256
        == hashlib.sha256((config_path.parent / "model.model").read_bytes()).hexdigest()
    )
    assert not (config_path.parent / "runs").exists()
    assert any("No model was loaded" in note for note in result.notes)


def test_preflight_reports_wrong_environment_and_missing_checkpoint(config_path, monkeypatch):
    monkeypatch.setattr(
        "matsim_agents.workflows.exploration_config.importlib.util.find_spec", lambda name: None
    )
    cfg = load_exploration_config(config_path)
    Path(cfg.mlip.model).unlink()
    result = check_exploration(cfg)
    assert not result.ready
    assert any("Missing mace" in error for error in result.errors)
    assert any("Required mace model file does not exist" in error for error in result.errors)


def test_hash_mismatch_blocks_execution(config_path, emt_workflow):
    cfg = load_exploration_config(config_path)
    Path(cfg.mlip.model).write_text("changed checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        run_exploration(cfg)
    assert emt_workflow["loads"] == 0
    assert not cfg.output_dir.exists()


def test_run_uses_existing_engine_budget_cache_and_provenance(config_path, emt_workflow):
    cfg = load_exploration_config(config_path)
    summary = run_exploration(cfg)
    assert summary.status == "completed"
    assert summary.counts["generated"] == 3
    assert summary.counts["attempted"] == 2
    assert summary.counts["converged"] == 2
    assert summary.calculator_loads == emt_workflow["loads"] == 1
    assert emt_workflow["actions"][0] == cfg.generation_action().model_copy(
        update={"action_id": "round-0001"}
    )
    run = Path(summary.run_directory)
    assert summarize_exploration(run) == summary
    saved = json.loads((run / "resolved_config.json").read_text())
    assert saved == cfg.model_dump(mode="json")
    state = json.loads((run / "rounds-state.json").read_text())
    assert state["checkpoint_sha256"] == cfg.checkpoint_sha256
    assert summary.rounds_completed == summary.rounds_requested == 1
    assert resume_exploration(run) == summary
    assert emt_workflow["loads"] == 1
    assert (run / "exploration-summary.json").is_file()
    # Repeating the request creates a fresh run, never overwrites prior evidence.
    second = run_exploration(cfg)
    assert second.run_directory != summary.run_directory


def test_full_relaxation_options_reach_calculator(config_path, emt_workflow, monkeypatch):
    import matsim_agents.workflows.exploration_rounds as workflow
    from matsim_agents.workflows.exploration import ExplorationRelaxationConfig

    captured = []
    original = workflow.relax_with_mlip

    def relax(args, **kwargs):
        captured.append(args)
        return original(args, **kwargs)

    monkeypatch.setattr(workflow, "relax_with_mlip", relax)
    cfg = load_exploration_config(config_path)
    cfg.relaxation = ExplorationRelaxationConfig(
        relax_cell=True,
        require_native_stress=True,
        stress_tolerance_eV_per_A3=0.01,
        max_cell_step_strain=0.1,
        max_cell_total_strain=0.5,
        relative_increase_threshold=None,
        max_optimization_seconds=1,
        maxiter=2,
    )
    run_exploration(cfg)
    assert len(captured) == 2
    for request in captured:
        for name, value in cfg.relaxation.model_dump().items():
            assert getattr(request, name) == value
        assert request.mlip == cfg.mlip


def test_backend_errors_are_persisted_and_cli_returns_failure(
    config_path, emt_workflow, monkeypatch
):
    def fail(args):
        raise RuntimeError("model initialization failed")

    monkeypatch.setattr("matsim_agents.backends.mlip.relaxation._relaxation_calculator", fail)
    result = CliRunner().invoke(app, ["explore", "run", str(config_path)])
    assert result.exit_code == 1, result.output
    assert "model initialization failed" in result.output
    assert "backend_errors: 2" in result.output
    run = next((config_path.parent / "runs").iterdir())
    assert summarize_exploration(run).status == "failed"


def test_cli_check_run_and_summary(config_path, emt_workflow):
    runner = CliRunner()
    check = runner.invoke(app, ["explore", "check", str(config_path)])
    assert check.exit_code == 0, check.output
    assert "Preflight: passed" in check.output
    run = runner.invoke(app, ["explore", "run", str(config_path)])
    assert run.exit_code == 0, run.output
    assert "converged: 2" in run.output
    directory = next((config_path.parent / "runs").iterdir())
    summary = runner.invoke(app, ["explore", "summary", str(directory)])
    assert summary.exit_code == 0, summary.output
    assert "does not launch LLM, DFT, or active-learning work." in summary.output
    assert emt_workflow["loads"] == 1


def test_nonconvergence_is_not_presented_as_successful_science(
    config_path, emt_workflow, monkeypatch
):
    from matsim_agents.backends.mlip.results import RelaxationResult

    def unconverged(request, **kwargs):
        output = Path(request.output_dir)
        output.mkdir(parents=True)
        trajectory = output / "test.traj"
        csv = output / "test.csv"
        trajectory.write_text("test trajectory")
        csv.write_text("test csv")
        return RelaxationResult(
            structure_path=request.structure_path,
            optimized_structure_path=request.structure_path,
            trajectory_path=str(trajectory),
            log_csv_path=str(csv),
            final_energy_eV=-1.0,
            final_max_force_eV_per_A=1.0,
            num_steps=2,
            converged=False,
            termination_reason="max_steps",
        )

    monkeypatch.setattr("matsim_agents.workflows.exploration_rounds.relax_with_mlip", unconverged)
    summary = run_exploration(load_exploration_config(config_path))
    assert summary.status == "completed"
    assert summary.counts["converged"] == 0
    assert summary.counts["nonconverged"] == 2
    assert summary.termination_reasons == {"max_steps": 2}
    report = json.loads((Path(summary.run_directory) / "cumulative-ranking.json").read_text())
    assert report is None


def test_generation_failure_keeps_complete_requested_config(config_path, emt_workflow, monkeypatch):
    def fail(action, output_dir):
        raise ValueError("invalid generator geometry")

    monkeypatch.setattr("matsim_agents.workflows.exploration_rounds.generate_seed_action", fail)
    cfg = load_exploration_config(config_path)
    with pytest.raises(ValueError, match="invalid generator geometry"):
        run_exploration(cfg)
    run = next(cfg.output_dir.iterdir())
    saved = json.loads((run / "resolved_config.json").read_text())
    assert saved == cfg.model_dump(mode="json")
    summary = summarize_exploration(run)
    assert summary.status == "failed"
    assert summary.errors == ["ValueError: invalid generator geometry"]


def test_empty_generation_is_a_failure(config_path, emt_workflow, monkeypatch):
    monkeypatch.setattr(
        "matsim_agents.workflows.exploration_rounds.generate_seed_action", lambda *_args: []
    )
    summary = run_exploration(load_exploration_config(config_path))
    assert summary.status == "failed"
    assert summary.counts["generated"] == 0
    assert any("No candidates" in error for error in summary.errors)


def test_cli_bad_config_and_missing_evidence_are_readable(config_path, tmp_path):
    runner = CliRunner()
    config_path.write_text("formula: [")
    result = runner.invoke(app, ["explore", "check", str(config_path)])
    assert result.exit_code == 2
    assert "Configuration error:" in result.output
    result = runner.invoke(app, ["explore", "summary", str(tmp_path)])
    assert result.exit_code == 1
    assert "Cannot read exploration evidence:" in result.output


def test_cli_preflight_failure_is_nonzero(config_path, available_dependencies):
    runner = CliRunner()
    (config_path.parent / "model.model").unlink()
    for action in ("check", "run"):
        result = runner.invoke(app, ["explore", action, str(config_path)])
        assert result.exit_code == 2
        assert "requires model to be an existing file" in result.output


def test_example_contract_is_valid():
    root = Path(__file__).resolve().parents[1]
    cfg = load_exploration_config(root / "examples/workflows/explore-nb2o5.yaml")
    assert cfg.formula == "Nb2O5"
    assert cfg.max_relaxations == 2
    assert cfg.relaxation.fmax == 0.02


def test_public_exploration_imports_remain_available():
    from matsim_agents.workflows import exploration, exploration_config, exploration_reporting

    for name in (
        "ExplorationCheck",
        "ExplorationConfig",
        "ExplorationGenerationConfig",
        "ExplorationRelaxationConfig",
        "check_exploration",
        "load_exploration_config",
    ):
        assert getattr(exploration, name) is getattr(exploration_config, name)
    assert exploration.ExplorationSummary is exploration_reporting.ExplorationSummary
    assert exploration.summarize_exploration is exploration_reporting.summarize_exploration


@pytest.mark.parametrize("action", ["summary", "resume"])
def test_legacy_evidence_is_explicitly_unsupported(tmp_path, action):
    (tmp_path / "provenance.json").write_text("{}")
    (tmp_path / "results.json").write_text('{"legacy": true}')
    result = CliRunner().invoke(app, ["explore", action, str(tmp_path)])
    assert result.exit_code == 1
    assert "Unsupported exploration run format" in result.output
    assert "explore run CONFIG" in result.output


def test_single_round_cli_resumes_interrupted_work(config_path, emt_workflow, monkeypatch):
    import matsim_agents.workflows.exploration_rounds as engine

    original = engine.relax_with_mlip
    attempted = []

    def interrupt(request, **kwargs):
        attempted.append(request.structure_path)
        if len(attempted) == 2:
            raise KeyboardInterrupt("single-round interruption")
        return original(request, **kwargs)

    monkeypatch.setattr(engine, "relax_with_mlip", interrupt)
    cfg = load_exploration_config(config_path)
    progress = []
    with pytest.raises(KeyboardInterrupt):
        run_exploration(cfg, on_progress=progress.append)
    directory = next(cfg.output_dir.iterdir())
    assert progress[0] == f"Run directory: {directory}"
    partial = summarize_exploration(directory)
    assert partial.outcome == "incomplete"
    assert partial.counts["attempted"] == 1
    monkeypatch.setattr(engine, "relax_with_mlip", original)
    output = CliRunner().invoke(app, ["explore", "resume", str(directory)])
    assert output.exit_code == 0, output.output
    summary = summarize_exploration(directory)
    assert summary.rounds_completed == summary.rounds_requested == 1
    assert summary.counts["attempted"] == 2
    assert summary.calculator_loads == emt_workflow["loads"] == 2
    assert len(emt_workflow["actions"]) == 1
    before = {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert resume_exploration(directory) == summary
    assert before == {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def test_hydragnn_exploration_requires_a_pinned_checkpoint(tmp_path, available_dependencies):
    block = _backend_mlip_block("hydragnn", tmp_path)
    block.pop("checkpoint")
    cfg = ExplorationConfig(
        formula="Cu", mlip=HydraGNNConfig(**{**block, "logdir": tmp_path / block["logdir"]})
    )
    check = check_exploration(cfg)
    assert not check.ready
    assert any("set mlip.checkpoint" in error for error in check.errors)


def test_resume_refuses_a_changed_model_file(tmp_path, emt_workflow):
    path = tmp_path / "explore.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "formula": "Cu",
                "mlip": _backend_mlip_block("hydragnn", tmp_path),
                "output_dir": "runs",
                "generation": {"pyxtal_count": 0, "max_aflow_candidates": 2},
                "relaxation": {"maxiter": 2},
            }
        )
    )
    run = Path(run_exploration(load_exploration_config(path)).run_directory)
    (tmp_path / "hydragnn-logdir" / "best_model.pk").write_bytes(b"replaced weights")
    with pytest.raises(ValueError, match="model file changed or missing: .*best_model.pk"):
        resume_exploration(run)


def test_relative_model_paths_are_persisted_absolute(tmp_path, emt_workflow, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    block = _backend_mlip_block("hydragnn", work)
    monkeypatch.chdir(work)
    summary = run_exploration(
        ExplorationConfig(
            formula="Cu",
            mlip=block,
            output_dir="runs",
            generation={"pyxtal_count": 0, "max_aflow_candidates": 2},
            relaxation={"maxiter": 2},
        )
    )
    run = Path(summary.run_directory)
    stored = json.loads((run / "resolved_config.json").read_text())["mlip"]
    assert stored["logdir"] == str(work / "hydragnn-logdir")
    assert stored["mlp_checkpoint"] == str(work / "branch.pt")
    monkeypatch.chdir(tmp_path)
    assert summarize_exploration(run) == summary
    assert resume_exploration(run) == summary
