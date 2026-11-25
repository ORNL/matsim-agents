from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from matsim_agents.workflows import runner
from matsim_agents.workflows.run_records import find_run, list_runs, load_record

REPO = Path(__file__).resolve().parents[1]


def test_alloy_spec_runs_and_records(tmp_path) -> None:
    spec = tmp_path / "alloy.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "crystal_structure": "bcc",
                "repeats": [2, 2, 2],
                "lattice_parameter": 3.16,
                "ordering": "random",
                "seed": 1,
                "counts": {"W": 12, "Ta": 4},
            }
        )
    )
    directory, record = runner.run(spec, runs_dir=tmp_path / "runs")
    assert record.state == "complete"
    assert record.summary["atoms"] == 16
    assert (directory / "structure.extxyz").is_file()
    assert load_record(directory).run_id == record.run_id
    assert find_run(record.run_id, [tmp_path / "runs"]) == directory.resolve()
    assert [r.run_id for _, r in list_runs([tmp_path / "runs"])] == [record.run_id]
    assert runner.exit_code(record) == 0


def test_inconsistent_alloy_is_recorded_as_failed(tmp_path) -> None:
    spec = tmp_path / "alloy.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "crystal_structure": "bcc",
                "repeats": [2, 2, 2],
                "lattice_parameter": 3.16,
                "counts": {"W": 12},
            }
        )
    )
    with pytest.raises(ValueError):
        runner.validate(spec)


def test_relaxation_with_missing_structure_is_blocked(tmp_path) -> None:
    config = tmp_path / "relax.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "dft",
                "structure_path": str(tmp_path / "absent.cif"),
                "dft": {"backend": "qe", "pseudo_dir": str(tmp_path)},
            }
        )
    )
    _, record = runner.run(config, runs_dir=tmp_path / "runs")
    assert record.state == "blocked"
    assert any("absent.cif" in item for item in record.blockers)
    assert any("approval" in item for item in record.blockers)
    assert runner.exit_code(record) == 3


def test_workflow_wrapper_resolves_to_native_kind() -> None:
    kind, _ = runner._resolve(REPO / "examples/workflows/debate-workflow.yaml")
    assert kind == "debate"


@pytest.mark.parametrize(
    "path",
    [
        "examples/mlip/mace-mp-medium.yaml",
        "examples/fusion_pfc/policy.yaml",
        "examples/mcp/mcp.example.yaml",
    ],
)
def test_auxiliary_inputs_are_refused_with_guidance(path) -> None:
    with pytest.raises(runner.UnrunnableInput, match="not run directly"):
        runner.validate(REPO / path)


def test_unknown_file_is_refused(tmp_path) -> None:
    stray = tmp_path / "stray.yaml"
    stray.write_text("alpha: 1\n")
    with pytest.raises(runner.UnrunnableInput, match="not a recognized"):
        runner.validate(stray)


def test_unset_example_inputs_are_recorded_as_blocked(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ELEMENTAL_REFERENCE_MANIFEST", raising=False)
    source = REPO / "examples/paper_cases/al_hea_fcc_cantor.yaml"
    _, record = runner.run(source, runs_dir=tmp_path)
    assert record.state == "blocked"
    assert any("ELEMENTAL_REFERENCE_MANIFEST" in blocker for blocker in record.blockers)
    assert runner.exit_code(record) == 3


def test_invalid_config_is_recorded_as_failed(tmp_path) -> None:
    source = tmp_path / "al.yaml"
    source.write_text("random_seed: 1\nmd: 3\nacquisition: {}\n")
    _, record = runner.run(source, runs_dir=tmp_path / "runs")
    assert record.state == "failed"
    assert "invalid or incomplete configuration" in (record.failure_reason or "")
    assert runner.exit_code(record) == 1


def test_neb_example_runs_as_dry_run(tmp_path) -> None:
    source = REPO / "examples/radiation_damage/neb_vacancy_hop.example.yaml"
    directory, record = runner.run(source, runs_dir=tmp_path)
    assert record.state == "complete"
    assert record.summary["dry_run"] is True
    assert (directory / "neb").is_dir()


def test_external_campaign_dry_run_does_not_execute(tmp_path) -> None:
    source = REPO / "examples/external/spectra-pka/campaign.yaml"
    _, record = runner.run(source, runs_dir=tmp_path)
    assert record.state in {"complete", "failed"}
    assert record.summary.get("numerical_validation", "NOT_VALIDATED") == "NOT_VALIDATED"


def test_external_execution_is_blocked_when_the_tool_is_missing(tmp_path) -> None:
    source = REPO / "examples/external/spectra-pka/campaign.yaml"
    options = runner.RunOptions(dispatch=True)
    _, record = runner.run(source, runs_dir=tmp_path, options=options)
    assert record.state == "blocked"
    assert any("SPECTRA-PKA" in item for item in record.blockers)


def _application(tmp_path, source, **overrides):
    import yaml

    data = yaml.safe_load((REPO / source).read_text())
    for key in ("policy", "workflow_config"):
        if key in data:
            data[key] = str((REPO / source).parent / data[key])
    data.update(overrides)
    target = tmp_path / "application.yaml"
    target.write_text(yaml.safe_dump(data))
    return target


def test_cantor_application_builds_the_exact_fcc_structure(tmp_path) -> None:
    from ase.io import read

    config = _application(tmp_path, "examples/cantor_fcc_256/application.yaml", mode="generate")
    directory, record = runner.run(config, runs_dir=tmp_path / "runs")
    assert record.state == "complete"
    atoms = read(directory / "structures" / "Co52Cr51Fe51Mn51Ni51.extxyz")
    symbols = atoms.get_chemical_symbols()
    assert len(atoms) == 256
    assert {el: symbols.count(el) for el in set(symbols)} == {
        "Co": 52,
        "Cr": 51,
        "Fe": 51,
        "Mn": 51,
        "Ni": 51,
    }


def test_perovskite_application_generates_deduplicated_candidates(tmp_path) -> None:
    config = _application(tmp_path, "examples/perovskite_oxides/application.yaml", mode="generate")
    _, record = runner.run(config, runs_dir=tmp_path / "runs")
    assert record.state == "complete"
    labels = [item["candidate"] for item in record.summary["structures"]]
    assert len(labels) == len(set(labels)) > 0
    assert {label.split("/")[0] for label in labels} == {
        "SrTiO3",
        "BaTiO3",
        "CaZrO3",
        "LaAlO3",
    }


def test_relax_mode_without_relaxation_config_is_rejected(tmp_path) -> None:
    config = _application(
        tmp_path, "examples/perovskite_oxides/application.yaml", mode="mlip_relax"
    )
    with pytest.raises(ValueError, match="needs a relaxation config"):
        runner.validate(config)


def test_mlip_relax_mode_reports_a_missing_backend_as_blocked(tmp_path) -> None:
    import yaml

    relaxation = tmp_path / "relax.yaml"
    relaxation.write_text(
        yaml.safe_dump({"mode": "mlip", "structure_path": "unused", "mlip": {"backend": "mace"}})
    )
    config = _application(
        tmp_path,
        "examples/perovskite_oxides/application.yaml",
        mode="mlip_relax",
        relaxation=str(relaxation),
    )
    _, record = runner.run(config, runs_dir=tmp_path / "runs")
    assert record.state in {"blocked", "failed"}
    assert record.native_run_directory is None or record.state == "failed"


@pytest.mark.parametrize("command", ["validate", "plan", "run"])
def test_malformed_yaml_is_a_config_error_not_a_traceback(tmp_path, command):
    from typer.testing import CliRunner

    from matsim_agents.cli import app

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n  bad: : :\n")
    args = [command, str(bad)] + (
        ["--runs-dir", str(tmp_path / "runs")] if command == "run" else []
    )
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2, result.output
    assert not isinstance(result.exception, yaml.YAMLError)
    assert "malformed YAML" in result.output


EXAMPLES = sorted((REPO / "examples").rglob("*.y*ml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: str(p.relative_to(REPO)))
def test_every_example_validates_or_names_what_is_missing(path, monkeypatch) -> None:
    for name in ("MODEL_ROOT", "PROJ_ROOT", "RUNS_ROOT", "MACE_CHECKPOINT", "PW_BIN"):
        monkeypatch.delenv(name, raising=False)
    try:
        kind, preflight = runner.validate(path)
    except runner.UnrunnableInput:
        return
    assert kind in runner.HANDLERS
    assert isinstance(preflight.blockers, list)


def test_unset_variables_are_blockers_not_invalid_configs(tmp_path, monkeypatch) -> None:
    from matsim_agents.execution.variables import unset_variables

    for name in ("MODEL_ROOT", "MACE_CHECKPOINT", "ELEMENTAL_REFERENCE_MANIFEST", "PROJ_ROOT"):
        monkeypatch.delenv(name, raising=False)
    for relative, names in {
        "debate/equal-independent-plan-servers.yaml": {"MODEL_ROOT"},
        "active_learning/e3_qe_mace.yaml": {"MACE_CHECKPOINT"},
        "paper_cases/al_zn_formate.yaml": {"ELEMENTAL_REFERENCE_MANIFEST"},
    }.items():
        path = REPO / "examples" / relative
        _, preflight = runner.validate(path)
        assert names <= set(unset_variables(yaml.safe_load(path.read_text())))
        for name in names:
            assert any(name in blocker for blocker in preflight.blockers), (relative, name)
    monkeypatch.setenv("MACE_CHECKPOINT", str(tmp_path / "mace.model"))
    document = yaml.safe_load((REPO / "examples/active_learning/e3_qe_mace.yaml").read_text())
    assert "MACE_CHECKPOINT" not in unset_variables(document)

    run_dir, record = runner.run(
        REPO / "examples/debate/equal-independent-plan-servers.yaml", runs_dir=tmp_path / "runs"
    )
    assert record.state == "blocked" and runner.exit_code(record) == 3
    assert any("MODEL_ROOT" in blocker for blocker in record.blockers)


def test_unset_variable_hints_defaults_and_vars() -> None:
    from matsim_agents.execution.variables import unset_variables

    document = {
        "vars": {"HOME_DIR": "${HOME_DIR:-/data}", "NEEDED": "${NEEDED:?export NEEDED}", "X": "1"},
        "a": ["${HOME_DIR}/a", "${X}", "${NEEDED}", "${ONLY_DEFAULT:-1}", "${PLAIN}"],
    }
    assert unset_variables(document) == {"NEEDED": "export NEEDED", "PLAIN": ""}


def test_missing_stability_reference_file_is_a_blocker() -> None:
    _, preflight = runner.validate(REPO / "examples/characterization/thermodynamic_stability.yaml")
    assert any("reference_energies" in blocker for blocker in preflight.blockers)


def test_investigation_examples_are_blocked_without_a_model(monkeypatch) -> None:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("single_llm.yaml", "multi_llm_debate.yaml"):
        kind, preflight = runner.validate(REPO / "examples" / "investigation" / name)
        assert kind == "investigation"
        assert preflight.blockers


@pytest.mark.parametrize(
    "relative",
    ["evaluation/composition-holdout.yaml", "characterization/fusion_alloy_selection.yaml"],
)
def test_auxiliary_inputs_are_refused_with_their_role(relative) -> None:
    with pytest.raises(runner.UnrunnableInput, match="not run directly"):
        runner.validate(REPO / "examples" / relative)


def test_server_backed_llm_check_is_detected_and_not_blocked_on_a_provider(
    tmp_path, monkeypatch
) -> None:
    # The shape deployments/common/run-llm-check.sh generates: a plan server, no provider/model.
    monkeypatch.setattr(runner, "backend_blockers", lambda names: [f"unavailable: {list(names)}"])
    config = tmp_path / "check.json"
    config.write_text(
        '{"server": "qwen", "output_root": "out", "temperature": 0.0, "timeout_sec": 120,'
        ' "concurrent_requests": 2, "context_length": 8192}'
    )
    kind, preflight = runner.validate(config)
    assert kind == "llm-check"
    assert preflight.summary == {"server": "qwen"}
    assert preflight.blockers == []


def test_llm_check_server_with_provider_is_invalid(tmp_path) -> None:
    config = tmp_path / "check.yaml"
    config.write_text("server: qwen\nprovider: ollama\n")
    with pytest.raises(ValueError, match="remove provider"):
        runner.validate(config)


def _perovskite_relaxation(tmp_path, mlip, mode="mlip_relax", relax_mode="mlip"):
    import yaml

    relaxation = tmp_path / "relax.yaml"
    relaxation.write_text(
        yaml.safe_dump({"mode": relax_mode, "structure_path": "unused", "mlip": mlip})
    )
    return _application(
        tmp_path,
        "examples/perovskite_oxides/application.yaml",
        mode=mode,
        relaxation=str(relaxation),
    )


def test_mlip_without_element_coverage_is_a_blocker(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "backend_blockers", lambda names: [])
    config = _perovskite_relaxation(tmp_path, {"backend": "mace", "family": "mace_off"})
    _, preflight = runner.validate(config)
    assert any(
        "does not cover element(s) Al, Ba, Ca, La, Sr, Ti, Zr" in b for b in preflight.blockers
    )
    _, record = runner.run(config, runs_dir=tmp_path / "runs")
    assert record.state == "blocked"


def test_mlip_covering_every_element_is_not_blocked(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "backend_blockers", lambda names: [])
    config = _perovskite_relaxation(tmp_path, {"backend": "mace", "family": "mace_mp"})
    _, preflight = runner.validate(config)
    assert preflight.blockers == []


def _fake_relaxation(converged_for):
    from matsim_agents.core.contracts import EvidenceLevel
    from matsim_agents.workflows.relaxation import (
        RelaxationMode,
        RelaxationStageResult,
        ScientificRelaxationResult,
    )

    def run_relaxation(cfg):
        converged = converged_for(cfg.structure_path)
        stage = RelaxationStageResult(
            stage="mlip_relaxation",
            backend="mace",
            evidence_level=EvidenceLevel.MLIP_RELAXATION,
            input_structure_path=cfg.structure_path,
            energy_eV=-1.0,
            max_force_eV_per_A=0.01 if converged else 0.5,
            converged=converged,
        )
        return ScientificRelaxationResult(
            run_id="r",
            run_directory=cfg.output_root,
            mode=RelaxationMode.MLIP,
            status="complete",
            stages=[stage],
        )

    return run_relaxation


@pytest.mark.parametrize(
    ("converged", "state"), [(True, "complete"), (False, "partial")], ids=["converged", "partial"]
)
def test_perovskite_mlip_relaxation_records_convergence_and_evidence(
    tmp_path, monkeypatch, converged, state
) -> None:
    from matsim_agents.workflows import relaxation

    monkeypatch.setattr(runner, "backend_blockers", lambda names: [])
    monkeypatch.setattr(relaxation, "run_relaxation", _fake_relaxation(lambda path: converged))
    config = _perovskite_relaxation(tmp_path, {"backend": "mace", "family": "mace_mp"})
    _, record = runner.run(config, runs_dir=tmp_path / "runs")
    assert record.state == state
    results = record.summary["relaxations"]
    labels = [r["candidate"] for r in results]
    assert len(labels) == len(set(labels)) > 0
    assert all(r["converged"] is converged for r in results)
    assert {r["evidence_level"] for r in results} == {"mlip_relaxation"}
    assert record.summary["not_converged"] == (0 if converged else len(results))
    assert "not scientifically qualified" in record.summary["scientific_status"]


def test_dft_mode_rejects_an_mlip_relaxation_config(tmp_path) -> None:
    config = _perovskite_relaxation(
        tmp_path, {"backend": "mace", "family": "mace_mp"}, mode="dft_relax"
    )
    with pytest.raises(ValueError, match="needs relaxation mode in"):
        runner.validate(config)
