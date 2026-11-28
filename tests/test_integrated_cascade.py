"""End-to-end integrated cascade workflow: gate, ensemble, ledger, revision, report."""

import json
from pathlib import Path

import pytest

from matsim_agents.core.hashing import sha256_file
from matsim_agents.execution.ledger import EvidenceLedger
from matsim_agents.science.cascade.qualification import Level
from matsim_agents.science.irradiation.contracts import (
    IrradiationScenario,
    PKASourceProvenance,
    PKASpectrum,
    RecoilSpecies,
    SpeciesSpectrum,
)
from matsim_agents.workflows import runner
from matsim_agents.workflows.radiation import integrated as integ

CONFIG = Path(__file__).parents[1] / "examples/radiation_damage/integrated_cascade_demo.yaml"


@pytest.fixture(scope="module")
def cfg():
    return integ.IntegratedCascadeConfig.from_yaml(CONFIG)


@pytest.fixture(scope="module")
def report_and_dir(cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("cascade")
    return integ.run_investigation(cfg, out), out


def test_demo_runs_the_full_chain(report_and_dir):
    report, out = report_and_dir
    assert report["classification"] == integ.EXPLORATORY_LABEL
    assert report["gate"]["level"] == "UNASSESSED"
    stats = report["statistics"]
    assert stats["n_run"] >= 2 and stats["n_usable"] <= stats["n_run"]
    assert (out / "gate.json").is_file() and (out / "report.json").is_file()
    assert json.loads((out / "report.json").read_text()) == report
    # Every task record carries the explicit qualification label.
    records = [json.loads(p.read_text()) for p in sorted((out / "tasks").glob("task-*.json"))]
    assert records and all(r["qualification"] == integ.EXPLORATORY_LABEL for r in records)


def test_evidence_and_revision_are_in_a_verifiable_ledger(report_and_dir):
    report, out = report_and_dir
    ledger = EvidenceLedger(out / "ledger.jsonl")
    evidence = ledger.get(report["evidence_id"])
    revision = ledger.get(report["hypothesis_revision_id"])
    assert evidence is not None and revision is not None
    assert f"classification: {integ.EXPLORATORY_LABEL}" in evidence.physical_assumptions
    assert revision.evidence_ids == [report["evidence_id"]]
    assert revision.status.value == report["hypothesis_status"]
    assert report["hypothesis_status"] in ("unresolved", "proposed")


def test_production_is_refused_without_cascade_qualification(cfg, tmp_path):
    production = cfg.model_copy(update={"classification": "production"})
    with pytest.raises(integ.GateRefusal, match="CASCADE_QUALIFIED"):
        integ.run_investigation(production, tmp_path)
    # The refusal is recorded, not silent.
    assert json.loads((tmp_path / "gate.json").read_text())["level"] == "UNASSESSED"


def test_gate_resolves_a_recorded_domain(cfg, tmp_path):
    checks = [
        {"name": n, "passed": True, "criterion": "test", "value": 0.0, "dataset": "synthetic"}
        for level in integ.Level
        if level > Level.UNASSESSED
        for n in __import__(
            "matsim_agents.science.cascade.qualification", fromlist=["REQUIRED_CHECKS"]
        ).REQUIRED_CHECKS[level]
    ]
    record = {
        "checkpoint": cfg.potential.checkpoint,
        "domain": {
            "elements": ["Cu"],
            "temperature_K": [0.0, 600.0],
            "recoil_energy_eV": [0.0, 100.0],
            "n_atoms": [1, 10000],
            "precision": cfg.potential.precision,
            "short_range_model": cfg.potential.short_range,
        },
        "checks": checks,
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    qualified = cfg.model_copy(
        update={"potential": cfg.potential.model_copy(update={"qualification_record": path})}
    )
    assert integ.resolve_gate(qualified).level is Level.CASCADE_QUALIFIED
    production = qualified.model_copy(update={"classification": "production"})
    config_path = tmp_path / "production.json"
    config_path.write_text(json.dumps(production.model_dump(mode="json")))
    kind, preflight = runner.validate(config_path)
    assert kind == "cascade"
    assert any("exploratory only" in blocker for blocker in preflight.blockers)
    # Outside the recorded domain the level collapses to UNASSESSED.
    hot = qualified.model_copy(update={"pka": qualified.pka.model_copy(update={"energy_eV": 1e4})})
    assert integ.resolve_gate(hot).level is Level.UNASSESSED


def test_qualification_rejects_non_boolean_passed(cfg, tmp_path):
    record = {
        "checkpoint": cfg.potential.checkpoint,
        "domain": {
            "elements": ["Cu"],
            "temperature_K": [0.0, 600.0],
            "recoil_energy_eV": [0.0, 100.0],
            "n_atoms": [1, 10000],
            "precision": cfg.potential.precision,
            "short_range_model": cfg.potential.short_range,
        },
        "checks": [
            {"name": "lattice", "passed": "false", "criterion": "test", "dataset": "synthetic"}
        ],
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="boolean 'passed'"):
        integ.load_qualification_record(path)


def test_required_realizations_formula():
    assert integ.required_realizations([5.0, 5.0, 5.0], 0.2) == 2  # zero variance
    assert integ.required_realizations([5.0], 0.2) is None
    assert integ.required_realizations([0.0, 0.0], 0.2) is None  # zero mean: inapplicable
    n = integ.required_realizations([4.0, 6.0], 0.1)
    assert n == pytest.approx((integ.Z_95 * (2.0) ** 0.5 / 0.5) ** 2, abs=1), n


def test_eam_fs_integrated_run_and_restart_identity(cfg, tmp_path):
    potential = tmp_path / "synthetic.fs"
    potential.write_text(
        "synthetic\nsynthetic\nsynthetic\n"
        "1 W\n5 1.0 5 1.0 4.0\n74 183.84 3.14 BCC\n"
        "0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0\n"
    )
    spec = integ.PotentialSpec(
        backend="eam_fs",
        checkpoint=f"sha256:{sha256_file(potential)}",
        potential_file=potential,
    )
    config = cfg.model_copy(
        update={
            "material": integ.MaterialSpec(element="W", lattice_parameter_A=3.14, n_cells=3),
            "potential": spec,
            "phases": integ.PhaseDurations(ballistic_fs=0.1, thermal_spike_fs=0.1, cooling_fs=0.1),
            "ensemble": integ.EnsembleSpec(
                initial_realizations=2, max_realizations=2, relative_ci_half_width=0.5
            ),
        }
    )
    config_path = tmp_path / "cascade.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json")))
    run_dir, run_record = runner.run(config_path, runs_dir=tmp_path / "runs")
    assert run_record.kind == "cascade"
    assert run_record.state in {"complete", "partial"}
    out = run_dir / "cascade"
    report = json.loads((out / "report.json").read_text())
    assert report["classification"] == integ.EXPLORATORY_LABEL
    assert report["potential"]["checkpoint"] == spec.checkpoint
    assert report["statistics"]["n_run"] == 2
    evidence = EvidenceLedger(out / "ledger.jsonl").get(report["evidence_id"])
    assert evidence.method.model == spec.checkpoint
    assert evidence.method.model_sha256 == spec.checkpoint[7:]
    assert evidence.evidence_level.value == "empirical_potential_prediction"
    revision = EvidenceLedger(out / "ledger.jsonl").get(report["hypothesis_revision_id"])
    assert "qualification record" in revision.proposed_tests[0]
    assert revision.falsifiable_prediction is None
    changed = config.model_copy(update={"seed": config.seed + 1})
    with pytest.raises(ValueError, match="different settings"):
        integ.run_investigation(changed, out)
    wrong_species = config.model_copy(
        update={"material": integ.MaterialSpec(element="Cu", lattice_parameter_A=3.615, n_cells=3)}
    )
    with pytest.raises(ValueError, match="does not cover elements"):
        integ.run_investigation(wrong_species, tmp_path / "wrong-species")
    potential.write_text(potential.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        integ.run_investigation(config, tmp_path / "other")


def test_eam_fs_rejects_production_even_if_record_claims_qualification(cfg, tmp_path):
    spec = integ.PotentialSpec(
        backend="eam_fs",
        checkpoint="sha256:" + "0" * 64,
        potential_file=tmp_path / "not-needed.fs",
    )
    config = cfg.model_copy(update={"potential": spec, "classification": "production"})
    with pytest.raises(integ.GateRefusal, match="exploratory only"):
        integ.classify(config, integ.Resolution(Level.CASCADE_QUALIFIED, (), {}))
    with pytest.raises(integ.GateRefusal, match="exploratory only"):
        integ.classify(
            cfg.model_copy(update={"classification": "production"}),
            integ.Resolution(Level.CASCADE_QUALIFIED, (), {}),
        )


def test_ensemble_excludes_unstable_and_boundary_contaminated_tasks(cfg, tmp_path, monkeypatch):
    results = [
        {
            "numerically_stable": True,
            "boundary_ok": True,
            "boundary_findings": [],
            "n_frenkel_pairs": 2,
        },
        {
            "numerically_stable": False,
            "boundary_ok": True,
            "boundary_findings": [],
            "n_frenkel_pairs": 9,
        },
        {
            "numerically_stable": True,
            "boundary_ok": False,
            "boundary_findings": ["boundary contaminated"],
            "n_frenkel_pairs": 8,
        },
    ]

    def run_task(task, calculators, out_dir):
        return {"id": task["id"], **results[task["id"]]}

    monkeypatch.setattr(integ.cascade_study, "run_task", run_task)
    config = cfg.model_copy(
        update={
            "ensemble": integ.EnsembleSpec(
                initial_realizations=3, max_realizations=3, relative_ci_half_width=0.5
            )
        }
    )
    report = integ.run_investigation(config, tmp_path)
    assert report["statistics"]["n_usable"] == 1
    assert report["statistics"]["n_excluded"] == 2
    assert report["statistics"]["mean"] == 2
    assert report["excluded"] == [
        {"id": 1, "reasons": ["numerically unstable"]},
        {"id": 2, "reasons": ["boundary contaminated"]},
    ]
    evidence = EvidenceLedger(tmp_path / "ledger.jsonl").get(report["evidence_id"])
    assert next(v.value for v in evidence.values if v.name == "n_realizations") == 1


def test_cascade_task_failure_is_recorded_as_failed_by_runner(cfg, tmp_path, monkeypatch):
    def fail_task(task, calculators, out_dir):
        raise RuntimeError("synthetic dynamics failure")

    monkeypatch.setattr(integ.cascade_study, "run_task", fail_task)
    config_path = tmp_path / "cascade.json"
    config_path.write_text(json.dumps(cfg.model_dump(mode="json")))
    run_dir, record = runner.run(config_path, runs_dir=tmp_path / "runs")
    assert record.state == "failed"
    assert "synthetic dynamics failure" in record.failure_reason
    assert not (run_dir / "cascade" / "report.json").exists()


def test_stopping_recalculates_target_after_the_last_allowed_task(cfg, tmp_path, monkeypatch):
    values = [3, 7, 5]

    def run_task(task, calculators, out_dir):
        return {
            "id": task["id"],
            "numerically_stable": True,
            "boundary_ok": True,
            "boundary_findings": [],
            "n_frenkel_pairs": values[task["id"]],
        }

    monkeypatch.setattr(integ.cascade_study, "run_task", run_task)
    config = cfg.model_copy(
        update={
            "ensemble": integ.EnsembleSpec(
                initial_realizations=2, max_realizations=3, relative_ci_half_width=0.5
            )
        }
    )
    statistics = integ.run_investigation(config, tmp_path)["statistics"]
    assert statistics["n_run"] == 3
    assert statistics["required_realizations_95"] == integ.required_realizations(values, 0.5)
    assert statistics["stopping"] == "target_met"


def test_saved_channel_spectrum_runs_through_runner_and_ledger(cfg, tmp_path):
    isotope = RecoilSpecies(element="Cu", mass_number=63)
    spectrum = PKASpectrum(
        scenario=IrradiationScenario(
            scenario_id="synthetic",
            parent_alloy_id="synthetic-cu",
            composition={"Cu": 1.0},
            neutron_spectrum_identity="synthetic",
            flux_normalization="synthetic",
        ),
        spectra=(
            SpeciesSpectrum(
                species=isotope,
                parent=isotope,
                bin_low_mev=(1e-6,),
                bin_high_mev=(2e-6,),
                rates=(1.0,),
            ),
        ),
        rate_unit="synthetic",
        provenance=PKASourceProvenance(source="synthetic", nuclear_data_identity="synthetic"),
    )
    source = integ.DisplacementSpectrum(channel="elastic", spectrum=spectrum)
    source_path = tmp_path / "spectrum.json"
    source_path.write_text(source.model_dump_json())
    config = cfg.model_copy(
        update={
            "pka": integ.PKASpec(spectrum_file=source_path, temperature_K=100.0),
            "phases": integ.PhaseDurations(ballistic_fs=0.1, thermal_spike_fs=0.1, cooling_fs=0.1),
            "ensemble": integ.EnsembleSpec(
                initial_realizations=2, max_realizations=2, relative_ci_half_width=0.5
            ),
        }
    )
    config_path = tmp_path / "cascade.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json")))
    run_dir, record = runner.run(config_path, runs_dir=tmp_path / "runs")
    assert record.state in {"complete", "partial"}, record.failure_reason
    out = run_dir / "cascade"
    report = json.loads((out / "report.json").read_text())
    assert report["gate"]["level"] == "UNASSESSED"
    assert report["pka_source"]["sha256"] == sha256_file(source_path)
    assert report["pka_source"]["channel"] == "elastic"
    tasks = [json.loads(p.read_text()) for p in (out / "tasks").glob("task-*.json")]
    assert len(tasks) == 2
    assert all(t["recoil_species"] == isotope.key and 1 <= t["energy_eV"] <= 2 for t in tasks)
    assert all(
        t["event_probability_basis"] == "sampled_species_energy_bin_probability" for t in tasks
    )
    evidence = EvidenceLedger(out / "ledger.jsonl").get(report["evidence_id"])
    assert evidence.method.settings["spectrum_sha256"] == sha256_file(source_path)
    assert evidence.input_sha256["pka_spectrum"] == sha256_file(source_path)

    source_path.write_text(source.model_dump_json() + "\n")
    with pytest.raises(ValueError, match="different settings"):
        integ.run_investigation(config, out)

    ambiguous = source.model_copy(
        update={
            "spectrum": spectrum.model_copy(
                update={"spectra": (spectrum.spectra[0].model_copy(update={"parent": None}),)}
            )
        }
    )
    source_path.write_text(ambiguous.model_dump_json())
    with pytest.raises(ValueError, match="unchanged isotope"):
        integ.run_investigation(config, tmp_path / "ambiguous")

    source_path.write_text(spectrum.model_dump_json())
    with pytest.raises(ValueError, match="channel"):
        integ.run_investigation(config, tmp_path / "no-channel")
