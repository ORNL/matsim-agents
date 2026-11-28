"""End-to-end integrated cascade workflow: gate, ensemble, ledger, revision, report."""

import json
from pathlib import Path

import pytest

from matsim_agents.execution.ledger import EvidenceLedger
from matsim_agents.science.cascade.qualification import Level
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
    # Outside the recorded domain the level collapses to UNASSESSED.
    hot = qualified.model_copy(update={"pka": qualified.pka.model_copy(update={"energy_eV": 1e4})})
    assert integ.resolve_gate(hot).level is Level.UNASSESSED


def test_required_realizations_formula():
    assert integ.required_realizations([5.0, 5.0, 5.0], 0.2) == 2  # zero variance
    assert integ.required_realizations([5.0], 0.2) is None
    assert integ.required_realizations([0.0, 0.0], 0.2) is None  # zero mean: inapplicable
    n = integ.required_realizations([4.0, 6.0], 0.1)
    assert n == pytest.approx((integ.Z_95 * (2.0) ** 0.5 / 0.5) ** 2, abs=1), n
