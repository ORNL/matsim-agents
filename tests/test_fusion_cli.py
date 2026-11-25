import json

import yaml
from typer.testing import CliRunner

from matsim_agents.cli import app
from matsim_agents.fusion import synthetic

runner = CliRunner()
H = "a" * 64


def _bundle(tmp_path):
    data = {
        "scenario": synthetic.SCENARIO.model_dump(mode="json"),
        "candidates": {
            cid: [r.model_dump(mode="json") for r in recs]
            for cid, recs in synthetic.candidates().items()
        },
        "benchmarks": synthetic.BENCHMARKS.model_dump(mode="json"),
    }
    path = tmp_path / "bundle.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_score_synthetic_json_matches_bundle(tmp_path):
    a = runner.invoke(app, ["materials", "score", "--synthetic", "--json"])
    assert a.exit_code == 0, a.output
    b = runner.invoke(app, ["materials", "score", str(_bundle(tmp_path)), "--json"])
    assert b.exit_code == 0, b.output
    assert json.loads(a.output)["digest"] == json.loads(b.output)["digest"]
    c = next(x for x in json.loads(a.output)["candidates"] if x["candidate_id"] == "C")
    assert c["score"] is None and c["ranking_status"] == "incomplete"


def test_score_report_roundtrip(tmp_path):
    out = tmp_path / "score.json"
    r = runner.invoke(app, ["materials", "score", "--synthetic", "--out", str(out)])
    assert r.exit_code == 0 and "SYNTHETIC" in r.output.upper()
    rep = runner.invoke(app, ["materials", "report", str(out)])
    assert rep.exit_code == 0 and rep.output.strip() == r.output.strip()


def test_validate_pareto_sensitivity_packet(tmp_path):
    for cmd in ("validate", "pareto", "sensitivity"):
        r = runner.invoke(app, ["materials", cmd, "--synthetic", "--json"])
        assert r.exit_code == 0, r.output
        json.loads(r.output)
    out = tmp_path / "packet.json"
    r = runner.invoke(app, ["materials", "packet-export", "--synthetic", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert json.loads(out.read_text())["score_digest"].startswith("fsc-")


def test_needs_exactly_one_source(tmp_path):
    r = runner.invoke(app, ["materials", "score"])
    assert r.exit_code == 1 and "not both or neither" in r.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("scenario: {}\n")
    r = runner.invoke(app, ["materials", "validate", str(bad)])
    assert r.exit_code == 1


def test_import_nuclear_evidence_to_ledger(tmp_path):
    ev = {
        "family": "pka_dpa",
        "external_application": "SPECTRA-PKA",
        "application_version": "1.3",
        "nuclear_data_library": "TENDL-2019",
        "input_sha256": {"in": H},
        "output_sha256": {"out": H},
        "material_id": "W640Ta128Cr64V96Hf32Mo64",
        "irradiation_scenario_id": "demo-fw",
        "quantities": [{"name": "dpa", "value": 1.0, "unit": "dpa"}],
        "dpa_convention": "NRT",
        "synthetic": True,
    }
    path = tmp_path / "ev.json"
    path.write_text(json.dumps(ev))
    ledger = tmp_path / "ledger.jsonl"
    r = runner.invoke(app, ["materials", "import", str(path), "--ledger", str(ledger), "--json"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert out["evidence_id"].startswith("nuc-") and out["ledger_id"]
    assert ledger.read_text().strip()
    ev.pop("dpa_convention")
    path.write_text(json.dumps(ev))
    r = runner.invoke(app, ["materials", "import", str(path)])
    assert r.exit_code == 1 and "dpa_convention" in r.output
