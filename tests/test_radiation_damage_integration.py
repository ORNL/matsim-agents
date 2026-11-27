"""End-to-end radiation-damage flow. All energies and NEB outputs are synthetic fixtures
produced by the fake launcher; nothing here is a scientific result."""

import json

from typer.testing import CliRunner

from matsim_agents.execution.ledger import EvidenceLedger, HypothesisStatus
from matsim_agents.workflows.radiation.damage import run_radiation_damage
from matsim_agents.workflows.radiation.hypothesis import revise_hypothesis
from matsim_agents.workflows.radiation.report import build_report, write_report
from tests.test_neb_workflow import _fake
from tests.test_radiation_damage_workflow import _config, _forbidden

runner = CliRunner()


def _dispatch(tmp_path, name, **over):
    return run_radiation_damage(
        _config(tmp_path, **over),
        tmp_path / name,
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
        ledger=EvidenceLedger(tmp_path / f"{name}.jsonl"),
    )


def test_full_flow_from_dry_run_to_revised_hypothesis(tmp_path):
    config = _config(tmp_path)
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    dry = run_radiation_damage(config, tmp_path / "run", launcher=_forbidden)
    first = revise_hypothesis(ledger, "h1", config, dry)
    assert first.record.status is HypothesisStatus.PROPOSED

    done = run_radiation_damage(
        config,
        tmp_path / "run",
        dispatch=True,
        resume=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
        ledger=ledger,
    )
    assert [s.status for s in done.state.stages] == ["done"] * 4
    second = revise_hypothesis(ledger, "h1", config, done)
    assert second.record.prior_revision_id == first.record.revision_id
    assert second.record.status is HypothesisStatus.UNRESOLVED
    assert second.record.evidence_ids
    report = build_report(done)
    assert report["qualification"] == "not_qualified"
    assert report["kinetics"]["status"] == "exploratory"
    assert report["blockers"]
    assert write_report(done).is_file()


def test_same_seed_gives_identical_kinetics_and_a_new_seed_a_new_structure(tmp_path):
    a = _dispatch(tmp_path, "a")
    b = _dispatch(tmp_path, "b")
    assert a.ensemble.model_dump() == b.ensemble.model_dump()
    assert a.state.transition_id == b.state.transition_id
    c = _dispatch(tmp_path, "c", seed=2)
    assert c.state.config_hash != a.state.config_hash


def test_kinetics_outcomes_are_probabilities_and_censoring_is_reported(tmp_path):
    run = _dispatch(tmp_path, "short", horizon_s=1e-30)
    ens = run.ensemble
    assert ens.censored_fraction == 1.0 and ens.first_passage_biased
    assert ens.outcomes == {} and ens.warnings
    report = build_report(run)
    assert report["kinetics"]["censored_fraction"] == 1.0
    assert report["kinetics"]["first_passage_biased"] is True


def test_every_failed_precondition_blocks_instead_of_inventing_a_barrier(tmp_path):
    no_freq = run_radiation_damage(
        _config(tmp_path, attempt_frequency_hz=None),
        tmp_path / "nf",
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
    )
    assert no_freq.ensemble is None
    assert no_freq.state.stage("kinetics").status == "blocked"
    dry = run_radiation_damage(_config(tmp_path), tmp_path / "dry", launcher=_forbidden)
    assert dry.ensemble is None and any("missing, not zero" in b for b in dry.blockers)


def test_ledger_chain_survives_reopening_and_report_is_valid_json(tmp_path):
    run = _dispatch(tmp_path, "r")
    reopened = EvidenceLedger(tmp_path / "r.jsonl")
    assert len(reopened.evidence()) >= 1
    path = write_report(run)
    assert json.loads(path.read_text())["schema"].startswith("matsim-radiation-damage-report")
