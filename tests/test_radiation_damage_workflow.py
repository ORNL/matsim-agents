"""End-to-end workflow with a fake launcher. Energies are synthetic fixtures."""

import json
import math

import pytest

from matsim_agents.execution.ledger import EvidenceLedger
from matsim_agents.workflows.radiation.damage import (
    STATE_FILE,
    RadiationDamageConfig,
    RadiationDamageError,
    read_state,
    run_radiation_damage,
)
from tests.test_neb_workflow import _cfg, _fake

COUNTS = {"Mo": 4, "Ta": 4, "W": 8}


def _config(tmp_path, **over):
    base = {
        "counts": COUNTS,
        "seed": 1,
        "qe": _cfg(tmp_path),
        "temperature_K": 900.0,
        "horizon_s": 1.0,
        "n_trajectories": 50,
        "attempt_frequency_hz": 1e13,
    }
    base.update(over)
    return RadiationDamageConfig(**base)


def _forbidden(*args, **kwargs):
    raise AssertionError("must not launch")


def test_dry_run_prepares_and_blocks_kinetics_without_evidence(tmp_path):
    run = run_radiation_damage(_config(tmp_path), tmp_path / "r", launcher=_forbidden)
    state = run.state
    assert state.dry_run
    assert [s.status for s in state.stages] == ["done", "done", "skipped", "blocked"]
    assert run.ensemble is None and run.network_report is None
    assert any("missing, not zero" in b for b in run.blockers)
    assert run.qualification == "not_qualified"
    assert (tmp_path / "r" / "neb" / "neb-workflow.json").is_file()


def test_mocked_dispatch_runs_exploratory_kinetics_never_qualified(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    run = run_radiation_damage(
        _config(tmp_path),
        tmp_path / "r",
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
        ledger=ledger,
    )
    assert [s.status for s in run.state.stages] == ["done"] * 4
    assert run.qualification == "not_qualified"
    assert run.network_report.ready_for_exploratory_kmc
    assert not run.network_report.ready_for_qualified_kmc
    assert run.ensemble.n_trajectories == 50
    assert any("not tested" in b for b in run.blockers)
    assert len(ledger.evidence()) == 1


def test_unrelaxed_endpoints_stop_dispatch_and_record_failure(tmp_path):
    with pytest.raises(RadiationDamageError, match="not relaxed"):
        run_radiation_damage(_config(tmp_path), tmp_path / "r", dispatch=True, launcher=_fake())
    assert read_state(tmp_path / "r").stage("neb_run").status == "failed"


def test_missing_attempt_frequency_blocks_instead_of_defaulting(tmp_path):
    run = run_radiation_damage(
        _config(tmp_path, attempt_frequency_hz=None),
        tmp_path / "r",
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
    )
    assert run.state.stage("kinetics").status == "blocked"
    assert any("attempt frequency" in b for b in run.blockers)
    assert run.ensemble is None


def test_resume_skips_finished_stages_and_is_reproducible(tmp_path):
    config = _config(tmp_path)
    first = run_radiation_damage(config, tmp_path / "r", launcher=_forbidden)
    assert first.state.stage("kinetics").status == "blocked"
    second = run_radiation_damage(
        config,
        tmp_path / "r",
        dispatch=True,
        resume=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
    )
    assert second.state.stage("kinetics").status == "done"
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return _fake()(*args, **kwargs)

    third = run_radiation_damage(
        config, tmp_path / "r", dispatch=True, resume=True, launcher=counting
    )
    assert calls == []
    assert third.ensemble == second.ensemble


def test_same_seed_in_a_new_directory_gives_the_same_ensemble(tmp_path):
    kwargs = {"dispatch": True, "allow_unrelaxed_endpoints": True, "launcher": _fake()}
    a = run_radiation_damage(_config(tmp_path), tmp_path / "a", **kwargs)
    b = run_radiation_damage(_config(tmp_path), tmp_path / "b", **kwargs)
    assert a.ensemble == b.ensemble


def test_existing_directory_needs_resume_and_resume_needs_a_run(tmp_path):
    config = _config(tmp_path)
    run_radiation_damage(config, tmp_path / "r", launcher=_forbidden)
    with pytest.raises(RadiationDamageError, match="resume=True"):
        run_radiation_damage(config, tmp_path / "r", launcher=_forbidden)
    with pytest.raises(RadiationDamageError, match="no run to resume"):
        run_radiation_damage(config, tmp_path / "none", resume=True, launcher=_forbidden)


def test_resume_refuses_a_changed_configuration(tmp_path):
    run_radiation_damage(_config(tmp_path), tmp_path / "r", launcher=_forbidden)
    with pytest.raises(RadiationDamageError, match="differs"):
        run_radiation_damage(
            _config(tmp_path, temperature_K=1000.0),
            tmp_path / "r",
            resume=True,
            launcher=_forbidden,
        )


def test_bad_mover_index_is_reported(tmp_path):
    with pytest.raises(RadiationDamageError, match="mover_index"):
        run_radiation_damage(_config(tmp_path, mover_index=99), tmp_path / "r", launcher=_forbidden)


def test_state_file_is_valid_json_with_schema(tmp_path):
    run_radiation_damage(_config(tmp_path), tmp_path / "r", launcher=_forbidden)
    data = json.loads((tmp_path / "r" / STATE_FILE).read_text())
    assert data["schema"] == "matsim-radiation-damage-run/1"
    assert math.isfinite(len(data["stages"]))
    assert [s["name"] for s in data["stages"]] == [
        "structure",
        "neb_prepare",
        "neb_run",
        "kinetics",
    ]
