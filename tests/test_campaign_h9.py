"""H9: validated config, resumable planner, CLI and the 10-item synthetic campaign.

These are SOFTWARE tests on synthetic data. They show the planner can detect transfer, detect
failed transfer, respect budgets and resume. They are not evidence of real-alloy transferability.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from matsim_agents.campaign.hierarchy_campaign import (
    CampaignWorkspace,
    compare_acquisition,
    dispatch,
    inventory,
    load_proposals,
    monitor,
    plan,
    review,
    transferability,
)
from matsim_agents.campaign.hierarchy_config import CampaignConfig
from matsim_agents.campaign.registry import NOT_YET_ESTABLISHED
from matsim_agents.campaign.synthetic import SyntheticSpec, make_synthetic_hierarchy

METHOD = {
    "functional": "PBE",
    "pseudopotentials": {e: f"{e}.upf" for e in ("W", "Ta", "Cr", "V", "Hf", "Mo")},
    "ecutwfc_ry": 60.0,
    "ecutrho_ry": 480.0,
    "smearing": "mv",
    "degauss_ry": 0.02,
    "kpts": (1, 1, 1),
    "magnetic_convention": "collinear",
}
SMALL = {"compositions_per_system": 2, "n_orderings": 3}
TARGET = {"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}


def _proposal(cid, interactions=(), hours=8.0, order=2, **kw):
    elements = ["W", "Ta", "Cr", "V", "Hf", "Mo"][:order]
    comp = {e: 1024 // order for e in elements}
    comp[elements[0]] += 1024 - sum(comp.values())
    return {
        "candidate_id": cid,
        "structure_hash": f"hash-{cid}",
        "category": kw.pop("category", "bridges"),
        "composition": comp,
        "interactions": list(interactions),
        "estimated_node_hours": hours,
        **kw,
    }


def _config(tmp_path: Path, **sections) -> dict:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    cfg = {
        "campaign": {"campaign_id": "synthetic-h9"},
        "data": {"candidates": str(tmp_path / "candidates.json")},
        "transfer_evaluation": {
            "experiments": ["2,3,4->5"],
            "n_new": [12],
            "min_eval_rows": 6,
            "required_properties": ["energy"],
        },
        "qe": {
            "method": METHOD,
            "resources": {"nodes": 4, "walltime_hours": 2.0, "memory_gb_per_node": 100.0},
            "limits": {"max_nodes": 8, "max_walltime_hours": 4.0, "node_memory_gb": 512.0},
            "input_ready": True,
        },
        "budget": {"total_node_hours": 100.0},
        "sampling": {"policy": {"batch_size": 3}},
    }
    for key, value in sections.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **value}
        else:
            cfg[key] = value
    return cfg


def _write_candidates(tmp_path: Path, proposals: list[dict]) -> None:
    for p in proposals:
        f = tmp_path / "inputs" / f"{p['candidate_id']}.in"
        f.parent.mkdir(exist_ok=True)
        f.write_text(f"&control\n calculation='scf' ! {p['candidate_id']}\n/\n")
        p["input_path"] = str(f)
    (tmp_path / "candidates.json").write_text(json.dumps(proposals))


def _workspace(tmp_path, proposals, name="ws", live=False, **sections):
    _write_candidates(tmp_path, proposals)
    raw = _config(tmp_path, **sections)
    if live:
        raw["execution"] = {
            "dry_run": False,
            "approved_by": "tester",
            "approved_node_hours": 100.0,
            "max_jobs_per_dispatch": 5,
            **raw.get("execution", {}),
        }
    cfg = CampaignConfig(**raw)
    return CampaignWorkspace(tmp_path / name, cfg), cfg


def _transfer_data(mismatch=False):
    kw = {"mismatched_orders": (2, 3, 4), "mismatch": 4.0} if mismatch else {}
    return make_synthetic_hierarchy(SyntheticSpec(seed=3, orders=(2, 3, 4, 5), **SMALL, **kw))


class FakeScheduler:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, spec):
        self.calls.append(spec.calc_id)
        return f"job-{len(self.calls)}"


# ----------------------------------------------------------------------------- config
def test_defaults_are_a_dry_run_on_the_1024_site_target(tmp_path):
    cfg = CampaignConfig(**_config(tmp_path))
    assert cfg.execution.dry_run is True
    assert cfg.execution.prevent_duplicate_jobs is True
    assert cfg.target.n_sites == 1024
    assert cfg.target.composition == TARGET


def test_unknown_keys_are_rejected_in_every_section(tmp_path):
    for section in ("campaign", "target", "qe", "budget", "execution", "sampling"):
        raw = _config(tmp_path)
        raw.setdefault(section, {})
        raw[section] = {**raw[section], "not_a_switch": True}
        with pytest.raises(ValidationError):
            CampaignConfig(**raw)
    with pytest.raises(ValidationError):
        CampaignConfig(**{**_config(tmp_path), "top_level_typo": 1})


def test_wrong_target_size_or_composition_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="1024"):
        CampaignConfig(**_config(tmp_path, target={"supercell": [4, 4, 4]}))
    with pytest.raises(ValidationError):
        CampaignConfig(**_config(tmp_path, target={"composition": {**TARGET, "W": 641}}))


def test_live_run_needs_an_approver_and_a_bounded_allowance(tmp_path):
    with pytest.raises(ValidationError, match="approved_by"):
        CampaignConfig(**_config(tmp_path, execution={"dry_run": False}))
    with pytest.raises(ValidationError, match="exceeds"):
        CampaignConfig(
            **_config(
                tmp_path,
                execution={"dry_run": False, "approved_by": "me", "approved_node_hours": 1e6},
            )
        )


def test_missing_pseudopotential_is_rejected(tmp_path):
    raw = _config(tmp_path)
    raw["qe"]["method"] = {**METHOD, "pseudopotentials": {"W": "W.upf"}}
    with pytest.raises(ValidationError, match="pseudopotentials"):
        CampaignConfig(**raw)


# ----------------------------------------------------------------------------- planner basics
def test_dry_run_dispatch_submits_nothing(tmp_path):
    ws, _ = _workspace(tmp_path, [_proposal("a", ["pair1:Hf-W"])])
    scheduler = FakeScheduler()
    out = dispatch(
        ws, load_proposals(tmp_path / "candidates.json"), scheduler, approved_by="tester"
    )
    assert scheduler.calls == [] and out.submitted == []
    assert "dry_run" in out.stopped_because


def test_wrong_approver_submits_nothing(tmp_path):
    ws, _ = _workspace(tmp_path, [_proposal("a", ["pair1:Hf-W"])], live=True)
    scheduler = FakeScheduler()
    out = dispatch(
        ws, load_proposals(tmp_path / "candidates.json"), scheduler, approved_by="someone"
    )
    assert scheduler.calls == [] and "approved_by" in out.stopped_because


def test_missing_input_is_input_blocked_not_dispatched(tmp_path):
    ws, _ = _workspace(
        tmp_path, [_proposal("a", ["pair1:Hf-W"])], live=True, qe={"input_ready": False}
    )
    scheduler = FakeScheduler()
    out = dispatch(
        ws, load_proposals(tmp_path / "candidates.json"), scheduler, approved_by="tester"
    )
    assert scheduler.calls == []
    assert any("INPUT_BLOCKED" in why for why in out.blocked.values())


def test_inventory_reuses_datasets_and_flags_incompatible_methods(tmp_path):
    entries = [
        {
            "structure_hash": "h1",
            "composition": {"W": 8, "Ta": 8},
            "local_environments": {"interactions": ["pair1:Ta-W"]},
            "dft_backend": "quantum_espresso",
        }
    ]
    (tmp_path / "ok.json").write_text(json.dumps(entries))
    (tmp_path / "bad.json").write_text(
        json.dumps([{**entries[0], "structure_hash": "h2", "composition": {"W": 8, "Hf": 8}}])
    )
    raw = _config(tmp_path)
    raw["data"]["datasets"] = [
        {"name": "ok", "path": str(tmp_path / "ok.json"), "method": METHOD},
        {
            "name": "bad",
            "path": str(tmp_path / "bad.json"),
            "method": {**METHOD, "functional": "LDA"},
        },
    ]
    _write_candidates(tmp_path, [_proposal("a")])
    ws = CampaignWorkspace(tmp_path / "ws", CampaignConfig(**raw))
    rows = {r["name"]: r for r in inventory(ws)}
    assert rows["ok"]["compatibility"]["compatible"] is True
    assert rows["bad"]["compatibility"]["compatible"] is False
    # the functional-mismatched dataset must not count towards coverage
    from matsim_agents.campaign.hierarchy_campaign import coverage

    assert "pair1:Ta-W" not in coverage(ws)["report"].missing


# ----------------------------------------------------------------------------- the 10-item campaign
def test_01_transferable_lower_dataset_is_validated(tmp_path):
    ws, _ = _workspace(tmp_path, [_proposal("a")])
    runs = transferability(ws, {"energy": _transfer_data()})
    energy = [r for r in runs if r.property == "energy"]
    assert [v.registry_status for r in energy for v in r.verdicts] == ["validated"]
    # properties without data are never inferred from the energy result
    for prop in ("magnetic_moment", "defect_formation_energy"):
        assert {v.registry_status for r in runs if r.property == prop for v in r.verdicts} == {
            NOT_YET_ESTABLISHED
        }


def test_02_useless_lower_dataset_is_not_validated(tmp_path):
    ws, _ = _workspace(tmp_path, [_proposal("a")])
    runs = transferability(ws, {"energy": _transfer_data(mismatch=True)})
    statuses = {v.registry_status for r in runs if r.property == "energy" for v in r.verdicts}
    assert statuses == {"unsupported"}


def test_03_bridge_supplies_a_missing_interaction(tmp_path):
    bridge = _proposal("bridge", ["pair1:Hf-W"], order=6)
    ws, _ = _workspace(tmp_path, [bridge], live=True)
    from matsim_agents.campaign.hierarchy_campaign import coverage

    before = coverage(ws)["report"]
    assert "pair1:Hf-W" in before.missing
    out = dispatch(
        ws, load_proposals(tmp_path / "candidates.json"), FakeScheduler(), approved_by="tester"
    )
    assert out.submitted
    monitor(ws, lambda job: ("COMPLETED", 8.0))
    after = coverage(ws)["report"]
    assert "pair1:Hf-W" not in after.missing
    assert after.fraction > before.fraction


def test_04_policy_identifies_the_useful_data(tmp_path):
    proposals = [
        _proposal("useful", ["pair1:Hf-W", "pair1:Hf-Ta"], hours=8.0),
        _proposal("filler1", [], hours=1.0, relevance=0.0),
        _proposal("filler2", [], hours=1.0, relevance=0.0),
    ]
    ws, _ = _workspace(tmp_path, proposals)
    result = plan(ws, load_proposals(tmp_path / "candidates.json"), acquisition="policy")
    assert result.selected[0] == "useful"
    assert set(result.missing_interactions_targeted) == {"pair1:Hf-W", "pair1:Hf-Ta"}


def test_05_no_repeated_redundant_selection(tmp_path):
    twin_a = _proposal("twin_a", ["pair1:Hf-W"])
    twin_b = {
        **twin_a,
        "candidate_id": "twin_b",
    }  # same structure hash and method: same calculation
    ws, _ = _workspace(tmp_path, [twin_a, twin_b, _proposal("other", ["pair1:Cr-W"])], live=True)
    proposals = load_proposals(tmp_path / "candidates.json")
    first = plan(ws, proposals)
    assert not {"twin_a", "twin_b"} <= set(first.selected)
    scheduler = FakeScheduler()
    dispatch(ws, proposals, scheduler, approved_by="tester")
    again = dispatch(ws, proposals, scheduler, approved_by="tester")
    assert len(scheduler.calls) == len(set(scheduler.calls))
    assert not again.submitted


def test_06_hard_budget_prevents_overspending(tmp_path):
    proposals = [
        _proposal(f"c{i}", [f"pair1:Hf-{e}"], hours=8.0)
        for i, e in enumerate(["W", "Ta", "Cr", "V", "Mo"])
    ]
    ws, _ = _workspace(
        tmp_path,
        proposals,
        live=True,
        budget={"total_node_hours": 20.0},
        execution={"approved_node_hours": 20.0},
    )
    scheduler = FakeScheduler()
    dispatch(ws, load_proposals(tmp_path / "candidates.json"), scheduler, approved_by="tester")
    summary = ws.ledger.summary()
    assert summary.predicted <= 20.0 + 1e-9
    assert summary.remaining >= -1e-9
    assert len(scheduler.calls) * 8.0 <= 20.0


def test_06b_approved_allowance_is_a_second_hard_cap(tmp_path):
    proposals = [
        _proposal(f"c{i}", [f"pair1:Hf-{e}"], hours=8.0) for i, e in enumerate(["W", "Ta", "Cr"])
    ]
    ws, _ = _workspace(
        tmp_path,
        proposals,
        live=True,
        execution={"approved_node_hours": 10.0, "max_jobs_per_dispatch": 5},
    )
    out = dispatch(
        ws, load_proposals(tmp_path / "candidates.json"), FakeScheduler(), approved_by="tester"
    )
    assert len(out.submitted) <= 1
    assert out.stopped_because is None or "approved_node_hours" in out.stopped_because


def test_07_reserved_validation_examples_are_never_selected(tmp_path):
    proposals = [
        _proposal("reserved", ["pair1:Hf-W", "pair1:Hf-Ta", "pair1:Hf-Cr"], relevance=1.0),
        _proposal("normal", ["pair1:Cr-W"]),
    ]
    ws, _ = _workspace(
        tmp_path, proposals, live=True, data={"reserved_validation_ids": ["reserved"]}
    )
    scheduler = FakeScheduler()
    result = plan(ws, load_proposals(tmp_path / "candidates.json"))
    assert "reserved" not in result.selected
    dispatch(ws, load_proposals(tmp_path / "candidates.json"), scheduler, approved_by="tester")
    assert all("reserved" not in call for call in scheduler.calls)
    assert {
        c.candidate_id
        for c in plan(ws, load_proposals(tmp_path / "candidates.json")).calculations
        if c.selected
    } <= {"normal"}


def test_08_interrupted_campaign_resumes_without_duplicate_submissions(tmp_path):
    proposals = [_proposal(f"c{i}", [f"pair1:Hf-{e}"]) for i, e in enumerate(["W", "Ta", "Cr"])]
    ws, cfg = _workspace(tmp_path, proposals, live=True)
    props = load_proposals(tmp_path / "candidates.json")
    submitted: list[str] = []

    def interrupted(spec):
        if len(submitted) == 1:
            raise KeyboardInterrupt  # killed after the intent was written, before the answer
        submitted.append(spec.calc_id)
        return f"job-{len(submitted)}"

    with pytest.raises(KeyboardInterrupt):
        dispatch(ws, props, interrupted, approved_by="tester")
    first_batch = list(submitted)
    resumed = CampaignWorkspace(tmp_path / "ws", cfg)  # a fresh process reading the saved state
    scheduler = FakeScheduler()
    out = dispatch(resumed, props, scheduler, approved_by="tester")
    everything = first_batch + scheduler.calls
    assert len(everything) == len(set(everything))
    assert first_batch and first_batch[0] not in scheduler.calls
    assert out.needs_manual_check, "the unconfirmed submission must be flagged, not retried"
    unconfirmed = out.needs_manual_check[0]
    assert unconfirmed not in scheduler.calls


def test_09_policy_beats_the_deterministic_baseline_on_the_synthetic_case(tmp_path):
    proposals = [
        _proposal("cheap1", [], hours=1.0),
        _proposal("cheap2", [], hours=1.0),
        _proposal("cheap3", [], hours=1.0),
        _proposal("good1", ["pair1:Hf-W", "pair1:Hf-Ta"], hours=6.0),
        _proposal("good2", ["pair1:Cr-W", "pair1:Cr-Ta"], hours=6.0),
    ]
    ws, _ = _workspace(tmp_path, proposals)
    result = compare_acquisition(ws, load_proposals(tmp_path / "candidates.json"))
    assert result["policy_beats_baseline"] is True
    assert result["policy"]["covered"] > result["baseline"]["covered"]


def test_10_failed_transfer_is_detected_and_reported(tmp_path):
    ws, _ = _workspace(tmp_path, [_proposal("a", ["pair1:Hf-W"])])
    transferability(ws, {"energy": _transfer_data(mismatch=True)})
    report, decision = review(ws, load_proposals(tmp_path / "candidates.json"))
    assert report.transfer_verdicts["energy:2,3,4->5"] == "unsupported"
    assert report.transfer_improvement is None
    assert decision.decision != "stop"
    assert 2 not in decision.completed_stages
    assert any("NOT_YET_ESTABLISHED" in v for v in report.scientific_coverage.values())


def test_report_table_has_every_required_field_and_no_invented_evidence(tmp_path):
    ws, _ = _workspace(tmp_path, [_proposal("a", ["pair1:Hf-W"])])
    report, _ = review(ws, load_proposals(tmp_path / "candidates.json"))
    keys = {k for k, _ in report.table()}
    for needed in (
        "datasets reused",
        "subspaces covered",
        "target coverage",
        "missing interactions",
        "proposed lower-order calcs",
        "proposed higher-order calcs",
        "predicted node-hours",
        "actual node-hours",
        "transfer improvement",
        "held-out accuracy (MAE)",
        "magnetic transferability",
        "defect transferability",
        "remaining allocation",
    ):
        assert needed in keys
    table = dict(report.table())
    assert table["transfer improvement"] == NOT_YET_ESTABLISHED
    assert table["held-out accuracy (MAE)"] == NOT_YET_ESTABLISHED
    assert report.magnetic_transferability == "pending"


def test_failed_job_is_charged_and_not_resubmitted(tmp_path):
    ws, _ = _workspace(tmp_path, [_proposal("a", ["pair1:Hf-W"])], live=True)
    props = load_proposals(tmp_path / "candidates.json")
    scheduler = FakeScheduler()
    dispatch(ws, props, scheduler, approved_by="tester")
    monitor(ws, lambda job: ("FAILED", 3.0))
    assert ws.ledger.summary().actual == pytest.approx(3.0)
    dispatch(ws, props, scheduler, approved_by="tester")
    assert len(scheduler.calls) == 1


# ----------------------------------------------------------------------------- CLI
def _cli(tmp_path, proposals=None, **sections):
    from matsim_agents.cli import app

    _write_candidates(tmp_path, proposals or [_proposal("a", ["pair1:Hf-W"])])
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(_config(tmp_path, **sections)))
    return CliRunner(), app, path, tmp_path / "ws"


def test_cli_check_and_unknown_key(tmp_path):
    runner, app, path, _ = _cli(tmp_path)
    ok = runner.invoke(app, ["campaign", "check", str(path)])
    assert ok.exit_code == 0 and "1024 sites" in ok.output and "dry run" in ok.output
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({**_config(tmp_path), "oops": 1}))
    assert runner.invoke(app, ["campaign", "check", str(bad)]).exit_code == 1


def test_cli_preview_cost_review_are_dry_and_approve_refuses(tmp_path):
    runner, app, path, ws = _cli(tmp_path)
    for cmd in ("preview", "cost", "review", "coverage", "inventory", "transferability"):
        result = runner.invoke(app, ["campaign", cmd, str(path), "-w", str(ws)])
        assert result.exit_code == 0, (cmd, result.output)
    review_out = runner.invoke(app, ["campaign", "review", str(path), "-w", str(ws)]).output
    assert NOT_YET_ESTABLISHED in review_out
    approve = runner.invoke(
        app, ["campaign", "approve", str(path), "-w", str(ws), "--approved-by", "tester"]
    )
    assert approve.exit_code == 2 and "dry_run" in approve.output
    assert (
        not (ws / "state.json").exists()
        or not json.loads((ws / "state.json").read_text())["submissions"]
    )
