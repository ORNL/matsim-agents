"""R9: full-size gas/transmutation QE campaign gates, status and dispatch (no job is run)."""

import json
import re
from pathlib import Path

import pytest

from matsim_agents.execution.budget import BudgetLedger
from matsim_agents.workflows import full_supercell_campaign as fsc
from matsim_agents.workflows.hierarchy.qe_plan import ResourceLimits, ResourceRequest
from matsim_agents.workflows.radiation import gas_supercells as gq

CONFIG = Path(__file__).parents[1] / "deployments/frontier/campaigns/qe-1024/campaign.yaml"
UPF = (
    '<UPF version="2.0.1">\n<PP_INFO/>\n<PP_HEADER element=" {el}" pseudo_type="PAW" '
    'is_paw="true" has_so="false" core_correction="true" functional=" SLA PW PBX PBC" '
    'z_valence="10.0" wfc_cutoff="5.0e1" rho_cutoff="4.0e2"/>\n</UPF>\n'
)
HOST = ("W", "Ta", "Cr", "V", "Hf", "Mo")
EXTRA = {"Re": "Re.fake.UPF", "Os": "Os.fake.UPF", "He": "He.fake.UPF", "H": "H.fake.UPF"}
LIMITS_OK = ResourceLimits(max_nodes=64, max_walltime_hours=24.0, node_memory_gb=512)
LIMITS_DEBUG = ResourceLimits(max_nodes=8, max_walltime_hours=2.0, node_memory_gb=512)
REQUEST = ResourceRequest(nodes=16, walltime_hours=6.0, memory_gb_per_node=400)


def write_pseudos(directory: Path, elements) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    names = {}
    for el in elements:
        name = f"{el}.fake.UPF"
        (directory / name).write_text(UPF.format(el=el))
        names[el] = name
    return names


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    base = fsc.FullSupercellCampaign.from_yaml(CONFIG)
    pdir = tmp_path_factory.mktemp("pseudo")
    host = write_pseudos(pdir, HOST)
    return fsc.FullSupercellCampaign.model_validate(
        {
            **json.loads(base.model_dump_json()),
            "qe": {**base.qe.model_dump(), "pseudo_dir": str(pdir), "pseudopotentials": host},
        }
    )


@pytest.fixture(scope="module")
def examples(cfg):
    return gq.build_gas_examples(cfg)


def test_ten_examples_with_exact_atom_counts(examples, cfg):
    assert len(examples) == 10
    assert len({e.calc_id for e in examples}) == 10
    assert len({e.config_id for e in examples}) == 10
    n = cfg.parent_supercell.expected_parent_atoms
    for e in examples:
        assert e.n_atoms == n - e.n_vacancies + e.n_he + e.n_h
    by = {e.calc_id: e.n_atoms for e in examples}
    assert by["V1"] == 1023
    assert by["He1"] == 1025
    assert by["V1He1"] == 1024
    assert by["V1He2H1"] == 1026
    assert by["Re5"] == 1024


def test_transmutation_examples_keep_the_site_count_and_cap(examples):
    for e in examples:
        if e.n_re or e.n_os:
            symbols = e.atoms.get_chemical_symbols()
            assert symbols.count("Re") == e.n_re
            assert symbols.count("Os") == e.n_os
            assert (e.n_re + e.n_os) / 1024 <= 0.01


def test_missing_gas_and_transmutation_pseudos_block_input(cfg, examples, tmp_path):
    entries = gq.prepare_gas_campaign(
        cfg, tmp_path, git_revision="x", request=REQUEST, limits=LIMITS_OK, examples=examples
    )
    status = {e["calc_id"]: e["status"] for e in entries}
    assert status["V1"] == "READY"
    for name in ("He1", "H1", "V1He1", "V1H1", "V2He2", "V1He2H1", "Re5", "Re3Os2"):
        assert status[name] == "INPUT_BLOCKED"
    blocked = next(e for e in entries if e["calc_id"] == "He1")
    assert blocked["input_sha256"] is None
    assert any("He" in b for b in blocked["blockers"])
    assert not (tmp_path / "He1" / "pw.in").exists()
    assert (tmp_path / "He1" / "structure.extxyz").exists()


def test_manifest_records_required_fields(cfg, examples, tmp_path):
    entries = gq.prepare_gas_campaign(
        cfg, tmp_path, git_revision="abc", request=REQUEST, limits=LIMITS_OK, examples=examples
    )
    v1 = next(e for e in entries if e["calc_id"] == "V1")
    assert re.fullmatch(r"[0-9a-f]{64}", v1["input_sha256"])
    assert v1["config_id"].startswith("gascomplex-")
    assert v1["job_id"] is None
    assert v1["qe_atoms"] == 1023
    assert v1["settings"]["kpts"] == [2, 2, 2]
    assert v1["cost"]["requested_node_hours"] == 96.0
    assert v1["cost"]["actual_node_hours"] is None
    assert v1["numerical_qualification"] == "not_run"
    assert v1["pseudopotential_sha256"].keys() == v1["pseudopotentials"].keys()
    saved = gq.load_gas_manifest(tmp_path)
    assert saved == json.loads(json.dumps(entries))
    text = (tmp_path / "V1" / "pw.in").read_text()
    assert re.search(r"nat\s*=\s*1023", text)


def test_all_ten_ready_when_every_pseudo_exists(cfg, examples, tmp_path):
    pdir = Path(cfg.qe.pseudo_dir)
    extra = write_pseudos(pdir, EXTRA)
    entries = gq.prepare_gas_campaign(
        cfg,
        tmp_path,
        git_revision="x",
        request=REQUEST,
        limits=LIMITS_OK,
        extra_pseudopotentials=extra,
        examples=examples,
    )
    assert {e["status"] for e in entries} == {"READY"}
    assert len({e["input_sha256"] for e in entries}) == 10
    for name in extra.values():
        (pdir / name).unlink()


def test_debug_queue_marks_resource_blocked_and_keeps_full_size_input(cfg, examples, tmp_path):
    entries = gq.prepare_gas_campaign(
        cfg, tmp_path, git_revision="x", request=REQUEST, limits=LIMITS_DEBUG, examples=examples
    )
    v1 = next(e for e in entries if e["calc_id"] == "V1")
    assert v1["status"] == "RESOURCE_BLOCKED"
    assert any("nodes" in b for b in v1["blockers"])
    assert v1["input_sha256"] is not None
    assert re.search(r"nat\s*=\s*1023", (tmp_path / "V1" / "pw.in").read_text())


def test_exhausted_ledger_marks_resource_blocked(cfg, examples, tmp_path):
    ledger = BudgetLedger(total_node_hours=100.0, allocation={"defect": 10.0})
    entries = gq.prepare_gas_campaign(
        cfg,
        tmp_path,
        git_revision="x",
        request=REQUEST,
        limits=LIMITS_OK,
        ledger=ledger,
        examples=examples,
    )
    v1 = next(e for e in entries if e["calc_id"] == "V1")
    assert v1["status"] == "RESOURCE_BLOCKED"
    assert any("node-hours" in b for b in v1["blockers"])


def test_wrong_atom_count_is_rejected(cfg, examples, tmp_path):
    import dataclasses

    bad = dataclasses.replace(examples[0], n_he=1)
    with pytest.raises(gq.GasCampaignError, match="expected"):
        gq.prepare_gas_campaign(
            cfg, tmp_path, git_revision="x", request=REQUEST, limits=LIMITS_OK, examples=[bad]
        )


@pytest.fixture
def prepared(cfg, examples, tmp_path):
    gq.prepare_gas_campaign(
        cfg, tmp_path, git_revision="x", request=REQUEST, limits=LIMITS_OK, examples=examples
    )
    return tmp_path


def make_ledger():
    return BudgetLedger(total_node_hours=10_000.0, allocation={"defect": 5000.0})


def test_dispatch_dry_run_submits_nothing(prepared):
    calls = []
    out = gq.dispatch_gas_campaign(prepared, make_ledger(), lambda e: calls.append(e) or "1")
    assert calls == []
    assert out.submitted == []
    assert "dry_run" in out.stopped_because


def test_dispatch_requires_the_expected_approver(prepared):
    calls = []
    out = gq.dispatch_gas_campaign(
        prepared,
        make_ledger(),
        lambda e: calls.append(e) or "1",
        dry_run=False,
        approved_by="someone",
        expected_approver="owner",
    )
    assert calls == []
    assert "approved_by" in out.stopped_because


def test_dispatch_submits_ready_once_records_job_id_and_never_polls(prepared):
    ledger = make_ledger()
    calls = []

    def submitter(entry):
        calls.append(entry["calc_id"])
        return "4242"

    kw = dict(dry_run=False, approved_by="owner", expected_approver="owner")
    out = gq.dispatch_gas_campaign(prepared, ledger, submitter, **kw)
    assert out.submitted == ["V1"]
    assert set(out.blocked) == {
        "He1",
        "H1",
        "V1He1",
        "V1H1",
        "V2He2",
        "V1He2H1",
        "Re5",
        "Re3Os2",
        "Re2Os1_V1He1",
    }
    saved = {e["calc_id"]: e for e in gq.load_gas_manifest(prepared)}
    assert saved["V1"]["status"] == "SUBMITTED"
    assert saved["V1"]["job_id"] == "4242"
    again = gq.dispatch_gas_campaign(prepared, ledger, submitter, **kw)
    assert again.submitted == []
    assert "V1" in again.skipped_duplicates
    assert calls == ["V1"]
    assert ledger.entries[0].job_id == "4242"
    assert (prepared / gq.GAS_LEDGER_FILE).exists()


def test_submitter_failure_is_not_retried_blindly(prepared):
    def boom(entry):
        raise RuntimeError("sbatch timeout")

    kw = dict(dry_run=False, approved_by="owner", expected_approver="owner")
    out = gq.dispatch_gas_campaign(prepared, make_ledger(), boom, **kw)
    assert out.needs_manual_check == ["V1"]
    saved = {e["calc_id"]: e for e in gq.load_gas_manifest(prepared)}
    assert saved["V1"]["status"] == "NEEDS_MANUAL_CHECK"
    calls = []
    gq.dispatch_gas_campaign(prepared, make_ledger(), lambda e: calls.append(1) or "7", **kw)
    assert calls == []
