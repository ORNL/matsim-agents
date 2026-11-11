import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from benchmarks.e3 import run
from matsim_agents.execution.ledger import EvidenceLedger

REPO = Path(__file__).resolve().parents[1]
OBSOLETE = re.compile(r"\b[Pp]olicy[ _-]?[CD]\b|\bC-vs-D\b|\bC-versus-D\b|\bC/D\b")


def _pool(n=20):
    entries = []
    for i in range(n):
        atoms = Atoms(
            "Nb2", positions=[[0, 0, 0], [1.6 + 0.01 * i, 0, 0]], cell=[5, 5, 5], pbc=True
        )
        entries.append(
            run.PoolEntry(f"cand-{i:02d}", f"group-{i // 2}", atoms, run._structure_hash(atoms))
        )
    return entries


def test_protocol_uses_manuscript_strategy_names_and_budget():
    protocol, digest = run.load_protocol()
    names = {protocol["strategies"][s]["name"] for s in run.STRATEGIES}
    assert names == {"LLM-guided adaptation", "Rule-based adaptation"}
    budget = protocol["budget"]
    assert (budget["rounds"], budget["max_attempts_per_round"]) == (4, 2)
    assert budget["max_attempts_per_strategy"] == 8 and budget["charge_failed_attempts"]
    assert len(protocol["materials"]) == 6 and len(digest) == 64


def test_candidate_preparation_uses_all_candidates_without_dft_or_splits(monkeypatch, tmp_path):
    protocol, _ = run.load_protocol()

    def candidate_pool_without_exclusions(spec, excluded, **kwargs):
        assert excluded == set()
        assert kwargs["include_seed_pools"] is True
        return _pool(40)

    monkeypatch.setattr(run, "candidate_pool", candidate_pool_without_exclusions)
    monkeypatch.setattr(
        run,
        "_split",
        lambda *_args, **_kwargs: pytest.fail("candidate-only preparation made a data split"),
    )
    out = tmp_path / "bcc-candidate-only"
    result = run.prepare_candidate_material("bcc_refractory", protocol, out)

    assert result["candidate_frames"] == 40
    assert result["dft_outputs_used"] is False
    assert result["dft_labels_used"] is False
    assert result["dft_reference_frames_used"] is False
    assert result["train_validation_test_split"] is False
    assert (out / "candidates.extxyz").is_file()
    assert not (out / "references.extxyz").exists()
    assert not (out / "split").exists()


def test_zn_formate_candidate_preparation_uses_verified_cod_geometry(tmp_path):
    protocol, _ = run.load_protocol()
    out = tmp_path / "zn-formate-candidate-only"

    result = run.prepare_candidate_material("zn_formate_mof", protocol, out)
    frames = read(out / "candidates.extxyz", index=":", format="extxyz")

    assert result["candidate_frames"] == 1
    assert result["geometry_sources"][0]["candidate_id"] == "COD-7205949"
    assert result["geometry_sources"][0]["group"] == "COD-7205949/Pbcn"
    assert result["source_cif_sha256"] == run.COD_ZN_FORMATE_CIF_SHA256
    assert len(frames) == 1 and len(frames[0]) == 216
    assert Counter(frames[0].get_chemical_symbols()) == Counter(
        {"Zn": 24, "C": 48, "H": 48, "O": 96}
    )
    assert frames[0].info["candidate_id"] == "COD-7205949"
    assert result["dft_outputs_used"] is False
    assert result["dft_labels_used"] is False


def test_zn_formate_candidate_preparation_validates_source_before_output(monkeypatch, tmp_path):
    protocol, _ = run.load_protocol()

    def reject_source():
        raise ValueError("verified COD 7205949 source is invalid")

    monkeypatch.setattr(run, "_zn_formate_cod_pool", reject_source)
    out = tmp_path / "invalid-zn-formate"

    with pytest.raises(ValueError, match="verified COD 7205949"):
        run.prepare_candidate_material("zn_formate_mof", protocol, out)
    assert not out.exists()


def test_seed_only_candidate_pools_are_opt_in(monkeypatch, tmp_path):
    from ase.io import write

    seed_path = tmp_path / "seed.extxyz"
    write(seed_path, Atoms("P2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 10], pbc=True))
    monkeypatch.setattr(run, "REPO", tmp_path)
    spec = {"kind": "seeds_only", "seeds": ["seed.extxyz"]}

    assert run.candidate_pool(spec, set()) == []
    assert len(run.candidate_pool(spec, set(), include_seed_pools=True)) == 1


@pytest.mark.parametrize("material", ["phosphorene", "zn_formate_mof"])
def test_legacy_dry_run_skips_seed_only_materials_without_splitting(
    material, tmp_path, monkeypatch
):
    protocol, digest = run.load_protocol()
    monkeypatch.setattr(
        run,
        "_split",
        lambda *_args, **_kwargs: pytest.fail("legacy dry-run must not split seed-only pools"),
    )

    result = run.dry_run_material(material, protocol, digest, tmp_path / material, run.DryRunChat())

    assert result["status"] == "skipped"
    assert not (tmp_path / material).exists()


@pytest.mark.parametrize(
    "path",
    [
        "benchmarks/e3/protocol.yaml",
        "benchmarks/e3/run.py",
        "benchmarks/e3/README.md",
        "research/paper/e3/README.md",
        "docs/e3-vasp-primary-design.md",
    ],
)
def test_no_obsolete_strategy_labels(path):
    assert not OBSOLETE.search((REPO / path).read_text())


def test_rule_based_selection_is_frozen_and_bounded():
    pool = _pool()
    first, _ = run.rule_based_select(pool, 2, seed=7)
    again, _ = run.rule_based_select(pool, 2, seed=7)
    assert first == again and len(first) == 2 and set(first) <= {e.candidate_id for e in pool}


class _Reply:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, prompt):
        return type("R", (), {"content": json.dumps(self.payload)})()


@pytest.mark.parametrize(
    "payload",
    [
        {"selected": ["outside"], "cited_evidence_ids": ["SEED"]},
        {"selected": ["cand-00", "cand-01", "cand-02"], "cited_evidence_ids": ["SEED"]},
        {"selected": ["cand-00"], "cited_evidence_ids": []},
        {"selected": ["cand-00"], "cited_evidence_ids": ["ev-not-in-ledger"]},
    ],
)
def test_llm_guided_selection_rejects_ungrounded_or_out_of_pool(tmp_path, payload):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    seed = ledger.append(run._seed_evidence("m", "a" * 64, "b" * 64, 20))
    payload = json.loads(json.dumps(payload).replace("SEED", seed.evidence_id))
    with pytest.raises(ValueError):
        run.llm_guided_select(_Reply(payload), "m", _pool(), ledger, 2)


def test_paired_dry_run_accounting_and_separate_ledgers(tmp_path, monkeypatch):
    protocol, digest = run.load_protocol()
    monkeypatch.setattr(run, "candidate_pool", lambda spec, excluded: _pool(40))
    result = run.dry_run_material("lifepo4", protocol, digest, tmp_path / "out", run.DryRunChat())
    assert result["logical_attempts"] == {run.LLM: 8, run.RULE: 8}
    assert 8 <= result["unique_physical_calculations"] <= 16
    assert all(status != "fail" for status in result["leakage"]["split_checks"].values())
    ledgers = [EvidenceLedger(tmp_path / "out" / s / "ledger.jsonl") for s in run.STRATEGIES]
    seeds = [{e.evidence_id for e in ledger.evidence()} for ledger in ledgers]
    assert seeds[0] == seeds[1] == {result["identical_initial_evidence_id"]}
    for ledger, strategy in zip(ledgers, run.STRATEGIES, strict=True):
        decisions = ledger.decisions()
        assert len(decisions) == 8 and {d.iteration for d in decisions} == {1, 2, 3, 4}
        assert all(d.actor.startswith(strategy) for d in decisions)
    roles = json.loads((tmp_path / "out" / "split" / "split-manifest.json").read_text())
    withheld = {f["index"] for f in roles["frames"] if f["role"] in {"validation", "test"}}
    picked = {r["candidate_id"] for s in result["strategies"].values() for r in s["rounds"]}
    assert len(picked) >= 8 and not picked & {f"cand-{i:02d}" for i in withheld}


def test_llm_replay_admissibility_requires_purpose_and_ledger_citations(tmp_path):
    ledger = EvidenceLedger(tmp_path / "ledger.jsonl")
    seed = ledger.append(run._seed_evidence("m", "a" * 64, "b" * 64, 20))
    good = {
        "selections": [
            {
                "candidate_id": "c1",
                "scientific_purpose": "probe Fe/Li antisite strain",
                "cited_evidence_ids": [seed.evidence_id],
            }
        ],
        "round_rationale": "r",
    }
    selections, _ = run.parse_llm_selections(json.dumps(good), {"c1", "c2"}, ledger, 2)
    assert selections[0]["candidate_id"] == "c1"
    for broken in (
        {"candidate_id": "c9"},
        {"scientific_purpose": ""},
        {"cited_evidence_ids": ["ev-missing"]},
    ):
        bad = json.loads(json.dumps(good))
        bad["selections"][0].update(broken)
        with pytest.raises(ValueError):
            run.parse_llm_selections(json.dumps(bad), {"c1", "c2"}, ledger, 2)


def test_rule_based_ensemble_uses_stored_committee_forces():
    from matsim_agents.active_learning.candidates import Candidate
    from matsim_agents.active_learning.config import AcquisitionConfig
    from matsim_agents.active_learning.uncertainty import select_candidates

    pool = _pool(6)
    rows_a, rows_b = {}, {}
    for i, entry in enumerate(pool):
        entry.atoms.info["frame_id"] = entry.candidate_id
        zeros = [[0.0, 0.0, 0.0]] * 2
        rows_a[entry.candidate_id] = {"forces_eV_A": zeros}
        rows_b[entry.candidate_id] = {"forces_eV_A": [[0.1 * i, 0.0, 0.0]] * 2}
    candidates = [Candidate(e.candidate_id, e.atoms, e.candidate_id, 0) for e in pool]
    cfg = AcquisitionConfig(strategy="ensemble", n_select=1, diversity_filter=False)
    chosen, scores = select_candidates(
        candidates, cfg, None, [run.StoredForces(rows_a), run.StoredForces(rows_b)]
    )
    assert chosen[0].candidate_id == "cand-05" and scores[0] == 0.0


def test_zero_shot_metrics_use_forces_and_group_relative_energies():
    from ase.calculators.singlepoint import SinglePointCalculator

    refs = []
    for i in range(2):
        atoms = _pool(1)[0].atoms.copy()
        atoms.calc = SinglePointCalculator(atoms, energy=-1.0 * i, forces=[[0, 0, 0]] * 2)
        atoms.info.update(frame_id=f"f{i}", e3_group="g", energy_eV=-1.0 * i)
        refs.append(atoms)
    predicted = {
        f"f{i}": {"energy_eV": 10.0 - 1.0 * i, "forces_eV_A": [[0.1, 0, 0]] * 2} for i in range(2)
    }
    metrics = run.zero_shot_metrics(refs, predicted)
    assert metrics["relative_energy_mae_meV_atom"] == pytest.approx(0.0)
    assert metrics["force_component_mae_eV_A"] == pytest.approx(0.1 / 3)
    assert metrics["absolute_energy_offset_eV_atom"] == pytest.approx(5.0)


def test_soap_rule_prefers_uncertainty_and_skips_near_duplicates():
    params = {
        "eligible_top_fraction": 0.5,
        "diversity_weight": 0.5,
        "duplicate_fraction_of_median_nn": 0.05,
    }
    basis = np.eye(4)
    descriptors = [basis[0], basis[0].copy(), basis[1], basis[2], basis[3], basis[3] * -1]
    descriptors = [d / np.linalg.norm(d) for d in descriptors]
    ids = [f"c{i}" for i in range(6)]
    uncertainty = np.array([0.9, 0.89, 0.5, 0.4, 0.1, 0.05])
    first, table = run.soap_rule_select(ids, uncertainty, descriptors, [], 2, params)
    again, _ = run.soap_rule_select(ids, uncertainty, descriptors, [], 2, params)
    assert first == again
    picked = [p["candidate_id"] for p in first]
    assert picked == ["c0", "c2"]  # c1 is a SOAP duplicate of c0; low-u c4/c5 ineligible
    assert any(r.get("near_duplicate") for r in table[1:] if r["candidate_id"] == "c1")


def test_unsupported_numerical_claims_are_detected():
    prompt = "disagreement 0.09725 eV/A; median 0.08229"
    assert run.unsupported_numbers("about 0.097 and 0.0823", prompt) == []
    assert run.unsupported_numbers("DFT error 0.512 eV/A", prompt) == ["0.512"]


def test_submission_helper_batch_envelopes_are_frozen():
    import importlib.util

    path = REPO / "deployments/aurora/jobs/e4-first-batch/submit_batch.py"
    spec = importlib.util.spec_from_file_location("submit_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    e4 = module.BATCHES["E4-first-production-batch-01"]
    e4_hydragnn = module.BATCHES["E4-first-production-batch-01-hydragnn"]
    e3 = module.BATCHES["E3-round1-bcc-01"]
    assert sum(e4["caps"].values()) == 480 and sum(e4["counts"].values()) == 20
    assert e4_hydragnn == {
        "id_pattern": r"E4-(BCC|FCC|LFP)-[0-9]{4}",
        "counts": {"HydraGNN": 20},
        "caps": {"HydraGNN": 480},
    }
    assert e3 == {
        "id_pattern": r"E3-R1-BCC-[0-9]{4}",
        "counts": {"HydraGNN": 3},
        "caps": {"HydraGNN": 72},
    }
    with pytest.raises(SystemExit):
        module.batch_spec({"batch_id": "unknown"})


def test_round1_staging_ids_and_frozen_manifest_consistency():
    from benchmarks.e3 import stage_round1

    assert stage_round1.e3_id("E4-BCC-1077") == "E3-R1-BCC-1077"
    manifest_path = stage_round1.DEFAULT_MANIFEST
    if not manifest_path.is_file():
        pytest.skip("Round 1 batch not staged in this checkout")
    manifest = json.loads(manifest_path.read_text())
    round1 = json.loads((REPO / "research/paper/e3/round1-bcc.json").read_text())
    if manifest["execution_authorized"]:
        assert manifest["authorization"]["granted_utc"] and manifest["authorization"]["scope"]
    assert manifest["total_resource_cap_node_hours"] == 72
    ids = {c["e3"]["candidate_id"] for c in manifest["cases"]}
    frozen = {d for s in round1["selections"].values() for d in s}
    assert ids == frozen and len(ids) == round1["unique_physical_calculations_requested"] == 3
    assert manifest["e3_accounting"]["logical_attempts"] == {
        "llm_guided_adaptation": 2,
        "rule_based_adaptation": 2,
    }
