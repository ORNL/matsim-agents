from __future__ import annotations

import numpy as np
from ase.build import bulk

from matsim_agents.campaign.coverage import (
    coverage_report,
    pair_priority,
    rank_by_coverage_gain,
    required_interactions,
    structure_descriptor,
)
from matsim_agents.campaign.registry import CandidateRegistry, DatasetEntry, ingest_dataset

A = 3.2


def _alloy(elements, seed=0, repeat=3):
    atoms = bulk(elements[0], "bcc", a=A, cubic=True).repeat((repeat,) * 3)
    symbols = [elements[k % len(elements)] for k in range(len(atoms))]
    np.random.default_rng(seed).shuffle(symbols)
    atoms.set_chemical_symbols(symbols)
    return atoms


def _desc(elements, **kw):
    return structure_descriptor(_alloy(elements), A, **kw)


def test_descriptor_normalised_and_reports_all_requested_groups():
    d = _desc(["W", "Ta", "Cr"])
    assert abs(sum(d.composition.values()) - 1) < 1e-12
    assert set(d.pair_frequency) == {"1", "2"}
    assert abs(sum(d.pair_frequency["1"].values()) - 1) < 1e-9
    assert set(d.local_composition) == {"Cr", "Ta", "W"}
    assert d.motifs and d.distortion < 1e-9
    assert "cluster:Cr-Ta-W" in d.interactions


def test_sro_regime_separates_ordered_from_random():
    b2 = bulk("W", "bcc", a=A, cubic=True).repeat((3, 3, 3))
    frac = b2.get_scaled_positions() * 6
    b2.set_chemical_symbols(["W" if int(round(x + y + z)) % 2 == 0 else "Ta" for x, y, z in frac])
    assert structure_descriptor(b2, A).sro_regime == "strong"


def test_binary_population_leaves_higher_order_interactions_missing():
    report = coverage_report([_desc(["W", "Ta"])])
    assert "pair1:Ta-W" in report.covered
    assert "cluster:Ta-W" in report.covered
    assert report.fraction_by_order[2] > 0 and report.fraction_by_order[3] == 0
    assert "cluster:Cr-Ta-W" in report.missing_higher_order
    assert "cluster:Cr-Hf-Mo-Ta-V-W" in report.missing_higher_order
    assert report.fraction < 1


def test_higher_order_source_covers_its_sub_clusters():
    report = coverage_report([_desc(["W", "Ta", "Cr"])])
    assert {"cluster:Ta-W", "cluster:Cr-W", "cluster:Cr-Ta", "cluster:Cr-Ta-W"} <= set(
        report.covered
    )
    assert "cluster:Cr-Ta-V" in report.missing


def test_full_target_population_reaches_full_cluster_coverage():
    d = structure_descriptor(_alloy(["W", "Ta", "Cr", "V", "Hf", "Mo"], repeat=4), A)
    report = coverage_report([d])
    assert all(k in report.covered for k in required_interactions() if k.startswith("cluster:"))


def test_ranking_prefers_new_interactions_and_is_deterministic():
    source = [_desc(["W", "Ta"])]
    cands = {
        "dup": _desc(["W", "Ta"]),
        "ternary": _desc(["W", "Ta", "Cr"]),
        "other": _desc(["Mo", "V"]),
    }
    ranked = rank_by_coverage_gain(cands, source)
    assert ranked[0].candidate_id == "ternary" and ranked[-1].candidate_id == "dup"
    assert ranked[-1].gain == 0
    assert ranked == rank_by_coverage_gain(dict(reversed(list(cands.items()))), source)


def test_defect_and_magnetic_descriptors_are_recorded():
    atoms = _alloy(["W", "Cr"])
    moments = np.where(np.array(atoms.get_chemical_symbols()) == "Cr", 1.0, 0.0)
    d = structure_descriptor(atoms, A, magnetic_moments=moments, defect_site_ids=[0, 1])
    assert d.defect_interactions
    assert any(k.startswith("Cr@+1.0") for k in d.magnetic_environments)
    assert sum(d.magnetic_environments.values()) == len(atoms)


def test_pair_priority_lists_least_covered_pairs_first():
    reg = CandidateRegistry()
    ingest_dataset(reg, "d", [DatasetEntry(structure_hash="h", composition={"W": 4, "Ta": 4})])
    order = pair_priority(reg)
    assert len(order) == 9 and order[-1] == ("Ta-W", 1)
    assert all(count == 0 for _, count in order[:-1])
