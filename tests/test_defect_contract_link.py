import numpy as np
import pytest

from matsim_agents.science.defects.contract_link import surviving_defects_from_ensemble
from matsim_agents.science.defects.survival import summarize_surviving_defects
from matsim_agents.science.defects.wigner_seitz import wigner_seitz_analysis
from matsim_agents.science.irradiation.contracts import DefectStatus
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy

SPEC = AlloySpec(counts={"Mo": 18, "Ta": 18, "W": 18}, repeats=(3, 3, 3), seed=2)
A = SPEC.lattice_parameter
REF = generate_alloy(SPEC)
CUTOFF = 0.9 * A


def _summary(current):
    return summarize_surviving_defects(
        REF,
        current,
        wigner_seitz_analysis(REF, current),
        cluster_cutoff=CUTOFF,
        cluster_justification="between bcc shells 1 and 2",
    )


def _with_frenkel_pairs(n):
    sites = [0, 13, 27, 40][:n]
    taken = set(sites)
    current = REF.copy()
    del current[sites]
    for site in sites:
        d = REF.get_distances(site, range(len(REF)), mic=True)
        host = next(int(i) for i in np.flatnonzero((d > 0) & (d < CUTOFF)) if int(i) not in taken)
        taken.add(host)
        current.append("Mo")
        current.positions[-1] = REF.positions[host] + 0.1 * A
    return current


def test_known_pair_counts_give_their_mean_and_say_how_it_was_obtained():
    counts = [1, 2, 3, 2]
    summaries = [_summary(_with_frenkel_pairs(c)) for c in counts]
    assert [s.n_frenkel_pairs for s in summaries] == counts
    out = surviving_defects_from_ensemble(summaries, cascade_provenance="synthetic test pairs")
    assert out.status is DefectStatus.ESTABLISHED
    assert out.n_frenkel_pairs == pytest.approx(np.mean(counts))
    assert "synthetic test pairs" in out.reason and "not validated" in out.reason
    assert "4 cascade realizations" in out.reason


def test_weights_change_the_reported_mean():
    summaries = [_summary(_with_frenkel_pairs(c)) for c in (1, 3)]
    out = surviving_defects_from_ensemble(
        summaries, [3.0, 1.0], cascade_provenance="synthetic test pairs"
    )
    assert out.n_frenkel_pairs == pytest.approx((1 * 3 + 3 * 1) / 4)


@pytest.mark.parametrize("provenance", [None, "", "   "])
def test_without_provenance_nothing_is_established(provenance):
    summaries = [_summary(_with_frenkel_pairs(c)) for c in (1, 2)]
    out = surviving_defects_from_ensemble(summaries, cascade_provenance=provenance)
    assert out.status is DefectStatus.NOT_ESTABLISHED and out.n_frenkel_pairs is None


def test_fewer_than_two_realizations_is_not_established():
    out = surviving_defects_from_ensemble(
        [_summary(_with_frenkel_pairs(1))], cascade_provenance="synthetic test pairs"
    )
    assert out.status is DefectStatus.NOT_ESTABLISHED and out.n_frenkel_pairs is None
    assert surviving_defects_from_ensemble([], cascade_provenance="x").n_frenkel_pairs is None
