import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from matsim_agents.science.defects.records import build_defect_records
from matsim_agents.science.defects.survival import summarize_surviving_defects
from matsim_agents.science.defects.wigner_seitz import wigner_seitz_analysis

A = 3.165
REF = bulk("W", "bcc", a=A, cubic=True).repeat((4, 4, 4))
NN = A * np.sqrt(3) / 2


def _records(current, *, cutoff=3.0, **kwargs):
    result = wigner_seitz_analysis(REF, current)
    summary = summarize_surviving_defects(
        REF, current, result, cluster_cutoff=cutoff, cluster_justification="test cutoff"
    )
    return build_defect_records(
        REF,
        current,
        result,
        summary,
        environment_cutoff=3.0,
        reference_provenance="W bcc 4x4x4 test reference",
        **kwargs,
    )


def _second_neighbor(site):
    d = REF.get_distances(site, range(len(REF)), mic=True)
    return int(np.argsort(d)[1])


def test_monovacancy_record():
    current = REF.copy()
    del current[5]
    (rec,) = _records(current)
    assert rec.defect_type == "monovacancy"
    assert rec.site == 5 and rec.species == "W"
    assert rec.environment == {"W": 8}
    assert rec.cluster_size == 1
    assert rec.detection_method == "wigner_seitz"
    assert rec.reference_provenance == "W bcc 4x4x4 test reference"
    assert np.allclose(rec.position, REF.positions[5])


def test_divacancy_and_cluster_types():
    nn = _second_neighbor(0)
    two = REF.copy()
    del two[[0, nn]]
    assert {r.defect_type for r in _records(two)} == {"divacancy"}
    d = REF.get_distances(0, range(len(REF)), mic=True)
    first = [int(i) for i in np.argsort(d)[:4]]
    three = REF.copy()
    del three[first[:3]]
    assert {r.defect_type for r in _records(three)} == {"vacancy_cluster"}


def test_far_vacancies_stay_monovacancies():
    d = REF.get_distances(0, range(len(REF)), mic=True)
    far = int(np.argmax(d))
    current = REF.copy()
    del current[[0, far]]
    recs = _records(current)
    assert len(recs) == 2
    assert {r.defect_type for r in recs} == {"monovacancy"}


def test_self_and_foreign_interstitials_with_solute_tag():
    current = REF.copy()
    current.append("W")
    current.positions[-1] = REF.positions[0] + [0.5 * A, 0.25 * A, 0.0]
    recs = _records(current)
    interstitials = [r for r in recs if r.kind == "interstitial"]
    assert [r.defect_type for r in interstitials] == ["self_interstitial"]
    assert interstitials[0].complexes == ()

    foreign = REF.copy()
    foreign.append("Ta")
    foreign.positions[-1] = REF.positions[0] + [0.5 * A, 0.25 * A, 0.0]
    recs = _records(foreign, solute_species=("Ta",))
    inter = [r for r in recs if r.kind == "interstitial"][0]
    assert inter.defect_type == "foreign_interstitial"
    assert inter.complexes == ("solute",)


def test_gas_in_vacancy_environment_is_tagged_not_typed():
    current = REF.copy()
    del current[0]
    he = Atoms("He", positions=[REF.positions[0] + [0.9, 0.0, 0.0]], cell=REF.cell, pbc=True)
    current += he
    recs = _records(current, gas_species=("He",))
    assert recs
    assert all(
        r.defect_type != "monovacancy" or "gas" in r.complexes for r in recs if r.kind == "vacancy"
    )


def test_antisite_record():
    ref = REF.copy()
    ref.symbols[0] = "Ta"
    cur = ref.copy()
    cur.symbols[0] = "W"
    result = wigner_seitz_analysis(ref, cur)
    summary = summarize_surviving_defects(
        ref, cur, result, cluster_cutoff=3.0, cluster_justification="test cutoff"
    )
    recs = build_defect_records(
        ref, cur, result, summary, environment_cutoff=3.0, reference_provenance="p"
    )
    assert [r.defect_type for r in recs] == ["antisite"]
    assert recs[0].species == "W" and recs[0].cluster is None


def test_arguments_validated():
    current = REF.copy()
    del current[0]
    result = wigner_seitz_analysis(REF, current)
    summary = summarize_surviving_defects(
        REF, current, result, cluster_cutoff=3.0, cluster_justification="x"
    )
    with pytest.raises(ValueError):
        build_defect_records(
            REF, current, result, summary, environment_cutoff=3.0, reference_provenance=" "
        )
    with pytest.raises(ValueError):
        build_defect_records(
            REF, current, result, summary, environment_cutoff=0.0, reference_provenance="p"
        )
    other = REF.copy()
    del other[7]
    other_result = wigner_seitz_analysis(REF, other)
    with pytest.raises(ValueError):
        build_defect_records(
            REF, other, other_result, summary, environment_cutoff=3.0, reference_provenance="p"
        )
