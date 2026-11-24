"""Redundancy reduction with symmetry, spin reversal and rare-environment retention (M13)."""

from __future__ import annotations

import numpy as np
from ase.build import bulk

from matsim_agents.discovery import magnetism_generators as gen
from matsim_agents.discovery.candidates import CandidateRecord
from matsim_agents.discovery.reduction import PoolItem, reduce_pool

A = 2.87


def _base():
    atoms = bulk("Fe", "bcc", a=A, cubic=True) * (2, 2, 2)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def _item(atoms, magnetic=None, *, defect=None, stratum=None, recipe=None):
    record = CandidateRecord.from_atoms(
        atoms,
        generation_method="t",
        magnetic=magnetic,
        defect=defect,
        recipe=recipe,
        stratum=stratum,
    )
    return PoolItem(record, atoms, magnetic)


def _vacancy(index):
    atoms = _base()
    keep = [i for i in range(len(atoms)) if i != index]
    return atoms[keep]


def test_equivalent_vacancies_collapse_to_one_and_conserve_counts():
    base = _base()
    fm = gen.ferromagnetic(base, "s", {"Fe": 2.2})
    items = []
    for index in range(len(base)):
        vac = _vacancy(index)
        items.append(_item(vac, fm_for(vac), defect={"site": index}))
    report = reduce_pool(items)
    assert len(report.kept) == 1
    assert report.reasons == {"translation_equivalent": len(base) - 1}
    assert len(report.kept) + len(report.removed) == report.n_input
    assert fm is not None


def fm_for(atoms):
    return gen.ferromagnetic(atoms, "s", {"Fe": 2.2})


def test_translated_copy_and_exact_duplicate_are_removed_with_reasons():
    atoms = _base()
    fm = gen.ferromagnetic(atoms, "s", {"Fe": 2.2})
    translated = atoms.copy()
    translated.positions += A / 2
    translated.wrap()
    other = _item(translated, fm, recipe={"copy": 1})
    original = _item(atoms, fm)
    report = reduce_pool([original, other, original])
    assert report.reasons == {"exact_duplicate": 1, "translation_equivalent": 1}
    assert len(report.kept) == 1


def test_global_spin_reversal_is_distinguished_from_real_difference():
    atoms = _base()
    # Mostly +1 with a few distinct sites, so no lattice translation maps it onto its reverse.
    up = {int(i): 1.0 for i in atoms.get_array("atom_id")}
    up.update({0: 2.0, 1: 2.0, 2: -2.0})
    down = {i: -m for i, m in up.items()}
    partial = dict(up)
    partial[0] = -partial[0]  # one flipped spin is a different state, not a reversal
    items = [
        _item(atoms, gen.custom_collinear(atoms, "s", m), recipe={"n": n})
        for n, m in enumerate((up, down, partial))
    ]
    report = reduce_pool(items)
    assert report.reasons == {"spin_reversal_equivalent": 1}
    assert len(report.kept) == 2


def test_fm_and_afm_and_nm_are_all_kept():
    atoms = _base()
    items = [
        _item(atoms, gen.nonmagnetic(atoms, "s")),
        _item(atoms, gen.ferromagnetic(atoms, "s", {"Fe": 2.2})),
        _item(atoms, gen.bcc_afm(atoms, "s", A, {"Fe": 2.2})),
    ]
    assert len(reduce_pool(items).kept) == 3


def test_thinning_retains_rare_environments_and_is_order_independent():
    rng = np.random.default_rng(3)
    items = []
    for n in range(6):
        atoms = _base()
        atoms.positions += rng.normal(0, 0.01, atoms.positions.shape)
        items.append(_item(atoms, recipe={"n": n}, stratum="bulk"))
    solute = _base()
    solute.symbols[0] = "Cr"
    rare = _item(solute, stratum="bulk")
    pool = [*items, rare]

    report = reduce_pool(pool, max_per_stratum=2)
    assert rare.record.candidate_id in report.kept
    assert report.rare_retained == (rare.record.candidate_id,)
    assert len(report.kept) == 2  # the rare one plus one common candidate
    assert report.reasons.get("thinned") == 5
    assert report.n_input == len(report.kept) + len(report.removed)

    shuffled = reduce_pool(list(reversed(pool)), max_per_stratum=2)
    assert shuffled.kept == report.kept


def test_rare_candidates_survive_even_beyond_the_cap():
    pool = []
    for n in range(3):
        atoms = _base()
        atoms.symbols[n] = "Cr"
        atoms.symbols[n + 8] = "Cr"
        atoms.symbols[(n + 4) % 16] = "Ni"
        pool.append(_item(atoms, recipe={"n": n}, stratum="s"))
    report = reduce_pool(pool, max_per_stratum=1, rare_max_count=5)
    assert len(report.kept) == len(report.rare_retained) >= 1
