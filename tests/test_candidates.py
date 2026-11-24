"""Content-addressed candidates and the resumable manifest (M12)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.discovery import magnetism_generators as gen
from matsim_agents.discovery.candidates import (
    CandidateManifest,
    CandidateRecord,
    candidate_identity,
    structure_hash,
)


def _atoms():
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def _rec(atoms=None, **kw):
    return CandidateRecord.from_atoms(atoms or _atoms(), generation_method="test", **kw)


def test_identity_is_independent_of_order_and_wrapping():
    atoms = _atoms()
    shuffled = atoms[np.random.default_rng(0).permutation(len(atoms))]
    shifted = atoms.copy()
    shifted.positions[0] += shifted.cell.array[0]  # same site, one periodic image away
    assert structure_hash(atoms) == structure_hash(shuffled) == structure_hash(shifted)


def test_identity_changes_with_structure_magnetism_defect_and_recipe():
    atoms = _atoms()
    base = candidate_identity(atoms)
    moved = atoms.copy()
    moved.positions[3] += 0.05
    nm = gen.nonmagnetic(atoms, "s")
    fm = gen.ferromagnetic(atoms, "s", {"Fe": 2.2})
    ids = {
        base,
        candidate_identity(moved),
        candidate_identity(atoms, magnetic=nm),
        candidate_identity(atoms, magnetic=fm),
        candidate_identity(atoms, defect={"kind": "vacancy"}),
        candidate_identity(atoms, recipe={"seed": 1}),
    }
    assert len(ids) == 6
    assert candidate_identity(atoms, magnetic=fm) == candidate_identity(atoms, magnetic=fm)


def test_record_fields_and_validation():
    atoms = _atoms()
    fm = gen.ferromagnetic(atoms, "s", {"Fe": 2.2})
    rec = _rec(atoms, magnetic=fm, seed=7, stratum="bulk", parent_ids=["p"])
    assert rec.composition == "Fe16" and rec.n_atoms == 16
    assert rec.magnetic_mode == "ferromagnetic" and rec.magnetic_fingerprint == fm.fingerprint()
    assert rec.parent_ids == ("p",) and rec.seed == 7
    with pytest.raises(Exception):  # noqa: B017, PT011 - frozen model
        rec.seed = 8
    assert _rec(atoms, magnetic=fm, seed=7).candidate_id == rec.candidate_id


def test_manifest_dedups_and_persists(tmp_path):
    path = tmp_path / "m.jsonl"
    manifest = CandidateManifest(path)
    rec = _rec()
    assert manifest.add(rec) is True
    assert manifest.add(rec) is False
    assert len(path.read_text().splitlines()) == 1
    manifest.set_status(rec.candidate_id, "done", {"energy_eV": -1.0})
    reopened = CandidateManifest(path)
    assert len(reopened) == 1 and reopened.status[rec.candidate_id] == "done"
    assert reopened.details[rec.candidate_id] == {"energy_eV": -1.0}
    assert reopened.pending() == []


def test_resume_after_interruption_and_torn_line(tmp_path):
    path = tmp_path / "m.jsonl"
    manifest = CandidateManifest(path)
    a, b = _rec(), _rec(defect={"kind": "vacancy"})
    manifest.add(a)
    manifest.add(b)
    manifest.set_status(a.candidate_id, "running")
    with path.open("a") as handle:
        handle.write('{"event":"status","candidate_id":"' + b.candidate_id + '","sta')  # torn
    resumed = CandidateManifest(path)
    assert resumed.torn_lines == 1
    assert resumed.status[a.candidate_id] == "running"
    assert resumed.status[b.candidate_id] == "generated"
    assert resumed.recover_interrupted() == [a.candidate_id]
    assert resumed.status[a.candidate_id] == "queued"
    assert {r.candidate_id for r in resumed.pending()} == {a.candidate_id, b.candidate_id}


def test_terminal_states_cannot_be_reopened_and_unknown_ids_fail(tmp_path):
    manifest = CandidateManifest(tmp_path / "m.jsonl")
    rec = _rec()
    manifest.add(rec)
    manifest.set_status(rec.candidate_id, "failed", {"error": "x"})
    with pytest.raises(ValueError, match="terminal"):
        manifest.set_status(rec.candidate_id, "queued")
    with pytest.raises(KeyError):
        manifest.set_status("missing", "done")
    assert manifest.counts() == {"failed": 1}
