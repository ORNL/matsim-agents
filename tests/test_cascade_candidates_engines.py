import numpy as np
import pytest
from ase.build import bulk

from matsim_agents.science.cascade.engines import (
    EngineUnavailableError,
    InRepoEngine,
    LammpsEngine,
    size_convergence,
)
from matsim_agents.science.cascade.protocol import run_protocol, thermalize
from matsim_agents.science.cascade.qualification import Level
from matsim_agents.workflows.radiation.cascade_candidates import (
    cascade_candidates,
    min_pair_distance,
    short_distance_diagnostic,
)
from tests.test_cascade_protocol import _cell, _event, _phases, _qual


def _result(**kw):
    atoms = _cell()
    thermalize(atoms, 300.0, seed=1)
    return run_protocol(
        atoms,
        _event(atoms),
        direction=(1.0, 0.3, 0.1),
        phases=_phases(),
        qualification=_qual(Level.CASCADE_QUALIFIED),
        snapshot_every_fs=5.0,
        **kw,
    )


def test_snapshots_are_collected_and_tagged_as_cascade_candidates():
    res = _result()
    assert len(res.snapshots) >= 3
    cands = cascade_candidates(
        res,
        max_candidates=4,
        training_min_distance_A=2.5,
        boundary_assumptions={"periodic": True, "thermostat": "none"},
        model_id="m",
    )
    assert 1 <= len(cands) <= 4
    c = cands[0]
    assert c.sampling_method == "cascade" and c.parent_id == res.event.event_id
    assert c.provenance["seed"] == 7 and c.metadata["boundary_assumptions"]["periodic"]
    assert c.metadata["qualification"] == "CASCADE_QUALIFIED"
    assert len({x.candidate_id for x in cands}) == len(cands)
    assert c.predicted_max_force_eV_per_A is not None


def test_candidate_selection_is_deterministic_and_prefers_high_force():
    res = _result()
    a = cascade_candidates(
        res, max_candidates=2, training_min_distance_A=None, boundary_assumptions={"x": 1}
    )
    b = cascade_candidates(
        res, max_candidates=2, training_min_distance_A=None, boundary_assumptions={"x": 1}
    )
    assert [x.candidate_id for x in a] == [x.candidate_id for x in b]
    top = sorted(res.snapshots, key=lambda s: -s.max_force_eV_per_A)[:2]
    assert {x.predicted_max_force_eV_per_A for x in a} == {s.max_force_eV_per_A for s in top}


def test_missing_snapshots_or_boundary_assumptions_are_refused():
    atoms = _cell()
    thermalize(atoms, 300.0, seed=1)
    bare = run_protocol(
        atoms, _event(atoms), direction=(1, 0, 0), phases=_phases()[:1], require_level=None
    )
    with pytest.raises(ValueError, match="snapshots"):
        cascade_candidates(
            bare, max_candidates=1, training_min_distance_A=None, boundary_assumptions={"a": 1}
        )
    with pytest.raises(ValueError, match="boundary_assumptions"):
        cascade_candidates(
            _result(), max_candidates=1, training_min_distance_A=None, boundary_assumptions={}
        )


def test_short_distance_diagnostic():
    atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    d0 = min_pair_distance(atoms)
    assert d0 == pytest.approx(3.165 * np.sqrt(3) / 2, rel=1e-6)
    assert short_distance_diagnostic(atoms, d0)["ood_short_distance"] is False
    close = atoms.copy()
    close.positions[1] = close.positions[0] + [0.8, 0, 0]
    assert short_distance_diagnostic(close, d0)["ood_short_distance"] is True
    unknown = short_distance_diagnostic(close, None)
    assert unknown["ood_short_distance"] is None and "unknown" in unknown["note"]


def test_lammps_engine_fails_clearly_when_missing_and_in_repo_engine_is_available(monkeypatch):
    import importlib.util
    import shutil

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(EngineUnavailableError, match="LAMMPS is not installed"):
        LammpsEngine().check_available()
    InRepoEngine().check_available()


def test_size_convergence_requires_realizations_and_two_sizes():
    assert not size_convergence({100: [1.0, 1.1, 0.9]}, tolerance=0.1).converged
    few = size_convergence({100: [1.0], 200: [1.0, 1.0, 1.0]}, tolerance=0.1)
    assert not few.converged and "fewer than" in few.statement
    ok = size_convergence({100: [10, 11, 9], 200: [10.2, 9.9, 10.5]}, tolerance=0.05)
    assert ok.converged and ok.relative_change < 0.05
    bad = size_convergence({100: [5, 5, 5], 200: [10, 10, 10]}, tolerance=0.05)
    assert not bad.converged and "exceeds" in bad.statement
    with pytest.raises(ValueError):
        size_convergence({1: [1], 2: [1]}, tolerance=0)
