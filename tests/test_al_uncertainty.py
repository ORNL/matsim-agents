"""Unit tests for the active-learning uncertainty / acquisition module.

These tests use synthetic ASE Atoms + tiny mock "calculator" objects that
return canned forces, so they run in milliseconds and do not require
HydraGNN or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from ase import Atoms

from matsim_agents.active_learning.candidates import Candidate
from matsim_agents.active_learning.config import AcquisitionConfig
from matsim_agents.active_learning.uncertainty import (
    greedy_farthest_point,
    score_ensemble,
    score_random,
    select_candidates,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_candidate(symbol: str = "Si", n_atoms: int = 2, idx: int = 0) -> Candidate:
    atoms = Atoms(
        symbols=[symbol] * n_atoms,
        positions=np.random.default_rng(idx).normal(size=(n_atoms, 3)) * 0.1,
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    return Candidate(
        candidate_id=f"cand_{idx:03d}",
        atoms=atoms,
        seed_path=f"/seed/{symbol}.vasp",
        md_step=idx,
    )


@dataclass
class _ConstantForceCalc:
    """Tiny stand-in calculator that always returns a fixed forces array.

    Mimics just enough of an ASE calculator for the uncertainty scorer:
    `atoms.get_forces()` after `atoms.calc = self` must return our value.
    """

    forces: np.ndarray

    def get_forces(self, atoms: Atoms) -> np.ndarray:  # pragma: no cover
        return self.forces

    # ASE attaches the calculator and calls calculate()/results lookups.
    # The minimal interface used by uncertainty._forces_from is just
    # `atoms.get_forces()`, which dispatches via ASE -> calc.calculate.
    # ASE's BaseCalculator path is simpler if we expose a `calculate` method.
    def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
        self.results = {"forces": self.forces}

    implemented_properties = ["forces"]


# --------------------------------------------------------------------------- #
# score_ensemble                                                              #
# --------------------------------------------------------------------------- #


def test_score_ensemble_identical_models_yield_zero() -> None:
    """If every model predicts the same forces, disagreement must be zero."""
    cands = [_make_candidate(idx=i) for i in range(3)]
    f = np.zeros((2, 3))
    calcs = [_ConstantForceCalc(forces=f.copy()) for _ in range(3)]

    scores = score_ensemble(cands, calcs)
    assert scores.shape == (3,)
    assert np.allclose(scores, 0.0)


def test_score_ensemble_disagreement_is_positive() -> None:
    """Disagreeing models must produce strictly-positive scores."""
    cands = [_make_candidate(idx=i) for i in range(2)]
    calcs = [
        _ConstantForceCalc(forces=np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        _ConstantForceCalc(forces=np.array([[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])),
    ]
    scores = score_ensemble(cands, calcs)
    assert (scores > 0).all()


def test_score_ensemble_requires_two_models() -> None:
    cands = [_make_candidate()]
    with pytest.raises(ValueError, match=r">=2"):
        score_ensemble(cands, [_ConstantForceCalc(forces=np.zeros((2, 3)))])


# --------------------------------------------------------------------------- #
# score_random                                                                #
# --------------------------------------------------------------------------- #


def test_score_random_is_reproducible() -> None:
    cands = [_make_candidate(idx=i) for i in range(5)]
    a = score_random(cands, np.random.default_rng(0))
    b = score_random(cands, np.random.default_rng(0))
    assert np.allclose(a, b)
    assert a.shape == (5,)


# --------------------------------------------------------------------------- #
# greedy_farthest_point                                                       #
# --------------------------------------------------------------------------- #


def test_greedy_farthest_point_picks_diverse_compositions() -> None:
    """Greedy FPS must pick across multiple element types when available."""
    # Mix three element types so FPS is forced to spread its picks.
    cands = (
        [_make_candidate("Si", idx=i) for i in range(4)]
        + [_make_candidate("Li", idx=10 + i) for i in range(4)]
        + [_make_candidate("O", idx=20 + i) for i in range(4)]
    )
    scores = np.ones(len(cands))
    chosen = greedy_farthest_point(cands, scores, n_select=3)

    # Three picks across three distinct elements.
    chosen_symbols = {cands[i].atoms.get_chemical_symbols()[0] for i in chosen}
    assert len(chosen_symbols) == 3


def test_greedy_farthest_point_returns_all_when_n_geq_pool() -> None:
    cands = [_make_candidate(idx=i) for i in range(3)]
    scores = np.array([0.1, 0.2, 0.3])
    chosen = greedy_farthest_point(cands, scores, n_select=10)
    assert sorted(chosen) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# select_candidates: end-to-end via the acquisition strategies                #
# --------------------------------------------------------------------------- #


def test_select_candidates_random_strategy() -> None:
    cands = [_make_candidate(idx=i) for i in range(8)]
    cfg = AcquisitionConfig(strategy="random", n_select=3, diversity_filter=False)
    selected, scores = select_candidates(cands, cfg, primary_calculator=None, seed=0)
    assert len(selected) == 3
    assert scores.shape == (8,)


def test_select_candidates_ensemble_strategy_picks_high_disagreement() -> None:
    """The candidate with the largest model disagreement must be selected."""
    cands = [_make_candidate(idx=i, symbol="Si") for i in range(4)]
    # Model A always predicts zero; Model B predicts something different on
    # candidate index 2 only -> only that candidate has nonzero disagreement.
    forces_default = np.zeros((2, 3))

    @dataclass
    class _PerCandCalc:
        signature: str  # "A" or "B"

        def get_forces(self, atoms: Atoms) -> np.ndarray:
            if self.signature == "B" and atoms.info.get("idx") == 2:
                return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            return forces_default.copy()

        def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
            self.results = {"forces": self.get_forces(atoms)}

        implemented_properties = ["forces"]

    # Tag each Atoms with a stable index so the calculator can branch.
    for i, c in enumerate(cands):
        c.atoms.info["idx"] = i

    calcs = [_PerCandCalc(signature="A"), _PerCandCalc(signature="B")]
    cfg = AcquisitionConfig(
        strategy="ensemble", n_select=1, diversity_filter=False
    )
    selected, scores = select_candidates(
        cands, cfg, primary_calculator=None, ensemble_calculators=calcs, seed=0
    )
    assert len(selected) == 1
    assert selected[0].candidate_id == cands[2].candidate_id
    # The non-disagreeing candidates must score zero, the disagreeing one >0.
    nonzero = np.flatnonzero(scores > 0)
    assert nonzero.tolist() == [2]


def test_select_candidates_ensemble_requires_two_calculators() -> None:
    cands = [_make_candidate(idx=0)]
    cfg = AcquisitionConfig(strategy="ensemble", n_select=1)
    with pytest.raises(ValueError, match=r"ensemble"):
        select_candidates(
            cands,
            cfg,
            primary_calculator=None,
            ensemble_calculators=[_ConstantForceCalc(forces=np.zeros((2, 3)))],
            seed=0,
        )


def test_select_candidates_empty_input_returns_empty() -> None:
    cfg = AcquisitionConfig(strategy="random", n_select=5)
    selected, scores = select_candidates([], cfg, primary_calculator=None)
    assert selected == []
    assert scores.shape == (0,)
