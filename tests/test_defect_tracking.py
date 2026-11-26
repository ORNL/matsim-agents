import numpy as np
import pytest

from matsim_agents.science.defects.tracking import track_defects
from matsim_agents.science.defects.wigner_seitz import wigner_seitz_analysis
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy

SPEC = AlloySpec(counts={"W": 54}, repeats=(3, 3, 3), seed=0)
A = SPEC.lattice_parameter
REF = generate_alloy(SPEC)
INTERSTITIAL_OFFSET = np.array([0.5, 0.25, 0.0]) * A


def _frame(displaced):
    """Frame in which each atom in ``displaced`` sits in an interstitial position near its site."""
    atoms = REF.copy()
    for atom in displaced:
        atoms.positions[atom] = REF.positions[atom] + INTERSTITIAL_OFFSET
    return wigner_seitz_analysis(REF, atoms)


def _history(displacements, **kwargs):
    kwargs.setdefault("persistence_window", 1)
    return track_defects([_frame(d) for d in displacements], **kwargs)


def test_counts_peak_and_final_population_of_a_known_sequence():
    # one pair appears, a second appears (peak), the first recombines, the second persists
    history = _history([[], [3], [3, 20], [20], [20]], persistence_window=2)
    assert history.n_frenkel_pairs == (0, 1, 2, 1, 1)
    assert history.n_vacancies == history.n_interstitials == history.n_frenkel_pairs
    assert (history.peak_frenkel_frame, history.peak_frenkel_pairs) == (2, 2)
    assert history.persistent_vacancies == (20,)
    assert history.persistent_interstitials == (20,)
    assert history.last_change_frame == 3


def test_a_defect_that_anneals_inside_the_window_is_not_persistent():
    history = _history([[], [7], [7], []], persistence_window=2)
    assert history.persistent_vacancies == () and history.persistent_interstitials == ()
    assert history.peak_frenkel_pairs == 1 and history.last_change_frame == 3
    short = _history([[], [7], [7], []], persistence_window=1)
    assert short.persistent_vacancies == ()  # absent in the last frame itself


def test_tracks_report_lifetime_and_flicker():
    history = _history([[5], [], [5], [5]])
    (vacancy,) = [t for t in history.tracks if t.kind == "vacancy"]
    assert (vacancy.identity, vacancy.first_frame, vacancy.last_frame, vacancy.n_frames) == (
        5,
        0,
        3,
        3,
    )
    assert vacancy.intermittent
    steady = _history([[5], [5], [5]])
    assert not [t for t in steady.tracks if t.intermittent]


def test_unchanging_defects_report_no_change_after_the_first_frame():
    history = _history([[4], [4], [4]])
    assert history.last_change_frame == 0
    assert history.persistent_vacancies == (4,)


def test_times_are_carried_and_validated():
    history = _history([[], [1]], times=[0.0, 0.5])
    assert history.times == (0.0, 0.5)
    with pytest.raises(ValueError, match="times"):
        _history([[], [1]], times=[0.0])
    with pytest.raises(ValueError, match="increasing"):
        _history([[], [1]], times=[1.0, 1.0])


def test_input_validation():
    with pytest.raises(ValueError, match="at least one"):
        track_defects([], persistence_window=1)
    frames = [_frame([]), _frame([2])]
    for window in (0, 3):
        with pytest.raises(ValueError, match="persistence_window"):
            track_defects(frames, persistence_window=window)
    other = REF.copy()
    del other[0]
    mismatched = wigner_seitz_analysis(REF, other)
    with pytest.raises(ValueError, match="share"):
        track_defects([frames[0], mismatched], persistence_window=1)
