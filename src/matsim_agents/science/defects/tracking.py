"""Defect populations over a sequence of Wigner-Seitz analyses of one trajectory.

Vacancies are identified by their reference site and interstitials by their atom index, so a
vacancy that hops to a neighboring site is reported as one vanishing and one appearing: identity
is not carried across a hop. All frames must be analysed against the same reference, with the
same atom ordering. This is bookkeeping on given analyses; it does not decide which transient
defects are physically meaningful, so the persistence window is an explicit argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from matsim_agents.science.defects.wigner_seitz import WignerSeitzResult


@dataclass(frozen=True)
class DefectTrack:
    kind: str  # "vacancy" (identity = reference site) or "interstitial" (identity = atom index)
    identity: int
    first_frame: int
    last_frame: int
    n_frames: int  # frames in which it is present; less than the span if it flickers

    @property
    def intermittent(self) -> bool:
        return self.n_frames < self.last_frame - self.first_frame + 1


@dataclass(frozen=True)
class DefectHistory:
    n_frames: int
    times: tuple[float, ...]
    n_vacancies: tuple[int, ...]
    n_interstitials: tuple[int, ...]
    n_frenkel_pairs: tuple[int, ...]
    peak_frenkel_frame: int
    peak_frenkel_pairs: int
    tracks: tuple[DefectTrack, ...]
    last_change_frame: int  # last frame whose defect set differs from the previous frame
    persistent_vacancies: tuple[int, ...]
    persistent_interstitials: tuple[int, ...]
    persistence_window: int


def _present(result: WignerSeitzResult) -> tuple[frozenset[int], frozenset[int]]:
    return frozenset(result.vacant_sites), frozenset(result.interstitial_atoms)


def _tracks(kind: str, frames: Sequence[frozenset[int]]) -> list[DefectTrack]:
    seen: dict[int, list[int]] = {}
    for index, members in enumerate(frames):
        for identity in members:
            seen.setdefault(identity, []).append(index)
    return [
        DefectTrack(kind, identity, hits[0], hits[-1], len(hits))
        for identity, hits in sorted(seen.items())
    ]


def track_defects(
    results: Sequence[WignerSeitzResult],
    *,
    persistence_window: int,
    times: Sequence[float] | None = None,
) -> DefectHistory:
    """History of the analyses ``results`` (one per frame, in time order).

    A defect is *persistent* when it is present in every one of the last ``persistence_window``
    frames. The window is a modelling choice with no default: a transient that anneals within a
    few frames is a different object from one that outlives the whole window.
    """
    if not results:
        raise ValueError("at least one frame is required")
    if persistence_window < 1 or persistence_window > len(results):
        raise ValueError(f"persistence_window must be within [1, {len(results)}]")
    if times is None:
        times = tuple(float(i) for i in range(len(results)))
    elif len(times) != len(results):
        raise ValueError("times must have one entry per frame")
    else:
        times = tuple(float(t) for t in times)
        if not np.isfinite(times).all() or any(np.diff(times) <= 0):
            raise ValueError("times must be finite and strictly increasing")
    first = results[0]
    if any(r.n_sites != first.n_sites or r.n_atoms != first.n_atoms for r in results):
        raise ValueError("all frames must share the reference and the atom count")

    frames = [_present(r) for r in results]
    vacancies = [v for v, _ in frames]
    interstitials = [i for _, i in frames]
    pairs = tuple(r.n_frenkel_pairs for r in results)
    peak = int(np.argmax(pairs))  # first frame reaching the maximum
    changed = [k for k in range(1, len(frames)) if frames[k] != frames[k - 1]]
    window = slice(len(frames) - persistence_window, None)
    return DefectHistory(
        n_frames=len(results),
        times=times,
        n_vacancies=tuple(len(v) for v in vacancies),
        n_interstitials=tuple(len(i) for i in interstitials),
        n_frenkel_pairs=pairs,
        peak_frenkel_frame=peak,
        peak_frenkel_pairs=pairs[peak],
        tracks=tuple(_tracks("vacancy", vacancies) + _tracks("interstitial", interstitials)),
        last_change_frame=changed[-1] if changed else 0,
        persistent_vacancies=tuple(sorted(frozenset.intersection(*vacancies[window]))),
        persistent_interstitials=tuple(sorted(frozenset.intersection(*interstitials[window]))),
        persistence_window=persistence_window,
    )


__all__ = ["DefectHistory", "DefectTrack", "track_defects"]
