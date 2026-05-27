"""Backward-compatible shim: re-exports the unified seed generator.

The legacy ``enumerate_phases`` API (hand-coded prototype list with
``include_2d``/``num_layers``/``n_orderings``/``lattice_scales`` kwargs)
has been replaced by the unified pipeline in
:mod:`matsim_agents.discovery.seeds`, which decorates every applicable
AFLOW prototype and (optionally) supplements it with pyXtal random
structures.

This shim:

* re-exports :class:`PhaseCandidate` from the new module;
* exposes :func:`enumerate_phases` that delegates to
  :func:`matsim_agents.discovery.seeds.generate_seeds`, accepting the
  old keyword arguments as deprecated no-ops with a one-shot warning.

Existing call sites (e.g. the active-learning subsystem) continue to
work without code changes; their config knobs simply have no effect.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from matsim_agents.discovery.composition import Composition
from matsim_agents.discovery.seeds import PhaseCandidate, generate_seeds

__all__ = ["PhaseCandidate", "enumerate_phases"]


_DEPRECATED_KWARGS = (
    "supercell",
    "min_atoms",
    "include_2d",
    "num_layers",
    "vacuum",
    "interlayer",
    "n_orderings",
    "lattice_scales",
    "ordering_seed",
)

_warned_once = False


def _maybe_warn_deprecated(kwargs: dict) -> None:
    global _warned_once
    used = [k for k in _DEPRECATED_KWARGS if k in kwargs and kwargs[k] is not None]
    if used and not _warned_once:
        warnings.warn(
            "enumerate_phases: the following keyword arguments are deprecated "
            "and ignored by the new prototype + pyXtal pipeline: "
            f"{used}. The new pipeline decorates every applicable AFLOW "
            "prototype automatically; use n_random=N to supplement with "
            "pyXtal random structures.",
            DeprecationWarning,
            stacklevel=3,
        )
        _warned_once = True


def enumerate_phases(
    composition: Composition,
    output_dir: str,
    fmt: str = "vasp",
    *,
    n_random: int = 0,
    random_seed: int = 0,
    # Deprecated kwargs (ignored).
    supercell: tuple[int, int, int] | None = None,
    min_atoms: int | None = None,
    include_2d: bool | None = None,
    num_layers: int | None = None,
    vacuum: float | None = None,
    interlayer: float | None = None,
    n_orderings: int | None = None,
    lattice_scales: Sequence[float] | None = None,
    ordering_seed: int | None = None,
) -> list[PhaseCandidate]:
    """Generate seed structures for ``composition``.

    Thin wrapper around :func:`matsim_agents.discovery.seeds.generate_seeds`;
    see that function for the active parameter set. The legacy keyword
    arguments are accepted for backward compatibility and silently
    ignored (a one-shot DeprecationWarning is emitted).
    """
    _maybe_warn_deprecated(
        {
            "supercell": supercell,
            "min_atoms": min_atoms,
            "include_2d": include_2d,
            "num_layers": num_layers,
            "vacuum": vacuum,
            "interlayer": interlayer,
            "n_orderings": n_orderings,
            "lattice_scales": lattice_scales,
            "ordering_seed": ordering_seed,
        }
    )
    return generate_seeds(
        composition,
        output_dir,
        n_random=n_random,
        fmt=fmt,
        random_seed=random_seed if ordering_seed is None else int(ordering_seed),
    )
