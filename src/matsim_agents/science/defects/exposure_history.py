"""Piecewise-constant exposure history for cascade-derived and dpa-reference sources.

Each segment has a duration, temperature and flux. Within a segment the network is linear with
constant sources, so ``ClusterNetwork.evolve`` is exact; populations carry from one segment to
the next. This is how time-dependent flux, temperature and a current microstructural state
(the initial populations) enter. A shutdown is a segment with zero flux.

Scaling and checks:

* Source rates are defects per target atom per second at ``reference_flux_n_cm2_s``; a segment
  scales them by ``flux / reference_flux`` (linear in flux, an assumption stated in the result).
  Cascade overlap and dose-dependent yields are not modelled.
* A segment's ``spectrum_id``, when given, must equal each source's ``spectrum_source``. A source
  whose material differs from the history's material is rejected. A source computed at a
  different temperature than the segment is reported, not rejected.
* The double-counting guard of :mod:`cascade_injection` runs on every segment's network.
* ``dpa`` channels are simplified reference models and are only accepted with
  ``allow_dpa_reference=True``; the result records their use.
* A segment's network must contain every state that carries population into it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from matsim_agents.science.defects.cascade_injection import (
    InjectionError,
    SourceChannel,
    check_channels,
)
from matsim_agents.science.defects.source_terms import cluster_source_term
from matsim_agents.science.kinetics.cluster_dynamics import ClusterError, ClusterNetwork

ZERO_FLUX_NOTE = "zero flux: decay and migration only"


class HistoryError(ValueError):
    """The exposure history is inconsistent with its sources or network."""


@dataclass(frozen=True)
class ExposureSegment:
    duration_s: float
    temperature_K: float
    flux_n_cm2_s: float
    label: str = ""
    spectrum_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("duration_s", "temperature_K"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise HistoryError(f"{name} must be finite and positive")
        if not math.isfinite(self.flux_n_cm2_s) or self.flux_n_cm2_s < 0:
            raise HistoryError("flux must be finite and nonnegative")


@dataclass(frozen=True)
class SegmentRecord:
    label: str
    duration_s: float
    temperature_K: float
    flux_scale: float
    injected: dict[str, float]
    populations: dict[str, float]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class HistoryResult:
    final: dict[str, float]
    segments: tuple[SegmentRecord, ...]
    total_injected: dict[str, float]
    dpa_reference_used: bool
    assumptions: tuple[str, ...]
    not_modelled: tuple[str, ...]


def run_exposure_history(
    segments: Sequence[ExposureSegment],
    build_network: Callable[[float], ClusterNetwork],
    channels: Sequence[SourceChannel],
    *,
    reference_flux_n_cm2_s: float,
    n_target_atoms: float,
    material: str,
    initial: Mapping[str, float] | None = None,
    allow_dpa_reference: bool = False,
    reservoir: str = "irradiation",
) -> HistoryResult:
    """Evolve ``initial`` through the segments; ``build_network(T)`` makes a network at ``T``."""
    if not segments:
        raise HistoryError("an exposure history needs at least one segment")
    if not math.isfinite(reference_flux_n_cm2_s) or reference_flux_n_cm2_s <= 0:
        raise HistoryError("reference flux must be finite and positive")
    if not math.isfinite(n_target_atoms) or n_target_atoms <= 0:
        raise HistoryError("n_target_atoms must be finite and positive")
    uses_dpa = any(c.kind == "dpa" for c in channels)
    if uses_dpa and not allow_dpa_reference:
        raise HistoryError(
            "dpa channels are simplified reference models; "
            "pass allow_dpa_reference=True to use them"
        )
    for c in channels:
        if c.source.material != material:
            raise HistoryError(
                f"source {c.source.defect_class!r} is for {c.source.material!r}, not {material!r}"
            )
    populations = dict(initial or {})
    records: list[SegmentRecord] = []
    total: Counter = Counter()
    for n, seg in enumerate(segments):
        label = seg.label or f"segment-{n}"
        network = build_network(seg.temperature_K)
        if not math.isclose(network.temperature_K, seg.temperature_K, rel_tol=1e-9):
            raise HistoryError(f"{label}: build_network returned the wrong temperature")
        scale = seg.flux_n_cm2_s / reference_flux_n_cm2_s
        notes: list[str] = []
        injected: Counter = Counter()
        try:
            check_channels(network, list(channels))
            for c in channels:
                if seg.spectrum_id is not None and c.source.spectrum_source != seg.spectrum_id:
                    raise HistoryError(
                        f"{label}: source {c.source.defect_class!r} was built for spectrum "
                        f"{c.source.spectrum_source!r}, not {seg.spectrum_id!r}"
                    )
                if not math.isclose(c.source.temperature_K, seg.temperature_K, rel_tol=1e-9):
                    notes.append(
                        f"{c.source.defect_class} yields were computed at "
                        f"{c.source.temperature_K:g} K, used at {seg.temperature_K:g} K"
                    )
                if scale == 0:
                    continue
                term = cluster_source_term(
                    c.source,
                    state_key=c.state_key,
                    n_target_atoms=n_target_atoms,
                    reservoir=reservoir,
                )
                network.add_source(
                    term.model_copy(
                        update={
                            "rate_per_s": term.rate_per_s * scale,
                            "basis": f"{term.basis}; x{scale:g} of reference flux",
                        }
                    )
                )
                for species, count in network.state(c.state_key).inventory.items():
                    injected[species] += term.rate_per_s * scale * seg.duration_s * count
            if scale == 0:
                notes.append(ZERO_FLUX_NOTE)
            populations = network.evolve(populations, seg.duration_s)
        except InjectionError as exc:
            raise HistoryError(f"{label}: {exc}") from exc
        except ClusterError as exc:
            raise HistoryError(f"{label}: {exc}") from exc
        total.update(injected)
        records.append(
            SegmentRecord(
                label=label,
                duration_s=seg.duration_s,
                temperature_K=seg.temperature_K,
                flux_scale=scale,
                injected=dict(sorted(injected.items())),
                populations=dict(populations),
                notes=tuple(notes),
            )
        )
    return HistoryResult(
        final=populations,
        segments=tuple(records),
        total_injected=dict(sorted(total.items())),
        dpa_reference_used=uses_dpa,
        assumptions=(
            "source rate scales linearly with flux; no dose-dependent yield or cascade overlap",
            "sources are constant within a segment and the solution is exact per segment",
            *(("dpa-based reference sources are a simplified model",) if uses_dpa else ()),
        ),
        not_modelled=(
            "cascade overlap and defect-density dependence of the yield",
            "interstitial and self-interstitial-cluster balance unless states carry them",
            "spatial inhomogeneity",
        ),
    )


__all__ = [
    "ZERO_FLUX_NOTE",
    "ExposureSegment",
    "HistoryError",
    "HistoryResult",
    "SegmentRecord",
    "run_exposure_history",
]
