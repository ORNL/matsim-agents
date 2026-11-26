"""PKA-weighted defect production source terms.

``S_d = sum_species sum_bin R_bin * Y_d(species, bin)`` where ``R_bin`` is the PKA rate in an
energy bin (already a bin total, so no extra ``dE`` factor) and ``Y_d`` is a surviving-defect
yield per PKA event from a cascade ensemble. Yields are looked up by recoiling element and
energy bin; a spectrum bin counts as covered only if it lies inside one yield bin, so no
yield is extended to energies or species the cascades did not sample. Rates that are not
covered are either an error or excluded and reported (``on_uncovered``), or, only when asked
for, assigned the nearest yield bin and reported as extrapolated.

Propagated: yield statistics (independent bins) and an optional caller-stated fractional
spectrum uncertainty (independent per bin; real spectrum errors are usually correlated).
Not propagated: material, temperature and composition mismatch between spectrum and yields
(only recorded), isotope detail, and model error of the cascade potential. Nothing here is
validated against data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from matsim_agents.science.defects.ensemble import EnsembleEstimate
from matsim_agents.science.irradiation.contracts import (
    RATE_BASIS_PER_ATOM_PER_S,
    ContractError,
    PKASpectrum,
)
from matsim_agents.science.kinetics.cluster_dynamics import SourceTerm

OnUncovered = Literal["error", "exclude", "hold_nearest"]

UNIT_PER_ATOM_PER_S = "defects per target atom per second"
_EDGE_TOL = 1e-9


_ALWAYS_NOT_PROPAGATED = (
    "material/temperature/composition agreement between spectrum and yields",
    "cascade-potential model error",
)


@dataclass(frozen=True)
class YieldBin:
    """Surviving-defect yield (defects per PKA event) of one element over an energy bin."""

    element: str
    low_mev: float
    high_mev: float
    estimate: EnsembleEstimate

    def __post_init__(self) -> None:
        values = (self.low_mev, self.high_mev, self.estimate.mean, self.estimate.standard_error)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("yield bin values must be finite")
        if self.low_mev < 0 or self.high_mev <= self.low_mev:
            raise ValueError("each yield bin needs 0 <= low < high")
        if self.estimate.mean < 0 or self.estimate.standard_error < 0:
            raise ValueError("a yield and its standard error must be non-negative")


@dataclass(frozen=True)
class YieldTable:
    """Yields of one defect class, with the conditions the cascades were run at."""

    defect_class: str
    bins: tuple[YieldBin, ...]
    material: str
    temperature_K: float
    cascade_provenance: str

    def __post_init__(self) -> None:
        if not self.defect_class.strip() or not self.material.strip():
            raise ValueError("defect_class and material are required")
        if not self.cascade_provenance.strip():
            raise ValueError("cascade_provenance is required")
        if not math.isfinite(self.temperature_K) or self.temperature_K <= 0:
            raise ValueError("temperature_K must be positive and finite")
        if not self.bins:
            raise ValueError("a yield table needs at least one bin")
        by_element: dict[str, list[YieldBin]] = {}
        for b in self.bins:
            by_element.setdefault(b.element, []).append(b)
        for element, items in by_element.items():
            items.sort(key=lambda b: b.low_mev)
            for prev, nxt in zip(items, items[1:], strict=False):
                if nxt.low_mev < prev.high_mev - _EDGE_TOL:
                    raise ValueError(f"yield bins of {element} overlap")


@dataclass(frozen=True)
class SourceContribution:
    species: str
    low_mev: float
    high_mev: float
    pka_rate: float
    yield_per_pka: float
    rate: float  # defects per target atom per second
    extrapolated: bool


@dataclass(frozen=True)
class ScaledRate:
    value: float
    standard_error: float
    unit: str


@dataclass(frozen=True)
class DefectSource:
    defect_class: str
    rate_per_target_atom_per_s: float
    standard_error: float
    mean_yield_per_pka: float | None  # over the PKA rate that was assigned a yield
    total_pka_rate: float
    used_pka_rate: float
    excluded_rate_fraction: float
    extrapolated_rate_fraction: float
    contributions: tuple[SourceContribution, ...]
    spectrum_source: str
    spectrum_rate_unit: str
    material: str
    temperature_K: float
    cascade_provenance: str
    spectrum_relative_sigma: float
    unit: str = UNIT_PER_ATOM_PER_S
    propagated: tuple[str, ...] = ("yield statistics",)
    not_propagated: tuple[str, ...] = _ALWAYS_NOT_PROPAGATED

    def per_volume(self, number_density_per_m3: float) -> ScaledRate:
        """Defects per cubic metre per second for a target atom number density."""
        _positive(number_density_per_m3, "number_density_per_m3")
        return ScaledRate(
            self.rate_per_target_atom_per_s * number_density_per_m3,
            self.standard_error * number_density_per_m3,
            "defects per m^3 per second",
        )

    def per_dpa(self, dpa_rate_per_s: float) -> ScaledRate:
        """Defects per target atom per dpa; the dpa rate is a caller-supplied number."""
        _positive(dpa_rate_per_s, "dpa_rate_per_s")
        return ScaledRate(
            self.rate_per_target_atom_per_s / dpa_rate_per_s,
            self.standard_error / dpa_rate_per_s,
            "defects per target atom per dpa",
        )


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _find(
    table: dict[str, list[YieldBin]], element: str, low: float, high: float
) -> YieldBin | None:
    for b in table.get(element, ()):
        if low >= b.low_mev - _EDGE_TOL and high <= b.high_mev + _EDGE_TOL:
            return b
    return None


def _nearest(table: dict[str, list[YieldBin]], element: str, low: float, high: float):
    items = table.get(element)
    if not items:
        return None
    mid = 0.5 * (low + high)
    return min(items, key=lambda b: abs(0.5 * (b.low_mev + b.high_mev) - mid))


def pka_weighted_source(
    spectrum: PKASpectrum,
    yields: YieldTable,
    *,
    on_uncovered: OnUncovered = "error",
    spectrum_relative_sigma: float = 0.0,
) -> DefectSource:
    """Combine ``spectrum`` with ``yields`` into a per-target-atom source rate."""
    if spectrum.rate_basis != RATE_BASIS_PER_ATOM_PER_S:
        raise ContractError(
            f"spectrum rate_basis is {spectrum.rate_basis!r}; a source term needs "
            f"{RATE_BASIS_PER_ATOM_PER_S!r}"
        )
    if on_uncovered not in ("error", "exclude", "hold_nearest"):
        raise ValueError("on_uncovered must be 'error', 'exclude' or 'hold_nearest'")
    if not math.isfinite(spectrum_relative_sigma) or spectrum_relative_sigma < 0:
        raise ValueError("spectrum_relative_sigma must be non-negative and finite")

    table: dict[str, list[YieldBin]] = {}
    for b in yields.bins:
        table.setdefault(b.element, []).append(b)

    contributions: list[SourceContribution] = []
    variances: list[float] = []
    excluded = 0.0
    extrapolated = 0.0
    for s in spectrum.spectra:
        element = s.species.element
        for lo, hi, rate in zip(s.bin_low_mev, s.bin_high_mev, s.rates, strict=True):
            if rate == 0.0:
                continue
            chosen = _find(table, element, lo, hi)
            was_extrapolated = False
            if chosen is None:
                if on_uncovered == "error":
                    raise ContractError(
                        f"no yield covers {element} recoils in [{lo:g}, {hi:g}] MeV; "
                        "use on_uncovered to exclude or extrapolate explicitly"
                    )
                if on_uncovered == "hold_nearest":
                    chosen = _nearest(table, element, lo, hi)
                    was_extrapolated = chosen is not None
                if chosen is None:
                    excluded += rate
                    continue
            if was_extrapolated:
                extrapolated += rate
            y, se = chosen.estimate.mean, chosen.estimate.standard_error
            value = rate * y
            variances.append((rate * se) ** 2 + (value * spectrum_relative_sigma) ** 2)
            contributions.append(
                SourceContribution(
                    species=s.species.key,
                    low_mev=lo,
                    high_mev=hi,
                    pka_rate=rate,
                    yield_per_pka=y,
                    rate=value,
                    extrapolated=was_extrapolated,
                )
            )

    total = spectrum.total_rate
    used = math.fsum(c.pka_rate for c in contributions)
    source_rate = math.fsum(c.rate for c in contributions)
    propagated = ("yield statistics",) + (
        ("stated fractional spectrum uncertainty (independent per bin)",)
        if spectrum_relative_sigma > 0
        else ()
    )
    not_propagated = _ALWAYS_NOT_PROPAGATED
    if spectrum_relative_sigma == 0:
        not_propagated = ("spectrum uncertainty (none stated)", *not_propagated)
    return DefectSource(
        defect_class=yields.defect_class,
        rate_per_target_atom_per_s=source_rate,
        standard_error=math.sqrt(math.fsum(variances)),
        mean_yield_per_pka=source_rate / used if used > 0 else None,
        total_pka_rate=total,
        used_pka_rate=used,
        excluded_rate_fraction=excluded / total,
        extrapolated_rate_fraction=extrapolated / total,
        contributions=tuple(contributions),
        spectrum_source=spectrum.provenance.source,
        spectrum_rate_unit=spectrum.rate_unit,
        material=yields.material,
        temperature_K=yields.temperature_K,
        cascade_provenance=yields.cascade_provenance,
        spectrum_relative_sigma=spectrum_relative_sigma,
        propagated=propagated,
        not_propagated=not_propagated,
    )


def cluster_source_term(
    source: DefectSource,
    *,
    state_key: str,
    n_target_atoms: float,
    reservoir: str,
) -> SourceTerm:
    """Cluster-dynamics ``SourceTerm`` for a cell of ``n_target_atoms`` atoms.

    The rate is defects per second in that cell; ``basis`` records the scaling so a rate
    cannot be reused at a different cell size by accident.
    """
    _positive(n_target_atoms, "n_target_atoms")
    if source.rate_per_target_atom_per_s <= 0:
        raise ContractError("source rate is zero; a cluster source term must be positive")
    return SourceTerm(
        state_key=state_key,
        rate_per_s=source.rate_per_target_atom_per_s * n_target_atoms,
        reservoir=reservoir,
        basis=(
            f"{source.defect_class}: {UNIT_PER_ATOM_PER_S} x {n_target_atoms:g} target atoms; "
            f"extrapolated rate fraction {source.extrapolated_rate_fraction:.3g}, "
            f"excluded {source.excluded_rate_fraction:.3g}"
        ),
    )


__all__ = [
    "DefectSource",
    "ScaledRate",
    "SourceContribution",
    "YieldBin",
    "YieldTable",
    "cluster_source_term",
    "pka_weighted_source",
]
