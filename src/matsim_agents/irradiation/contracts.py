"""Application-independent irradiation contracts.

Nothing here knows any external program's file format, and nothing imports an adapter. A PKA
spectrum is a statement about recoil production, not about surviving defects: converting PKAs to
defects needs a cascade model, and without one the answer is ``NOT_ESTABLISHED``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

_ELEMENT = re.compile(r"^[A-Z][a-z]?$")
UNKNOWN_ELEMENT = "unknown"
RATE_BASIS_PER_ATOM_PER_S = "per_target_atom_per_s"
RATE_BASIS_UNSPECIFIED = "unspecified"
SUM_TOLERANCE = 1e-9


class ContractError(ValueError):
    """A contract invariant is violated."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ValidationStatus(StrEnum):
    NOT_VALIDATED = "NOT_VALIDATED"
    STRUCTURALLY_VALID = "STRUCTURALLY_VALID"
    STRUCTURALLY_INVALID = "STRUCTURALLY_INVALID"


class DefectStatus(StrEnum):
    NOT_ESTABLISHED = "NOT_ESTABLISHED"
    ESTABLISHED = "ESTABLISHED"


class RecoilSpecies(_Frozen):
    """An isotope (``mass_number`` set) or an element aggregate (``mass_number`` None)."""

    element: str
    mass_number: int | None = Field(default=None, gt=0, lt=400)

    @model_validator(mode="after")
    def _element_symbol(self) -> RecoilSpecies:
        if self.element != UNKNOWN_ELEMENT and not _ELEMENT.match(self.element):
            raise ValueError(f"{self.element!r} is not an element symbol")
        return self

    @property
    def key(self) -> str:
        return self.element if self.mass_number is None else f"{self.element}{self.mass_number}"

    def to_element(self) -> RecoilSpecies:
        return RecoilSpecies(element=self.element)


class SpeciesSpectrum(_Frozen):
    """PKA production of one recoiling species on explicit energy bins (MeV).

    ``parent`` is the target atom that recoiled, when the source says so. For elastic
    scattering the parent and the recoil are the same species; for (n,gamma) the recoil is
    the product but the lattice site belonged to the parent.
    """

    species: RecoilSpecies
    parent: RecoilSpecies | None = None
    bin_low_mev: tuple[float, ...]
    bin_high_mev: tuple[float, ...]
    rates: tuple[float, ...]

    @model_validator(mode="after")
    def _bins(self) -> SpeciesSpectrum:
        n = len(self.rates)
        if n == 0:
            raise ValueError("a spectrum needs at least one bin")
        if len(self.bin_low_mev) != n or len(self.bin_high_mev) != n:
            raise ValueError("bin edges and rates must have the same length")
        for lo, hi, r in zip(self.bin_low_mev, self.bin_high_mev, self.rates, strict=True):
            if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(r)):
                raise ValueError("bin edges and rates must be finite")
            if lo < 0 or hi <= lo:
                raise ValueError("each bin needs 0 <= low < high")
            if r < 0:
                raise ValueError("rates must be non-negative")
        for prev, nxt in zip(self.bin_high_mev, self.bin_low_mev[1:], strict=False):
            if nxt < prev - SUM_TOLERANCE * max(1.0, prev):
                raise ValueError("bins must be ordered and non-overlapping")
        return self

    @property
    def total_rate(self) -> float:
        return math.fsum(self.rates)

    @property
    def site_element(self) -> str:
        return (self.parent or self.species).element


class PKASourceProvenance(_Frozen):
    """Where a spectrum came from. Hashes tie it to bytes; nothing here is a result claim."""

    source: str
    executable_sha256: str | None = None
    executable_path: str | None = None
    nuclear_data_identity: str
    input_sha256: dict[str, str] = Field(default_factory=dict)
    output_sha256: dict[str, str] = Field(default_factory=dict)
    execution: dict[str, object] = Field(default_factory=dict)
    raw_output_dir: str | None = None
    numerical_validation: ValidationStatus = ValidationStatus.NOT_VALIDATED
    notes: tuple[str, ...] = ()


class IrradiationScenario(_Frozen):
    scenario_id: str = Field(min_length=1)
    parent_alloy_id: str = Field(min_length=1)
    composition: Mapping[str, float]
    isotopic_composition: Mapping[str, Mapping[int, float]] = Field(default_factory=dict)
    neutron_spectrum_identity: str = Field(min_length=1)
    flux_normalization: str = Field(min_length=1)
    energy_grid_mev: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _composition(self) -> IrradiationScenario:
        if not self.composition:
            raise ValueError("composition is empty")
        if any((not math.isfinite(v)) or v <= 0 for v in self.composition.values()):
            raise ValueError("composition fractions must be positive and finite")
        if abs(math.fsum(self.composition.values()) - 1.0) > 1e-6:
            raise ValueError("composition fractions must sum to 1")
        for element, isotopes in self.isotopic_composition.items():
            if element not in self.composition:
                raise ValueError(f"isotopic composition names {element}, not in the composition")
            if abs(math.fsum(isotopes.values()) - 1.0) > 1e-6:
                raise ValueError(f"isotope fractions of {element} must sum to 1")
        if any(
            b <= a for a, b in zip(self.energy_grid_mev, self.energy_grid_mev[1:], strict=False)
        ):
            raise ValueError("energy grid must be strictly increasing")
        return self


class PKASpectrum(_Frozen):
    """Species-resolved PKA production for one scenario."""

    scenario: IrradiationScenario
    spectra: tuple[SpeciesSpectrum, ...]
    rate_unit: str = Field(min_length=1)
    rate_basis: str = RATE_BASIS_UNSPECIFIED
    energy_unit: str = "MeV"
    provenance: PKASourceProvenance

    @model_validator(mode="after")
    def _consistent(self) -> PKASpectrum:
        if not self.spectra:
            raise ValueError("a PKA spectrum needs at least one species")
        if self.energy_unit != "MeV":
            raise ValueError("energy_unit must be 'MeV'; convert at the adapter boundary")
        seen: set[tuple[str, str | None]] = set()
        for s in self.spectra:
            ident = (s.species.key, s.parent.key if s.parent else None)
            if ident in seen:
                raise ValueError(f"duplicate spectrum for {ident}")
            seen.add(ident)
        if self.total_rate <= 0:
            raise ValueError("total PKA rate is zero; the output is empty or incomplete")
        return self

    @property
    def total_rate(self) -> float:
        return math.fsum(s.total_rate for s in self.spectra)

    def species_probabilities(self) -> dict[str, float]:
        """Probability that the next PKA is of each species (isotope or element key)."""
        total = self.total_rate
        out: dict[str, float] = {}
        for s in self.spectra:
            out[s.species.key] = out.get(s.species.key, 0.0) + s.total_rate / total
        return out

    def element_rates(self) -> dict[str, float]:
        """Isotope-to-element aggregation. Isotope identity is dropped here and only here."""
        out: dict[str, float] = {}
        for s in self.spectra:
            out[s.species.element] = out.get(s.species.element, 0.0) + s.total_rate
        return out

    def element_spectra(self) -> tuple[SpeciesSpectrum, ...]:
        """Per-element spectra. Bins must be identical across the summed isotopes."""
        groups: dict[str, list[SpeciesSpectrum]] = {}
        for s in self.spectra:
            groups.setdefault(s.species.element, []).append(s)
        merged = []
        for element, items in groups.items():
            ref = items[0]
            for other in items[1:]:
                if (other.bin_low_mev, other.bin_high_mev) != (ref.bin_low_mev, ref.bin_high_mev):
                    raise ContractError(
                        f"cannot aggregate {element} isotopes: their energy bins differ"
                    )
            rates = tuple(math.fsum(it.rates[i] for it in items) for i in range(len(ref.rates)))
            merged.append(
                SpeciesSpectrum(
                    species=RecoilSpecies(element=element),
                    bin_low_mev=ref.bin_low_mev,
                    bin_high_mev=ref.bin_high_mev,
                    rates=rates,
                )
            )
        return tuple(merged)


class PKAEvent(_Frozen):
    index: int = Field(ge=0)
    species: RecoilSpecies
    energy_mev: float = Field(gt=0.0)
    weight: float = Field(default=1.0, gt=0.0)
    time_s: float | None = None
    interval_s: float | None = None
    site_index: int | None = None
    site_element: str | None = None
    chemical_configuration: tuple[str, ...] | None = None
    probability: float = Field(gt=0.0, le=1.0)


class SurvivingDefects(_Frozen):
    status: DefectStatus
    reason: str
    n_frenkel_pairs: float | None = None


class CascadeResponse(Protocol):
    """Optional PKA-to-surviving-defects model. Absent by default."""

    def surviving_defects(self, event: PKAEvent) -> SurvivingDefects: ...


class NoCascadeModel:
    """The default: no cascade model, so surviving-defect production is not established."""

    def surviving_defects(self, event: PKAEvent) -> SurvivingDefects:
        return SurvivingDefects(
            status=DefectStatus.NOT_ESTABLISHED,
            reason="no cascade-response model is configured; a PKA is not a surviving defect",
        )


def elements_of(species: Iterable[RecoilSpecies]) -> set[str]:
    return {s.element for s in species}


__all__ = [
    "RATE_BASIS_PER_ATOM_PER_S",
    "RATE_BASIS_UNSPECIFIED",
    "UNKNOWN_ELEMENT",
    "CascadeResponse",
    "ContractError",
    "DefectStatus",
    "IrradiationScenario",
    "NoCascadeModel",
    "PKAEvent",
    "PKASourceProvenance",
    "PKASpectrum",
    "RecoilSpecies",
    "SpeciesSpectrum",
    "SurvivingDefects",
    "ValidationStatus",
    "elements_of",
]
