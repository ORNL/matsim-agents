"""Binding and insertion/solution energy contracts for gas and transmutation defects.

Sign convention (fixed, recorded in every result): a **positive binding energy means the bound
complex is lower in energy than its separated fragments (attractive)**.

    E_b = sum_f E(fragment_f) - E(complex) - (n_fragments - 1) * E(host)

All energies are total energies of cells with the same lattice-site count, so the host term
removes the extra reference cells. The reaction must conserve every species; this is checked
from the compositions, so a mis-specified reaction is rejected instead of giving a number.

Insertion/solution energy needs explicit chemical-potential references:

    E_ins = E(defect) - E(host) - sum_s dn_s * mu_s

Software never decides a value is trustworthy. Results carry ``comparable`` (same method,
numerical settings, cell and spin treatment, converged) and ``qualification`` which stays
``unqualified`` here; numerical validation needs separate convergence evidence.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
SIGN_CONVENTION = "positive binding energy = attractive (complex lower than separated fragments)"
INDEPENDENCE_NOTE = "uncertainty combined in quadrature assuming independent term errors"


class GasEnergeticsError(ValueError):
    """Raised when a reaction or its evidence cannot support a number."""


class EnergyTerm(BaseModel):
    """One total energy with the evidence needed to decide whether it can be combined."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1)
    config_id: str = Field(min_length=1)
    energy_eV: Finite
    composition: dict[str, int]
    n_lattice_sites: int = Field(gt=0)
    method: str = Field(min_length=1)
    settings_fingerprint: str = Field(min_length=1)
    spin_polarized: bool
    magnetic_config_id: str | None = None
    magnetic_class: str | None = None
    converged: bool
    uncertainty_eV: Finite | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> EnergyTerm:
        if any(n < 0 for n in self.composition.values()):
            raise ValueError("composition counts must be non-negative")
        if self.spin_polarized and self.magnetic_config_id is None:
            raise ValueError("a spin-polarized energy must name its magnetic configuration")
        return self


class FragmentTerm(BaseModel):
    model_config = ConfigDict(frozen=True)

    term: EnergyTerm
    multiplicity: int = Field(default=1, ge=1)


class ReferenceTerm(BaseModel):
    """Explicit chemical potential for one species."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    species: str
    mu_eV: Finite
    definition: str = Field(min_length=1)
    source: str = Field(min_length=1)
    method: str = Field(min_length=1)
    uncertainty_eV: Finite | None = Field(default=None, ge=0)


class EnergyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    quantity: Literal["binding", "insertion"]
    value_eV: float | None
    uncertainty_eV: float | None
    comparable: bool
    problems: tuple[str, ...]
    sign_convention: str
    terms: tuple[str, ...]
    reference_terms: tuple[str, ...] = ()
    magnetic_state: str
    qualification: Literal["unqualified"] = "unqualified"
    numerical_qualification: tuple[str, ...]
    notes: tuple[str, ...] = ()


def _sum_compositions(items: Sequence[tuple[Mapping[str, int], int]]) -> Counter:
    total: Counter = Counter()
    for composition, factor in items:
        for symbol, n in composition.items():
            total[symbol] += factor * n
    return total


def _comparability(terms: Sequence[EnergyTerm]) -> tuple[list[str], list[str], str]:
    problems: list[str] = []
    numerical: list[str] = []
    first = terms[0]
    for term in terms[1:]:
        if term.method != first.method:
            problems.append(f"method differs: {term.label} uses {term.method}, not {first.method}")
        if term.settings_fingerprint != first.settings_fingerprint:
            problems.append(f"numerical settings differ for {term.label}")
        if term.n_lattice_sites != first.n_lattice_sites:
            problems.append(f"cell size differs for {term.label}")
        if term.spin_polarized != first.spin_polarized:
            problems.append(f"spin treatment differs for {term.label}")
        if (
            term.spin_polarized
            and first.spin_polarized
            and term.magnetic_class is not None
            and first.magnetic_class is not None
            and term.magnetic_class != first.magnetic_class
        ):
            problems.append(
                f"magnetic state differs: {term.label} is {term.magnetic_class}, "
                f"{first.label} is {first.magnetic_class}"
            )
    for term in terms:
        if not term.converged:
            problems.append(f"{term.label} is not converged")
        if term.spin_polarized and term.magnetic_class is None:
            numerical.append(f"{term.label}: converged magnetic class not recorded")
        if term.uncertainty_eV is None:
            numerical.append(f"{term.label}: no uncertainty supplied")
    numerical.append(
        "k-point and cutoff convergence of the energy difference is not evidenced here"
    )
    state = (
        "spin_unpolarized"
        if not first.spin_polarized
        else f"spin_polarized:{first.magnetic_class or 'class_unrecorded'}"
    )
    return problems, numerical, state


def _combine_uncertainty(weighted: Sequence[tuple[float, float | None]]) -> float | None:
    if any(u is None for _, u in weighted):
        return None
    return math.sqrt(sum((w * u) ** 2 for w, u in weighted if u is not None))


def binding_energy(
    complex_term: EnergyTerm,
    fragments: Sequence[FragmentTerm],
    host: EnergyTerm,
) -> EnergyResult:
    """Binding energy of ``complex_term`` relative to separated ``fragments`` in the same cell."""
    if not fragments:
        raise GasEnergeticsError("a binding energy needs at least one fragment")
    n_frag = sum(f.multiplicity for f in fragments)
    if n_frag < 2:
        raise GasEnergeticsError("a binding energy needs at least two fragments")
    lhs = _sum_compositions(
        [(f.term.composition, f.multiplicity) for f in fragments]
        + [(host.composition, -(n_frag - 1))]
    )
    rhs = _sum_compositions([(complex_term.composition, 1)])
    lhs += Counter()
    if {k: v for k, v in lhs.items() if v} != {k: v for k, v in rhs.items() if v}:
        raise GasEnergeticsError(
            f"reaction does not conserve species: fragments give {dict(lhs)}, "
            f"complex has {dict(rhs)}"
        )
    all_terms = [complex_term, *[f.term for f in fragments], host]
    problems, numerical, state = _comparability(all_terms)
    value = (
        sum(f.multiplicity * f.term.energy_eV for f in fragments)
        - complex_term.energy_eV
        - (n_frag - 1) * host.energy_eV
    )
    weighted = [(1.0, complex_term.uncertainty_eV), (float(n_frag - 1), host.uncertainty_eV)]
    weighted += [(float(f.multiplicity), f.term.uncertainty_eV) for f in fragments]
    uncertainty = _combine_uncertainty(weighted)
    comparable = not problems
    return EnergyResult(
        quantity="binding",
        value_eV=value if comparable else None,
        uncertainty_eV=uncertainty if comparable else None,
        comparable=comparable,
        problems=tuple(problems),
        sign_convention=SIGN_CONVENTION,
        terms=tuple(t.config_id for t in all_terms),
        magnetic_state=state,
        numerical_qualification=tuple(numerical),
        notes=(INDEPENDENCE_NOTE,),
    )


def insertion_energy(
    defect: EnergyTerm,
    host: EnergyTerm,
    references: Mapping[str, ReferenceTerm],
) -> EnergyResult:
    """Insertion/solution energy with an explicit chemical potential per changed species."""
    delta = Counter(defect.composition)
    delta.subtract(Counter(host.composition))
    changed = {s: n for s, n in delta.items() if n}
    if not changed:
        raise GasEnergeticsError("defect and host have identical composition; nothing inserted")
    missing = sorted(set(changed) - set(references))
    if missing:
        raise GasEnergeticsError(f"no explicit chemical-potential reference for {missing}")
    extra_method = sorted(s for s in changed if references[s].method != defect.method)
    problems, numerical, state = _comparability([defect, host])
    problems += [
        f"reference for {s} uses a different method than the cell energies" for s in extra_method
    ]
    value = (
        defect.energy_eV - host.energy_eV - sum(changed[s] * references[s].mu_eV for s in changed)
    )
    weighted = [(1.0, defect.uncertainty_eV), (1.0, host.uncertainty_eV)]
    weighted += [(float(changed[s]), references[s].uncertainty_eV) for s in changed]
    uncertainty = _combine_uncertainty(weighted)
    comparable = not problems
    return EnergyResult(
        quantity="insertion",
        value_eV=value if comparable else None,
        uncertainty_eV=uncertainty if comparable else None,
        comparable=comparable,
        problems=tuple(problems),
        sign_convention="E_ins = E(defect) - E(host) - sum dn_s * mu_s; positive = unfavourable",
        terms=(defect.config_id, host.config_id),
        reference_terms=tuple(
            f"{s}: {references[s].definition} [{references[s].source}]" for s in sorted(changed)
        ),
        magnetic_state=state,
        numerical_qualification=tuple(numerical),
        notes=(INDEPENDENCE_NOTE, f"species change dn = {dict(sorted(changed.items()))}"),
    )
