"""Element-set -> formula enumeration for multi-composition campaigns.

Single-composition discovery (:mod:`matsim_agents.discovery.seeds`,
:mod:`matsim_agents.discovery.stability`) already covers AFLOW prototype
matching, pyXtal random generation, relaxation, and hull ranking for one
target formula. What is missing for a campaign spanning a whole element set
(e.g. Nb, Ta, O) is the layer that decides *which formulas* to generate seeds
for in the first place. This module is that layer: a deterministic,
reproducible generator that enumerates chemically-screened candidate
formulas from a user-supplied element set and coefficient/oxidation-state
policy.

Charge balance here is a screening heuristic, not proof of stability or
synthesizability: it only checks whether *some* combination of the allowed
oxidation states per element can sum to zero net charge.
"""

from __future__ import annotations

from itertools import combinations, product

from pydantic import BaseModel, Field, model_validator

from matsim_agents.discovery.composition import _hill_order, _reduce


class FormulaGenerationPolicy(BaseModel):
    """Deterministic constraints for enumerating candidate formulas."""

    elements: list[str] = Field(min_length=1)
    minimum_species: int = Field(2, ge=1)
    maximum_species: int = Field(3, ge=1)
    minimum_coefficient: int = Field(1, ge=1)
    maximum_coefficient: int = Field(6, ge=1)
    maximum_atoms_in_reduced_formula: int = Field(12, ge=1)
    include_binary_endmembers: bool = True
    include_mixed_oxides: bool = True
    require_charge_balance: bool = True
    oxidation_states: dict[str, list[int]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent_bounds(self) -> FormulaGenerationPolicy:
        if self.maximum_species < self.minimum_species:
            raise ValueError("maximum_species must be >= minimum_species")
        if self.maximum_coefficient < self.minimum_coefficient:
            raise ValueError("maximum_coefficient must be >= minimum_coefficient")
        return self


class FormulaCandidate(BaseModel):
    """One candidate formula tracked across a discovery campaign."""

    formula_id: str
    reduced_formula: str
    elements: dict[str, int]
    allowed_oxidation_states: dict[str, list[int]] = Field(default_factory=dict)
    charge_balanced: bool
    generation_source: str = "deterministic"  # "deterministic" | "llm" | "deterministic+llm"
    llm_contributors: list[str] = Field(default_factory=list)
    model_disagreements: dict[str, str] = Field(default_factory=dict)
    acceptance_reason: str | None = None
    rejection_reason: str | None = None
    iteration_created: int = 0
    active: bool = True


def _charge_balance_possible(
    elements: dict[str, int], oxidation_states: dict[str, list[int]]
) -> bool:
    """True if some choice of one oxidation state per element sums to zero net charge."""
    symbols = list(elements.keys())
    if any(sym not in oxidation_states or not oxidation_states[sym] for sym in symbols):
        return False
    for combo in product(*(oxidation_states[sym] for sym in symbols)):
        if sum(elements[sym] * state for sym, state in zip(symbols, combo, strict=True)) == 0:
            return True
    return False


def enumerate_formulas(policy: FormulaGenerationPolicy) -> list[FormulaCandidate]:
    """Deterministically enumerate candidate formulas from an element-set policy.

    Every structurally-valid formula (species count, coefficient range, atom
    limit) is returned. When ``require_charge_balance`` is set, formulas that
    fail the charge-balance screen are still returned but with ``active=False``
    and a ``rejection_reason`` so the caller can audit why they were excluded.
    """
    results: dict[str, FormulaCandidate] = {}
    max_species = min(policy.maximum_species, len(policy.elements))
    coeff_range = range(policy.minimum_coefficient, policy.maximum_coefficient + 1)
    for n_species in range(policy.minimum_species, max_species + 1):
        if n_species == 2 and not policy.include_binary_endmembers:
            continue
        if n_species >= 3 and not policy.include_mixed_oxides:
            continue
        for combo in combinations(policy.elements, n_species):
            for coeffs in product(coeff_range, repeat=n_species):
                raw = dict(zip(combo, coeffs, strict=True))
                reduced = _reduce(raw)
                if sum(reduced.values()) > policy.maximum_atoms_in_reduced_formula:
                    continue
                formula = "".join(
                    f"{el}{reduced[el] if reduced[el] > 1 else ''}" for el in _hill_order(reduced)
                )
                if formula in results:
                    continue
                charge_ok = True
                if policy.require_charge_balance:
                    charge_ok = _charge_balance_possible(reduced, policy.oxidation_states)
                results[formula] = FormulaCandidate(
                    formula_id=f"det-{formula}",
                    reduced_formula=formula,
                    elements=reduced,
                    allowed_oxidation_states={
                        el: policy.oxidation_states.get(el, []) for el in reduced
                    },
                    charge_balanced=charge_ok,
                    generation_source="deterministic",
                    active=charge_ok,
                    acceptance_reason=(
                        "within formula policy constraints and charge-balanced"
                        if charge_ok
                        else None
                    ),
                    rejection_reason=(
                        None if charge_ok else "fails charge balance for given oxidation states"
                    ),
                )
    return sorted(results.values(), key=lambda c: (len(c.elements), c.reduced_formula))
