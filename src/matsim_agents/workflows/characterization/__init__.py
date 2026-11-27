"""Characterize a material against an explicit contract with deterministic recipes.

A characterization answers one question about given structures of one
composition, with no optimization objective. Its contract
(:class:`CharacterizationContract`) declares the objective, the observables
with their units and minimum fidelity, the methods, the numerical tolerances,
the references, the budget, and the approvals. A recipe plans the calculations
as an ordered task graph (:func:`plan_recipe`); the numerical work is done by
the existing deterministic operators (:func:`matsim_agents.workflows.relaxation.
run_relaxation`, :func:`matsim_agents.science.thermodynamics.stability.score_stability`), never
by an LLM. The answer (:class:`~matsim_agents.core.contracts.
CharacterizationReport`) states a conclusion only when the evidence suffices
and says otherwise what is missing:

- ``relaxation``: relaxed structures with converged residual forces;
- ``energy_ranking``: the lowest-energy structure among same-composition
  candidates, which is not a thermodynamic-stability claim;
- ``thermodynamic_stability``: the energy above the convex hull of declared,
  method-compatible DFT elemental and competing-phase references. MLIP
  energies, clean relaxations, or a low total energy are never sufficient; a
  provisional reference set gives ``inconclusive``.

DFT runs only when a required observable needs DFT fidelity. Every step's
outcome is saved as it completes, so running the same contract again resumes
with the outstanding steps and never repeats a completed calculation.
"""

from __future__ import annotations

from matsim_agents.workflows.characterization.contract import (
    CharacterizationContract,
    CharacterizationMethods,
    NumericalTolerances,
    load_contract,
)
from matsim_agents.workflows.characterization.execute import run_characterization
from matsim_agents.workflows.characterization.plan import RecipeStep, plan_recipe
from matsim_agents.workflows.characterization.state import outstanding_steps

__all__ = [
    "CharacterizationContract",
    "CharacterizationMethods",
    "NumericalTolerances",
    "RecipeStep",
    "load_contract",
    "outstanding_steps",
    "plan_recipe",
    "run_characterization",
]
