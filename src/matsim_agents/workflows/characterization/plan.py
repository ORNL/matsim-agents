"""The ordered task graph of a recipe."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from matsim_agents.science.properties.recipes import (
    PROPERTY_RECIPES,
)
from matsim_agents.workflows.characterization.contract import CharacterizationContract


class RecipeStep(BaseModel):
    """One node of a recipe's task graph."""

    step_id: str
    operator: Literal["relax", "rank", "hull"]
    fidelity: Literal["mlip", "dft"] | None = None
    structure: str | None = None
    requires: list[str] = Field(default_factory=list)
    produces: list[str]
    needs_dft_approval: bool = False
    max_attempts: int = Field(
        1,
        description="Attempts before recovery; failed DFT attempts are retried per "
        "execution.recovery.DEFAULT_POLICY within the budget.",
    )


def plan_recipe(contract: CharacterizationContract) -> list[RecipeStep]:
    """The ordered task graph of a contract's recipe (deterministic).

    Property recipes have no relaxation graph: they assess existing evidence
    and report the calculations that are missing.
    """
    if contract.recipe in PROPERTY_RECIPES:
        return []
    fidelity: Literal["mlip", "dft"] = "dft" if contract.needs_dft() else "mlip"
    relax = [
        RecipeStep(
            step_id=f"relax-{fidelity}-{index:03d}",
            operator="relax",
            fidelity=fidelity,
            structure=str(path),
            produces=["relaxed_structure", "energy_per_atom", "residual_max_force"],
            needs_dft_approval=fidelity == "dft" and contract.approvals.before_dft,
        )
        for index, path in enumerate(contract.structures)
    ]
    if contract.recipe == "relaxation":
        return relax
    final = (
        RecipeStep(
            step_id="rank",
            operator="rank",
            requires=[step.step_id for step in relax],
            produces=["relative_energy_per_atom"],
        )
        if contract.recipe == "energy_ranking"
        else RecipeStep(
            step_id="hull",
            operator="hull",
            requires=[step.step_id for step in relax],
            produces=["formation_energy_per_atom", "energy_above_hull"],
        )
    )
    return [*relax, final]
