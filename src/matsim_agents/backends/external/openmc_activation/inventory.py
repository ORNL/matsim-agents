"""Activation inventory from an OpenMC ``openmc.deplete`` run, as evidence for the nuclear contract.

OpenMC is open source, so unlike FISPACT-II this backend can be run here. The inventory is the
number of atoms per nuclide at each time, plus the mass, activity and decay heat computed from
the depletion chain. Nothing in it is a measurement: there is no uncertainty and no dose rate.
This module does not import OpenMC, so a stored inventory can be converted anywhere.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from matsim_agents.science.irradiation.contracts import RecoilSpecies
from matsim_agents.science.irradiation.nuclear_evidence import (
    ExternalNuclearEvidence,
    NuclearEvidenceError,
    NuclearFamily,
    NuclearQuantity,
)

APPLICATION = "OpenMC"
GAS_ELEMENTS = ("H", "He")
_APPM = 1e6
_NUCLIDE = re.compile(r"^([A-Z][a-z]?)(\d+)(?:_[em]\d+)?$")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OpenMCStep(_Frozen):
    irradiation_time_s: float = Field(ge=0, allow_inf_nan=False)
    cooling_time_s: float = Field(ge=0, allow_inf_nan=False)
    atoms: dict[str, float]
    mass_kg: float = Field(gt=0, allow_inf_nan=False)
    activity_bq: float = Field(ge=0, allow_inf_nan=False)
    decay_heat_w: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("atoms")
    @classmethod
    def _finite_atoms(cls, atoms: dict[str, float]) -> dict[str, float]:
        for name, value in atoms.items():
            if not _NUCLIDE.match(name):
                raise ValueError(f"{name!r} is not a nuclide name such as Fe56 or Ag110_m1")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name}: atoms must be finite and non-negative")
        return atoms


class OpenMCInventory(_Frozen):
    openmc_version: str = Field(min_length=1)
    nuclear_data_library: str = Field(min_length=1)
    material_name: str = Field(min_length=1)
    flux_cm2_s: float = Field(gt=0, allow_inf_nan=False)
    temperature_K: float = Field(gt=0, allow_inf_nan=False)
    input_sha256: dict[str, str] = Field(min_length=1)
    output_sha256: dict[str, str] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()
    steps: tuple[OpenMCStep, ...] = Field(min_length=1)


def parse_inventory(data: Any) -> OpenMCInventory:
    try:
        return OpenMCInventory.model_validate(data)
    except ValidationError as exc:
        raise NuclearEvidenceError(f"invalid OpenMC inventory: {exc}") from None


def read_inventory(path: str | Path) -> OpenMCInventory:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise NuclearEvidenceError(f"{source}: {exc}") from None
    try:
        return OpenMCInventory.model_validate_json(text)
    except ValidationError as exc:
        raise NuclearEvidenceError(f"{source}: invalid OpenMC inventory: {exc}") from None


def select_step(inventory: OpenMCInventory, cooling_time_s: float) -> OpenMCStep:
    tolerance = 1e-9 * max(1.0, abs(cooling_time_s))
    matches = [s for s in inventory.steps if abs(s.cooling_time_s - cooling_time_s) <= tolerance]
    if not matches:
        available = sorted({s.cooling_time_s for s in inventory.steps})
        raise NuclearEvidenceError(
            f"no step at cooling time {cooling_time_s:g} s; available: "
            + ", ".join(f"{t:g}" for t in available)
        )
    return max(matches, key=lambda s: s.irradiation_time_s)


def _evidence_kwargs(
    inventory: OpenMCInventory, material_id: str, irradiation_scenario_id: str, synthetic: bool
) -> dict[str, Any]:
    return {
        "external_application": APPLICATION,
        "application_version": inventory.openmc_version,
        "nuclear_data_library": inventory.nuclear_data_library,
        "input_sha256": dict(inventory.input_sha256),
        "output_sha256": dict(inventory.output_sha256),
        "material_id": material_id,
        "irradiation_scenario_id": irradiation_scenario_id,
        "synthetic": synthetic,
    }


def to_activation_evidence(
    inventory: OpenMCInventory,
    *,
    cooling_time_s: float,
    material_id: str,
    irradiation_scenario_id: str,
    synthetic: bool = False,
) -> ExternalNuclearEvidence:
    """Family-C evidence at one cooling time: specific activity and decay heat, no dose rate."""
    if not inventory.output_sha256:
        raise NuclearEvidenceError("the inventory has no output_sha256 for its result files")
    step = select_step(inventory, cooling_time_s)
    return ExternalNuclearEvidence(
        family=NuclearFamily.ACTIVATION,
        exposure_time_s=step.irradiation_time_s or None,
        cooling_time_s=step.cooling_time_s,
        quantities=(
            NuclearQuantity(
                name="specific_activity",
                value=step.activity_bq / step.mass_kg,
                unit="Bq/kg",
            ),
            NuclearQuantity(name="decay_heat", value=step.decay_heat_w / step.mass_kg, unit="W/kg"),
        ),
        notes=(
            f"openmc_material: {inventory.material_name}",
            f"one-group flux {inventory.flux_cm2_s:.6g} n/cm2/s, {inventory.temperature_K:g} K",
            f"inventory mass at this step: {step.mass_kg:.6g} kg",
            "activity and decay heat are computed from the depletion chain's decay data",
            "no uncertainty and no shutdown dose rate are reported by this result",
            *inventory.notes,
        ),
        **_evidence_kwargs(inventory, material_id, irradiation_scenario_id, synthetic),
    )


def _fold(atoms: dict[str, float]) -> dict[str, float]:
    """Atoms per isotope key; metastable states fold into the ground-state key."""
    out: dict[str, float] = {}
    for name, count in atoms.items():
        match = _NUCLIDE.match(name)
        if match is None:
            raise NuclearEvidenceError(f"{name!r} is not a nuclide name")
        key = f"{match.group(1)}{match.group(2)}"
        out[key] = out.get(key, 0.0) + count
    return out


def to_transmutation_evidence(
    inventory: OpenMCInventory,
    *,
    cooling_time_s: float,
    material_id: str,
    irradiation_scenario_id: str,
    elements: tuple[str, ...] = GAS_ELEMENTS,
    synthetic: bool = False,
) -> ExternalNuclearEvidence:
    """Family-B evidence: appm of the chosen elements' isotopes, per initial atom."""
    if not inventory.output_sha256:
        raise NuclearEvidenceError("the inventory has no output_sha256 for its result files")
    initial = inventory.steps[0]
    if initial.irradiation_time_s != 0 or initial.cooling_time_s != 0:
        raise NuclearEvidenceError("the first step must be the unirradiated material")
    step = select_step(inventory, cooling_time_s)
    if step.irradiation_time_s <= 0:
        raise NuclearEvidenceError("the selected step has not been irradiated")
    start = _fold(initial.atoms)
    total = math.fsum(start.values())
    if total <= 0:
        raise NuclearEvidenceError("the initial material has no atoms")
    now = _fold(step.atoms)
    wanted = {e.lower() for e in elements}
    quantities = []
    for key in sorted(now):
        match = _NUCLIDE.match(key)
        if match is None or match.group(1).lower() not in wanted:
            continue
        quantities.append(
            NuclearQuantity(
                name="concentration_appm",
                value=now[key] / total * _APPM,
                unit="appm",
                isotope=RecoilSpecies(element=match.group(1), mass_number=int(match.group(2))),
            )
        )
    if not quantities:
        raise NuclearEvidenceError(f"no {', '.join(elements)} isotopes in the selected step")
    return ExternalNuclearEvidence(
        family=NuclearFamily.TRANSMUTATION,
        isotope_inventory={key: value / total for key, value in sorted(start.items())},
        exposure_time_s=step.irradiation_time_s,
        cooling_time_s=step.cooling_time_s,
        quantities=tuple(quantities),
        notes=(
            f"openmc_material: {inventory.material_name}",
            f"one-group flux {inventory.flux_cm2_s:.6g} n/cm2/s, {inventory.temperature_K:g} K",
            "appm per initial atom; no uncertainty reported",
            *inventory.notes,
        ),
        **_evidence_kwargs(inventory, material_id, irradiation_scenario_id, synthetic),
    )


__all__ = [
    "APPLICATION",
    "GAS_ELEMENTS",
    "OpenMCInventory",
    "OpenMCStep",
    "parse_inventory",
    "read_inventory",
    "select_step",
    "to_activation_evidence",
    "to_transmutation_evidence",
]
