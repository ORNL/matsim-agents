"""Reader for FISPACT-II inventory results, as the JSON that pypact writes from a ``.out`` file.

This reads a finished result. It never runs or installs FISPACT-II (licensed software) or pypact.
Units follow the FISPACT-II output: grams per nuclide, activity in Bq for the whole material,
heat in kW, dose rate in Sv/h. The JSON carries no uncertainties, library or code version, so
those are supplied by the caller or reported as missing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from matsim_agents.core.hashing import sha256_file
from matsim_agents.science.irradiation.contracts import RecoilSpecies
from matsim_agents.science.irradiation.nuclear_evidence import (
    ExternalNuclearEvidence,
    NuclearEvidenceError,
    NuclearFamily,
    NuclearQuantity,
)

APPLICATION = "FISPACT-II"
_G_PER_KG = 1000.0
_W_PER_KW = 1000.0
_APPM = 1e6
GAS_ELEMENTS = ("H", "He")


class _Reported(BaseModel):
    # Upstream adds fields over time; ignore the ones this reader does not use.
    model_config = ConfigDict(frozen=True, extra="ignore")


def _num() -> Any:
    return Field(0.0, ge=0, allow_inf_nan=False)


def _optional() -> Any:
    # A value the file does not report stays None; only a reported 0 is a measured zero.
    return Field(None, ge=0, allow_inf_nan=False)


class FispactNuclide(_Reported):
    element: str
    isotope: int
    state: str = ""
    grams: float | None = _optional()
    activity: float | None = _optional()
    heat: float | None = _optional()

    @property
    def name(self) -> str:
        return f"{self.element}{self.isotope}{self.state}"


class FispactDoseRate(_Reported):
    type: str = ""
    distance: float = _num()
    dose: float | None = _optional()


class FispactStep(_Reported):
    irradiation_time: float = _num()
    cooling_time: float = _num()
    flux: float = _num()
    total_heat: float | None = _optional()
    dose_rate: FispactDoseRate = FispactDoseRate()
    nuclides: tuple[FispactNuclide, ...] = ()

    @property
    def mass_kg(self) -> float | None:
        """None unless every nuclide reports its mass."""
        grams = [n.grams for n in self.nuclides]
        if not grams or any(g is None for g in grams):
            return None
        return math.fsum(g for g in grams if g is not None) / _G_PER_KG

    @property
    def total_activity_bq(self) -> float | None:
        """None unless every nuclide reports its activity."""
        values = [n.activity for n in self.nuclides]
        if not values or any(v is None for v in values):
            return None
        return math.fsum(v for v in values if v is not None)


class FispactInventory(_Reported):
    run_name: str = ""
    flux_name: str = ""
    steps: tuple[FispactStep, ...] = Field(min_length=1)
    source_sha256: str = ""


def parse_inventory(data: Mapping[str, Any], *, source_sha256: str = "") -> FispactInventory:
    """Validate pypact-style JSON (``run_data`` and ``inventory_data``)."""
    try:
        run = data.get("run_data", {})
        return FispactInventory(
            run_name=run.get("run_name", ""),
            flux_name=run.get("flux_name", ""),
            steps=tuple(data["inventory_data"]),
            source_sha256=source_sha256,
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise NuclearEvidenceError(f"not a FISPACT-II inventory JSON: {exc!r}") from None
    except ValidationError as exc:
        raise NuclearEvidenceError(f"invalid FISPACT-II inventory: {exc}") from None


def read_inventory(path: str | Path) -> FispactInventory:
    """Read a FISPACT-II inventory JSON file; the file's SHA-256 is kept for provenance."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NuclearEvidenceError(f"{source}: {exc}") from None
    if not isinstance(data, dict):
        raise NuclearEvidenceError(f"{source}: the top level must be a JSON object")
    return parse_inventory(data, source_sha256=sha256_file(source))


def select_step(inventory: FispactInventory, cooling_time_s: float) -> FispactStep:
    """The step at this cooling time; the one with the longest irradiation if several match."""
    tolerance = 1e-9 * max(1.0, abs(cooling_time_s))
    matches = [s for s in inventory.steps if abs(s.cooling_time - cooling_time_s) <= tolerance]
    if not matches:
        available = sorted({s.cooling_time for s in inventory.steps})
        raise NuclearEvidenceError(
            f"no step at cooling time {cooling_time_s:g} s; available: "
            + ", ".join(f"{t:g}" for t in available)
        )
    return max(matches, key=lambda s: s.irradiation_time)


def to_activation_evidence(
    inventory: FispactInventory,
    *,
    cooling_time_s: float,
    material_id: str,
    irradiation_scenario_id: str,
    application_version: str,
    nuclear_data_library: str,
    input_files: Mapping[str, str | Path],
    output_name: str = "inventory.json",
    dose_model: str | None = None,
    dose_geometry: str | None = None,
    synthetic: bool = False,
) -> ExternalNuclearEvidence:
    """Family-C evidence (activity, decay heat, dose rate) at one cooling time.

    ``input_files`` are the FISPACT-II input files (for example the ``.i`` and ``.files``);
    they are hashed because the JSON does not contain them. The dose model and geometry default
    to what the result file states; no uncertainty is reported.
    """
    if not inventory.source_sha256:
        raise NuclearEvidenceError("read the inventory with read_inventory so it has a hash")
    if not input_files:
        raise NuclearEvidenceError("input_files are required to hash the FISPACT-II inputs")
    step = select_step(inventory, cooling_time_s)
    mass_kg = step.mass_kg
    if mass_kg is None or mass_kg <= 0:
        raise NuclearEvidenceError(
            "the selected step does not report a positive mass for every nuclide; "
            "cannot normalise per kg"
        )

    quantities: list[NuclearQuantity] = []
    notes = [
        f"fispact_run: {inventory.run_name}".rstrip(),
        f"fispact_flux: {inventory.flux_name}".rstrip(),
        f"inventory mass at this step: {mass_kg:.6g} kg",
        "decay heat is the file's total_heat, converted from kW",
        "no uncertainty is reported by this result",
    ]
    activity = step.total_activity_bq
    if activity is None:
        notes.append("activity not reported for every nuclide in this step; omitted")
    else:
        quantities.append(
            NuclearQuantity(name="specific_activity", value=activity / mass_kg, unit="Bq/kg")
        )
    if step.total_heat is None:
        notes.append("total_heat not reported in this step; decay heat omitted")
    else:
        quantities.append(
            NuclearQuantity(
                name="decay_heat", value=step.total_heat * _W_PER_KW / mass_kg, unit="W/kg"
            )
        )
    dose = step.dose_rate
    if dose.type and dose.dose is not None:
        quantities.append(NuclearQuantity(name="shutdown_dose_rate", value=dose.dose, unit="Sv/h"))
        dose_model = dose_model or f"FISPACT-II dose rate ({dose.type})"
        dose_geometry = dose_geometry or f"{dose.type}, distance {dose.distance:g} m"
    else:
        dose_model = dose_geometry = None
        notes.append("no dose rate in this step")
    if not quantities:
        raise NuclearEvidenceError("the selected step reports no activity, heat or dose rate")

    return ExternalNuclearEvidence(
        family=NuclearFamily.ACTIVATION,
        external_application=APPLICATION,
        application_version=application_version,
        nuclear_data_library=nuclear_data_library,
        input_sha256={Path(name).name: sha256_file(path) for name, path in input_files.items()},
        output_sha256={output_name: inventory.source_sha256},
        material_id=material_id,
        irradiation_scenario_id=irradiation_scenario_id,
        exposure_time_s=step.irradiation_time or None,
        cooling_time_s=step.cooling_time,
        quantities=tuple(quantities),
        dose_model=dose_model,
        dose_geometry=dose_geometry,
        synthetic=synthetic,
        notes=tuple(notes),
    )


def _atoms(nuclides: tuple[FispactNuclide, ...]) -> dict[str, float]:
    """Relative atom counts (mol of atoms) per isotope key; metastable states fold into one."""
    out: dict[str, float] = {}
    for n in nuclides:
        if n.isotope <= 0:
            raise NuclearEvidenceError(f"{n.name}: the isotope must be a positive mass number")
        if n.grams is None:
            raise NuclearEvidenceError(f"{n.name}: the file does not report its mass")
        key = f"{n.element}{n.isotope}"
        out[key] = out.get(key, 0.0) + n.grams / n.isotope
    return out


def to_transmutation_evidence(
    inventory: FispactInventory,
    *,
    cooling_time_s: float,
    material_id: str,
    irradiation_scenario_id: str,
    application_version: str,
    nuclear_data_library: str,
    input_files: Mapping[str, str | Path],
    output_name: str = "inventory.json",
    elements: tuple[str, ...] = GAS_ELEMENTS,
    synthetic: bool = False,
) -> ExternalNuclearEvidence:
    """Family-B evidence: appm of the chosen elements' isotopes after the selected step.

    The JSON gives grams, not atoms, so atoms are taken as grams / mass number; the isotope mass
    in u is not in the file, which biases each count by under 1 % (mass defect). appm is per
    atom of the initial material, taken from the first step.
    """
    if not inventory.source_sha256:
        raise NuclearEvidenceError("read the inventory with read_inventory so it has a hash")
    if not input_files:
        raise NuclearEvidenceError("input_files are required to hash the FISPACT-II inputs")
    initial = inventory.steps[0]
    if initial.irradiation_time != 0 or initial.cooling_time != 0:
        raise NuclearEvidenceError("the first step must be the unirradiated material")
    step = select_step(inventory, cooling_time_s)
    if step.irradiation_time <= 0:
        raise NuclearEvidenceError("the selected step has not been irradiated")
    start = _atoms(initial.nuclides)
    total = math.fsum(start.values())
    if total <= 0:
        raise NuclearEvidenceError("the initial material has no atoms")
    now = _atoms(step.nuclides)
    wanted = {e.lower() for e in elements}
    seen = {n.element + str(n.isotope): n for n in step.nuclides if n.element.lower() in wanted}
    quantities = tuple(
        NuclearQuantity(
            name="concentration_appm",
            value=now[key] / total * _APPM,
            unit="appm",
            isotope=RecoilSpecies(element=n.element, mass_number=n.isotope),
        )
        for key, n in sorted(seen.items())
    )
    if not quantities:
        raise NuclearEvidenceError(f"no {', '.join(elements)} isotopes in the selected step")
    return ExternalNuclearEvidence(
        family=NuclearFamily.TRANSMUTATION,
        external_application=APPLICATION,
        application_version=application_version,
        nuclear_data_library=nuclear_data_library,
        input_sha256={Path(name).name: sha256_file(path) for name, path in input_files.items()},
        output_sha256={output_name: inventory.source_sha256},
        material_id=material_id,
        isotope_inventory={key: value / total for key, value in sorted(start.items())},
        irradiation_scenario_id=irradiation_scenario_id,
        exposure_time_s=step.irradiation_time,
        cooling_time_s=step.cooling_time,
        quantities=quantities,
        synthetic=synthetic,
        notes=(
            f"fispact_run: {inventory.run_name}".rstrip(),
            "atoms = grams / mass number; appm per initial atom; no uncertainty reported",
        ),
    )


__all__ = [
    "APPLICATION",
    "GAS_ELEMENTS",
    "FispactInventory",
    "FispactNuclide",
    "FispactStep",
    "parse_inventory",
    "read_inventory",
    "select_step",
    "sha256_file",
    "to_activation_evidence",
    "to_transmutation_evidence",
]
