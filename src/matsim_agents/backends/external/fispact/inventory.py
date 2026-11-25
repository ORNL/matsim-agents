"""Reader for FISPACT-II inventory results, as the JSON that pypact writes from a ``.out`` file.

This reads a finished result. It never runs or installs FISPACT-II (licensed software) or pypact.
Units follow the FISPACT-II output: grams per nuclide, activity in Bq for the whole material,
heat in kW, dose rate in Sv/h. The JSON carries no uncertainties, library or code version, so
those are supplied by the caller or reported as missing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from matsim_agents.science.irradiation.nuclear_evidence import (
    ExternalNuclearEvidence,
    NuclearEvidenceError,
    NuclearFamily,
    NuclearQuantity,
)

APPLICATION = "FISPACT-II"
_G_PER_KG = 1000.0
_W_PER_KW = 1000.0


class _Reported(BaseModel):
    # Upstream adds fields over time; ignore the ones this reader does not use.
    model_config = ConfigDict(frozen=True, extra="ignore")


def _num() -> Any:
    return Field(0.0, ge=0, allow_inf_nan=False)


class FispactNuclide(_Reported):
    element: str
    isotope: int
    state: str = ""
    grams: float = _num()
    activity: float = _num()
    heat: float = _num()

    @property
    def name(self) -> str:
        return f"{self.element}{self.isotope}{self.state}"


class FispactDoseRate(_Reported):
    type: str = ""
    distance: float = _num()
    dose: float = _num()


class FispactStep(_Reported):
    irradiation_time: float = _num()
    cooling_time: float = _num()
    flux: float = _num()
    total_heat: float = _num()
    dose_rate: FispactDoseRate = FispactDoseRate()
    nuclides: tuple[FispactNuclide, ...] = ()

    @property
    def mass_kg(self) -> float:
        return math.fsum(n.grams for n in self.nuclides) / _G_PER_KG

    @property
    def total_activity_bq(self) -> float:
        return math.fsum(n.activity for n in self.nuclides)


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if mass_kg <= 0:
        raise NuclearEvidenceError("the selected step has no mass; cannot normalise per kg")

    quantities = [
        NuclearQuantity(
            name="specific_activity", value=step.total_activity_bq / mass_kg, unit="Bq/kg"
        ),
        NuclearQuantity(
            name="decay_heat", value=step.total_heat * _W_PER_KW / mass_kg, unit="W/kg"
        ),
    ]
    dose = step.dose_rate
    notes = [
        f"fispact_run: {inventory.run_name}".rstrip(),
        f"fispact_flux: {inventory.flux_name}".rstrip(),
        f"inventory mass at this step: {mass_kg:.6g} kg",
        "decay heat is the file's total_heat, converted from kW",
        "no uncertainty is reported by this result",
    ]
    if dose.type:
        quantities.append(NuclearQuantity(name="shutdown_dose_rate", value=dose.dose, unit="Sv/h"))
        dose_model = dose_model or f"FISPACT-II dose rate ({dose.type})"
        dose_geometry = dose_geometry or f"{dose.type}, distance {dose.distance:g} m"
    else:
        dose_model = dose_geometry = None
        notes.append("no dose rate in this step")

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


__all__ = [
    "APPLICATION",
    "FispactInventory",
    "FispactNuclide",
    "FispactStep",
    "parse_inventory",
    "read_inventory",
    "select_step",
    "sha256_file",
    "to_activation_evidence",
]
