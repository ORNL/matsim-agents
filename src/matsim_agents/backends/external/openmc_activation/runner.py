"""Run ``openmc.deplete`` for one material in a fixed flux and write an ``OpenMCInventory``.

The neutron spectrum is collapsed to one group at one temperature, and the cross sections are
held fixed for the whole run. OpenMC is imported only when a run starts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

from matsim_agents.backends.external.openmc_activation.inventory import (
    OpenMCInventory,
    OpenMCStep,
)
from matsim_agents.science.irradiation.nuclear_evidence import NuclearEvidenceError

MATERIAL_ID = "1"
_MATERIAL_VOLUME_CM3 = 1.0  # makes the [n-cm/src] flux of the operator equal n/cm2/s per source n


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_file(value: str | os.PathLike[str] | None, variable: str) -> Path:
    path = Path(value) if value else Path(os.environ.get(variable, ""))
    if not path.is_file():
        raise NuclearEvidenceError(f"set {variable} (or pass the path) to an existing file")
    return path


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise NuclearEvidenceError(f"{name} must be finite and positive, got {value}")


def run_activation(
    *,
    material_name: str,
    composition: dict[str, float],
    density_g_cm3: float,
    energies_eV: Sequence[float],
    group_flux_cm2_s: Sequence[float],
    irradiation_s: float,
    cooling_times_s: Sequence[float],
    output_dir: str | os.PathLike[str],
    temperature_K: float = 293.6,
    nuclides: Sequence[str] | None = None,
    chain_file: str | os.PathLike[str] | None = None,
    cross_sections: str | os.PathLike[str] | None = None,
) -> OpenMCInventory:
    """Irradiate ``composition`` (element -> atom fraction), cool it, and store the inventory.

    ``energies_eV`` are the group boundaries (one more than ``group_flux_cm2_s``). ``nuclides``
    limits which nuclides get collapsed cross sections; None uses every chain nuclide that has
    data, which is the slow, complete choice. Writes ``inventory.json`` and the raw OpenMC
    results next to each other in ``output_dir``.
    """
    if not composition or any(f <= 0 or not math.isfinite(f) for f in composition.values()):
        raise NuclearEvidenceError("composition needs positive atom fractions")
    _positive("density_g_cm3", density_g_cm3)
    _positive("irradiation_s", irradiation_s)
    _positive("temperature_K", temperature_K)
    if len(energies_eV) != len(group_flux_cm2_s) + 1:
        raise NuclearEvidenceError("energies_eV must have one more entry than group_flux_cm2_s")
    if any(b <= a for a, b in zip(energies_eV, energies_eV[1:], strict=False)):
        raise NuclearEvidenceError("energies_eV must increase")
    if any(f < 0 or not math.isfinite(f) for f in group_flux_cm2_s):
        raise NuclearEvidenceError("group fluxes must be finite and non-negative")
    flux = float(sum(group_flux_cm2_s))
    _positive("total flux", flux)
    cooling = sorted(set(float(t) for t in cooling_times_s) - {0.0})
    if any(t < 0 or not math.isfinite(t) for t in cooling_times_s):
        raise NuclearEvidenceError("cooling times must be finite and non-negative")

    chain = _existing_file(chain_file, "OPENMC_CHAIN_FILE")
    xs_xml = _existing_file(cross_sections, "OPENMC_CROSS_SECTIONS")

    import openmc
    import openmc.deplete as dep

    openmc.config["chain_file"] = str(chain)
    openmc.config["cross_sections"] = str(xs_xml)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "depletion_results.h5"

    material = openmc.Material(material_id=int(MATERIAL_ID), name=material_name)
    for element, fraction in composition.items():
        material.add_element(element, fraction)
    material.set_density("g/cm3", density_g_cm3)
    material.volume = _MATERIAL_VOLUME_CM3
    material.depletable = True

    micro = dep.MicroXS.from_multigroup_flux(
        energies=list(energies_eV),
        multigroup_flux=list(group_flux_cm2_s),
        chain_file=str(chain),
        temperature=temperature_K,
        nuclides=list(nuclides) if nuclides is not None else None,
    )
    operator = dep.IndependentOperator(
        openmc.Materials([material]), [[1.0]], [micro], normalization_mode="source-rate"
    )
    marks = [0.0, *cooling]
    steps = [irradiation_s, *(b - a for a, b in zip(marks, marks[1:], strict=False))]
    rates = [flux, *([0.0] * len(cooling))]
    dep.CECMIntegrator(operator, steps, source_rates=rates, timestep_units="s").integrate(
        path=results_path
    )

    results = dep.Results(results_path)
    states = [(0.0, 0.0), (irradiation_s, 0.0), *((irradiation_s, t) for t in cooling)]
    stored = []
    for index, (irradiated, cooled) in enumerate(states):
        state = results[index].get_material(MATERIAL_ID)
        state.volume = _MATERIAL_VOLUME_CM3
        atoms = {k: v for k, v in state.get_nuclide_atoms().items() if v > 0}
        stored.append(
            OpenMCStep(
                irradiation_time_s=irradiated,
                cooling_time_s=cooled,
                atoms=atoms,
                mass_kg=state.get_mass() / 1000.0,
                activity_bq=state.get_activity("Bq"),
                decay_heat_w=state.get_decay_heat("W"),
            )
        )

    spec = {
        "composition": composition,
        "density_g_cm3": density_g_cm3,
        "energies_eV": list(energies_eV),
        "group_flux_cm2_s": list(group_flux_cm2_s),
        "irradiation_s": irradiation_s,
        "cooling_times_s": cooling,
        "temperature_K": temperature_K,
        "nuclides": list(nuclides) if nuclides is not None else None,
    }
    spec_path = out / "run_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    inventory = OpenMCInventory(
        openmc_version=openmc.__version__,
        nuclear_data_library=_library_name(xs_xml),
        material_name=material_name,
        flux_cm2_s=flux,
        temperature_K=temperature_K,
        input_sha256={
            chain.name: _sha256(chain),
            xs_xml.name: _sha256(xs_xml),
            spec_path.name: _sha256(spec_path),
        },
        output_sha256={results_path.name: _sha256(results_path)},
        notes=(
            "one-group cross sections collapsed from the supplied spectrum, fixed over the run",
            "half-lives are OpenMC's bundled ENDF/B-VIII.0 table; decay energies from the chain",
        ),
        steps=tuple(stored),
    )
    (out / "inventory.json").write_text(inventory.model_dump_json(indent=2), encoding="utf-8")
    return inventory


def _library_name(xs_xml: Path) -> str:
    # The cross_sections.xml file carries no library name, so the directory name stands in.
    return xs_xml.parent.name or xs_xml.name
