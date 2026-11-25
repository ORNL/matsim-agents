"""Activate iron in a two-group flux with OpenMC and print the evidence (needs OpenMC and data).

Setup: docs/external/openmc.md. Needs OPENMC_CROSS_SECTIONS and OPENMC_CHAIN_FILE.
The two-group flux below is a placeholder for illustration, not a reactor or fusion spectrum.
"""

from __future__ import annotations

import sys
from pathlib import Path

from matsim_agents.backends.external.openmc_activation.inventory import (
    to_activation_evidence,
    to_transmutation_evidence,
)
from matsim_agents.backends.external.openmc_activation.runner import run_activation
from matsim_agents.science.irradiation.nuclear_evidence import NuclearEvidenceError

DAY = 86400.0


def main(output_dir: str = "outputs/openmc_fe") -> int:
    try:
        inventory = run_activation(
            material_name="Fe",
            composition={"Fe": 1.0},
            density_g_cm3=7.874,
            energies_eV=[1e-5, 1e5, 2e7],
            group_flux_cm2_s=[5e13, 5e13],
            irradiation_s=30 * DAY,
            cooling_times_s=[DAY, 30 * DAY],
            output_dir=Path(output_dir),
            nuclides=["Fe54", "Fe56", "Fe57", "Fe58"],
        )
    except NuclearEvidenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for cooling in (0.0, DAY, 30 * DAY):
        activation = to_activation_evidence(
            inventory, cooling_time_s=cooling, material_id="Fe", irradiation_scenario_id="demo"
        )
        values = {q.name: q.value for q in activation.quantities}
        print(f"cooling {cooling / DAY:5.1f} d  {values}")
    gas = to_transmutation_evidence(
        inventory, cooling_time_s=0.0, material_id="Fe", irradiation_scenario_id="demo"
    )
    for quantity in gas.quantities:
        print(
            f"{quantity.isotope.element}{quantity.isotope.mass_number}: {quantity.value:.4g} appm"
        )
    print(f"wrote {output_dir}/inventory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
