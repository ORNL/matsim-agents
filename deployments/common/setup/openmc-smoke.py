"""Smoke test for an OpenMC install: Co-59 activation against the analytic Co-60 inventory.

Uses a flux and time small enough that burn-up of Co-59 and Co-60 is negligible, so the Co-60
(plus Co-60m, which decays into it) atom count after irradiation is N59 * sigma * phi * t.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import openmc
import openmc.deplete as dep

FLUX = 1.0e11  # n/cm2/s
IRRADIATION_S = 86400.0
TOLERANCE = 1e-2


def main() -> int:
    xs_xml = os.environ.get("OPENMC_CROSS_SECTIONS", "")
    chain = os.environ.get("OPENMC_CHAIN_FILE", "")
    if not (Path(xs_xml).is_file() and Path(chain).is_file()):
        print("set OPENMC_CROSS_SECTIONS and OPENMC_CHAIN_FILE to existing files", file=sys.stderr)
        return 2
    openmc.config["cross_sections"] = xs_xml
    openmc.config["chain_file"] = chain

    mat = openmc.Material(material_id=1, name="Co")
    mat.add_element("Co", 1.0)
    mat.set_density("g/cm3", 8.9)
    mat.volume = 1.0
    mat.depletable = True

    # One group spanning the full range collapses the library to a flat-spectrum average.
    micro = dep.MicroXS.from_multigroup_flux(
        energies=[1.0e-5, 2.0e7], multigroup_flux=[1.0], chain_file=chain, nuclides=["Co59"]
    )
    sigma_b = float(micro["Co59", "(n,gamma)"][0])
    operator = dep.IndependentOperator(
        openmc.Materials([mat]), [np.array([1.0])], [micro], normalization_mode="source-rate"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "depletion_results.h5"
        dep.CECMIntegrator(
            operator, [IRRADIATION_S, 86400.0], source_rates=[FLUX, 0.0], timestep_units="s"
        ).integrate(path=path)
        results = dep.Results(path)
        _, co60 = results.get_atoms("1", "Co60")
        _, co60m = results.get_atoms("1", "Co60_m1")
        _, co59 = results.get_atoms("1", "Co59")

    n59 = co59[0]
    expected = n59 * sigma_b * 1e-24 * FLUX * IRRADIATION_S
    got = co60[1] + co60m[1]
    error = abs(got - expected) / expected
    print(f"Co59 (n,gamma) one-group xs: {sigma_b:.4g} b")
    print(f"Co60+Co60m atoms after irradiation: {got:.6e}  analytic: {expected:.6e}")
    print(f"relative error: {error:.3e}  (tolerance {TOLERANCE:g})")
    if not error <= TOLERANCE:
        print("FAIL", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
