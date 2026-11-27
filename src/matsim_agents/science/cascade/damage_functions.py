"""NRT and arc-dpa damage functions (Nordlund et al., Nat. Commun. 9, 1084, 2018, Eqs. 5-7).

Parameters are always supplied by the caller together with their source; none are built in.
``damage_energy_eV`` is the energy available for atomic displacements, which equals the PKA
energy only if no energy is lost to electrons.
"""

from __future__ import annotations

import math


def nrt_displacements(damage_energy_eV: float, ed_eV: float) -> float:
    if damage_energy_eV < 0 or ed_eV <= 0:
        raise ValueError("need a non-negative damage energy and a positive Ed")
    if damage_energy_eV < ed_eV:
        return 0.0
    if damage_energy_eV < 2.0 * ed_eV / 0.8:
        return 1.0
    return 0.8 * damage_energy_eV / (2.0 * ed_eV)


def arc_dpa_efficiency(damage_energy_eV: float, ed_eV: float, b: float, c: float) -> float:
    knee = 2.0 * ed_eV / 0.8
    if damage_energy_eV < knee:
        return 1.0
    return (1.0 - c) / knee**b * damage_energy_eV**b + c


def arc_dpa_displacements(damage_energy_eV: float, ed_eV: float, b: float, c: float) -> float:
    if not (math.isfinite(b) and math.isfinite(c)):
        raise ValueError("b and c must be finite")
    return nrt_displacements(damage_energy_eV, ed_eV) * arc_dpa_efficiency(
        damage_energy_eV, ed_eV, b, c
    )


__all__ = ["arc_dpa_displacements", "arc_dpa_efficiency", "nrt_displacements"]
