"""Point-defect formation energies referenced to one method-consistent convex hull.

A defect that changes stoichiometry (vacancy, interstitial, substitution) exchanges
atoms with reservoirs, so

    E_f = E_defect - E_host - sum_i dn_i * mu_i

where dn_i is the change in the count of element i and mu_i its absolute chemical
potential. The admissible mu_i are the vertices of the host's stability region on a
hull built from the same method as both supercells; E_f is reported at every vertex.
Stoichiometry-conserving defects (antisites) give the same value at every vertex.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

MAGNETISM_LIMITATION = (
    "Magnetism is not handled: energies are not resolved over magnetic orderings or spin "
    "states (MLIPs carry no explicit spin; DFT uses only the initialized spin "
    "configuration), so magnetic contributions to formation, hull and defect energies, "
    "including spin-polarized references such as triplet O2, are unaccounted for."
)
CHARGE_LIMITATION = (
    "Only neutral defects are treated: charged defects need Fermi-level referencing and "
    "finite-size electrostatic corrections, which are not implemented."
)
MAGNETIC_ELEMENTS = frozenset(
    {
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "U",
        "Np",
        "Pu",
    }
)


def magnetic_elements(symbols: Iterable[str]) -> list[str]:
    return sorted(set(symbols) & MAGNETIC_ELEMENTS)


def stoichiometry_change(
    host_symbols: Iterable[str], defect_symbols: Iterable[str]
) -> dict[str, int]:
    host = Counter(host_symbols)
    defect = Counter(defect_symbols)
    return {
        element: defect[element] - host[element]
        for element in sorted(set(host) | set(defect))
        if defect[element] != host[element]
    }


def host_chemical_potentials(diagram: Any, host_formula: str) -> dict[str, dict[str, float]]:
    """Absolute chemical potentials (eV/atom) at each vertex of the host stability region."""
    from pymatgen.core import Composition

    return {
        str(facet): {str(element): float(mu) for element, mu in chempots.items()}
        for facet, chempots in diagram.get_all_chempots(Composition(host_formula)).items()
    }


def defect_formation_energies(
    host_symbols: Iterable[str],
    host_energy_eV: float,
    defect_symbols: Iterable[str],
    defect_energy_eV: float,
    chemical_potentials: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Neutral defect formation energy (eV) at each chemical-potential vertex."""
    change = stoichiometry_change(host_symbols, defect_symbols)
    energies = {}
    for vertex, mu in chemical_potentials.items():
        missing = set(change) - mu.keys()
        if missing:
            raise ValueError(f"no chemical potential for {sorted(missing)} at {vertex}")
        energies[vertex] = float(
            defect_energy_eV
            - host_energy_eV
            - sum(count * mu[element] for element, count in change.items())
        )
    return energies


def defect_report(
    entries: list[Any],
    host_formula: str,
    host_symbols: list[str],
    host_energy_eV: float,
    defects: list[dict[str, Any]],
    *,
    host_hull_tolerance_eV_per_atom: float = 0.0,
) -> dict[str, Any]:
    """Score defects against a hull of ``entries`` plus the host, all from one method.

    Each defect is ``{"defect_id", "symbols", "energy_eV"}`` in the host supercell.
    """
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    host_entry = ComputedEntry(
        Composition(dict(Counter(host_symbols))), host_energy_eV, entry_id="defect-host"
    )
    diagram = PhaseDiagram([*entries, host_entry])
    host_e_hull = float(diagram.get_e_above_hull(host_entry))
    report: dict[str, Any] = {
        "host_formula": host_formula,
        "host_energy_above_hull_eV_per_atom": host_e_hull,
        "limitations": [MAGNETISM_LIMITATION, CHARGE_LIMITATION],
        "magnetic_elements_present": magnetic_elements(
            [*host_symbols, *(s for d in defects for s in d["symbols"])]
        ),
        "defects": [],
    }
    # Chemical potentials exist only where the host is stable against the references.
    if host_e_hull > host_hull_tolerance_eV_per_atom:
        report["status"] = "host_not_on_hull"
        return report
    chemical_potentials = host_chemical_potentials(diagram, host_formula)
    report["status"] = "evaluated"
    report["chemical_potentials_eV_per_atom"] = chemical_potentials
    for defect in defects:
        energies = defect_formation_energies(
            host_symbols,
            host_energy_eV,
            defect["symbols"],
            defect["energy_eV"],
            chemical_potentials,
        )
        report["defects"].append(
            {
                "defect_id": defect["defect_id"],
                "stoichiometry_change": stoichiometry_change(host_symbols, defect["symbols"]),
                "formation_energy_eV_by_vertex": energies,
                "formation_energy_min_eV": min(energies.values()),
                "formation_energy_max_eV": max(energies.values()),
            }
        )
    return report


def reference_set_entries(reference_energies: Any) -> list[Any]:
    """Hull entries (total energies) of a DFT ``ReferenceEnergySet``."""
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    entries = [
        ComputedEntry(element, energy)
        for element, energy in reference_energies.elemental_energies_eV_per_atom.items()
    ]
    for phase_id, formula, formation_per_atom in reference_energies.competing_phase_records():
        composition = Composition(formula)
        elemental_total = sum(
            amount * reference_energies.elemental_energies_eV_per_atom[element]
            for element, amount in composition.as_dict().items()
        )
        entries.append(
            ComputedEntry(
                composition,
                elemental_total + formation_per_atom * composition.num_atoms,
                entry_id=None if phase_id.startswith("formula:") else phase_id,
            )
        )
    return entries


def evaluate_dft_defects(
    reference_energies: Any,
    method_signature: str,
    host: Any,
    defects: dict[str, Any],
    *,
    host_hull_tolerance_eV_per_atom: float = 0.0,
) -> dict[str, Any]:
    """Defects from converged DFT relaxations against a same-method reference set.

    ``host`` and each value of ``defects`` are ``RelaxationResult`` objects of one
    supercell, relaxed with ``method_signature``.
    """
    from ase.io import read

    if method_signature != reference_energies.method_signature:
        raise ValueError("defect and reference energies must share method_signature")
    relaxations = {"host": host, **defects}
    unconverged = sorted(name for name, result in relaxations.items() if not result.converged)
    if unconverged:
        raise ValueError(f"defect energies need converged relaxations: {unconverged}")
    host_atoms = read(host.optimized_structure_path)
    host_symbols = host_atoms.get_chemical_symbols()
    defect_inputs = []
    for defect_id, result in defects.items():
        atoms = read(result.optimized_structure_path)
        defect_inputs.append(
            {
                "defect_id": defect_id,
                "symbols": atoms.get_chemical_symbols(),
                "energy_eV": result.final_energy_eV,
                "volume_ratio_to_host": atoms.get_volume() / host_atoms.get_volume(),
            }
        )
    report = defect_report(
        reference_set_entries(reference_energies),
        host_atoms.get_chemical_formula(mode="hill", empirical=True),
        host_symbols,
        host.final_energy_eV,
        defect_inputs,
        host_hull_tolerance_eV_per_atom=host_hull_tolerance_eV_per_atom,
    )
    report["method_signature"] = method_signature
    report["reference_set_id"] = reference_energies.identifier
    for item in report["defects"]:
        item["volume_ratio_to_host"] = next(
            d["volume_ratio_to_host"] for d in defect_inputs if d["defect_id"] == item["defect_id"]
        )
    return report
