"""Problems that make calculations pointless, found before any of them runs."""

from __future__ import annotations

from matsim_agents.core.composition import parse_composition
from matsim_agents.core.contracts import (
    CharacterizationStatus,
)
from matsim_agents.workflows.characterization.contract import CharacterizationContract, _references


def _preflight(
    contract: CharacterizationContract,
) -> tuple[CharacterizationStatus, list[str]] | None:
    """Problems that make calculations pointless, found before any of them runs."""
    from ase.io import read

    expected = parse_composition(contract.formula)
    assert expected is not None
    wrong = []
    for path in contract.structures:
        try:
            atoms = read(path)
        except Exception as exc:  # noqa: BLE001 - any unreadable file is a protocol error
            wrong.append(f"{path} is not a readable structure ({type(exc).__name__}: {exc})")
            continue
        found = parse_composition(atoms.get_chemical_formula(empirical=True))
        if found is None or found.formula != expected.formula:
            wrong.append(f"{path} is not {expected.formula}")
    if wrong:
        return CharacterizationStatus.INVALID_PROTOCOL, wrong
    if contract.recipe != "thermodynamic_stability":
        return None
    try:
        references = _references(contract)
    except (OSError, ValueError) as exc:  # pydantic's ValidationError is a ValueError
        return CharacterizationStatus.INVALID_PROTOCOL, [
            f"a valid reference-energy set: {contract.reference_energies} is not one "
            f"({type(exc).__name__}: {exc})"
        ]
    if references.method_signature != contract.methods.dft_method_signature:
        return CharacterizationStatus.INVALID_PROTOCOL, [
            f"references computed with the contract's DFT method: the reference set uses "
            f"{references.method_signature!r}, the relaxations "
            f"{contract.methods.dft_method_signature!r}"
        ]
    backend = contract.methods.dft.backend if contract.methods.dft is not None else None
    declared = [("the reference set", references.backend)]
    declared += [(f"competing phase {e.phase_id}", e.backend) for e in references.phase_entries]
    declared += [
        (f"elemental reference {e.phase_id}", e.backend)
        for e in [
            *references.elemental_entries.values(),
            *references.elemental_reference_candidates,
        ]
    ]
    foreign = [
        f"references computed with the contract's DFT backend: {source} uses "
        f"{value!r}, the relaxations {backend!r}"
        for source, value in declared
        if value is not None and value != backend
    ]
    if foreign:
        return CharacterizationStatus.INVALID_PROTOCOL, foreign
    from pymatgen.core import Composition as PMGComposition

    uncovered = []
    for phase_id, formula, _energy in references.competing_phase_records():
        try:
            elements = {str(element) for element in PMGComposition(formula).elements}
        except Exception:  # noqa: BLE001 - pymatgen raises several types for bad formulas
            elements = None
        missing = (
            sorted(elements - set(references.elemental_energies_eV_per_atom))
            if elements
            else [formula]
        )
        if missing:
            uncovered.append(
                f"elemental energies for {missing}, needed by competing phase {phase_id}"
            )
    if uncovered:
        return CharacterizationStatus.INVALID_PROTOCOL, uncovered
    completeness = references.audit_completeness(expected.elements)
    if completeness.provisional:
        # The hull would be provisional whatever the relaxations give: no DFT is spent.
        return CharacterizationStatus.INCONCLUSIVE, (
            [f"elemental reference: {e}" for e in completeness.missing_elemental_references]
            + [f"required phase: {f}" for f in completeness.missing_required_formulas]
            + [
                f"competing phase in the {pair} subsystem"
                for pair in completeness.missing_binary_subsystems
            ]
            + (["a ternary competing phase"] if completeness.missing_ternary_competitor else [])
        )
    return None
