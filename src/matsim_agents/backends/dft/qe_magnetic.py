"""Map a collinear magnetic configuration onto Quantum ESPRESSO atomic types.

QE attaches ``starting_magnetization(i)`` to an atomic *type*, so chemically identical atoms
with different starting polarizations need distinct types that share one pseudopotential.
Atoms are grouped by (element, starting polarization); distinct initializations are never
merged, and one type per atom is not created unless every atom really differs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ase import Atoms

from matsim_agents.backends.dft.qe_relax import _ATOMIC_MASS  # noqa: PLC2701
from matsim_agents.discovery.magnetism import MagneticConfiguration

MAX_LABEL_LENGTH = 3
_ROUND = 12


@dataclass(frozen=True)
class QEMagneticType:
    label: str
    element: str
    magnetic_group: str
    starting_magnetization: float | None
    mass: float
    pseudopotential: str
    atom_ids: tuple[int, ...]


@dataclass(frozen=True)
class QEMagneticTypes:
    types: tuple[QEMagneticType, ...]
    spin_polarized: bool
    label_by_atom_id: Mapping[int, str]

    @property
    def ntyp(self) -> int:
        return len(self.types)

    @property
    def nspin(self) -> int:
        return 2 if self.spin_polarized else 1

    def labels_for_atoms(self, atoms: Atoms) -> list[str]:
        """Type label per atom in the current order of ``atoms`` (matched by atom_id)."""
        if "atom_id" not in atoms.arrays:
            raise ValueError("structure has no atom_id array; stable identities are required")
        ids = [int(i) for i in atoms.get_array("atom_id")]
        if set(ids) != set(self.label_by_atom_id):
            raise ValueError("atom_ids of the structure do not match the magnetic type mapping")
        return [self.label_by_atom_id[i] for i in ids]

    def species_lines(self) -> list[str]:
        return [f"  {t.label}  {t.mass:.4f}  {t.pseudopotential}" for t in self.types]

    def position_lines(self, atoms: Atoms) -> list[str]:
        labels = self.labels_for_atoms(atoms)
        return [
            f"  {label}  {pos[0]:18.10f} {pos[1]:18.10f} {pos[2]:18.10f}"
            for label, pos in zip(labels, atoms.get_positions(), strict=True)
        ]

    def system_entries(self) -> dict[str, object]:
        """Values for the ``&SYSTEM`` namelist; empty for spin-unpolarized calculations."""
        if not self.spin_polarized:
            return {}
        entries: dict[str, object] = {"nspin": 2}
        for index, qe_type in enumerate(self.types, start=1):
            entries[f"starting_magnetization({index})"] = float(qe_type.starting_magnetization)
        return entries


def _label(element: str, ordinal: int, count: int) -> str:
    if count == 1:
        return element
    label = f"{element}{ordinal}"
    if len(label) > MAX_LABEL_LENGTH:
        raise ValueError(
            f"{element} needs {count} magnetic types but QE labels allow at most "
            f"{MAX_LABEL_LENGTH} characters ({element} + digits); merge or reduce the "
            "initializations"
        )
    return label


def build_magnetic_types(
    atoms: Atoms,
    config: MagneticConfiguration | None,
    pseudopotentials: Mapping[str, str],
) -> QEMagneticTypes:
    """Build the type table. ``config=None`` means a spin-unpolarized calculation."""
    if "atom_id" not in atoms.arrays:
        raise ValueError("structure has no atom_id array; stable identities are required")
    ids = [int(i) for i in atoms.get_array("atom_id")]
    symbols = atoms.get_chemical_symbols()
    missing = sorted(set(symbols) - set(pseudopotentials))
    if missing:
        raise ValueError(f"no pseudopotential for {missing}; every type needs its element's file")

    polarization: dict[int, float | None]
    group_of: dict[int, str]
    if config is None or not config.spin_polarized:
        if config is not None:
            config.validate_against(atoms)
        polarization = dict.fromkeys(ids)
        group_of = {i: "nm" for i in ids}
        spin = False
    else:
        config.validate_against(atoms)
        if config.qe_starting_magnetization is None:
            raise ValueError(
                "configuration has no QE starting polarization; convert moments explicitly with "
                "qe_polarization_from_moment (verified valence count and QE version required)"
            )
        values = dict(zip(config.atom_ids, config.qe_starting_magnetization, strict=True))
        names = dict(zip(config.atom_ids, config.group_labels, strict=True))
        polarization = {i: round(values[i], _ROUND) for i in ids}
        group_of = {i: names[i] for i in ids}
        spin = True

    members: dict[tuple[str, float | None], list[int]] = {}
    for atom_id, symbol in zip(ids, symbols, strict=True):
        members.setdefault((symbol, polarization[atom_id]), []).append(atom_id)

    by_element: dict[str, list[float | None]] = {}
    for element, value in members:
        by_element.setdefault(element, []).append(value)

    types: list[QEMagneticType] = []
    label_by_atom_id: dict[int, str] = {}
    for element in sorted(by_element):
        values = sorted(by_element[element], key=lambda v: (v is None, -(v or 0.0)))
        for ordinal, value in enumerate(values, start=1):
            label = _label(element, ordinal, len(values))
            atom_ids = tuple(sorted(members[(element, value)]))
            groups = sorted({group_of[i] for i in atom_ids})
            types.append(
                QEMagneticType(
                    label=label,
                    element=element,
                    magnetic_group="+".join(groups),
                    starting_magnetization=value,
                    mass=_ATOMIC_MASS.get(element, 1.0),
                    pseudopotential=pseudopotentials[element],
                    atom_ids=atom_ids,
                )
            )
            label_by_atom_id.update(dict.fromkeys(atom_ids, label))

    return QEMagneticTypes(tuple(types), spin, label_by_atom_id)
