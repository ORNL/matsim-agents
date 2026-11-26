"""Cascade-ready PKA event: identity, site, direction source and reaction channel checks.

Wraps the existing ``PKAEvent`` with what a cascade needs and that SPECTRA-PKA does not
supply (direction, structure identity, seed). Three kinds are kept apart:
``displacement`` (an ordinary lattice recoil), ``transmutation_daughter`` (must sit on the
parent nucleus's site, never on an unrelated site of the same element) and
``emitted_light_ion`` (H or He from a reaction; not a lattice recoil, so refused here).
Unsupported reaction channels raise with a named reason instead of falling back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ase import Atoms

from matsim_agents.core.hashing import sha256_json
from matsim_agents.science.irradiation.contracts import PKAEvent, RecoilSpecies

EventKind = Literal["displacement", "transmutation_daughter", "emitted_light_ion"]
DirectionSource = Literal["isotropic", "user", "crystallographic", "external"]

SUPPORTED_DISPLACEMENT_CHANNELS = frozenset({"elastic", "inelastic", "(n,2n)", "(n,gamma)"})
LIGHT_IONS = frozenset({"H", "He"})


class EventError(ValueError):
    """The event cannot be turned into a consistent cascade initial condition."""


class UnsupportedChannelError(EventError):
    pass


@dataclass(frozen=True)
class CascadeEvent:
    event_id: str
    parent_material_id: str
    parent_structure_hash: str
    kind: EventKind
    recoil_species: RecoilSpecies
    parent_species: RecoilSpecies | None
    site_index: int
    energy_eV: float
    scenario_id: str
    channel: str
    seed: int
    nuclear_data_provenance: str
    rate_or_probability: float
    rate_or_probability_basis: str
    direction_source: DirectionSource | None = None
    direction: tuple[float, float, float] | None = None
    parent_site_index: int | None = None


def structure_hash(atoms: Atoms) -> str:
    return sha256_json(
        {
            "numbers": atoms.numbers.tolist(),
            "cell": [[round(float(x), 8) for x in row] for row in atoms.cell.array],
            "positions": [[round(float(x), 6) for x in p] for p in atoms.positions],
        }
    )


def validate_event(event: CascadeEvent, atoms: Atoms) -> None:
    """Raise unless the event is consistent with ``atoms`` and a supported channel."""
    light = event.recoil_species.element in LIGHT_IONS
    if event.kind == "emitted_light_ion" or (light and event.kind != "transmutation_daughter"):
        raise UnsupportedChannelError(
            f"{event.recoil_species.key} from channel {event.channel!r} is an emitted light "
            "ion, not a lattice recoil; model it as implanted gas, not a PKA"
        )
    if event.kind == "displacement" and event.channel not in SUPPORTED_DISPLACEMENT_CHANNELS:
        raise UnsupportedChannelError(
            f"channel {event.channel!r} is not supported; supported: "
            f"{sorted(SUPPORTED_DISPLACEMENT_CHANNELS)}"
        )
    if not event.nuclear_data_provenance.strip():
        raise EventError("an event needs nuclear-data provenance")
    if event.parent_structure_hash != structure_hash(atoms):
        raise EventError("the parent structure does not match the event's structure hash")
    if not 0 <= event.site_index < len(atoms):
        raise EventError(f"site index {event.site_index} is outside the structure")
    here = atoms.get_chemical_symbols()[event.site_index]
    if event.kind == "displacement":
        if here != event.recoil_species.element:
            raise EventError(
                f"site {event.site_index} holds {here}, but the recoil is "
                f"{event.recoil_species.element}"
            )
    else:
        if event.parent_species is None or event.parent_site_index is None:
            raise EventError("a transmutation daughter needs its parent species and site")
        if event.site_index != event.parent_site_index:
            raise EventError(
                "a transmutation daughter must start on the parent nucleus's site "
                f"({event.parent_site_index}), not site {event.site_index}"
            )
        if atoms.get_chemical_symbols()[event.parent_site_index] != event.parent_species.element:
            raise EventError("the parent site does not hold the stated parent species")
    if not event.energy_eV > 0:
        raise EventError("recoil energy must be positive")


def event_from_pka(
    pka: PKAEvent,
    atoms: Atoms,
    *,
    parent_material_id: str,
    channel: str,
    seed: int,
    nuclear_data_provenance: str,
    scenario_id: str,
    rate_basis: str,
    kind: EventKind = "displacement",
    parent_species: RecoilSpecies | None = None,
    parent_site_index: int | None = None,
) -> CascadeEvent:
    if pka.site_index is None:
        raise EventError("the PKA event carries no site; place it on the lattice first")
    ev = CascadeEvent(
        event_id=f"{scenario_id}:{pka.index}:{seed}",
        parent_material_id=parent_material_id,
        parent_structure_hash=structure_hash(atoms),
        kind=kind,
        recoil_species=pka.species,
        parent_species=parent_species,
        site_index=pka.site_index,
        energy_eV=pka.energy_mev * 1.0e6,
        scenario_id=scenario_id,
        channel=channel,
        seed=seed,
        nuclear_data_provenance=nuclear_data_provenance,
        rate_or_probability=pka.probability,
        rate_or_probability_basis=rate_basis,
        parent_site_index=parent_site_index,
    )
    validate_event(ev, atoms)
    return ev


__all__ = [
    "CascadeEvent",
    "EventError",
    "UnsupportedChannelError",
    "event_from_pka",
    "structure_hash",
    "validate_event",
]
