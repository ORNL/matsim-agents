"""Gas, vacancy-cluster and trap mechanisms with explicit inventory accounting.

Inventory is a species count that includes the pseudo-species ``V`` for vacancies. A closed
event conserves it exactly. An open-system event changes it only through explicit source/sink
terms that name the reservoir and the basis; the change must equal the sum of those terms.
NEB preparation keeps the inventory fixed, so any pathway that needs an open term, an
unregistered mechanism, or a mechanism that has no fixed-inventory NEB meaning is rejected.
No rate, barrier or prefactor is created here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.science.kinetics.graph import TransitionNetwork
from matsim_agents.science.kinetics.transitions import (
    add_candidate_transition,
    register_mechanism,
)
from matsim_agents.science.magnetism.config import MagneticConfiguration
from matsim_agents.science.magnetism.defects import check_neb_magnetic_compatibility
from matsim_agents.science.neb.contracts import NebEndpoint, NebTransition

GAS = ("H", "He")
VACANCY = "V"


class GasMechanismError(ValueError):
    """Raised for an unsupported pathway or an inventory that the event does not justify."""


@dataclass(frozen=True)
class GasMechanismSpec:
    name: str
    reversible: bool
    min_inventory: Mapping[str, int]
    min_gas_total: int
    neb_supported: bool
    description: str


def _spec(name, description, *, reversible=False, need=None, gas_total=0, neb=True):
    return GasMechanismSpec(
        name=name,
        reversible=reversible,
        min_inventory=dict(need or {}),
        min_gas_total=gas_total,
        neb_supported=neb,
        description=description,
    )


GAS_MECHANISMS: Mapping[str, GasMechanismSpec] = {
    s.name: s
    for s in (
        _spec(
            "hydrogen_migration", "H hop between interstitial sites", reversible=True, need={"H": 1}
        ),
        _spec(
            "helium_migration", "He hop between interstitial sites", reversible=True, need={"He": 1}
        ),
        _spec("hydrogen_trapping", "H moves from an interstitial site into a trap", need={"H": 1}),
        _spec("helium_trapping", "He moves from an interstitial site into a trap", need={"He": 1}),
        _spec("hydrogen_detrapping", "H leaves a trap for an interstitial site", need={"H": 1}),
        _spec("helium_detrapping", "He leaves a trap for an interstitial site", need={"He": 1}),
        _spec("gas_cluster_association", "gas atoms or clusters combine", gas_total=2),
        _spec(
            "gas_cluster_dissociation", "a gas cluster releases a gas atom or cluster", gas_total=2
        ),
        _spec("vacancy_cluster_growth", "a vacancy joins a vacancy cluster", need={"V": 2}),
        _spec("vacancy_cluster_dissociation", "a vacancy leaves a vacancy cluster", need={"V": 2}),
        _spec(
            "solute_vacancy_association",
            "a solute and a vacancy become nearest neighbours",
            need={"V": 1},
        ),
        _spec("solute_vacancy_dissociation", "a solute-vacancy pair separates", need={"V": 1}),
        _spec(
            "trap_mutation",
            "a trap changes its occupancy or morphology at fixed inventory",
            neb=True,
        ),
    )
}
GAS_MECHANISM_IDS: tuple[str, ...] = tuple(GAS_MECHANISMS)

for _s in GAS_MECHANISMS.values():
    register_mechanism(_s.name, reversible=_s.reversible)


def get_gas_mechanism(name: str) -> GasMechanismSpec:
    try:
        return GAS_MECHANISMS[name]
    except KeyError:
        raise GasMechanismError(
            f"unsupported mechanism {name!r}; supported: {', '.join(GAS_MECHANISM_IDS)}"
        ) from None


# --------------------------------------------------------------------------- #
# Inventory                                                                   #
# --------------------------------------------------------------------------- #


def inventory_of(species: Sequence[str], n_lattice_sites: int) -> Counter:
    """Species counts plus ``V``: sites minus atoms that sit on lattice sites (not H/He)."""
    counts = Counter(species)
    on_lattice = sum(n for s, n in counts.items() if s not in GAS)
    vacancies = n_lattice_sites - on_lattice
    if vacancies < 0:
        raise GasMechanismError("more lattice atoms than lattice sites")
    if vacancies:
        counts[VACANCY] = vacancies
    return counts


def inventory_of_endpoint(endpoint: NebEndpoint, n_lattice_sites: int) -> Counter:
    return inventory_of(endpoint.species, n_lattice_sites)


class OpenSystemTerm(BaseModel):
    """An explicit source or sink; the change in one species and where it comes from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    species: str = Field(min_length=1)
    delta: int
    reservoir: str = Field(min_length=1)
    basis: str = Field(min_length=1)

    def model_post_init(self, __context) -> None:
        if self.delta == 0:
            raise ValueError("an open-system term must change the inventory")


class InventoryCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    mechanism: str
    system: Literal["closed", "open"]
    delta: dict[str, int]
    terms: tuple[OpenSystemTerm, ...] = ()


def _delta(source: Mapping[str, int], target: Mapping[str, int]) -> dict[str, int]:
    out = {}
    for key in set(source) | set(target):
        d = target.get(key, 0) - source.get(key, 0)
        if d:
            out[key] = d
    return dict(sorted(out.items()))


def check_event_inventory(
    mechanism: str,
    source: Mapping[str, int],
    target: Mapping[str, int],
    open_terms: Sequence[OpenSystemTerm] = (),
) -> InventoryCheck:
    """Validate one event's inventory change; raise rather than accept an unexplained one."""
    spec = get_gas_mechanism(mechanism)
    for key, minimum in spec.min_inventory.items():
        if source.get(key, 0) < minimum:
            raise GasMechanismError(
                f"{mechanism} needs at least {minimum} {key} in the source state, "
                f"found {source.get(key, 0)}"
            )
    if sum(source.get(g, 0) for g in GAS) < spec.min_gas_total:
        raise GasMechanismError(f"{mechanism} needs at least {spec.min_gas_total} gas atoms")
    if any(v < 0 for v in (*source.values(), *target.values())):
        raise GasMechanismError("inventory counts must be non-negative")
    change = _delta(source, target)
    if not open_terms:
        if change:
            raise GasMechanismError(
                f"{mechanism} changes the inventory by {change} without explicit source/sink "
                "terms; closed-system events must conserve it"
            )
        return InventoryCheck(mechanism=mechanism, system="closed", delta={})
    explained = Counter()
    for term in open_terms:
        explained[term.species] += term.delta
    explained = {k: v for k, v in explained.items() if v}
    if dict(sorted(explained.items())) != change:
        raise GasMechanismError(
            f"{mechanism}: source/sink terms {dict(sorted(explained.items()))} do not equal "
            f"the inventory change {change}"
        )
    return InventoryCheck(mechanism=mechanism, system="open", delta=change, terms=tuple(open_terms))


# --------------------------------------------------------------------------- #
# Network and NEB                                                             #
# --------------------------------------------------------------------------- #


def add_gas_transition(
    network: TransitionNetwork, transition: NebTransition, n_lattice_sites: int
) -> tuple:
    """Add a closed-system candidate event after checking its inventory is conserved."""
    kind = transition.mechanism.kind
    check_event_inventory(
        kind,
        inventory_of_endpoint(transition.initial, n_lattice_sites),
        inventory_of_endpoint(transition.final, n_lattice_sites),
    )
    return add_candidate_transition(network, transition)


class GasNebPreparation(BaseModel):
    model_config = ConfigDict(frozen=True)

    mechanism: str
    inventory: dict[str, int]
    inventory_fixed: Literal[True] = True
    magnetic_provenance_checked: bool
    magnetic_config_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def prepare_gas_neb(
    transition: NebTransition,
    n_lattice_sites: int,
    *,
    magnetic_images: Sequence[MagneticConfiguration] | None = None,
    open_terms: Sequence[OpenSystemTerm] = (),
) -> GasNebPreparation:
    """Accept a pathway for NEB only when the inventory is fixed and the mechanism is supported."""
    kind = transition.mechanism.kind
    spec = get_gas_mechanism(kind)
    if open_terms:
        raise GasMechanismError(
            f"{kind}: NEB keeps the inventory fixed; open-system source/sink terms are rejected"
        )
    if not spec.neb_supported:
        raise GasMechanismError(f"{kind} has no fixed-inventory NEB interpretation")
    source = inventory_of_endpoint(transition.initial, n_lattice_sites)
    target = inventory_of_endpoint(transition.final, n_lattice_sites)
    check_event_inventory(kind, source, target)
    notes: list[str] = []
    ids: tuple[str, ...] = ()
    checked = False
    if magnetic_images is not None:
        result = check_neb_magnetic_compatibility(magnetic_images)
        if not result.compatible:
            raise GasMechanismError(
                f"{kind}: magnetic provenance differs along the path: {list(result.problems)}"
            )
        checked = True
        ids = tuple(c.config_id for c in magnetic_images)
    else:
        notes.append("magnetic provenance was not checked")
    return GasNebPreparation(
        mechanism=kind,
        inventory=dict(sorted(source.items())),
        magnetic_provenance_checked=checked,
        magnetic_config_ids=ids,
        notes=tuple(notes),
    )
