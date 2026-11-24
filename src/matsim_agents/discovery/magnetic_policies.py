"""Material-family policies that decide which collinear magnetic trials to generate (M9).

A policy lists starting states worth evaluating. It is not a claim about the true magnetic
ground state, and all moments are starting values (muB) that DFT may change or collapse.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ase import Atoms

from matsim_agents.discovery import magnetism_generators as gen
from matsim_agents.discovery.magnetism import MagneticConfiguration

Trial = Literal["nonmagnetic", "ferromagnetic", "antiparallel_solute", "bcc_afm"]
STATEMENT = "starting-state policy; not a ground-state claim and not qualified"


@dataclass(frozen=True)
class MagneticPolicy:
    name: str
    elements: frozenset[str]
    required_any: frozenset[str]
    trials: tuple[Trial, ...]
    host_moments_muB: Mapping[str, float]
    solute_moments_muB: Mapping[str, float]
    rationale: str
    default_trial: Trial = "nonmagnetic"
    statement: str = STATEMENT

    def matches(self, symbols: set[str]) -> bool:
        if not symbols <= self.elements:
            return False
        return not self.required_any or bool(symbols & self.required_any)


_REFRACTORY = frozenset({"W", "Ta", "Cr", "V", "Hf", "Mo"})
POLICIES: tuple[MagneticPolicy, ...] = (
    MagneticPolicy(
        name="ods_ferritic",
        elements=frozenset({"Fe", "Cr", "Y", "O", "Ti", "W", "C"}),
        required_any=frozenset({"Y", "O", "Ti"}),
        trials=("ferromagnetic", "antiparallel_solute", "nonmagnetic"),
        host_moments_muB={"Fe": 2.2},
        solute_moments_muB={"Cr": -1.0},
        default_trial="ferromagnetic",
        rationale="Fe-based matrix with oxide-forming solutes; Y, O and Ti start unpolarized.",
    ),
    MagneticPolicy(
        name="fe_cr_rafm",
        elements=frozenset({"Fe", "Cr", "W", "V", "Ta", "C", "Mn"}),
        required_any=frozenset({"Fe"}),
        trials=("ferromagnetic", "antiparallel_solute", "nonmagnetic"),
        host_moments_muB={"Fe": 2.2},
        solute_moments_muB={"Cr": -1.0, "Mn": -1.0},
        default_trial="ferromagnetic",
        rationale="Fe-Cr is known to show Cr moments antiparallel to Fe in dilute Cr; both "
        "alignments are trialled because the preferred state depends on composition.",
    ),
    MagneticPolicy(
        name="v_cr_ti",
        elements=frozenset({"V", "Cr", "Ti"}),
        required_any=frozenset({"V"}),
        trials=("nonmagnetic",),
        host_moments_muB={},
        solute_moments_muB={},
        rationale="V-4Cr-4Ti is treated as nonmagnetic; a spin-polarized check is a separate "
        "explicit request.",
    ),
    MagneticPolicy(
        name="refractory_w_ta_cr_v_hf_mo",
        elements=_REFRACTORY,
        required_any=frozenset(),
        trials=("nonmagnetic", "ferromagnetic"),
        host_moments_muB={"Cr": 1.0},
        solute_moments_muB={},
        rationale="W-rich refractory alloys default to nonmagnetic; Cr-containing cells add a "
        "small-moment ferromagnetic trial to test for spin polarization.",
    ),
    MagneticPolicy(
        name="sic_sic",
        elements=frozenset({"Si", "C"}),
        required_any=frozenset(),
        trials=("nonmagnetic",),
        host_moments_muB={},
        solute_moments_muB={},
        rationale="Wide-gap pristine SiC is nonmagnetic. Defect spin states are not guessed; "
        "request an explicit moment for a spin-polarized defect study.",
    ),
)


def select_policy(atoms: Atoms) -> MagneticPolicy | None:
    """First matching policy in registry order (most specific first), else ``None``."""
    symbols = set(atoms.get_chemical_symbols())
    return next((p for p in POLICIES if p.matches(symbols)), None)


def configurations_for_policy(
    atoms: Atoms,
    structure_id: str,
    policy: MagneticPolicy,
    *,
    lattice_a: float | None = None,
) -> list[MagneticConfiguration]:
    """Generate the policy's trial configurations for ``atoms`` (needs ``atom_id``)."""
    symbols = set(atoms.get_chemical_symbols())
    note = f"policy {policy.name}: {policy.rationale}"
    out: list[MagneticConfiguration] = []
    for trial in policy.trials:
        if trial == "nonmagnetic":
            out.append(gen.nonmagnetic(atoms, structure_id))
        elif trial == "ferromagnetic":
            moments = {
                el: m for el, m in {**policy.host_moments_muB}.items() if el in symbols and m > 0
            }
            if moments:
                out.append(gen.ferromagnetic(atoms, structure_id, moments))
        elif trial == "antiparallel_solute":
            solutes = {el: m for el, m in policy.solute_moments_muB.items() if el in symbols}
            if solutes:
                values = {**policy.host_moments_muB, **solutes}
                by_id = {
                    int(i): values.get(sym, 0.0)
                    for i, sym in zip(
                        atoms.get_array("atom_id"), atoms.get_chemical_symbols(), strict=True
                    )
                }
                out.append(gen.custom_collinear(atoms, structure_id, by_id))
        elif trial == "bcc_afm" and lattice_a is not None:
            out.append(gen.bcc_afm(atoms, structure_id, lattice_a, dict(policy.host_moments_muB)))
    return [
        c.model_copy(update={"assumptions": (*c.assumptions, note, policy.statement)}) for c in out
    ]


def policy_readiness() -> dict[str, dict[str, object]]:
    """Capability reporting: software status per policy, never a scientific qualification."""
    return {
        p.name: {
            "trials": list(p.trials),
            "default_trial": p.default_trial,
            "software_implemented": True,
            "numerically_validated": False,
            "scientifically_qualified": False,
            "statement": p.statement,
        }
        for p in POLICIES
    }
