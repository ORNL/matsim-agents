"""Checks on an OpenMC depletion inventory before it feeds H/He production and activation.

The inventory is one-group at one temperature and flux, from a stated depletion chain. This
module does not run OpenMC. It does four things and nothing else:

* It refuses to treat a thermal-fission chain (for example ``chain_endfb80_pwr.xml``) as a
  model for a hard spectrum. A chain is accepted only if the caller declares a ``hard`` spectrum
  kind with a citation; the declaration is a claim, not a validation, and is reported as such.
* It reports which expected transmutation products are present or absent in the irradiated
  material (a coverage check on products, not a check of the chain's reaction list).
* It turns the irradiation steps into H/He production rates per initial atom per second and
  lists the activation and decay-heat series for the irradiation and cooling schedule.
* It reports nuclear-data uncertainty as unavailable, never zero.

No hard-spectrum-spectrum chain or inventory has been run in this repository, so nothing here is a
validated hard-spectrum prediction.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from matsim_agents.backends.external.openmc_activation.inventory import (
    GAS_ELEMENTS,
    OpenMCInventory,
    OpenMCStep,
    _fold,
)

UNCERTAINTY = "unavailable: OpenMC depletion reports no nuclear-data uncertainty"
_THERMAL_NAME = re.compile(r"pwr|bwr|lwr|thermal|fission|candu", re.IGNORECASE)
_NUCLIDE = re.compile(r"^([A-Z][a-z]?)(\d+)$")

# Product elements by channel for hosts with a documented transmutation path.
EXPECTED_PRODUCTS: dict[str, dict[str, tuple[str, ...]]] = {
    "W": {
        "(n,gamma) then beta decay": ("Re",),
        "second-order (n,gamma) and decay": ("Os",),
        "(n,p)": ("Ta",),
        "(n,alpha)": ("Hf",),
    },
}


class ChainError(ValueError):
    """The inventory cannot be used as a hard-spectrum transmutation source."""


@dataclass(frozen=True)
class ChainDeclaration:
    spectrum_kind: str  # thermal, fission, hard-spectrum or unknown
    citation: str = ""


@dataclass(frozen=True)
class ChainVerdict:
    chain_names: tuple[str, ...]
    usable_for_declared_spectrum: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CoverageEntry:
    channel: str
    elements: tuple[str, ...]
    present: tuple[str, ...]
    absent: tuple[str, ...]


@dataclass(frozen=True)
class GasProductionRate:
    start_s: float
    end_s: float
    appm_per_s: dict[str, float]


@dataclass(frozen=True)
class ScheduleRow:
    irradiation_time_s: float
    cooling_time_s: float
    activity_bq_per_kg: float
    decay_heat_w_per_kg: float


@dataclass(frozen=True)
class InventoryAssessment:
    chain: ChainVerdict
    host: str
    coverage: tuple[CoverageEntry, ...]
    schedule: tuple[ScheduleRow, ...]
    gas_rates: tuple[GasProductionRate, ...]
    uncertainty: str
    caveats: tuple[str, ...]


def chain_names(inventory: OpenMCInventory) -> tuple[str, ...]:
    return tuple(sorted(k for k in inventory.input_sha256 if "chain" in k.lower()))


def judge_chain(inventory: OpenMCInventory, declaration: ChainDeclaration) -> ChainVerdict:
    names = chain_names(inventory)
    reasons: list[str] = []
    if not names:
        reasons.append("the inventory records no depletion chain file in its input hashes")
    for name in names:
        if _THERMAL_NAME.search(name):
            reasons.append(
                f"chain {name!r} is a thermal-fission chain, not a model for a hard spectrum"
            )
    if declaration.spectrum_kind != "hard":
        reasons.append(f"declared spectrum kind is {declaration.spectrum_kind!r}, not 'hard'")
    elif not declaration.citation.strip():
        reasons.append("a hard-spectrum declaration needs a citation for the chain and spectrum")
    return ChainVerdict(names, not reasons, tuple(reasons))


def _host(step: OpenMCStep) -> str:
    totals: dict[str, float] = {}
    for key, count in _fold(step.atoms).items():
        element = _NUCLIDE.match(key)
        if element is not None:
            totals[element.group(1)] = totals.get(element.group(1), 0.0) + count
    if not totals:
        raise ChainError("the initial material has no atoms")
    return max(totals, key=lambda e: totals[e])


def _elements_with_atoms(step: OpenMCStep) -> set[str]:
    out = set()
    for key, count in _fold(step.atoms).items():
        match = _NUCLIDE.match(key)
        if match is not None and count > 0:
            out.add(match.group(1))
    return out


def _gas_appm(step: OpenMCStep, total: float) -> dict[str, float]:
    out = dict.fromkeys(GAS_ELEMENTS, 0.0)
    for key, count in _fold(step.atoms).items():
        match = _NUCLIDE.match(key)
        if match is not None and match.group(1) in out:
            out[match.group(1)] += count / total * 1e6
    return out


def assess_inventory(
    inventory: OpenMCInventory,
    declaration: ChainDeclaration,
    *,
    expected_products: Mapping[str, tuple[str, ...]] | None = None,
) -> InventoryAssessment:
    initial = inventory.steps[0]
    if initial.irradiation_time_s != 0 or initial.cooling_time_s != 0:
        raise ChainError("the first step must be the unirradiated material")
    host = _host(initial)
    total = math.fsum(_fold(initial.atoms).values())
    chain = judge_chain(inventory, declaration)
    caveats = [
        "one-group flux at one temperature; the spectrum collapse is not checked here",
        "flux is constant across steps; use the exposure history for time-dependent flux",
        "no hard-spectrum chain or inventory has been run or validated in this repository",
    ]
    if chain.usable_for_declared_spectrum:
        caveats.append(
            f"hard-spectrum use rests on the caller's declaration: {declaration.citation.strip()!r}"
        )
    irradiated = [s for s in inventory.steps if s.cooling_time_s == 0 and s.irradiation_time_s > 0]
    final_element_set = (
        _elements_with_atoms(max(irradiated, key=lambda s: s.irradiation_time_s))
        if irradiated
        else set()
    )
    channels = dict(expected_products) if expected_products else EXPECTED_PRODUCTS.get(host)
    coverage: list[CoverageEntry] = []
    if channels is None:
        caveats.append(f"no expected-product table for host {host!r}; only H/He are checked")
        channels = {}
    channels = {**channels, "(n,p) and (n,alpha) gas": GAS_ELEMENTS}
    for channel, elements in channels.items():
        present = tuple(e for e in elements if e in final_element_set)
        coverage.append(
            CoverageEntry(
                channel,
                tuple(elements),
                present,
                tuple(e for e in elements if e not in present),
            )
        )
    if any(c.absent for c in coverage):
        caveats.append(
            "an absent product can mean a missing chain reaction or too short an irradiation"
        )
    schedule = tuple(
        ScheduleRow(
            s.irradiation_time_s,
            s.cooling_time_s,
            s.activity_bq / s.mass_kg,
            s.decay_heat_w / s.mass_kg,
        )
        for s in inventory.steps
    )
    rates: list[GasProductionRate] = []
    previous = None
    for step in sorted(irradiated, key=lambda s: s.irradiation_time_s):
        if previous is not None:
            before, after = _gas_appm(previous, total), _gas_appm(step, total)
            span = step.irradiation_time_s - previous.irradiation_time_s
            rates.append(
                GasProductionRate(
                    previous.irradiation_time_s,
                    step.irradiation_time_s,
                    {e: (after[e] - before[e]) / span for e in GAS_ELEMENTS},
                )
            )
        previous = step
    if not irradiated:
        caveats.append("the inventory has no irradiated step")
    elif len(irradiated) < 2:
        caveats.append("one irradiated step: gas production rate cannot be resolved")
    if any(v < 0 for r in rates for v in r.appm_per_s.values()):
        caveats.append("a negative gas production rate means decay or burn-out between steps")
    return InventoryAssessment(
        chain=chain,
        host=host,
        coverage=tuple(coverage),
        schedule=schedule,
        gas_rates=tuple(rates),
        uncertainty=UNCERTAINTY,
        caveats=tuple(caveats),
    )


def require_declared_spectrum(assessment: InventoryAssessment) -> InventoryAssessment:
    if not assessment.chain.usable_for_declared_spectrum:
        raise ChainError("; ".join(assessment.chain.reasons))
    return assessment


__all__ = [
    "EXPECTED_PRODUCTS",
    "UNCERTAINTY",
    "ChainDeclaration",
    "ChainVerdict",
    "CoverageEntry",
    "ChainError",
    "InventoryAssessment",
    "GasProductionRate",
    "ScheduleRow",
    "assess_inventory",
    "chain_names",
    "judge_chain",
    "require_declared_spectrum",
]
