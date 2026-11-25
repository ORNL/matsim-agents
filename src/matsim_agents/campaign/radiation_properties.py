"""MP9: radiation-resistance properties as evidence in the multi-property selection.

This connects the existing radiation workflows (the 1,024-site gas/transmutation QE campaign,
the SPECTRA-PKA adapter and the pseudopotential gates) to ``PropertyEvidence`` and to
:func:`~matsim_agents.campaign.property_acquisition.plan_property_acquisition`. It computes no
radiation physics.

Rules:

* Swelling, DBTT shift and conductivity degradation are never estimated here. A property with
  no supplied record is ``MISSING`` and says what is absent.
* A supplied record is used only if its conditions (temperature, irradiation, transmutation and
  defect state) match the requested radiation state. A pristine record never stands in for an
  irradiated one.
* A conductivity degradation is derived only from two records, a pristine and an irradiated
  conductivity at the same temperature. The irradiated one must itself carry an irradiation or
  defect state. The result is ``APPROXIMATED``, never qualified.
* Readiness (SPECTRA-PKA installed, pseudopotentials for Re, Os, H, He) is reported next to the
  evidence and turned into acquisition blockers. It is never turned into a value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from matsim_agents.campaign.gas_qe_campaign import GasExample
from matsim_agents.campaign.property_recipes import conditions_match
from matsim_agents.core.contracts import PROPERTY_UNITS, EvidenceLevel
from matsim_agents.discovery.species_support import assess_pseudopotentials
from matsim_agents.execution.properties import (
    PropertyConditions,
    PropertyEvidence,
    PropertyStatus,
)
from matsim_agents.integrations.external.spectra_pka.availability import (
    Status,
    check_availability,
)
from matsim_agents.integrations.external.spectra_pka.config import SpectraPKAConfig
from matsim_agents.workflows.full_supercell_campaign import FullSupercellCampaign

RADIATION_PROPERTIES = (
    "irradiation_swelling",
    "embrittlement_dbtt_shift",
    "thermal_conductivity_degradation",
)
REFERENCE_ALLOY_COUNTS: dict[str, int] = {
    "W": 640,
    "Ta": 128,
    "Cr": 64,
    "V": 96,
    "Hf": 32,
    "Mo": 64,
}
REFERENCE_REPEATS = (8, 8, 8)
REFERENCE_SITES = 1024
# Where the inputs come from, for re-acquisition:
# * UPFs: pslibrary 1.0.0 PBE PAW, https://pseudopotentials.quantum-espresso.org/upf_files/<file>
#   (Re/Os.pbe-spn-kjpaw_psl.1.0.0.UPF, H/He.pbe-kjpaw_psl.1.0.0.UPF). The Frontier copy is in
#   /lustre/orion/lrn110/proj-shared/cache/pseudopotentials/pslibrary-1.0.0-pbe-paw with
#   SOURCE.txt and SHA256SUMS; point MATSIM_PSEUDO_DIR there.
# * SPECTRA-PKA build and PKA matrices: see docs/external/spectra-pka.md.
RADIATION_SPECIES = ("Re", "Os", "H", "He")

_MISSING_REASON = {
    "irradiation_swelling": (
        "no validated swelling model or irradiation measurement has been ingested for this "
        "alloy and state; vacancy and helium energetics are not a swelling prediction"
    ),
    "embrittlement_dbtt_shift": (
        "no validated embrittlement model or measured DBTT shift has been ingested for this "
        "alloy and state; defect energetics are not a ductile-to-brittle shift"
    ),
    "thermal_conductivity_degradation": (
        "needs a pristine and an irradiated-state conductivity at the same temperature, from "
        "a transport calculation of a defected structure or a measurement; none ingested"
    ),
}


class RadiationError(ValueError):
    """A radiation state or reference alloy is not usable."""


class RadiationState(BaseModel):
    """The irradiated state a property is requested for. Nothing here is a damage estimate."""

    model_config = ConfigDict(frozen=True)

    irradiation: str = Field(min_length=1, description="Dose/spectrum label, e.g. from SPECTRA.")
    temperature_K: float = Field(gt=0)
    transmutation_state: str | None = None
    defect_state: str | None = None

    def conditions(self) -> PropertyConditions:
        return PropertyConditions(
            temperature_K=self.temperature_K,
            irradiation=self.irradiation,
            transmutation_state=self.transmutation_state,
            defect_state=self.defect_state,
        )


def state_for_example(
    example: GasExample, irradiation: str, temperature_K: float
) -> RadiationState:
    """The radiation state of one 1,024-site gas/transmutation configuration."""
    transmuted = f"Re{example.n_re}Os{example.n_os}" if example.n_re or example.n_os else None
    has_defect = bool(example.n_vacancies or example.n_he or example.n_h)
    return RadiationState(
        irradiation=irradiation,
        temperature_K=temperature_K,
        transmutation_state=transmuted,
        defect_state=f"{example.notation}@{example.config_id}" if has_defect else None,
    )


def check_reference_alloy(cfg: FullSupercellCampaign) -> dict[str, int]:
    """Refuse a campaign whose parent is not the 8x8x8, 1,024-site reference alloy."""
    p = cfg.parent_supercell
    if tuple(p.conventional_repeats) != REFERENCE_REPEATS or p.expected_parent_atoms != (
        REFERENCE_SITES
    ):
        raise RadiationError(
            f"parent is {p.conventional_repeats} / {p.expected_parent_atoms} sites, "
            f"not {REFERENCE_REPEATS} / {REFERENCE_SITES}"
        )
    if dict(p.composition) != REFERENCE_ALLOY_COUNTS:
        raise RadiationError(
            f"parent composition {dict(p.composition)} is not {REFERENCE_ALLOY_COUNTS}"
        )
    return dict(p.composition)


class RadiationReadiness(BaseModel):
    """What the radiation workflows can and cannot run here."""

    spectra_pka: str
    pseudopotentials: dict[str, str]
    blockers: dict[str, str]
    statement: str = (
        "readiness only: no radiation property is computed, and no degradation is claimed"
    )


def assess_radiation_readiness(
    spectra: SpectraPKAConfig,
    pseudo_dir: str | Path,
    pseudopotentials: Mapping[str, str],
    species: Sequence[str] = RADIATION_SPECIES,
) -> RadiationReadiness:
    """SPECTRA-PKA and Re/Os/H/He pseudopotential status, as acquisition blockers."""
    availability = check_availability(spectra, None, invoke=False)
    pseudo = assess_pseudopotentials(species, pseudo_dir, pseudopotentials)
    status = {r.species: r.status for r in pseudo.records}
    blockers: dict[str, str] = {}
    if availability.status in {Status.NOT_INSTALLED, Status.EXECUTION_FAILED}:
        blockers["spectra_pka"] = f"SPECTRA-PKA is {availability.status.value}"
    bad = sorted(s for s, v in status.items() if v != "VERIFIED")
    if bad:
        blockers["qe_gas_defects"] = "pseudopotential not verified for " + ", ".join(bad)
    return RadiationReadiness(
        spectra_pka=availability.status.value, pseudopotentials=status, blockers=blockers
    )


def _missing(
    prop: str, structure_id: str, composition: str, state: RadiationState, why: str
) -> PropertyEvidence:
    return PropertyEvidence(
        structure_id=structure_id,
        composition=composition,
        property=prop,
        status=PropertyStatus.MISSING,
        unit=PROPERTY_UNITS[prop],
        conditions=state.conditions(),
        reason=why,
    )


def derive_conductivity_degradation(
    pristine: PropertyEvidence, irradiated: PropertyEvidence, state: RadiationState
) -> PropertyEvidence:
    """``1 - k_irradiated / k_pristine`` from two scoreable records at the same temperature."""
    for rec, name in ((pristine, "pristine"), (irradiated, "irradiated")):
        if rec.property != "thermal_conductivity_total" or not rec.is_scoreable():
            raise RadiationError(f"the {name} record is not a scoreable total conductivity")
        if rec.conditions.temperature_K != state.temperature_K:
            raise RadiationError(f"the {name} record is not at {state.temperature_K} K")
    if pristine.conditions.irradiation or pristine.conditions.defect_state:
        raise RadiationError("the pristine record carries an irradiation or defect state")
    problems = conditions_match(state.conditions(), irradiated.conditions)
    if problems:
        raise RadiationError(
            "the irradiated record does not match the state: " + "; ".join(problems)
        )
    k0, u0 = float(pristine.value), float(pristine.uncertainty or 0.0)  # type: ignore[arg-type]
    k1, u1 = float(irradiated.value), float(irradiated.uncertainty or 0.0)  # type: ignore[arg-type]
    if k0 <= 0:
        raise RadiationError("pristine conductivity must be positive")
    value = 1.0 - k1 / k0
    uncertainty = math.hypot(u1 / k0, k1 * u0 / k0**2)
    ids = sorted({*pristine.evidence_ids, *irradiated.evidence_ids})
    return PropertyEvidence(
        structure_id=irradiated.structure_id,
        composition=irradiated.composition,
        property="thermal_conductivity_degradation",
        status=PropertyStatus.APPROXIMATED,
        value=value,
        unit=PROPERTY_UNITS["thermal_conductivity_degradation"],
        uncertainty=uncertainty,
        conditions=state.conditions(),
        method="ratio of irradiated to pristine total conductivity",
        evidence_ids=ids,
        evidence_level=min(
            (pristine.evidence_level, irradiated.evidence_level),
            key=list(EvidenceLevel).index,
        ),
        derived_from=["thermal_conductivity_total"],
        approximation=(
            "first-order uncertainty propagation of two independent records; the irradiated "
            "record is only as representative as its defect state"
        ),
        limitations=["not qualified: inherits the weaker of the two records"],
    )


def radiation_evidence(
    structure_id: str,
    composition: str,
    state: RadiationState,
    supplied: Sequence[PropertyEvidence] = (),
    readiness: RadiationReadiness | None = None,
) -> list[PropertyEvidence]:
    """One record per radiation property: a matching supplied record, a derivation, or MISSING."""
    out: list[PropertyEvidence] = []
    for prop in RADIATION_PROPERTIES:
        match = [
            e
            for e in supplied
            if e.property == prop
            and e.structure_id == structure_id
            and not conditions_match(state.conditions(), e.conditions)
        ]
        if match:
            out.append(match[0])
            continue
        if prop == "thermal_conductivity_degradation":
            derived = _try_derive(structure_id, supplied, state)
            if derived is not None:
                out.append(derived)
                continue
        why = _MISSING_REASON[prop]
        if readiness is not None and readiness.blockers:
            why += "; backend blockers: " + "; ".join(sorted(readiness.blockers.values()))
        out.append(_missing(prop, structure_id, composition, state, why))
    return out


def _try_derive(
    structure_id: str, supplied: Sequence[PropertyEvidence], state: RadiationState
) -> PropertyEvidence | None:
    kappa = [
        e
        for e in supplied
        if e.property == "thermal_conductivity_total"
        and e.is_scoreable()
        and e.conditions.temperature_K == state.temperature_K
    ]
    pristine = [e for e in kappa if not (e.conditions.irradiation or e.conditions.defect_state)]
    irradiated = [
        e
        for e in kappa
        if e.structure_id == structure_id and not conditions_match(state.conditions(), e.conditions)
    ]
    if not pristine or not irradiated:
        return None
    return derive_conductivity_degradation(pristine[0], irradiated[0], state)


__all__ = [
    "RADIATION_PROPERTIES",
    "RADIATION_SPECIES",
    "REFERENCE_ALLOY_COUNTS",
    "RadiationError",
    "RadiationReadiness",
    "RadiationState",
    "assess_radiation_readiness",
    "check_reference_alloy",
    "derive_conductivity_degradation",
    "radiation_evidence",
    "state_for_example",
]
