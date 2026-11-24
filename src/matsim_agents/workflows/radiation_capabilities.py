"""Capability registry for the radiation-damage workflow, with readiness levels.

Readiness separates three things that must not be conflated:

* ``software``: the entry points exist, import, and are covered by tests;
* ``runtime``: the external backends a real run needs (QE, an MLIP) are present
  in this environment;
* ``qualification``: scientific qualification, which this registry never grants.
  Every capability is ``NOT_QUALIFIED`` until the qualification matrix records
  evidence for it.
"""

from __future__ import annotations

import importlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from matsim_agents.execution.capabilities import check, get_capability


class SoftwareReadiness(StrEnum):
    """Whether the code for a capability is present and loadable."""

    IMPLEMENTED = "implemented"
    UNAVAILABLE = "unavailable"


class RuntimeReadiness(StrEnum):
    """Whether the external backends needed for a real run are present."""

    NOT_REQUIRED = "not_required"
    READY = "ready"
    MISSING_BACKEND = "missing_backend"


class QualificationLevel(StrEnum):
    """Scientific qualification; only the qualification matrix may raise this."""

    NOT_QUALIFIED = "not_qualified"


class RadiationCapability(BaseModel):
    """Static description of one radiation-damage capability."""

    model_config = ConfigDict(frozen=True)

    name: str
    summary: str
    entry_points: tuple[str, ...]
    backends: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    caveats: str = ""


class CapabilityReadiness(BaseModel):
    """Evaluated readiness of one capability in the current environment."""

    name: str
    software: SoftwareReadiness
    runtime: RuntimeReadiness
    qualification: QualificationLevel = QualificationLevel.NOT_QUALIFIED
    problems: tuple[str, ...] = ()
    missing_backends: tuple[str, ...] = ()


RADIATION_CAPABILITIES: tuple[RadiationCapability, ...] = (
    RadiationCapability(
        name="alloy-generation",
        summary="Seeded BCC multi-element supercells with exact composition and SRO descriptors.",
        entry_points=(
            "matsim_agents.discovery.alloys:build_bcc_supercell",
            "matsim_agents.discovery.alloys:warren_cowley_sro",
            "matsim_agents.discovery.alloy_ensemble:AlloyEnsemble",
        ),
        tests=("tests/test_alloys.py", "tests/test_alloy_ensemble.py"),
    ),
    RadiationCapability(
        name="defect-generation",
        summary="Vacancies, interstitials, crowdions, Frenkel pairs and divacancies.",
        entry_points=(
            "matsim_agents.discovery.radiation_defects:generate_vacancy",
            "matsim_agents.discovery.radiation_defects:generate_frenkel_pair",
            "matsim_agents.discovery.radiation_defect_mapping:identify_defect_sites",
        ),
        tests=("tests/test_radiation_defects.py", "tests/test_radiation_defect_mapping.py"),
    ),
    RadiationCapability(
        name="defect-energetics",
        summary="Formation energies and outcome classification from supplied energies.",
        entry_points=(
            "matsim_agents.discovery.radiation_defect_energies:radiation_defect_formation_energy",
            "matsim_agents.discovery.radiation_defect_classification:classify_defect_outcome",
        ),
        backends=("qe",),
        tests=(
            "tests/test_radiation_defect_energies.py",
            "tests/test_radiation_defect_classification.py",
        ),
        caveats="Energies must come from a qualified calculator; none is qualified.",
    ),
    RadiationCapability(
        name="neb-preparation",
        summary="QE NEB endpoints, images, input and result ingestion; dry run by default.",
        entry_points=(
            "matsim_agents.workflows.neb:prepare_neb_workflow",
            "matsim_agents.workflows.neb:run_neb_workflow",
            "matsim_agents.workflows.neb:ingest_neb_results",
        ),
        backends=("qe",),
        tests=("tests/test_neb_workflow.py", "tests/test_qe_neb.py"),
        caveats="Running neb.x needs QE and an explicit dispatch.",
    ),
    RadiationCapability(
        name="transition-network",
        summary="Directed transition network with NEB evidence ingestion and validation.",
        entry_points=(
            "matsim_agents.kinetics.graph:TransitionNetwork",
            "matsim_agents.kinetics.evidence:attach_neb_result",
            "matsim_agents.kinetics.validation:validate_network",
        ),
        tests=("tests/test_kinetics_graph.py", "tests/test_kinetics_validation.py"),
        caveats="Barriers are only as good as the supplied evidence.",
    ),
    RadiationCapability(
        name="kinetics",
        summary="HTST rates, seeded CT-KMC, ensembles and uncertainty propagation.",
        entry_points=(
            "matsim_agents.kinetics.rates:htst_rate",
            "matsim_agents.kinetics.kmc:run_kmc",
            "matsim_agents.kinetics.ensemble:run_ensemble",
            "matsim_agents.kinetics.uncertainty:propagate_uncertainty",
        ),
        tests=("tests/test_kinetics_kmc.py", "tests/test_kinetics_ensemble.py"),
        caveats="Outcomes are conditional on the supplied network.",
    ),
)


def get_radiation_capability(name: str) -> RadiationCapability:
    """The registered capability called ``name``."""
    for capability in RADIATION_CAPABILITIES:
        if capability.name == name:
            return capability
    known = ", ".join(c.name for c in RADIATION_CAPABILITIES)
    raise KeyError(f"unknown radiation-damage capability {name!r}; known: {known}")


def _resolve(entry_point: str) -> str | None:
    """A problem description when ``module:attribute`` does not resolve, else None."""
    module_name, _, attribute = entry_point.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return f"{entry_point}: import failed ({exc})"
    if not hasattr(module, attribute):
        return f"{entry_point}: missing attribute"
    return None


def evaluate_capability(capability: RadiationCapability) -> CapabilityReadiness:
    """Resolve entry points and probe backends; never raises for a missing backend."""
    problems = tuple(p for p in map(_resolve, capability.entry_points) if p)
    missing = tuple(
        name for name in capability.backends if not check(get_capability(name)).available
    )
    if not capability.backends:
        runtime = RuntimeReadiness.NOT_REQUIRED
    elif missing:
        runtime = RuntimeReadiness.MISSING_BACKEND
    else:
        runtime = RuntimeReadiness.READY
    return CapabilityReadiness(
        name=capability.name,
        software=SoftwareReadiness.UNAVAILABLE if problems else SoftwareReadiness.IMPLEMENTED,
        runtime=runtime,
        problems=problems,
        missing_backends=missing,
    )


def evaluate_all() -> list[CapabilityReadiness]:
    """Readiness of every registered capability, in registration order."""
    return [evaluate_capability(c) for c in RADIATION_CAPABILITIES]


__all__ = [
    "RADIATION_CAPABILITIES",
    "CapabilityReadiness",
    "QualificationLevel",
    "RadiationCapability",
    "RuntimeReadiness",
    "SoftwareReadiness",
    "evaluate_all",
    "evaluate_capability",
    "get_radiation_capability",
]
