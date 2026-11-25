"""Property characterization recipes: declared requirements and evidence-based assessment.

The recipes ``mechanical_elasticity``, ``electrical_transport``,
``thermal_transport``, and ``multiproperty_characterization`` extend
:mod:`matsim_agents.workflows.characterization`. Each :class:`PropertyRecipe`
declares its observables, supported methods, fidelity, operating conditions,
convergence requirements, cost, dependencies, and limitations. Assessment is
deterministic and never runs a calculation: it finds existing
:class:`~matsim_agents.science.properties.calculators.PropertyEvidence` that matches the
structure and operating conditions, reuses it when valid, and otherwise reports
exactly which property is missing and whether a backend, the budget, and the
approvals allow computing it. A missing property is never replaced by a related
one (thermal conductivity by formation energy, electrical conductivity by a gap,
strength by Young's modulus).
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from matsim_agents.core.contracts import (
    PROPERTY_UNITS,
    CharacterizationReport,
    CharacterizationStatus,
    EvidenceItem,
    EvidenceLevel,
    ObservableValue,
    at_least,
)
from matsim_agents.science.properties.calculators import (
    PropertyConditions,
    PropertyEvidence,
    PropertyStatus,
)

PROPERTY_RECIPES = (
    "mechanical_elasticity",
    "electrical_transport",
    "thermal_transport",
    "multiproperty_characterization",
)
# Level at which evidence is reported as qualified rather than screening.
QUALIFIED_LEVEL = EvidenceLevel.CONVERGED_DFT


class BackendReadiness(StrEnum):
    AVAILABLE = "available"
    NOT_CONFIGURED = "not_configured"
    METHOD_UNSUPPORTED = "method_unsupported"
    RESOURCE_BLOCKED = "resource_blocked"


@dataclass(frozen=True)
class MethodSpec:
    """A numerical method that can produce a recipe's properties."""

    name: str
    properties: tuple[str, ...]
    executable: str | None = None
    kind: str = "external"  # "external" executable, "internal" workflow, "data" ingestion
    needs_stress: bool = False
    limitations: tuple[str, ...] = ()


METHODS: dict[str, MethodSpec] = {
    "existing_data": MethodSpec(
        "existing_data",
        tuple(PROPERTY_UNITS),
        kind="data",
        limitations=("Only as good as the ingested record's provenance.",),
    ),
    "stress_strain_dft": MethodSpec(
        "stress_strain_dft",
        (
            "elastic_stiffness_tensor",
            "bulk_modulus",
            "shear_modulus",
            "youngs_modulus",
            "poissons_ratio",
            "elastic_anisotropy",
            "elastic_stability_min_eigenvalue",
        ),
        kind="internal",
        limitations=("Small-strain stiffness only; not strength, toughness, or creep.",),
    ),
    "stress_strain_mlip": MethodSpec(
        "stress_strain_mlip",
        (
            "elastic_stiffness_tensor",
            "bulk_modulus",
            "shear_modulus",
            "youngs_modulus",
            "poissons_ratio",
            "elastic_anisotropy",
            "elastic_stability_min_eigenvalue",
        ),
        kind="internal",
        needs_stress=True,
        limitations=("Screening only: needs DFT reference evidence before qualification.",),
    ),
    "epw": MethodSpec(
        "epw",
        (
            "electrical_conductivity",
            "electrical_conductivity_tensor",
            "electrical_resistivity",
            "thermal_conductivity_electronic",
        ),
        executable="epw.x",
        limitations=(
            "Electron-phonon scattering only: no chemical-disorder or defect scattering.",
        ),
    ),
    "kubo_greenwood": MethodSpec(
        "kubo_greenwood",
        ("electrical_conductivity", "electrical_resistivity", "thermal_conductivity_electronic"),
        limitations=("Needs an explicitly configured, validated Kubo-Greenwood executable.",),
    ),
    "shengbte": MethodSpec(
        "shengbte",
        ("thermal_conductivity_lattice", "thermal_conductivity_tensor"),
        executable="ShengBTE",
        limitations=("Phonon transport of a periodic crystal; disorder needs a validated model.",),
    ),
    "almabte": MethodSpec(
        "almabte",
        ("thermal_conductivity_lattice", "thermal_conductivity_tensor"),
        limitations=("Needs an explicitly configured almaBTE executable.",),
    ),
    "wiedemann_franz": MethodSpec(
        "wiedemann_franz",
        ("thermal_conductivity_electronic",),
        kind="internal",
        limitations=("Approximation kappa_e = L sigma T; the Lorenz number is not universal.",),
    ),
}


@dataclass(frozen=True)
class PropertyRecipe:
    """Declared requirements of one property characterization recipe."""

    name: str
    description: str
    observables: tuple[tuple[str, str, EvidenceLevel], ...]
    methods: tuple[str, ...]
    inputs: str
    required_conditions: tuple[str, ...]
    convergence: tuple[str, ...]
    cost: str
    dependencies: tuple[str, ...]
    limitations: tuple[str, ...]
    forbidden_fallbacks: tuple[str, ...] = field(default_factory=tuple)

    def required_properties(self) -> list[str]:
        return [name for name, _unit, _level in self.observables]


def _obs(*names: str, level: EvidenceLevel = EvidenceLevel.MLIP_PREDICTION):
    return tuple((n, PROPERTY_UNITS[n], level) for n in names)


_ELASTIC = (
    "elastic_stiffness_tensor",
    "bulk_modulus",
    "shear_modulus",
    "elastic_stability_min_eigenvalue",
)
RECIPES: dict[str, PropertyRecipe] = {
    "mechanical_elasticity": PropertyRecipe(
        "mechanical_elasticity",
        "Small-strain elastic stiffness tensor, moduli, and mechanical stability.",
        _obs(*_ELASTIC),
        ("stress_strain_dft", "stress_strain_mlip", "existing_data"),
        "Relaxed structures of one composition (production DFT: the full parent supercell).",
        ("temperature_K optional (0 K elasticity by default)",),
        (
            "residual stress",
            "linear stress-strain response across amplitudes",
            "Cij fit residual",
            "full 6x6 tensor retained unless symmetry is justified",
        ),
        "Number of strained calculations x the calculation cost (a benchmark is required).",
        ("QE or VASP executable for DFT", "an MLIP with stress support for screening"),
        (
            "Elastic constants are not yield strength, fracture toughness, creep, ductility, "
            "or embrittlement.",
        ),
        ("yield_strength", "fracture_toughness", "creep_rate", "ductility_elongation"),
    ),
    "electrical_transport": PropertyRecipe(
        "electrical_transport",
        "Electrical conductivity or resistivity at a stated temperature.",
        _obs("electrical_conductivity"),
        ("epw", "kubo_greenwood", "existing_data"),
        "Relaxed structures with electronic-structure inputs for the transport method.",
        ("temperature_K",),
        ("k/q-mesh convergence", "scattering model convergence", "temperature-dependent inputs"),
        "Electron-phonon or Kubo-Greenwood runs are far costlier than ground-state DFT.",
        ("epw.x or a configured Kubo-Greenwood executable",),
        (
            "An electronic band structure, a DOS, or sigma/tau is not an absolute conductivity.",
            "Large disordered supercells may exceed the allocation.",
        ),
        ("band_gap", "density_of_states", "formation_energy_per_atom"),
    ),
    "thermal_transport": PropertyRecipe(
        "thermal_transport",
        "Total thermal conductivity, kappa = kappa_e + kappa_ph, at a stated temperature.",
        _obs("thermal_conductivity_total"),
        ("epw", "shengbte", "almabte", "wiedemann_franz", "existing_data"),
        "Relaxed structures with force constants and electronic-transport inputs.",
        ("temperature_K",),
        ("q-mesh convergence", "force-constant cutoff", "scattering-model convergence"),
        "Anharmonic transport on large disordered cells is prohibitively expensive.",
        ("ShengBTE/almaBTE for kappa_ph", "EPW or evidence-backed sigma for kappa_e"),
        ("Wiedemann-Franz gives kappa_e only, as an approximation; it is not kappa_ph.",),
        ("formation_energy_per_atom", "elastic_stiffness_tensor", "electrical_conductivity"),
    ),
    "multiproperty_characterization": PropertyRecipe(
        "multiproperty_characterization",
        "Thermal, electrical, and mechanical evidence of one candidate at shared conditions.",
        _obs("thermal_conductivity_total", "electrical_conductivity", *_ELASTIC),
        (
            "epw",
            "shengbte",
            "almabte",
            "kubo_greenwood",
            "stress_strain_dft",
            "stress_strain_mlip",
            "wiedemann_franz",
            "existing_data",
        ),
        "Relaxed structures of one composition.",
        ("temperature_K",),
        ("each constituent recipe's convergence requirements",),
        "Sum of the constituent recipes' costs.",
        ("each constituent recipe's dependencies",),
        (
            "Radiation-damage resistance is evaluated by the radiation-damage workflows, not here.",
            "Thermal and electrical conductivity are correlated through electronic transport.",
        ),
        ("any one property standing in for another",),
    ),
}


def recipe_spec(name: str) -> PropertyRecipe:
    if name not in RECIPES:
        raise KeyError(f"{name!r} is not a property recipe; known: {sorted(RECIPES)}")
    return RECIPES[name]


def list_recipes() -> list[dict]:
    """Discoverable summary of every property recipe and its missing dependencies."""
    return [
        {
            "recipe": r.name,
            "required_observables": r.required_properties(),
            "methods": list(r.methods),
            "required_conditions": list(r.required_conditions),
            "convergence": list(r.convergence),
            "cost": r.cost,
            "dependencies": list(r.dependencies),
            "limitations": list(r.limitations),
            "forbidden_fallbacks": list(r.forbidden_fallbacks),
        }
        for r in RECIPES.values()
    ]


@dataclass
class BackendContext:
    """What this deployment can run for property methods."""

    mlip_supports_stress: bool = False
    dft_backend: str | None = None
    executables: Mapping[str, str] = field(default_factory=dict)  # method -> path
    calculation_cost_node_hours: float | None = None
    remaining_node_hours: float | None = None


def method_readiness(
    method: str, prop: str, ctx: BackendContext, *, n_calculations: int = 1
) -> tuple[BackendReadiness, str]:
    """Whether ``method`` could evaluate ``prop`` here, and why not."""
    spec = METHODS[method]
    if prop not in spec.properties:
        return BackendReadiness.METHOD_UNSUPPORTED, f"{method} does not evaluate {prop}"
    if spec.kind == "data":
        return BackendReadiness.AVAILABLE, "ingest validated property records"
    if spec.needs_stress and not ctx.mlip_supports_stress:
        return BackendReadiness.METHOD_UNSUPPORTED, "the MLIP has no stress support"
    if method == "stress_strain_dft" and ctx.dft_backend is None:
        return BackendReadiness.NOT_CONFIGURED, "methods.dft is not configured"
    if spec.kind == "external":
        configured = ctx.executables.get(method)
        found = configured or (spec.executable and shutil.which(spec.executable))
        if not found or not os.access(found, os.X_OK):
            need = spec.executable or "an explicit executable path"
            return BackendReadiness.NOT_CONFIGURED, f"{method} executable not found ({need})"
    if method in {"stress_strain_dft", "epw", "kubo_greenwood", "shengbte", "almabte"}:
        cost = ctx.calculation_cost_node_hours
        if cost is None:
            return BackendReadiness.RESOURCE_BLOCKED, (
                f"no measured cost per calculation for {method}: benchmark before dispatch"
            )
        total = cost * n_calculations
        if ctx.remaining_node_hours is not None and total > ctx.remaining_node_hours:
            return BackendReadiness.RESOURCE_BLOCKED, (
                f"{total:.1f} node-hours exceed the remaining {ctx.remaining_node_hours:.1f}"
            )
    return BackendReadiness.AVAILABLE, "ready"


def load_property_evidence(paths: list[Path]) -> list[PropertyEvidence]:
    """Read JSON files holding one PropertyEvidence or a list of them."""
    out: list[PropertyEvidence] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        for item in payload if isinstance(payload, list) else [payload]:
            out.append(PropertyEvidence.model_validate(item))
    return out


def conditions_match(
    wanted: PropertyConditions, have: PropertyConditions, *, temperature_tolerance_K: float = 1.0
) -> list[str]:
    """Differences that make evidence incompatible with the requested conditions."""
    problems = []
    if wanted.temperature_K is not None and (
        have.temperature_K is None
        or abs(have.temperature_K - wanted.temperature_K) > temperature_tolerance_K
    ):
        problems.append(f"temperature {have.temperature_K} K != {wanted.temperature_K} K")
    for name in ("irradiation", "defect_state", "transmutation_state", "magnetic_state", "loading"):
        want, got = getattr(wanted, name), getattr(have, name)
        if want is not None and want != got:
            problems.append(f"{name} {got!r} != {want!r}")
        if want is None and name in {"irradiation", "defect_state", "transmutation_state"} and got:
            problems.append(f"{name} {got!r} was not requested")
    return problems


@dataclass
class PropertyFinding:
    structure_id: str
    property: str
    evidence: PropertyEvidence | None
    problems: list[str]
    hint: str = ""


def _structure_ids(structures: list[Path]) -> list[str]:
    return [Path(p).stem for p in structures]


# A total is the sum of components: it is computed from them, not by one method.
COMPONENTS = {
    "thermal_conductivity_total": (
        "thermal_conductivity_electronic",
        "thermal_conductivity_lattice",
    )
}


def _methods_for(recipe: PropertyRecipe, prop: str) -> tuple[str, ...]:
    direct = [m for m in recipe.methods if prop in METHODS[m].properties]
    for part in COMPONENTS.get(prop, ()):
        direct += [m for m in recipe.methods if part in METHODS[m].properties and m not in direct]
    return tuple(direct)


def find_property(
    evidence: list[PropertyEvidence],
    structure_id: str,
    composition: str,
    prop: str,
    minimum: EvidenceLevel,
    wanted: PropertyConditions,
    supported_methods: tuple[str, ...],
) -> PropertyFinding:
    """The best reusable evidence for one property, or the reasons it cannot be used."""
    problems: list[str] = []
    for item in evidence:
        if item.property != prop or item.structure_id != structure_id:
            continue
        if item.composition != composition:
            problems.append(f"composition {item.composition} != {composition}")
            continue
        diff = conditions_match(wanted, item.conditions)
        if diff:
            problems.append("incompatible conditions: " + "; ".join(diff))
            continue
        if item.status in {PropertyStatus.MISSING, PropertyStatus.UNSUPPORTED}:
            problems.append(f"{item.status.value}: {item.reason}")
            continue
        if item.status == PropertyStatus.UNCONVERGED:
            problems.append(f"unconverged: {item.reason}")
            continue
        if not at_least(item.evidence_level, minimum):
            problems.append(
                f"evidence level {item.evidence_level.value} is below the required {minimum.value}"
            )
            continue
        if (
            item.method
            and item.method not in {*supported_methods, "measurement"}
            and (item.status != PropertyStatus.MEASURED)
        ):
            problems.append(f"method {item.method!r} is not one of {list(supported_methods)}")
            continue
        return PropertyFinding(structure_id, prop, item, [])
    return PropertyFinding(structure_id, prop, None, problems or ["no evidence"])


def _related_hint(evidence: list[PropertyEvidence], structure_id: str, prop: str) -> str:
    present = {
        e.property for e in evidence if e.structure_id == structure_id and e.value is not None
    }
    if (
        prop == "electrical_conductivity"
        and "electrical_conductivity_over_relaxation_time" in present
    ):
        return "only sigma/tau is available; no relaxation time, so no absolute conductivity"
    if (
        prop == "thermal_conductivity_total"
        and {
            "thermal_conductivity_electronic",
            "thermal_conductivity_lattice",
        }
        <= present
    ):
        return "kappa_e and kappa_ph are available: combine them with matching conditions"
    return ""


def assess_property_recipe(
    *,
    contract_id: str,
    contract_sha256: str,
    recipe_name: str,
    structures: list[Path],
    composition: str,
    conditions: PropertyConditions,
    evidence: list[PropertyEvidence],
    minimum_levels: Mapping[str, EvidenceLevel] | None = None,
    backend: BackendContext | None = None,
    approved: bool = False,
    recipe_properties: list[str] | None = None,
    temperature_tolerance_K: float = 1.0,
) -> CharacterizationReport:
    """Reuse valid evidence; otherwise report INCONCLUSIVE with the exact missing calculation."""
    recipe = recipe_spec(recipe_name)
    ctx = backend or BackendContext()
    levels = {n: lvl for n, _u, lvl in recipe.observables}
    levels.update(minimum_levels or {})
    wanted_props = recipe_properties or recipe.required_properties()
    methods_for = {p: _methods_for(recipe, p) for p in wanted_props}
    values: list[ObservableValue] = []
    items: dict[str, EvidenceItem] = {}
    missing: list[str] = []
    next_steps: list[str] = []
    limitations = [*recipe.limitations]
    screening = False
    for structure_id in _structure_ids(structures):
        for prop in wanted_props:
            found = find_property(
                evidence,
                structure_id,
                composition,
                prop,
                levels[prop],
                conditions,
                methods_for[prop],
            )
            if found.evidence is not None:
                e = found.evidence
                pid = e.property_id()
                items.setdefault(
                    pid,
                    EvidenceItem(
                        evidence_id=pid,
                        kind="property",
                        evidence_level=e.evidence_level,
                        method=e.method or "unspecified",
                        converged=e.converged,
                    ),
                )
                values.append(
                    ObservableValue(
                        name=prop,
                        value=e.value if isinstance(e.value, float | int) else json.dumps(e.value),
                        unit=e.unit,
                        evidence_level=e.evidence_level,
                        structure=structure_id,
                        evidence_ids=[pid],
                    )
                )
                screening = screening or not at_least(e.evidence_level, QUALIFIED_LEVEL)
                continue
            hint = _related_hint(evidence, structure_id, prop)
            reason = "; ".join(found.problems) + (f" ({hint})" if hint else "")
            missing.append(f"{prop} of {structure_id}: {reason}")
            options = []
            for method in methods_for[prop]:
                if method == "existing_data":
                    continue
                state, why = method_readiness(method, prop, ctx)
                options.append(f"{method}: {state.value} ({why})")
            if options:
                next_steps.append(f"{prop}: " + "; ".join(options))
    if missing:
        gate = "" if approved else " Production calculations also need dft_approved."
        return CharacterizationReport(
            contract_id=contract_id,
            contract_sha256=contract_sha256,
            recipe=recipe_name,
            status=CharacterizationStatus.INCONCLUSIVE,
            observables=values,
            methods={"evidence_source": "existing property evidence"},
            evidence=list(items.values()),
            missing=missing,
            limitations=[*limitations, "Missing properties are unknown, not zero and not failed."],
            recommended_next_test=("; ".join(next_steps) + gate).strip() or None,
        )
    if screening:
        limitations.append(
            "Some properties are screening-level (below converged DFT): not qualified."
        )
    return CharacterizationReport(
        contract_id=contract_id,
        contract_sha256=contract_sha256,
        recipe=recipe_name,
        status=CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED,
        conclusion=(
            f"{len(values)} required property values are available for {composition} at the "
            f"stated conditions; each carries its own evidence level and uncertainty."
        ),
        observables=values,
        methods={"evidence_source": "existing property evidence"},
        evidence=list(items.values()),
        limitations=limitations,
    )


__all__ = [
    "METHODS",
    "PROPERTY_RECIPES",
    "RECIPES",
    "BackendContext",
    "BackendReadiness",
    "MethodSpec",
    "PropertyFinding",
    "PropertyRecipe",
    "assess_property_recipe",
    "conditions_match",
    "find_property",
    "list_recipes",
    "load_property_evidence",
    "method_readiness",
    "recipe_spec",
]
