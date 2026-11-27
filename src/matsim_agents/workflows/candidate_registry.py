"""Content-addressed candidate lineage and structural evidence for campaigns."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase.io import read as ase_read
from pydantic import BaseModel, Field, model_validator

from matsim_agents.core.seeding import python_rng
from matsim_agents.science.acquisition.acquisition import FormulaAcquisitionMetrics
from matsim_agents.science.structures.seed_structures import PhaseCandidate
from matsim_agents.workflows.exploration.composition import CompositionExplorationResult

NOT_YET_ESTABLISHED = "NOT_YET_ESTABLISHED"
Origin = Literal["existing", "new"]
AllowedUse = Literal["training", "validation", "held_out_test", "reference", "transfer_source"]
TransferStatus = Literal["validated", "unsupported", "pending", "NOT_YET_ESTABLISHED"]


def chemical_system(elements: Iterable[str]) -> str:
    return "-".join(sorted(set(elements)))


def hierarchy_subspaces(
    elements: Iterable[str],
) -> dict[int, list[tuple[str, ...]]]:
    """Candidate chemical subspaces by component count (2..n); not an execution list."""
    ordered = sorted(set(elements))
    return {k: list(itertools.combinations(ordered, k)) for k in range(2, len(ordered) + 1)}


def composition_formula(composition: dict[str, int]) -> str:
    return "".join(f"{el}{n}" for el, n in sorted(composition.items()))


def calculation_id(
    structure_hash: str,
    backend: str | None,
    settings: dict[str, Any],
    magnetic_initial: dict[str, Any] | None,
) -> str:
    """Stable ID of one calculation; other numerics or magnetism give another ID."""
    payload = json.dumps(
        {
            "structure": structure_hash,
            "backend": backend,
            "settings": settings,
            "magnetic_initial": magnetic_initial or {},
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class StructureCandidateRecord(BaseModel):
    candidate_id: str
    formula: str
    structure_path: str
    structure_hash: str
    source: str
    prototype_id: str | None = None
    space_group: int | None = None
    decoration_mapping: str | None = None
    random_seed: int | None = None
    parent_candidate_id: str | None = None
    iteration_created: int = 0
    needs_dft_verification: bool = False
    optimized_structure_path: str | None = None
    optimized_structure_hash: str | None = None
    relaxed_family_id: str | None = None
    chemical_system: str | None = None
    element_set: list[str] = Field(default_factory=list)
    n_components: int | None = None
    composition: dict[str, int] = Field(default_factory=dict)
    parent_lattice: str | None = None
    lattice_parameter_A: float | None = None
    supercell: tuple[int, int, int] | None = None
    atom_count: int | None = None
    ordering: str | None = None
    sro: dict[str, Any] = Field(default_factory=dict)
    local_environments: dict[str, Any] = Field(default_factory=dict)
    defect_config: str | None = None
    origin: Origin = "new"
    source_dataset: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)
    allowed_uses: list[AllowedUse] = Field(default_factory=list)
    transferability_evidence: dict[str, TransferStatus] = Field(default_factory=dict)

    def transferability(self, prop: str) -> str:
        return self.transferability_evidence.get(prop, NOT_YET_ESTABLISHED)


class CandidateEvaluationRecord(BaseModel):
    candidate_id: str
    backend: str | None = None
    model_identifier: str | None = None
    model_checkpoint_hash: str | None = None
    converged: bool
    final_energy_eV: float | None = None
    energy_per_atom_eV: float | None = None
    residual_force_eV_per_A: float | None = None
    optimization_steps: int | None = None
    uncertainty: float | None = None
    structural_novelty: float | None = None
    distance_from_labeled_data: float | None = None
    predicted_formation_energy_eV_per_atom: float | None = None
    predicted_energy_above_hull_eV_per_atom: float | None = None
    failure_reason: str | None = None
    calculation_id: str | None = None
    dft_settings: dict[str, Any] = Field(default_factory=dict)
    magnetic_initial: dict[str, Any] | None = None
    magnetic_final: dict[str, Any] | None = None
    properties: dict[str, float] = Field(default_factory=dict)
    cost_node_hours: float | None = Field(default=None, ge=0.0)
    origin: Origin = "new"
    source_dataset: str | None = None
    provenance: dict[str, str] = Field(default_factory=dict)


class DatasetEntry(BaseModel):
    """One structure+calculation from an external or historical dataset."""

    structure_hash: str
    composition: dict[str, int]
    structure_path: str = ""
    supercell: tuple[int, int, int] | None = None
    lattice_parameter_A: float | None = None
    parent_lattice: str | None = "bcc"
    ordering: str | None = None
    defect_config: str | None = None
    sro: dict[str, Any] = Field(default_factory=dict)
    local_environments: dict[str, Any] = Field(default_factory=dict)
    dft_backend: str | None = None
    dft_settings: dict[str, Any] = Field(default_factory=dict)
    magnetic_initial: dict[str, Any] | None = None
    magnetic_final: dict[str, Any] | None = None
    properties: dict[str, float] = Field(default_factory=dict)
    converged: bool = True
    cost_node_hours: float | None = Field(default=None, ge=0.0)
    allowed_uses: list[AllowedUse] = Field(default_factory=lambda: ["transfer_source"])
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_composition(self) -> DatasetEntry:
        if not self.composition or any(n <= 0 for n in self.composition.values()):
            raise ValueError("composition needs positive atom counts")
        return self


class DatasetRecord(BaseModel):
    """Element coverage of one ingested dataset, kept separate from the target hierarchy."""

    name: str
    elements: list[str]
    target_elements_covered: list[str]
    off_target_elements: list[str]
    covered_subspaces: list[str]
    n_entries: int = 0
    n_new_calculations: int = 0
    n_duplicate_calculations: int = 0
    n_new_structures: int = 0


class CandidateRegistry(BaseModel):
    candidates: dict[str, StructureCandidateRecord] = Field(default_factory=dict)
    evaluations: dict[str, CandidateEvaluationRecord] = Field(default_factory=dict)
    relaxed_families: dict[str, list[str]] = Field(default_factory=dict)
    calculations: dict[str, CandidateEvaluationRecord] = Field(default_factory=dict)
    datasets: dict[str, DatasetRecord] = Field(default_factory=dict)

    def find_by_structure_hash(self, structure_hash: str) -> str | None:
        for candidate_id, record in self.candidates.items():
            if record.structure_hash == structure_hash:
                return candidate_id
        return None

    def target_coverage(self, subspaces: dict[int, list[tuple[str, ...]]]) -> dict[str, int]:
        """Records per target subspace, counted only on an exact element-set match."""
        counts = {chemical_system(sub): 0 for subs in subspaces.values() for sub in subs}
        for record in self.candidates.values():
            key = chemical_system(record.element_set)
            if key in counts:
                counts[key] += 1
        return counts


def ingest_dataset(
    registry: CandidateRegistry,
    name: str,
    entries: Iterable[DatasetEntry],
    *,
    target_elements: Iterable[str],
    origin: Origin = "existing",
) -> DatasetRecord:
    """Ingest a dataset with explicit element coverage and duplicate detection.

    A structure covers a hierarchy subspace only if its element set equals that subspace, so
    off-target elements never count as coverage of the target subspaces. Structures dedupe by hash;
    a calculation dedupes only on an identical calculation ID, so changed numerics or magnetic
    initialisation keep separate records.
    """
    target = set(target_elements)
    subspace_keys = {
        chemical_system(sub) for subs in hierarchy_subspaces(target).values() for sub in subs
    }
    elements: set[str] = set()
    covered: set[str] = set()
    n_entries = n_new_calc = n_dup = n_new_struct = 0
    for entry in entries:
        n_entries += 1
        element_set = sorted(entry.composition)
        elements.update(element_set)
        if chemical_system(element_set) in subspace_keys:
            covered.add(chemical_system(element_set))
        candidate_id = registry.find_by_structure_hash(entry.structure_hash)
        if candidate_id is None:
            candidate_id = f"{name}:{entry.structure_hash[:16]}"
            n_new_struct += 1
            registry.candidates[candidate_id] = StructureCandidateRecord(
                candidate_id=candidate_id,
                formula=composition_formula(entry.composition),
                structure_path=entry.structure_path,
                structure_hash=entry.structure_hash,
                source=name,
                chemical_system=chemical_system(element_set),
                element_set=element_set,
                n_components=len(element_set),
                composition=dict(sorted(entry.composition.items())),
                parent_lattice=entry.parent_lattice,
                lattice_parameter_A=entry.lattice_parameter_A,
                supercell=entry.supercell,
                atom_count=sum(entry.composition.values()),
                ordering=entry.ordering,
                sro=entry.sro,
                local_environments=entry.local_environments,
                defect_config=entry.defect_config,
                origin=origin,
                source_dataset=name,
                provenance=entry.provenance,
                allowed_uses=entry.allowed_uses,
            )
        cid = calculation_id(
            entry.structure_hash, entry.dft_backend, entry.dft_settings, entry.magnetic_initial
        )
        if cid in registry.calculations:
            n_dup += 1
            continue
        n_new_calc += 1
        registry.calculations[cid] = CandidateEvaluationRecord(
            candidate_id=candidate_id,
            backend=entry.dft_backend,
            converged=entry.converged,
            calculation_id=cid,
            dft_settings=entry.dft_settings,
            magnetic_initial=entry.magnetic_initial,
            magnetic_final=entry.magnetic_final,
            properties=entry.properties,
            cost_node_hours=entry.cost_node_hours,
            origin=origin,
            source_dataset=name,
            provenance=entry.provenance,
        )
    record = DatasetRecord(
        name=name,
        elements=sorted(elements),
        target_elements_covered=sorted(elements & target),
        off_target_elements=sorted(elements - target),
        covered_subspaces=sorted(covered),
        n_entries=n_entries,
        n_new_calculations=n_new_calc,
        n_duplicate_calculations=n_dup,
        n_new_structures=n_new_struct,
    )
    registry.datasets[name] = record
    return record


class CandidateSelectionPolicy(BaseModel):
    enabled: bool = False
    mode: str = "adaptive"
    lambda_value: float = Field(0.5, ge=0.0, le=1.0)
    minimum_exploitation_fraction: float = Field(0.2, ge=0.0, le=1.0)
    minimum_exploration_fraction: float = Field(0.2, ge=0.0, le=1.0)
    maximum_per_relaxed_family: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _valid_mode_and_quotas(self) -> CandidateSelectionPolicy:
        if self.mode not in {"random", "exploitation", "exploration", "adaptive"}:
            raise ValueError(f"unsupported candidate acquisition mode: {self.mode}")
        if self.minimum_exploitation_fraction + self.minimum_exploration_fraction > 1.0:
            raise ValueError("minimum candidate acquisition fractions must sum to at most one")
        return self


class CandidateSelectionScore(BaseModel):
    candidate_id: str
    exploitation_score: float
    exploration_score: float
    combined_score: float
    assigned_branch: str
    relaxed_family_id: str


def structure_content_hash(path: str | Path) -> str:
    """Hash chemistry and geometry independently of atom ordering and translation."""
    from ase.io import read

    atoms = read(str(path))
    numbers = np.asarray(atoms.numbers, dtype=int)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    scaled = np.asarray(atoms.get_scaled_positions(wrap=False))
    scaled[:, pbc] = np.mod(scaled[:, pbc], 1.0)
    metric = np.asarray(atoms.cell) @ np.asarray(atoms.cell).T
    origins = scaled if len(scaled) else np.zeros((1, 3))
    representations: list[str] = []
    for origin in origins:
        shifted = scaled - origin
        shifted[:, pbc] = np.mod(shifted[:, pbc], 1.0)
        sites = sorted(
            (int(number), *(round(float(value), 8) for value in position))
            for number, position in zip(numbers, shifted, strict=True)
        )
        payload = {
            "sites": sites,
            "metric": np.round(metric, decimals=8).tolist(),
            "pbc": pbc.tolist(),
        }
        representations.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return hashlib.sha256(min(representations).encode("utf-8")).hexdigest()


def soap_descriptor(
    path: str | Path,
    *,
    species: list[str],
    r_cut: float = 6.0,
    n_max: int = 8,
    l_max: int = 6,
) -> np.ndarray:
    """Return a normalized, structure-averaged DScribe SOAP descriptor."""
    from ase.io import read
    from dscribe.descriptors import SOAP

    atoms = read(str(path))
    descriptor = SOAP(
        species=sorted(set(species)),
        periodic=bool(np.any(atoms.pbc)),
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average="inner",
        sparse=False,
    ).create(atoms)
    values = np.asarray(descriptor, dtype=float).reshape(-1)
    norm = np.linalg.norm(values)
    return values / norm if norm else values


def soap_novelty(descriptor: np.ndarray, references: list[np.ndarray]) -> float:
    """Cosine distance to the nearest reference descriptor, normalized to [0, 1]."""
    if not references:
        return 1.0
    distances = [1.0 - float(np.clip(np.dot(descriptor, ref), -1.0, 1.0)) for ref in references]
    return float(np.clip(min(distances) / 2.0, 0.0, 1.0))


def _candidate_id(
    registry: CandidateRegistry,
    candidate: PhaseCandidate,
    index: int,
    iteration: int,
) -> tuple[str, str]:
    base = candidate.candidate_id or f"{candidate.formula}-{candidate.source[0].upper()}{index:04d}"
    if base not in registry.candidates:
        return base, base
    candidate_id = f"{base}-iter{iteration:04d}"
    suffix = 2
    while candidate_id in registry.candidates:
        candidate_id = f"{base}-iter{iteration:04d}-{suffix}"
        suffix += 1
    return candidate_id, base


def _assign_relaxed_families(registry: CandidateRegistry, candidate_ids: list[str]) -> None:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    matcher = StructureMatcher(primitive_cell=True, attempt_supercell=True)
    representatives: list[tuple[str, Structure]] = []
    for family_id, family_members in sorted(registry.relaxed_families.items()):
        for candidate_id in family_members:
            record = registry.candidates.get(candidate_id)
            if record is not None and record.optimized_structure_path is not None:
                representatives.append(
                    (family_id, Structure.from_file(record.optimized_structure_path))
                )
                break
    for candidate_id in sorted(candidate_ids):
        record = registry.candidates[candidate_id]
        if record.optimized_structure_path is None:
            continue
        structure = Structure.from_file(record.optimized_structure_path)
        family_id = None
        for existing_id, representative in representatives:
            if matcher.fit(representative, structure):
                family_id = existing_id
                break
        if family_id is None:
            family_id = f"family-{record.optimized_structure_hash[:16]}"
            representatives.append((family_id, structure))
        record.relaxed_family_id = family_id
        registry.relaxed_families.setdefault(family_id, []).append(candidate_id)


def ingest_exploration_result(
    registry: CandidateRegistry,
    exploration: CompositionExplorationResult,
    *,
    iteration: int,
    backend: str | None = None,
    model_identifier: str | None = None,
    model_checkpoint_hash: str | None = None,
    uncertainty_by_candidate: dict[str, float] | None = None,
    novelty_reference_paths: list[str] | None = None,
) -> list[str]:
    """Ingest seed and relaxation lineage, then group equivalent relaxed minima."""
    relaxation_by_path = {item.structure_path: item for item in exploration.relaxations}
    candidate_ids: list[str] = []
    for index, candidate in enumerate(exploration.phase_candidates):
        candidate_id, source_candidate_id = _candidate_id(registry, candidate, index, iteration)
        candidate_ids.append(candidate_id)
        relaxation = relaxation_by_path.get(candidate.structure_path)
        optimized_path = relaxation.optimized_structure_path if relaxation is not None else None
        record = StructureCandidateRecord(
            candidate_id=candidate_id,
            formula=candidate.formula,
            structure_path=candidate.structure_path,
            structure_hash=(
                candidate.structure_hash or structure_content_hash(candidate.structure_path)
            ),
            source=candidate.source,
            prototype_id=candidate.prototype_id,
            space_group=candidate.space_group,
            decoration_mapping=candidate.decoration_mapping,
            random_seed=candidate.random_seed,
            parent_candidate_id=candidate.parent_candidate_id,
            iteration_created=iteration,
            needs_dft_verification=candidate.needs_dft_verification,
            optimized_structure_path=optimized_path,
            optimized_structure_hash=(
                structure_content_hash(optimized_path) if optimized_path else None
            ),
        )
        registry.candidates[candidate_id] = record
        atom_count = candidate.num_atoms or (
            len(ase_read(optimized_path or candidate.structure_path))
            if relaxation is not None
            else 0
        )
        registry.evaluations[candidate_id] = CandidateEvaluationRecord(
            candidate_id=candidate_id,
            backend=backend,
            model_identifier=model_identifier,
            model_checkpoint_hash=model_checkpoint_hash,
            converged=bool(relaxation and relaxation.converged),
            final_energy_eV=(relaxation.final_energy_eV if relaxation else None),
            energy_per_atom_eV=(
                relaxation.final_energy_eV / atom_count if relaxation and atom_count else None
            ),
            residual_force_eV_per_A=(relaxation.final_max_force_eV_per_A if relaxation else None),
            optimization_steps=(relaxation.num_steps if relaxation else None),
            uncertainty=(uncertainty_by_candidate or {}).get(source_candidate_id),
            failure_reason=None if relaxation is not None else "no relaxation result",
        )
    _assign_relaxed_families(registry, candidate_ids)

    relaxed = [
        registry.candidates[candidate_id]
        for candidate_id in candidate_ids
        if registry.candidates[candidate_id].optimized_structure_path is not None
    ]
    if relaxed and novelty_reference_paths is not None:
        from ase.io import read

        reference_paths = list(novelty_reference_paths)
        species = set(exploration.composition.elements)
        for path in reference_paths:
            species.update(read(str(path)).get_chemical_symbols())
        soap_species = sorted(species)
        references = [soap_descriptor(path, species=soap_species) for path in reference_paths]
        for record in relaxed:
            descriptor = soap_descriptor(record.optimized_structure_path, species=soap_species)
            novelty = soap_novelty(descriptor, references)
            evaluation = registry.evaluations[record.candidate_id]
            evaluation.structural_novelty = novelty
            evaluation.distance_from_labeled_data = novelty
    return candidate_ids


def formula_acquisition_metrics(
    registry: CandidateRegistry,
    formula: str,
    *,
    energy_above_hull_eV_per_atom: float | None = None,
    near_hull_threshold_eV_per_atom: float = 0.05,
    llm_disagreement: float = 0.0,
    llm_falsification_priority: float = 0.0,
) -> FormulaAcquisitionMetrics:
    evaluations = [
        item
        for candidate_id, item in registry.evaluations.items()
        if registry.candidates[candidate_id].formula == formula
    ]
    converged = [item for item in evaluations if item.converged]
    uncertainty = [item.uncertainty for item in converged if item.uncertainty is not None]
    novelty = [item.structural_novelty for item in converged if item.structural_novelty is not None]
    families = {
        registry.candidates[item.candidate_id].relaxed_family_id
        for item in converged
        if registry.candidates[item.candidate_id].relaxed_family_id is not None
    }
    hull_proximity = 0.0
    if energy_above_hull_eV_per_atom is not None:
        hull_proximity = max(
            0.0,
            1.0 - max(0.0, energy_above_hull_eV_per_atom) / near_hull_threshold_eV_per_atom,
        )
    return FormulaAcquisitionMetrics(
        predicted_hull_proximity=min(1.0, hull_proximity),
        mlip_uncertainty=float(np.clip(np.mean(uncertainty), 0.0, 1.0)) if uncertainty else 0.0,
        structural_novelty=float(np.clip(max(novelty), 0.0, 1.0)) if novelty else 0.0,
        structural_diversity=(len(families) / len(converged) if converged else 0.0),
        composition_coverage=1.0 / (1.0 + len(evaluations)),
        llm_disagreement=float(np.clip(llm_disagreement, 0.0, 1.0)),
        llm_falsification_priority=float(np.clip(llm_falsification_priority, 0.0, 1.0)),
        convergence=len(converged) / len(evaluations) if evaluations else 0.0,
    )


def select_dft_refinement_candidates(
    exploration: CompositionExplorationResult,
    *,
    max_candidates: int,
    policy: CandidateSelectionPolicy,
    uncertainty_by_candidate: dict[str, float] | None = None,
    random_seed: int,
) -> tuple[list[Any], dict[str, CandidateSelectionScore]]:
    """Select non-equivalent relaxed structures for DFT with branch quotas.

    Random mode shuffles with *random_seed*.
    """
    temporary = CandidateRegistry()
    ids = ingest_exploration_result(
        temporary,
        exploration,
        iteration=0,
        uncertainty_by_candidate=uncertainty_by_candidate,
        novelty_reference_paths=[],
    )
    relaxation_by_path = {item.structure_path: item for item in exploration.relaxations}
    converged_ids = [
        candidate_id for candidate_id in ids if temporary.evaluations[candidate_id].converged
    ]
    if not converged_ids:
        return [], {}
    prototype_paths = [
        temporary.candidates[candidate_id].optimized_structure_path
        for candidate_id in converged_ids
        if temporary.candidates[candidate_id].source == "prototype"
        and temporary.candidates[candidate_id].optimized_structure_path is not None
    ]
    species = sorted(exploration.composition.elements)
    prototype_descriptors = [soap_descriptor(path, species=species) for path in prototype_paths]
    energies = np.asarray(
        [temporary.evaluations[candidate_id].energy_per_atom_eV for candidate_id in converged_ids],
        dtype=float,
    )
    energy_span = float(np.ptp(energies))
    energy_quality = (
        np.ones(len(energies))
        if energy_span == 0
        else 1.0 - (energies - energies.min()) / energy_span
    )
    scores: dict[str, CandidateSelectionScore] = {}
    for index, candidate_id in enumerate(converged_ids):
        record = temporary.candidates[candidate_id]
        evaluation = temporary.evaluations[candidate_id]
        descriptor = soap_descriptor(record.optimized_structure_path, species=species)
        novelty = (
            0.0 if record.source == "prototype" else soap_novelty(descriptor, prototype_descriptors)
        )
        uncertainty = float(np.clip(evaluation.uncertainty or 0.0, 0.0, 1.0))
        exploitation = 0.7 * float(energy_quality[index]) + 0.3 * (1.0 - uncertainty)
        exploration = 0.5 * uncertainty + 0.3 * novelty + 0.2 * float(record.source == "random")
        combined = policy.lambda_value * exploitation + (1.0 - policy.lambda_value) * exploration
        scores[candidate_id] = CandidateSelectionScore(
            candidate_id=candidate_id,
            exploitation_score=exploitation,
            exploration_score=exploration,
            combined_score=combined,
            assigned_branch=("exploitation" if exploitation >= exploration else "exploration"),
            relaxed_family_id=record.relaxed_family_id or candidate_id,
        )

    limit = min(max_candidates, len(converged_ids))
    if policy.mode == "random":
        ordered = sorted(converged_ids)
        python_rng(random_seed).shuffle(ordered)
    elif policy.mode == "exploitation":
        ordered = sorted(converged_ids, key=lambda key: (-scores[key].exploitation_score, key))
    elif policy.mode == "exploration":
        ordered = sorted(converged_ids, key=lambda key: (-scores[key].exploration_score, key))
    else:
        n_exploit = min(limit, int(np.ceil(limit * policy.minimum_exploitation_fraction)))
        n_explore = min(
            limit - n_exploit,
            int(np.ceil(limit * policy.minimum_exploration_fraction)),
        )
        exploit = sorted(converged_ids, key=lambda key: (-scores[key].exploitation_score, key))
        explore = sorted(converged_ids, key=lambda key: (-scores[key].exploration_score, key))
        combined = sorted(converged_ids, key=lambda key: (-scores[key].combined_score, key))
        ordered = exploit[:n_exploit]
        ordered.extend(key for key in explore if key not in ordered)
        ordered = ordered[: n_exploit + n_explore]
        ordered.extend(key for key in combined if key not in ordered)

    selected_ids: list[str] = []
    family_counts: dict[str, int] = {}
    for candidate_id in ordered:
        family = scores[candidate_id].relaxed_family_id
        if family_counts.get(family, 0) >= policy.maximum_per_relaxed_family:
            continue
        selected_ids.append(candidate_id)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected_ids) == limit:
            break
    selected = [
        relaxation_by_path[temporary.candidates[candidate_id].structure_path]
        for candidate_id in selected_ids
    ]
    return selected, scores
