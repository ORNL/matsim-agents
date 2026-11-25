"""Deterministic structural screening, without energy or uncertainty predictions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from ase.io import read
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from pymatgen.core import Composition, Structure


class CandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: int = Field(ge=0)
    disposition: Literal["selected", "duplicate", "deferred"]
    reason: str
    duplicate_of: str | None = None
    novelty_distance_A: float | None = Field(None, ge=0, allow_inf_nan=False)


class CandidateSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["periodic-neighbor-diversity-v1"] = "periodic-neighbor-diversity-v1"
    references: list[str]
    selected: list[int]
    decisions: list[CandidateDecision]

    @model_validator(mode="after")
    def _validate_decisions(self) -> CandidateSelection:
        if [decision.candidate for decision in self.decisions] != list(range(len(self.decisions))):
            raise ValueError("Selection decisions must cover candidates in input order")
        selected = {
            decision.candidate for decision in self.decisions if decision.disposition == "selected"
        }
        if len(set(self.selected)) != len(self.selected) or set(self.selected) != selected:
            raise ValueError("Selection order must contain each selected candidate exactly once")
        for decision in self.decisions:
            if (decision.disposition == "duplicate") != (decision.duplicate_of is not None):
                raise ValueError("Only duplicate decisions must identify a matching structure")
        return self


def _read_structure(path: str, composition: Composition) -> Structure:
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = read(path)
    if (
        len(atoms) == 0
        or not atoms.pbc.all()
        or not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
        or abs(np.linalg.det(atoms.cell.array)) <= 1e-12
    ):
        raise ValueError(f"Selection requires a finite, nonempty periodic structure: {path}")
    structure = AseAtomsAdaptor.get_structure(atoms)
    if structure.composition.fractional_composition != composition.fractional_composition:
        raise ValueError(f"Selection structure does not match the requested composition: {path}")
    return structure


def _neighbor_fingerprint(structure: Structure, species: list[str]) -> np.ndarray:
    """Species-averaged first-eight periodic neighbor distances, in angstroms."""
    radius = max(1.0, 2 * (structure.volume / len(structure)) ** (1 / 3))
    for _ in range(8):
        neighbors = structure.get_all_neighbors(radius)
        if all(len(row) >= 8 for row in neighbors):
            distances = np.array(
                [sorted(neighbor.nn_distance for neighbor in row)[:8] for row in neighbors]
            )
            return np.concatenate(
                [
                    distances[[site.specie.symbol == element for site in structure]].mean(axis=0)
                    for element in species
                ]
            )
        radius *= 2
    raise ValueError("Cannot construct eight-neighbor periodic diversity fingerprint")


def select_diverse_candidates(
    paths: list[str], reference_paths: list[str], *, formula: str, budget: int
) -> CandidateSelection:
    """Deduplicate, then greedily maximize distance to prior work and this selection.

    StructureMatcher handles periodic equivalence; the inexpensive neighbor-distance
    fingerprint is only a diversity heuristic, never a duplicate or stability test.
    Input order breaks ties and seeds selection when there is no prior work.
    """
    if budget <= 0:
        raise ValueError("Selection budget must be positive")
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Composition

    composition = Composition(formula)
    species = sorted(element.symbol for element in composition.elements)
    references = list(dict.fromkeys(str(Path(path).resolve()) for path in reference_paths))
    structures = [_read_structure(path, composition) for path in paths]
    previous = [_read_structure(path, composition) for path in references]
    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
    )
    representatives = list(zip(references, previous, strict=True))
    decisions = []
    eligible = []
    for index, structure in enumerate(structures):
        duplicate = next(
            (path for path, reference in representatives if matcher.fit(reference, structure)), None
        )
        if duplicate is not None:
            decisions.append(
                CandidateDecision(
                    candidate=index,
                    disposition="duplicate",
                    reason="periodic_structure_match",
                    duplicate_of=duplicate,
                )
            )
        else:
            representatives.append((paths[index], structure))
            eligible.append(index)
            decisions.append(
                CandidateDecision(
                    candidate=index,
                    disposition="deferred",
                    reason="per_round_budget",
                )
            )
    fingerprints = {index: _neighbor_fingerprint(structures[index], species) for index in eligible}
    reference_fingerprints = [_neighbor_fingerprint(structure, species) for structure in previous]
    selected = []
    while eligible and len(selected) < budget:
        distances = (
            {
                index: min(
                    float(np.sqrt(np.mean((fingerprints[index] - reference) ** 2)))
                    for reference in reference_fingerprints
                )
                for index in eligible
            }
            if reference_fingerprints
            else {}
        )
        chosen = (
            max(eligible, key=lambda index: (distances[index], -index))
            if distances
            else eligible[0]
        )
        decisions[chosen] = CandidateDecision(
            candidate=chosen,
            disposition="selected",
            reason="farthest_from_prior_and_selected" if distances else "first_unique_candidate",
            novelty_distance_A=distances.get(chosen),
        )
        selected.append(chosen)
        eligible.remove(chosen)
        reference_fingerprints.append(fingerprints[chosen])
    return CandidateSelection(references=references, selected=selected, decisions=decisions)
