"""Unified crystal-structure seed generation for a target composition.

Two complementary sources are combined into one ranked list of seed
structures that downstream agents will relax and score:

1. **AFLOW prototype decoration** — for every known crystal-prototype
   topology (~288 entries from the pymatgen-bundled AFLOW
   encyclopedia, spanning all 230 3-D space groups and a wide range of
   stoichiometries) whose reduced stoichiometric ratios match the target
   composition, we substitute the prototype's placeholder species with
   the target's elements. When the prototype admits more than one
   element-to-placeholder assignment (e.g. ABX3 vs BAX3 for a 1:1:3
   ternary), we enumerate the symmetrically distinct assignments and
   keep all of them. This covers known crystal phases (fcc, bcc, hcp,
   rocksalt, zincblende, perovskite, spinel, Heusler, MAX phases, …)
   without any hand-coded rules.

2. **pyXtal random search** — for novelty / characterization of
   compositions that *don't* match any known prototype (e.g. high-entropy
   alloys, exotic stoichiometries), we draw ``n_random`` random
   crystal structures uniformly across the 230 space groups, respecting
   Wyckoff-position symmetry and minimum-distance constraints. This
   path is optional (pyxtal is an optional dependency) and any seed it
   produces is flagged with ``needs_dft_verification=True`` so the
   stability report can call them out as candidates requiring further
   DFT validation before publication.

The unified entry point is :func:`generate_seeds` which writes all
seeds to disk and returns a list of :class:`PhaseCandidate` objects
ready to feed into the relaxation pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from collections import Counter
from functools import lru_cache
from itertools import permutations
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from matsim_agents.discovery.composition import Composition, parse_composition

if TYPE_CHECKING:
    from pymatgen.core import Structure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class PhaseCandidate(BaseModel):
    """A single seed structure generated for a target composition.

    Carries the metadata of the unified prototype + random pipeline:
    ``source`` distinguishes AFLOW-prototype decorations from pyXtal
    random structures, and ``needs_dft_verification`` flags the latter
    as requiring further DFT validation before any stability claim.
    """

    formula: str
    candidate_id: str | None = None
    structure_path: str
    structure_hash: str | None = None
    source: Literal["prototype", "random"] = "prototype"
    prototype_id: str | None = Field(
        default=None,
        description="AFLOW prototype label, e.g. 'AB_cF8_216_c_a' (zincblende).",
    )
    space_group: int | None = None
    lattice_parameters: dict[str, float] | None = None
    cell_matrix: list[list[float]] | None = None
    decoration_mapping: str | None = None
    random_seed: int | None = None
    parent_candidate_id: str | None = None
    iteration_created: int = 0
    num_atoms: int | None = None
    formula_units: int | None = None
    needs_dft_verification: bool = Field(
        default=False,
        description="True for random-source seeds: known-prototype hits are "
        "structurally familiar but pyXtal-generated novel topologies should be "
        "DFT-validated before any stability claim.",
    )
    notes: str | None = None


class SeedGenerationAction(BaseModel):
    """Strategy-issued controls for one reproducible structure-generation batch."""

    action_id: str = Field(min_length=1)
    formula: str
    formula_units: int = Field(gt=0)
    max_aflow_candidates: int = Field(default=0, ge=0)
    prototype_ids: list[str] = Field(default_factory=list)
    space_groups: list[int] = Field(default_factory=list)
    pyxtal_count: int = Field(default=0, ge=0)
    random_seed: int = 0

    @model_validator(mode="after")
    def _validate_requested_sources(self) -> SeedGenerationAction:
        if self.max_aflow_candidates == 0 and self.pyxtal_count == 0:
            raise ValueError("an action must request AFLOW candidates or pyXtal structures")
        if self.prototype_ids and self.max_aflow_candidates == 0:
            raise ValueError("prototype_ids require a positive max_aflow_candidates")
        return self

    @field_validator("formula")
    @classmethod
    def _normalize_formula(cls, value: str) -> str:
        composition = parse_composition(value)
        if composition is None:
            raise ValueError(f"Could not parse chemical composition: {value!r}")
        return composition.formula

    @field_validator("space_groups")
    @classmethod
    def _validate_space_groups(cls, value: list[int]) -> list[int]:
        if any(group < 1 or group > 230 for group in value):
            raise ValueError("space_groups must contain integers from 1 to 230")
        return value

    @field_validator("action_id")
    @classmethod
    def _validate_action_id(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError("action_id may contain only letters, digits, '.', '_' and '-'")
        return value


# ---------------------------------------------------------------------------
# AFLOW prototype loading
# ---------------------------------------------------------------------------


class _LoadedPrototype:
    """Internal: a prototype structure plus its AFLOW metadata."""

    __slots__ = ("aflow", "pearson", "strukturbericht", "mineral", "structure", "signature")

    def __init__(
        self,
        aflow: str,
        pearson: str | None,
        strukturbericht: str | None,
        mineral: str | None,
        structure: Structure,
        signature: tuple[int, ...],
    ) -> None:
        self.aflow = aflow
        self.pearson = pearson
        self.strukturbericht = strukturbericht
        self.mineral = mineral
        self.structure = structure  # pymatgen Structure with placeholder species
        self.signature = signature  # tuple[int,...] of reduced sorted counts


def _reduce_counts(counts: list[int]) -> tuple[int, ...]:
    if not counts:
        return ()
    g = counts[0]
    for c in counts[1:]:
        g = gcd(g, c)
    g = max(g, 1)
    return tuple(sorted(c // g for c in counts))


@lru_cache(maxsize=1)
def load_prototypes() -> list[_LoadedPrototype]:
    """Load and cache the AFLOW prototype encyclopedia bundled in pymatgen.

    Returns 288 entries spanning the 3-D crystallographic space groups.
    """
    import gzip
    import json
    import pathlib

    import pymatgen.analysis.prototypes as _ap
    from pymatgen.core import Structure
    from pymatgen.util.provenance import StructureNL

    json_path = pathlib.Path(_ap.__file__).parent / "aflow_prototypes.json.gz"
    if not json_path.exists():
        raise FileNotFoundError(
            f"pymatgen AFLOW prototype encyclopedia not found at {json_path}. "
            "Install pymatgen>=2023 or check the package layout."
        )
    with gzip.open(json_path, "rt") as fh:
        data = json.load(fh)

    protos: list[_LoadedPrototype] = []
    for entry in data:
        try:
            snl = StructureNL.from_dict(entry["snl"])
            s = snl.structure
            if not isinstance(s, Structure):
                continue
        except Exception as exc:  # pragma: no cover - guards against future format changes
            logger.debug("Skipping prototype %r: %s", entry.get("tags", {}).get("aflow"), exc)
            continue
        per_site = Counter(site.specie.symbol for site in s)
        signature = _reduce_counts(list(per_site.values()))
        tags = entry.get("tags", {})
        protos.append(
            _LoadedPrototype(
                aflow=tags.get("aflow", ""),
                pearson=tags.get("pearson"),
                strukturbericht=tags.get("strukturbericht"),
                mineral=tags.get("mineral"),
                structure=s,
                signature=signature,
            )
        )
    logger.info("Loaded %d AFLOW prototypes.", len(protos))
    return protos


# ---------------------------------------------------------------------------
# Stoichiometry matching
# ---------------------------------------------------------------------------


def _composition_signature(comp: Composition) -> tuple[int, ...]:
    return _reduce_counts(list(comp.elements.values()))


def compatible_prototypes(comp: Composition) -> list[_LoadedPrototype]:
    """Return every prototype whose reduced stoichiometric ratios match ``comp``."""
    target = _composition_signature(comp)
    return [p for p in load_prototypes() if p.signature == target]


# ---------------------------------------------------------------------------
# Prototype decoration: substitute placeholders with target elements
# ---------------------------------------------------------------------------


def _placeholder_groups(structure) -> list[tuple[str, int]]:
    """Return [(placeholder_symbol, count), ...] in canonical (sorted by count) order."""
    per_site = Counter(site.specie.symbol for site in structure)
    # Sort by (count, symbol) so the canonical ordering is deterministic.
    return sorted(per_site.items(), key=lambda kv: (kv[1], kv[0]))


def _target_groups(comp: Composition) -> list[tuple[str, int]]:
    return sorted(comp.elements.items(), key=lambda kv: (kv[1], kv[0]))


def _enumerate_assignments(
    proto_groups: list[tuple[str, int]],
    target_groups: list[tuple[str, int]],
) -> list[dict[str, str]]:
    """Enumerate every placeholder->target mapping that preserves *reduced* counts.

    The prototype cell may be an integer multiple of the composition's
    formula unit (e.g. a Te4 elemental prototype decorated with formula
    unit Si yields 4 Si atoms). We therefore bin both sides by their
    counts divided by the per-side gcd, and require the reduced-count
    multisets to match (guaranteed by the caller via
    :func:`compatible_prototypes`).
    """
    proto_counts = [n for _, n in proto_groups]
    tgt_counts = [n for _, n in target_groups]
    if not proto_counts or not tgt_counts:
        return []
    proto_gcd = proto_counts[0]
    for c in proto_counts[1:]:
        proto_gcd = gcd(proto_gcd, c)
    proto_gcd = max(proto_gcd, 1)
    tgt_gcd = tgt_counts[0]
    for c in tgt_counts[1:]:
        tgt_gcd = gcd(tgt_gcd, c)
    tgt_gcd = max(tgt_gcd, 1)

    # Bin by reduced count value.
    proto_by_n: dict[int, list[str]] = {}
    tgt_by_n: dict[int, list[str]] = {}
    for ph, n in proto_groups:
        proto_by_n.setdefault(n // proto_gcd, []).append(ph)
    for el, n in target_groups:
        tgt_by_n.setdefault(n // tgt_gcd, []).append(el)

    if sorted(proto_by_n.keys()) != sorted(tgt_by_n.keys()) or any(
        len(proto_by_n[k]) != len(tgt_by_n[k]) for k in proto_by_n
    ):
        return []

    # For each reduced-count bin, enumerate permutations of targets onto placeholders.
    bins = sorted(proto_by_n.keys())
    per_bin_perms = [list(permutations(tgt_by_n[k])) for k in bins]

    assignments: list[dict[str, str]] = []
    from itertools import product

    for combo in product(*per_bin_perms):
        mapping: dict[str, str] = {}
        for bin_key, perm in zip(bins, combo, strict=True):
            for ph, el in zip(proto_by_n[bin_key], perm, strict=True):
                mapping[ph] = el
        assignments.append(mapping)
    return assignments


def decorate(proto: _LoadedPrototype, comp: Composition) -> list[tuple[Structure, str]]:
    """Yield every distinct decoration of ``proto`` with ``comp``'s elements.

    Returns a list of (pymatgen Structure, mapping_str). Duplicates that
    fold onto the same structure (by symmetry) are removed via pymatgen
    ``StructureMatcher``; if pymatgen is unavailable we keep them all.
    """
    pgroups = _placeholder_groups(proto.structure)
    tgroups = _target_groups(comp)
    assignments = _enumerate_assignments(pgroups, tgroups)
    if not assignments:
        return []

    from pymatgen.core import Structure

    out: list[tuple[Structure, str]] = []
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher

        matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False)
    except Exception:  # pragma: no cover
        matcher = None

    seen_structs: list[Structure] = []
    for mapping in assignments:
        # Build a fresh structure with substituted species.
        species = [mapping[site.specie.symbol] for site in proto.structure]
        new = Structure(
            lattice=proto.structure.lattice,
            species=species,
            coords=[site.frac_coords for site in proto.structure],
            coords_are_cartesian=False,
        )
        if matcher is not None and any(matcher.fit(prev, new) for prev in seen_structs):
            continue
        seen_structs.append(new)
        mapping_str = ",".join(f"{ph}->{el}" for ph, el in sorted(mapping.items()))
        out.append((new, mapping_str))
    return out


# ---------------------------------------------------------------------------
# pyXtal random search (optional dependency)
# ---------------------------------------------------------------------------


def _pyxtal_available() -> bool:
    try:
        import pyxtal  # noqa: F401

        return True
    except ImportError:
        return False


def random_search(
    comp: Composition,
    n: int = 50,
    *,
    seed: int = 0,
    formula_units: int = 1,
    space_groups: list[int] | None = None,
    max_attempts_factor: int = 4,
) -> list[tuple[Structure, int]]:
    """Generate up to ``n`` random crystal structures across the 230 space groups.

    Returns a list of ``(pymatgen Structure, space_group)`` tuples.
    Returns ``[]`` (with a warning) if pyXtal is not installed.
    """
    if formula_units < 1:
        raise ValueError("formula_units must be positive")
    if space_groups is not None and (
        not space_groups or any(group < 1 or group > 230 for group in space_groups)
    ):
        raise ValueError("space_groups must contain integers from 1 to 230")
    if n <= 0:
        return []
    if not _pyxtal_available():
        warnings.warn(
            "pyXtal is not installed; random-structure search will be skipped. "
            "Add `pyxtal` to your environment (pip install pyxtal) to enable it.",
            stacklevel=2,
        )
        return []

    import random as _random

    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pyxtal import pyxtal

    rng = _random.Random(seed)
    elements = list(comp.elements.keys())
    counts = [comp.elements[e] * formula_units for e in elements]
    allowed_groups = tuple(space_groups) if space_groups is not None else tuple(range(1, 231))

    results: list[tuple[Structure, int]] = []
    matcher = StructureMatcher(primitive_cell=True, attempt_supercell=False)
    max_attempts = max(n * max_attempts_factor, n + 20)
    attempts = 0
    duplicates = 0
    while len(results) < n and attempts < max_attempts:
        attempts += 1
        spg = rng.choice(allowed_groups)
        try:
            x = pyxtal()
            x.from_random(dim=3, group=spg, species=elements, numIons=counts)
            if not x.valid:
                continue
            pmg = x.to_pymatgen()
            if not isinstance(pmg, Structure):
                raise TypeError("pyXtal did not produce a periodic pymatgen Structure")
            if any(matcher.fit(existing, pmg) for existing, _ in results):
                duplicates += 1
                continue
            results.append((pmg, spg))
        except Exception as exc:
            logger.debug("pyXtal SG=%d failed: %s", spg, exc)
            continue
    if len(results) < n:
        logger.info(
            "pyXtal produced %d/%d unique structures after %d attempts (%d duplicates).",
            len(results),
            n,
            attempts,
            duplicates,
        )
    return results


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def generate_seeds(
    composition: Composition,
    output_dir: str,
    *,
    n_random: int = 50,
    fmt: str = "vasp",
    random_seed: int = 0,
    formula_units: int | None = None,
    include_aflow: bool = True,
    max_aflow_prototypes: int | None = None,
    max_aflow_candidates: int | None = None,
    prototype_ids: list[str] | None = None,
    space_groups: list[int] | None = None,
) -> list[PhaseCandidate]:
    """Generate the unified prototype + random seed list for one composition.

    Parameters
    ----------
    composition:
        Target composition.
    output_dir:
        Directory where seed structure files are written. Created if needed.
    n_random:
        Number of random pyXtal structures to draw (in addition to the
        prototype matches). Set to ``0`` to disable. When pyXtal is not
        installed, this silently degrades to ``0`` with a warning.
    fmt:
        ASE-compatible structure format (``"vasp"``, ``"cif"``, ``"xyz"``).
    random_seed:
        Seed for the pyXtal RNG (reproducibility).
    formula_units:
        Requested number of reduced formula units. AFLOW cells are replicated
        only when their existing formula-unit count divides this value; pyXtal
        structures are generated at the requested size.
    include_aflow:
        Whether to include AFLOW prototype decorations in this action.
    max_aflow_prototypes:
        Maximum number of matching AFLOW prototypes to use after filtering.
    max_aflow_candidates:
        Maximum number of generated AFLOW configurations to return. This is
        applied after decoration, size filtering, and structural deduplication.
    prototype_ids:
        Optional AFLOW prototype labels to include.
    space_groups:
        Optional allowed space groups applied to both AFLOW and pyXtal.
    """
    from ase.io import write
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.io.ase import AseAtomsAdaptor

    if n_random < 0:
        raise ValueError("n_random must be non-negative")
    if formula_units is not None and formula_units < 1:
        raise ValueError("formula_units must be positive")
    if max_aflow_prototypes is not None and max_aflow_prototypes < 1:
        raise ValueError("max_aflow_prototypes must be positive")
    if max_aflow_candidates is not None and max_aflow_candidates < 0:
        raise ValueError("max_aflow_candidates must be non-negative")
    if space_groups is not None and (
        not space_groups or any(group < 1 or group > 230 for group in space_groups)
    ):
        raise ValueError("space_groups must contain integers from 1 to 230")
    requested_prototypes = set(prototype_ids or [])
    allowed_space_groups = set(space_groups) if space_groups is not None else None
    os.makedirs(output_dir, exist_ok=True)
    ext = ".vasp" if fmt == "vasp" else f".{fmt}"
    candidates: list[PhaseCandidate] = []
    seen_by_size: dict[int, list[Structure]] = {}
    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False)

    # --- 1. Prototype decoration -------------------------------------------
    matches = compatible_prototypes(composition) if include_aflow else []
    if requested_prototypes:
        found = {proto.aflow for proto in matches}
        missing = requested_prototypes - found
        if missing:
            raise ValueError(
                f"Requested AFLOW prototype(s) do not match {composition.formula}: "
                f"{', '.join(sorted(missing))}"
            )
        matches = [proto for proto in matches if proto.aflow in requested_prototypes]
    if formula_units is not None:
        matches = [
            proto
            for proto in matches
            if proto.structure.num_sites % composition.total_atoms == 0
            and formula_units % (proto.structure.num_sites // composition.total_atoms) == 0
        ]
    if max_aflow_candidates == 0:
        matches = []
    if max_aflow_prototypes is not None:
        matches = matches[:max_aflow_prototypes]
    logger.info(
        "Composition %s: %d AFLOW prototypes match stoichiometry signature %s.",
        composition.formula,
        len(matches),
        _composition_signature(composition),
    )

    aflow_candidates_written = 0
    for proto in matches:
        if max_aflow_candidates is not None and aflow_candidates_written >= max_aflow_candidates:
            break
        decorations = decorate(proto, composition)
        for o_idx, (struct, mapping_str) in enumerate(decorations):
            if (
                max_aflow_candidates is not None
                and aflow_candidates_written >= max_aflow_candidates
            ):
                break
            base_formula_units, remainder = divmod(struct.num_sites, composition.total_atoms)
            if remainder:
                logger.warning(
                    "Skipping AFLOW prototype %s: atom count %d is not an integer "
                    "number of %s formula units.",
                    proto.aflow,
                    struct.num_sites,
                    composition.formula,
                )
                continue
            if formula_units is not None:
                multiplier, remainder = divmod(formula_units, base_formula_units)
                if remainder:
                    logger.debug(
                        "Skipping AFLOW prototype %s: its %d formula units do not "
                        "divide requested tier %d.",
                        proto.aflow,
                        base_formula_units,
                        formula_units,
                    )
                    continue
                if multiplier > 1:
                    struct.make_supercell(_balanced_supercell_matrix(struct, multiplier))
            actual_formula_units = struct.num_sites // composition.total_atoms
            spg = struct.get_space_group_info()[1]
            if allowed_space_groups is not None and spg not in allowed_space_groups:
                continue
            prior_structures = seen_by_size.setdefault(actual_formula_units, [])
            if any(matcher.fit(previous, struct) for previous in prior_structures):
                continue
            prior_structures.append(struct)
            atoms = AseAtomsAdaptor.get_atoms(struct)
            o_tag = f"_o{o_idx}" if len(decorations) > 1 else ""
            # AFLOW labels contain '_' already; replace with '-' in filename
            # to keep tags readable.
            safe_label = proto.aflow.replace("/", "-") or f"proto{matches.index(proto)}"
            size_tag = f"__fu{actual_formula_units}" if formula_units is not None else ""
            path = os.path.join(
                output_dir,
                f"{composition.formula}{size_tag}__{safe_label}{o_tag}{ext}",
            )
            write(path, atoms, format=fmt)
            candidate_id = f"{composition.formula}-P{len(candidates):04d}"

            notes_bits = [f"AFLOW prototype {proto.aflow} (SG {spg})."]
            if proto.mineral:
                notes_bits.append(f"Mineral: {proto.mineral}.")
            if proto.strukturbericht:
                notes_bits.append(f"Strukturbericht: {proto.strukturbericht}.")
            notes_bits.append(f"Decoration: {mapping_str}.")

            candidates.append(
                PhaseCandidate(
                    formula=composition.formula,
                    candidate_id=candidate_id,
                    structure_path=path,
                    structure_hash=_file_sha256(path),
                    source="prototype",
                    prototype_id=proto.aflow,
                    space_group=spg,
                    lattice_parameters=_lattice_parameters(struct),
                    cell_matrix=struct.lattice.matrix.tolist(),
                    decoration_mapping=mapping_str,
                    num_atoms=len(atoms),
                    formula_units=actual_formula_units,
                    needs_dft_verification=False,
                    notes=" ".join(notes_bits),
                )
            )
            aflow_candidates_written += 1

    # --- 2. pyXtal random search -------------------------------------------
    requested_formula_units = formula_units or 1
    random_hits = random_search(
        composition,
        n=n_random,
        seed=random_seed,
        formula_units=requested_formula_units,
        space_groups=space_groups,
    )
    for r_idx, (struct, spg) in enumerate(random_hits):
        actual_formula_units = struct.num_sites // composition.total_atoms
        prior_structures = seen_by_size.setdefault(actual_formula_units, [])
        if any(matcher.fit(previous, struct) for previous in prior_structures):
            continue
        prior_structures.append(struct)
        atoms = AseAtomsAdaptor.get_atoms(struct)
        size_tag = f"__fu{actual_formula_units}" if formula_units is not None else ""
        path = os.path.join(
            output_dir,
            f"{composition.formula}{size_tag}__pyxtal_sg{spg:03d}_r{r_idx:03d}{ext}",
        )
        write(path, atoms, format=fmt)
        candidates.append(
            PhaseCandidate(
                formula=composition.formula,
                candidate_id=f"{composition.formula}-R{r_idx:04d}",
                structure_path=path,
                structure_hash=_file_sha256(path),
                source="random",
                prototype_id=None,
                space_group=int(spg),
                lattice_parameters=_lattice_parameters(struct),
                cell_matrix=struct.lattice.matrix.tolist(),
                random_seed=random_seed,
                num_atoms=len(atoms),
                formula_units=actual_formula_units,
                needs_dft_verification=True,
                notes=(
                    f"Random pyXtal structure (space group {spg}). "
                    "Novel topology — requires independent validation before any stability claim."
                ),
            )
        )

    prototype_count = sum(candidate.source == "prototype" for candidate in candidates)
    random_count = sum(candidate.source == "random" for candidate in candidates)
    logger.info(
        "Composition %s: %d total seeds (%d prototype + %d random).",
        composition.formula,
        len(candidates),
        prototype_count,
        random_count,
    )
    return candidates


def _balanced_supercell_matrix(structure: Structure, multiplier: int) -> list[list[int]]:
    """Choose a diagonal replication that keeps cell-vector lengths balanced."""
    lengths = structure.lattice.abc
    best_factors = (multiplier, 1, 1)
    best_score = float("inf")
    for a in range(1, multiplier + 1):
        if multiplier % a:
            continue
        remaining = multiplier // a
        for b in range(1, remaining + 1):
            if remaining % b:
                continue
            c = remaining // b
            scaled = (lengths[0] * a, lengths[1] * b, lengths[2] * c)
            score = max(scaled) / min(scaled)
            if score < best_score:
                best_factors = (a, b, c)
                best_score = score
    a, b, c = best_factors
    return [[a, 0, 0], [0, b, 0], [0, 0, c]]


def _lattice_parameters(structure: Structure) -> dict[str, float]:
    lattice = structure.lattice
    return {
        "a": float(lattice.a),
        "b": float(lattice.b),
        "c": float(lattice.c),
        "alpha": float(lattice.alpha),
        "beta": float(lattice.beta),
        "gamma": float(lattice.gamma),
    }


def _file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_seed_action(
    action: SeedGenerationAction,
    output_dir: str,
    *,
    fmt: str = "vasp",
) -> list[PhaseCandidate]:
    """Execute one strategy-selected generation action."""
    composition = parse_composition(action.formula)
    if composition is None:
        raise ValueError(f"Could not parse chemical composition: {action.formula!r}")
    os.makedirs(output_dir, exist_ok=True)
    action_manifest = Path(output_dir) / "generation-action.json"
    action_manifest.write_text(action.model_dump_json(indent=2) + "\n", encoding="utf-8")
    candidates = generate_seeds(
        composition,
        output_dir,
        n_random=action.pyxtal_count,
        fmt=fmt,
        random_seed=action.random_seed,
        formula_units=action.formula_units,
        include_aflow=action.max_aflow_candidates > 0,
        max_aflow_candidates=action.max_aflow_candidates,
        prototype_ids=action.prototype_ids or None,
        space_groups=action.space_groups or None,
    )
    manifest = {
        "schema_version": 1,
        "action": action.model_dump(mode="json"),
        "candidate_count": len(candidates),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    (Path(output_dir) / "generation-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return candidates
