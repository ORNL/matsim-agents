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

import logging
import os
import warnings
from collections import Counter
from functools import lru_cache
from itertools import permutations
from math import gcd
from typing import Literal

from pydantic import BaseModel, Field

from matsim_agents.discovery.composition import Composition

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
    structure_path: str
    source: Literal["prototype", "random"] = "prototype"
    prototype_id: str | None = Field(
        default=None,
        description="AFLOW prototype label, e.g. 'AB_cF8_216_c_a' (zincblende).",
    )
    space_group: int | None = None
    num_atoms: int | None = None
    needs_dft_verification: bool = Field(
        default=False,
        description="True for random-source seeds: known-prototype hits are "
        "structurally familiar but pyXtal-generated novel topologies should be "
        "DFT-validated before any stability claim.",
    )
    notes: str | None = None


# ---------------------------------------------------------------------------
# AFLOW prototype loading
# ---------------------------------------------------------------------------


class _LoadedPrototype:
    """Internal: a prototype structure plus its AFLOW metadata."""

    __slots__ = ("aflow", "pearson", "strukturbericht", "mineral", "structure", "signature")

    def __init__(self, aflow, pearson, strukturbericht, mineral, structure, signature):
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


def decorate(proto: _LoadedPrototype, comp: Composition):
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
    max_attempts_factor: int = 4,
) -> list[tuple[object, int]]:
    """Generate up to ``n`` random crystal structures across the 230 space groups.

    Returns a list of ``(pymatgen Structure, space_group)`` tuples.
    Returns ``[]`` (with a warning) if pyXtal is not installed.
    """
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

    from pyxtal import pyxtal

    rng = _random.Random(seed)
    elements = list(comp.elements.keys())
    counts = [comp.elements[e] for e in elements]

    results: list[tuple[object, int]] = []
    max_attempts = max(n * max_attempts_factor, n + 20)
    attempts = 0
    while len(results) < n and attempts < max_attempts:
        attempts += 1
        spg = rng.randint(1, 230)
        try:
            x = pyxtal()
            x.from_random(dim=3, group=spg, species=elements, numIons=counts)
            if not x.valid:
                continue
            pmg = x.to_pymatgen()
            results.append((pmg, spg))
        except Exception as exc:
            logger.debug("pyXtal SG=%d failed: %s", spg, exc)
            continue
    if len(results) < n:
        logger.info(
            "pyXtal produced %d/%d structures after %d attempts.",
            len(results),
            n,
            attempts,
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
    """
    from ase.io import write
    from pymatgen.io.ase import AseAtomsAdaptor

    os.makedirs(output_dir, exist_ok=True)
    ext = ".vasp" if fmt == "vasp" else f".{fmt}"
    candidates: list[PhaseCandidate] = []

    # --- 1. Prototype decoration -------------------------------------------
    matches = compatible_prototypes(composition)
    logger.info(
        "Composition %s: %d AFLOW prototypes match stoichiometry signature %s.",
        composition.formula,
        len(matches),
        _composition_signature(composition),
    )

    for proto in matches:
        decorations = decorate(proto, composition)
        for o_idx, (struct, mapping_str) in enumerate(decorations):
            atoms = AseAtomsAdaptor.get_atoms(struct)
            spg = struct.get_space_group_info()[1]
            o_tag = f"_o{o_idx}" if len(decorations) > 1 else ""
            # AFLOW labels contain '_' already; replace with '-' in filename
            # to keep tags readable.
            safe_label = proto.aflow.replace("/", "-") or f"proto{matches.index(proto)}"
            path = os.path.join(
                output_dir,
                f"{composition.formula}__{safe_label}{o_tag}{ext}",
            )
            write(path, atoms, format=fmt)

            notes_bits = [f"AFLOW prototype {proto.aflow} (SG {spg})."]
            if proto.mineral:
                notes_bits.append(f"Mineral: {proto.mineral}.")
            if proto.strukturbericht:
                notes_bits.append(f"Strukturbericht: {proto.strukturbericht}.")
            notes_bits.append(f"Decoration: {mapping_str}.")

            candidates.append(
                PhaseCandidate(
                    formula=composition.formula,
                    structure_path=path,
                    source="prototype",
                    prototype_id=proto.aflow,
                    space_group=spg,
                    num_atoms=len(atoms),
                    needs_dft_verification=False,
                    notes=" ".join(notes_bits),
                )
            )

    # --- 2. pyXtal random search -------------------------------------------
    random_hits = random_search(composition, n=n_random, seed=random_seed)
    for r_idx, (struct, spg) in enumerate(random_hits):
        atoms = AseAtomsAdaptor.get_atoms(struct)
        path = os.path.join(
            output_dir,
            f"{composition.formula}__pyxtal_sg{spg:03d}_r{r_idx:03d}{ext}",
        )
        write(path, atoms, format=fmt)
        candidates.append(
            PhaseCandidate(
                formula=composition.formula,
                structure_path=path,
                source="random",
                prototype_id=None,
                space_group=int(spg),
                num_atoms=len(atoms),
                needs_dft_verification=True,
                notes=(
                    f"Random pyXtal structure (space group {spg}). "
                    "Novel topology — requires DFT verification before any stability claim."
                ),
            )
        )

    logger.info(
        "Composition %s: %d total seeds (%d prototype + %d random).",
        composition.formula,
        len(candidates),
        len(candidates) - len(random_hits),
        len(random_hits),
    )
    return candidates
