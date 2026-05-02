"""Enumerate candidate crystal phases for a chemical composition.

The goal is breadth, not exhaustiveness: for each composition we generate a
handful of physically meaningful prototypes that downstream agents can
relax and compare. Heavy structure prediction (USPEX, AIRSS, CALYPSO, ...)
is out of scope; this module produces seed structures.

Three families of prototypes are supported:

* **3-D bulk** (default for most compositions): fcc, bcc, hcp, sc, diamond
  for elements; rocksalt, CsCl, zincblende, wurtzite, fluorite, rutile for
  binaries; cubic perovskite, rocksalt-ordered double perovskite, normal
  spinel for ternaries / quaternaries.
* **2-D monolayer**: graphene-like (1 element), hexagonal h-BN-like (2
  elements, 1:1), and MX2 (2 elements, 1:2 — covers the MoS2 family).
  All 2-D prototypes are generated as periodic slabs with a configurable
  vacuum gap along z.
* **2-D multilayer**: any 2-D prototype can be stacked ``num_layers``
  times before the vacuum is applied.

In addition, every prototype (2-D or 3-D) can be tiled in-plane via
``supercell`` or auto-tiled to reach ``min_atoms``. For multi-species
prototypes, the tiled supercell can be re-decorated to sample
symmetrically-distinct site orderings (``n_orderings``), and each
ordering can be expanded into an isotropic lattice-constant sweep
(``lattice_scales``) to bracket the equilibrium volume before
relaxation.

Backends, in order of preference:
    1. ASE ``bulk`` / ``mx2`` / ``graphene`` builders.
    2. pymatgen for prototypes ASE does not cover well.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from matsim_agents.discovery.composition import Composition

PhaseTag = Literal[
    # Elemental 3-D
    "fcc",
    "bcc",
    "hcp",
    "sc",
    "diamond",
    # Binary 3-D
    "rocksalt",
    "cscl",
    "zincblende",
    "wurtzite",
    "fluorite",
    "rutile",
    # Ternary / quaternary 3-D
    "perovskite",
    "double_perovskite",
    "spinel",
    # 2-D
    "graphene",
    "hbn",
    "mx2_2h",
    "mx2_1t",
]

# Sensible default lattice constants (Å) per prototype family.
_DEFAULT_A: dict[str, float] = {
    "fcc": 4.0,
    "bcc": 3.2,
    "hcp": 3.0,
    "sc": 3.0,
    "diamond": 5.4,
    "rocksalt": 5.0,
    "cscl": 4.0,
    "zincblende": 5.5,
    "wurtzite": 3.5,
    "fluorite": 5.4,
    "rutile": 4.6,
    "perovskite": 4.0,
    "double_perovskite": 8.0,
    "spinel": 8.0,
    "graphene": 2.46,
    "hbn": 2.50,
    "mx2_2h": 3.18,
    "mx2_1t": 3.32,
}

# Default interlayer separation for 2-D prototypes (Å).
_DEFAULT_INTERLAYER: dict[str, float] = {
    "graphene": 3.35,
    "hbn": 3.33,
    "mx2_2h": 6.15,  # half c-lattice of 2H-MoS2
    "mx2_1t": 5.95,
}

_2D_PHASES: set[str] = {"graphene", "hbn", "mx2_2h", "mx2_1t"}


class PhaseCandidate(BaseModel):
    """A single seed structure generated for a composition."""

    formula: str
    phase: str = Field(..., description="Prototype tag (e.g. 'rocksalt', 'graphene').")
    structure_path: str
    space_group_hint: str | None = None
    notes: str | None = None
    num_atoms: int | None = None
    supercell: tuple[int, int, int] | None = None
    dimensionality: Literal["2D", "3D"] = "3D"
    num_layers: int | None = None
    ordering_index: int = 0
    lattice_scale: float = 1.0


# ---------------------------------------------------------------------------
# 3-D builders
# ---------------------------------------------------------------------------


def _ase_bulk_phase(composition: Composition, phase: PhaseTag):
    """Try to build a structure with ASE ``bulk`` for the given prototype."""
    from ase.build import bulk

    a = _DEFAULT_A[phase]
    elems = list(composition.elements.keys())

    if phase in ("fcc", "bcc", "hcp", "sc", "diamond"):
        if composition.num_elements != 1:
            return None
        return bulk(elems[0], crystalstructure=phase, a=a)

    if phase in ("rocksalt", "cscl", "zincblende", "wurtzite", "fluorite"):
        if composition.num_elements != 2:
            return None
        formula = elems[0] + elems[1]
        try:
            if phase == "wurtzite":
                return bulk(formula, crystalstructure="wurtzite", a=a, c=a * 1.6)
            return bulk(formula, crystalstructure=phase, a=a)
        except Exception:
            return None

    return None


def _pymatgen_phase(composition: Composition, phase: PhaseTag):
    """Optional pymatgen path for prototypes ASE does not cover well."""
    try:
        from pymatgen.core import Lattice, Structure
    except ImportError:
        return None

    elems = list(composition.elements.keys())
    counts = composition.elements

    if phase == "perovskite":
        if composition.num_elements < 2:
            return None
        a = _DEFAULT_A[phase]
        lattice = Lattice.cubic(a)
        species = list(elems)
        while len(species) < 3:
            species.append(species[-1])
        coords = [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
        all_species = [species[0], species[1]] + [species[2]] * 3
        return Structure(lattice, all_species, coords)

    if phase == "double_perovskite":
        # A2 B B' X6 in rocksalt-ordered Fm-3m (2x2x2 of cubic ABX3).
        if composition.num_elements != 4:
            return None
        try:
            A = next(e for e, n in counts.items() if n == 2)
            X = next(e for e, n in counts.items() if n == 6)
            BB = [e for e, n in counts.items() if n == 1]
        except StopIteration:
            return None
        if len(BB) != 2:
            return None
        a = _DEFAULT_A["double_perovskite"]
        lattice = Lattice.cubic(a)

        species: list[str] = []
        coords: list[tuple[float, float, float]] = []
        # A on (1/4 + i/2, 1/4 + j/2, 1/4 + k/2)
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    species.append(A)
                    coords.append((0.25 + 0.5 * i, 0.25 + 0.5 * j, 0.25 + 0.5 * k))
        # B / B' rocksalt-alternating on (i/2, j/2, k/2)
        B, Bp = BB
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    species.append(B if (i + j + k) % 2 == 0 else Bp)
                    coords.append((0.5 * i, 0.5 * j, 0.5 * k))
        # X at midpoints of every B-B' bond along x, y, z
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    coords.append((0.25 + 0.5 * i, 0.5 * j, 0.5 * k))
                    species.append(X)
                    coords.append((0.5 * i, 0.25 + 0.5 * j, 0.5 * k))
                    species.append(X)
                    coords.append((0.5 * i, 0.5 * j, 0.25 + 0.5 * k))
                    species.append(X)
        return Structure(lattice, species, coords)

    if phase == "rutile" and composition.num_elements == 2:
        a, c = _DEFAULT_A[phase], _DEFAULT_A[phase] * 0.65
        lattice = Lattice.tetragonal(a, c)
        u = 0.305
        coords = [
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (u, u, 0),
            (-u, -u, 0),
            (0.5 + u, 0.5 - u, 0.5),
            (0.5 - u, 0.5 + u, 0.5),
        ]
        species = [elems[0], elems[0], elems[1], elems[1], elems[1], elems[1]]
        return Structure(lattice, species, coords)

    if phase == "spinel" and composition.num_elements == 3:
        # Normal spinel AB2X4 (Fd-3m), 14-atom primitive seed.
        try:
            A = next(e for e, n in counts.items() if n == 1)
            B = next(e for e, n in counts.items() if n == 2)
            X = next(e for e, n in counts.items() if n == 4)
        except StopIteration:
            return None
        a = _DEFAULT_A[phase]
        lattice = Lattice.cubic(a)
        u = 0.387
        species = [A] * 2 + [B] * 4 + [X] * 8
        coords = [
            (0, 0, 0),
            (0.25, 0.25, 0.25),
            (0.5, 0.5, 0.5),
            (0.5, 0, 0),
            (0, 0.5, 0),
            (0, 0, 0.5),
            (u, u, u),
            (u, -u, -u),
            (-u, u, -u),
            (-u, -u, u),
            (0.25 + u, 0.25 + u, 0.25 + u),
            (0.25 + u, 0.25 - u, 0.25 - u),
            (0.25 - u, 0.25 + u, 0.25 - u),
            (0.25 - u, 0.25 - u, 0.25 + u),
        ]
        return Structure(lattice, species, coords)

    return None


# ---------------------------------------------------------------------------
# 2-D builders (monolayer; multilayer assembled in _stack_2d)
# ---------------------------------------------------------------------------


def _ase_2d_monolayer(composition: Composition, phase: PhaseTag):
    """Build a 2-D monolayer in a periodic cell with vacuum=0 (added later)."""
    elems = list(composition.elements.keys())
    counts = composition.elements

    if phase == "graphene":
        # Honeycomb single-element monolayer (graphene topology).
        if composition.num_elements != 1:
            return None
        try:
            from ase.build import graphene  # type: ignore

            return graphene(elems[0], a=_DEFAULT_A[phase], vacuum=0.0)
        except Exception:
            # Fallback: build manually using ase.Atoms.
            return _manual_honeycomb(elems[0], elems[0], _DEFAULT_A[phase])

    if phase == "hbn":
        # Hexagonal AB monolayer (h-BN topology). Requires 1:1 binary.
        if composition.num_elements != 2 or sorted(counts.values()) != [1, 1]:
            return None
        return _manual_honeycomb(elems[0], elems[1], _DEFAULT_A[phase])

    if phase in ("mx2_2h", "mx2_1t"):
        # MX2 trigonal prismatic (2H) or octahedral (1T) monolayer.
        # Requires 1:2 binary.
        if composition.num_elements != 2 or sorted(counts.values()) != [1, 2]:
            return None
        try:
            M = next(e for e, n in counts.items() if n == 1)
            X = next(e for e, n in counts.items() if n == 2)
        except StopIteration:
            return None
        try:
            from ase.build import mx2  # type: ignore

            kind = "2H" if phase == "mx2_2h" else "1T"
            # ASE's mx2 expects the formula as a string of M followed by X.
            atoms = mx2(
                formula=f"{M}{X}2", kind=kind, a=_DEFAULT_A[phase], thickness=3.19, vacuum=0.0
            )
            return atoms
        except Exception:
            return None

    return None


def _manual_honeycomb(elem_a: str, elem_b: str, a: float):
    """Build a 2-atom honeycomb monolayer in a hexagonal cell, vacuum=0."""
    import numpy as np
    from ase import Atoms

    # Hexagonal lattice vectors with c chosen as a placeholder; vacuum
    # is added by the caller.
    cell = np.array(
        [
            [a, 0.0, 0.0],
            [-a / 2.0, a * math.sqrt(3) / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # Two-atom basis at fractional (1/3, 2/3) and (2/3, 1/3) on z=0.
    scaled = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.0]])
    positions = scaled @ cell
    return Atoms(symbols=[elem_a, elem_b], positions=positions, cell=cell, pbc=True)


def _stack_2d(monolayer, num_layers: int, interlayer: float, vacuum: float):
    """Stack ``num_layers`` copies of a 2-D monolayer along z, then add vacuum."""
    import numpy as np
    from ase import Atoms

    if monolayer is None:
        return None
    if num_layers < 1:
        num_layers = 1

    base = monolayer.copy()
    cell = base.get_cell().array.copy()
    # Shift each copy by interlayer along z; final cell c = (n-1)*d + vacuum.
    z_offsets = [i * interlayer for i in range(num_layers)]
    symbols: list[str] = []
    positions = []
    for dz in z_offsets:
        for sym, pos in zip(base.get_chemical_symbols(), base.get_positions(), strict=False):
            symbols.append(sym)
            positions.append([pos[0], pos[1], pos[2] + dz])

    c_total = max(z_offsets) + vacuum
    cell[2, :] = [0.0, 0.0, c_total if c_total > 0 else vacuum]
    stacked = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, True])
    # Center the slab in z so vacuum is split symmetrically.
    z_pos = stacked.get_positions()[:, 2]
    shift = (c_total - (z_pos.max() - z_pos.min())) / 2.0 - z_pos.min()
    stacked.translate([0.0, 0.0, shift])
    return stacked


# ---------------------------------------------------------------------------
# Phase selection
# ---------------------------------------------------------------------------


def _candidate_phases(composition: Composition, *, include_2d: bool) -> list[PhaseTag]:
    n = composition.num_elements
    counts = composition.elements
    phases: list[PhaseTag] = []

    if n == 1:
        phases += ["fcc", "bcc", "hcp", "sc", "diamond"]
        if include_2d:
            phases += ["graphene"]
    elif n == 2:
        phases += ["rocksalt", "cscl", "zincblende", "wurtzite", "fluorite", "rutile"]
        if include_2d:
            ratios = sorted(counts.values())
            if ratios == [1, 1]:
                phases += ["hbn"]
            if ratios == [1, 2]:
                phases += ["mx2_2h", "mx2_1t"]
    elif n == 3:
        phases += ["perovskite"]
        if sorted(counts.values()) == [1, 2, 4]:
            phases += ["spinel"]
    elif n == 4 and sorted(counts.values()) == [1, 1, 2, 6]:
        phases += ["double_perovskite"]
    else:
        phases += ["perovskite"]

    return phases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_atoms(obj):
    if obj is None:
        return None
    try:
        from ase import Atoms

        if isinstance(obj, Atoms):
            return obj
    except Exception:
        pass
    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        return AseAtomsAdaptor.get_atoms(obj)
    except Exception:
        return None


def _auto_supercell(
    n_atoms_primitive: int, min_atoms: int, *, in_plane_only: bool
) -> tuple[int, int, int]:
    """Smallest uniform tile s.t. tiled cell has >= ``min_atoms`` atoms.

    For 2-D slabs we only tile in x and y; z stays at 1.
    """
    if n_atoms_primitive <= 0 or min_atoms <= n_atoms_primitive:
        return (1, 1, 1)
    if in_plane_only:
        n = max(1, math.ceil((min_atoms / n_atoms_primitive) ** (1 / 2)))
        return (n, n, 1)
    n = max(1, math.ceil((min_atoms / n_atoms_primitive) ** (1 / 3)))
    return (n, n, n)


def _apply_supercell(atoms, supercell, min_atoms, *, is_2d: bool):
    if atoms is None:
        return None, (1, 1, 1)
    primitive_n = len(atoms)
    if supercell is not None:
        nx, ny, nz = supercell
        if is_2d:
            nz = 1  # never tile through vacuum
        if (nx, ny, nz) != (1, 1, 1):
            atoms = atoms.repeat((nx, ny, nz))
        return atoms, (nx, ny, nz)
    nx, ny, nz = _auto_supercell(primitive_n, min_atoms, in_plane_only=is_2d)
    if (nx, ny, nz) != (1, 1, 1):
        atoms = atoms.repeat((nx, ny, nz))
    return atoms, (nx, ny, nz)


# ---------------------------------------------------------------------------
# Configurational sampling: orderings & lattice-constant sweep
# ---------------------------------------------------------------------------


def _to_pmg_structure(atoms):
    """ASE Atoms -> pymatgen Structure (best-effort, may return None)."""
    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        return AseAtomsAdaptor.get_structure(atoms)
    except Exception:
        return None


def _generate_orderings(atoms, n_orderings: int, *, seed: int = 0, max_attempts_factor: int = 40):
    """Return up to ``n_orderings`` symmetrically-distinct site decorations.

    The atomic positions and cell are kept fixed; only the species labels
    are permuted across sites of the supercell. The first entry is always
    the original (prototype-decorated) cell. Duplicates are filtered with
    pymatgen's ``StructureMatcher`` when available, otherwise by a coarse
    fractional-coordinate hash of the (sorted) (site, species) tuples.
    """
    if atoms is None or n_orderings <= 1:
        return [atoms] if atoms is not None else []

    syms = list(atoms.get_chemical_symbols())
    if len(set(syms)) < 2:
        # Single-element cells have no non-trivial decorations.
        return [atoms]

    results = [atoms.copy()]

    matcher = None
    try:
        from pymatgen.core.structure_matcher import StructureMatcher

        matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False)
    except Exception:
        matcher = None

    pmg_seen = []
    if matcher is not None:
        s0 = _to_pmg_structure(results[0])
        if s0 is not None:
            pmg_seen.append(s0)
        else:
            matcher = None  # fall back to hash dedup

    hash_seen: set[tuple] = set()
    if matcher is None:
        hash_seen.add(_ordering_hash(results[0]))

    rng = random.Random(seed)
    max_attempts = max(20, n_orderings * max_attempts_factor)
    attempts = 0
    while len(results) < n_orderings and attempts < max_attempts:
        attempts += 1
        shuffled = syms.copy()
        rng.shuffle(shuffled)
        cand = atoms.copy()
        cand.set_chemical_symbols(shuffled)

        if matcher is not None:
            s_cand = _to_pmg_structure(cand)
            if s_cand is None:
                continue
            if any(matcher.fit(s_prev, s_cand) for s_prev in pmg_seen):
                continue
            pmg_seen.append(s_cand)
        else:
            h = _ordering_hash(cand)
            if h in hash_seen:
                continue
            hash_seen.add(h)

        results.append(cand)

    return results


def _ordering_hash(atoms, decimals: int = 3) -> tuple:
    """Fallback de-dup key for orderings (ignores lattice symmetry)."""
    scaled = atoms.get_scaled_positions(wrap=True)
    syms = atoms.get_chemical_symbols()
    items = sorted(
        (
            round(float(p[0]) % 1.0, decimals),
            round(float(p[1]) % 1.0, decimals),
            round(float(p[2]) % 1.0, decimals),
            s,
        )
        for p, s in zip(scaled, syms, strict=False)
    )
    return tuple(items)


def _scale_atoms(atoms, scale: float):
    """Return a copy with cell vectors (and atomic positions) scaled by ``scale``."""
    if atoms is None or scale == 1.0:
        return atoms.copy() if atoms is not None else None
    scaled = atoms.copy()
    scaled.set_cell(scaled.get_cell() * scale, scale_atoms=True)
    return scaled


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def enumerate_phases(
    composition: Composition,
    output_dir: str,
    fmt: str = "vasp",
    *,
    supercell: tuple[int, int, int] | None = None,
    min_atoms: int = 32,
    include_2d: bool = False,
    num_layers: int = 1,
    vacuum: float = 15.0,
    interlayer: float | None = None,
    n_orderings: int = 1,
    lattice_scales: Sequence[float] | None = None,
    ordering_seed: int = 0,
) -> list[PhaseCandidate]:
    """Generate seed structures for every applicable prototype.

    Parameters
    ----------
    composition:
        Target composition.
    output_dir:
        Directory where structure files are written. Created if needed.
    fmt:
        ASE format identifier (``"vasp"``, ``"cif"``, ``"xyz"``, ...).
    supercell:
        Explicit ``(nx, ny, nz)`` tiling applied to every prototype after
        construction. For 2-D slabs ``nz`` is forced to 1. Overrides
        ``min_atoms``.
    min_atoms:
        When ``supercell`` is ``None``, each prototype is auto-tiled
        uniformly until it contains at least this many atoms (in-plane
        only for 2-D slabs). Default ``32``. Set to ``1`` to keep
        primitive cells.
    include_2d:
        If ``True``, also enumerate applicable 2-D prototypes (graphene,
        h-BN, MoS2-type 1T/2H). Default ``False`` so that 3-D bulk
        discovery is unchanged.
    num_layers:
        Number of monolayers to stack for every 2-D prototype before
        the vacuum gap is applied. ``1`` is a true monolayer.
    vacuum:
        Vacuum gap (Å) added along z for every 2-D prototype. Ignored
        for 3-D phases.
    interlayer:
        Override the per-prototype default interlayer separation (Å)
        used when ``num_layers > 1``.
    n_orderings:
        For multi-species prototypes, number of symmetrically-distinct
        site decorations to sample within the (already tiled) supercell.
        ``1`` (default) keeps only the canonical prototype decoration.
        Larger values explore site-disorder / antisite / inversion-like
        configurations that only become non-trivial once the supercell
        is bigger than the primitive cell. Single-element prototypes
        always yield exactly one ordering.
    lattice_scales:
        Iterable of isotropic cell-scaling factors applied per ordering
        (e.g. ``(0.96, 1.0, 1.04)`` to bracket the equilibrium lattice
        constant). Atoms are scaled coherently with the cell.
        ``None`` (default) means a single ``1.0`` scale.
    ordering_seed:
        Seed for the ordering-shuffle RNG (reproducibility).
    """
    from ase.io import write

    os.makedirs(output_dir, exist_ok=True)
    ext = ".vasp" if fmt == "vasp" else f".{fmt}"
    candidates: list[PhaseCandidate] = []

    scales: tuple[float, ...] = tuple(lattice_scales) if lattice_scales else (1.0,)
    n_orderings = max(1, int(n_orderings))

    for phase in _candidate_phases(composition, include_2d=include_2d):
        is_2d = phase in _2D_PHASES

        if is_2d:
            mono = _ase_2d_monolayer(composition, phase)
            if mono is None:
                continue
            d = interlayer if interlayer is not None else _DEFAULT_INTERLAYER[phase]
            atoms = _stack_2d(mono, num_layers=num_layers, interlayer=d, vacuum=vacuum)
            backend = "ASE-2D"
        else:
            ase_obj = _ase_bulk_phase(composition, phase)
            atoms = _to_atoms(ase_obj) or _to_atoms(_pymatgen_phase(composition, phase))
            if atoms is None:
                continue
            backend = "ASE" if ase_obj is not None else "pymatgen"

        atoms, used_sc = _apply_supercell(atoms, supercell, min_atoms, is_2d=is_2d)

        orderings = _generate_orderings(atoms, n_orderings, seed=ordering_seed)

        sc_tag = "" if used_sc == (1, 1, 1) else f"_sc{used_sc[0]}x{used_sc[1]}x{used_sc[2]}"
        layer_tag = f"_L{num_layers}" if (is_2d and num_layers > 1) else ""

        for o_idx, base in enumerate(orderings):
            o_tag = f"_o{o_idx}" if len(orderings) > 1 else ""
            for scale in scales:
                tiled = _scale_atoms(base, scale)
                s_tag = "" if scale == 1.0 else f"_s{int(round(scale * 1000)):04d}"
                path = os.path.join(
                    output_dir,
                    f"{composition.formula}_{phase}{layer_tag}{sc_tag}{o_tag}{s_tag}{ext}",
                )
                write(path, tiled, format=fmt)

                notes = f"Seed structure built via {backend}."
                if used_sc != (1, 1, 1):
                    notes += (
                        f" Tiled to {used_sc[0]}x{used_sc[1]}x{used_sc[2]} ({len(tiled)} atoms)."
                    )
                if is_2d:
                    notes += f" 2-D slab: {num_layers} layer(s), vacuum={vacuum:g} Å."
                if len(orderings) > 1:
                    notes += f" Ordering {o_idx + 1}/{len(orderings)}."
                if scale != 1.0:
                    notes += f" Lattice scale {scale:.3f}."

                candidates.append(
                    PhaseCandidate(
                        formula=composition.formula,
                        phase=phase,
                        structure_path=path,
                        notes=notes,
                        num_atoms=len(tiled),
                        supercell=used_sc,
                        dimensionality="2D" if is_2d else "3D",
                        num_layers=num_layers if is_2d else None,
                        ordering_index=o_idx,
                        lattice_scale=float(scale),
                    )
                )

    return candidates
