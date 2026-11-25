"""Generators of collinear magnetic initializations keyed by stable ``atom_id``.

Sublattices are never inferred from ASE atom order. For BCC they are derived from geometry
(cube corners versus body centres) or supplied by the user as ``atom_id -> label``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from ase import Atoms

from matsim_agents.discovery.magnetism.config import MagneticConfiguration
from matsim_agents.seeding import numpy_rng

Sublattices = Mapping[int, str]


def _atom_ids(atoms: Atoms) -> list[int]:
    if "atom_id" not in atoms.arrays:
        raise ValueError("structure has no atom_id array; stable identities are required")
    ids = [int(i) for i in atoms.get_array("atom_id")]
    if len(set(ids)) != len(ids):
        raise ValueError("atom_id values must be unique")
    return ids


def _label_groups(
    species: Sequence[str], subs: Sequence[str], moments: Sequence[float]
) -> list[str]:
    """Group label per atom; atoms in one group share species and initial moment."""
    by_base: dict[str, list[float]] = {}
    for sym, sub, moment in zip(species, subs, moments, strict=True):
        values = by_base.setdefault(f"{sym}_{sub}", [])
        if moment not in values:
            values.append(moment)
    ranks = {base: sorted(values) for base, values in by_base.items()}
    labels = []
    for sym, sub, moment in zip(species, subs, moments, strict=True):
        base = f"{sym}_{sub}"
        labels.append(base if len(ranks[base]) == 1 else f"{base}_{ranks[base].index(moment)}")
    return labels


def _build(
    atoms: Atoms,
    structure_id: str,
    mode: str,
    moments_by_id: Mapping[int, float],
    subs_by_id: Mapping[int, str],
    *,
    method: str,
    seed: int | None = None,
    assumptions: Sequence[str] = (),
    config_id: str | None = None,
) -> MagneticConfiguration:
    ids = _atom_ids(atoms)
    symbols = atoms.get_chemical_symbols()
    moments = [float(moments_by_id.get(i, 0.0)) for i in ids]
    subs = [subs_by_id.get(i, "all") for i in ids]
    cfg = MagneticConfiguration(
        structure_id=structure_id,
        config_id=config_id or f"{structure_id}-{mode}",
        mode=mode,
        atom_ids=tuple(ids),
        species=tuple(symbols),
        group_labels=tuple(_label_groups(symbols, subs, moments)),
        spin_polarized=mode != "nonmagnetic",
        initial_moments_muB=tuple(moments),
        generation_method=method,
        seed=seed,
        assumptions=tuple(assumptions),
    )
    if config_id is None:
        short = cfg.fingerprint()[:10]
        cfg = cfg.model_copy(update={"config_id": f"{structure_id}-{mode}-{short}"})
    return cfg


def nonmagnetic(atoms: Atoms, structure_id: str) -> MagneticConfiguration:
    return _build(
        atoms,
        structure_id,
        "nonmagnetic",
        {},
        {},
        method="nonmagnetic",
        assumptions=("spin-unpolarized reference",),
    )


def ferromagnetic(
    atoms: Atoms,
    structure_id: str,
    magnitudes: Mapping[str, float],
    *,
    other_moments: Mapping[str, float] | None = None,
) -> MagneticConfiguration:
    """Parallel moments on the selected species; others get zero or ``other_moments``."""
    if not magnitudes or any(v <= 0 for v in magnitudes.values()):
        raise ValueError("ferromagnetic magnitudes must be positive and non-empty")
    symbols = atoms.get_chemical_symbols()
    extra = dict(other_moments or {})
    moments = {
        i: float(magnitudes[s]) if s in magnitudes else float(extra.get(s, 0.0))
        for i, s in zip(_atom_ids(atoms), symbols, strict=True)
    }
    return _build(
        atoms, structure_id, "ferromagnetic", moments, {}, method="ferromagnetic_parallel"
    )


def bcc_two_sublattices(atoms: Atoms, lattice_parameter: float) -> dict[int, str]:
    """Corner ('A') versus body-centre ('B') sublattice of a cubic BCC supercell by position."""
    cell = atoms.cell.array
    lengths = np.linalg.norm(cell, axis=1)
    multiples = lengths / lattice_parameter
    unit = cell / lengths[:, None]
    if not np.allclose(unit @ unit.T, np.eye(3), atol=1e-6) or not np.allclose(
        multiples, np.rint(multiples), atol=1e-6
    ):
        raise ValueError("cell is not an orthogonal integer supercell of the cubic BCC cell")
    half = np.rint(2.0 * (atoms.positions @ unit.T) / lattice_parameter).astype(int)
    labels: dict[int, str] = {}
    for atom_id, row in zip(_atom_ids(atoms), half, strict=True):
        parity = set((row % 2).tolist())
        if len(parity) != 1:
            raise ValueError(f"atom_id {atom_id} is not on a BCC site of a={lattice_parameter}")
        labels[atom_id] = "B" if parity == {1} else "A"
    return labels


def _check_sublattices(atoms: Atoms, sublattices: Sublattices) -> None:
    missing = set(_atom_ids(atoms)) - set(sublattices)
    if missing:
        raise ValueError(f"sublattice not given for atom_ids {sorted(missing)[:5]}")


def _signed(
    atoms: Atoms,
    sublattices: Sublattices,
    signs: Mapping[str, int],
    magnitudes: Mapping[str, float],
    overrides: Mapping[str, Mapping[str, float]] | None,
) -> dict[int, float]:
    if set(signs.values()) - {-1, 1}:
        raise ValueError("sublattice signs must be +1 or -1")
    unknown = {sublattices[i] for i in _atom_ids(atoms)} - set(signs)
    if unknown:
        raise ValueError(f"no sign for sublattices {sorted(unknown)}")
    moments = {}
    for atom_id, sym in zip(_atom_ids(atoms), atoms.get_chemical_symbols(), strict=True):
        sub = sublattices[atom_id]
        by_sub = (overrides or {}).get(sub, {})
        magnitude = by_sub.get(sym, magnitudes.get(sym, 0.0))
        if magnitude < 0:
            raise ValueError("magnitudes must be non-negative")
        moments[atom_id] = signs[sub] * float(magnitude)
    return moments


def antiferromagnetic(
    atoms: Atoms,
    structure_id: str,
    sublattices: Sublattices,
    magnitudes: Mapping[str, float],
    *,
    signs: Mapping[str, int] | None = None,
) -> MagneticConfiguration:
    """User-supplied sublattices; the first sublattice (sorted) is +1 and the rest -1."""
    _check_sublattices(atoms, sublattices)
    labels = sorted(set(sublattices.values()))
    if len(labels) != 2 and signs is None:
        raise ValueError("pass explicit signs when there are not exactly two sublattices")
    chosen = dict(signs) if signs else {labels[0]: 1, labels[1]: -1}
    moments = _signed(atoms, sublattices, chosen, magnitudes, None)
    return _build(
        atoms,
        structure_id,
        "antiferromagnetic",
        moments,
        sublattices,
        method="antiferromagnetic_sublattices",
        assumptions=("net moment is not constrained to zero",),
    )


def bcc_afm(
    atoms: Atoms, structure_id: str, lattice_parameter: float, magnitudes: Mapping[str, float]
) -> MagneticConfiguration:
    """Two-sublattice (corner/body-centre) AFM of a cubic BCC supercell."""
    return antiferromagnetic(
        atoms, structure_id, bcc_two_sublattices(atoms, lattice_parameter), magnitudes
    )


def ferrimagnetic(
    atoms: Atoms,
    structure_id: str,
    sublattices: Sublattices,
    magnitudes_by_sublattice: Mapping[str, Mapping[str, float]],
    *,
    signs: Mapping[str, int],
) -> MagneticConfiguration:
    """Antiparallel sublattices with different magnitudes, per sublattice and species."""
    _check_sublattices(atoms, sublattices)
    moments = _signed(atoms, sublattices, signs, {}, magnitudes_by_sublattice)
    return _build(
        atoms,
        structure_id,
        "ferrimagnetic",
        moments,
        sublattices,
        method="ferrimagnetic_sublattices",
    )


def custom_collinear(
    atoms: Atoms,
    structure_id: str,
    moments_by_atom_id: Mapping[int, float],
    *,
    sublattices: Sublattices | None = None,
    config_id: str | None = None,
) -> MagneticConfiguration:
    """Explicit per-atom moments (e.g. environment-dependent moments of one element)."""
    ids = set(_atom_ids(atoms))
    if set(moments_by_atom_id) - ids:
        raise ValueError("moments given for atom_ids not present in the structure")
    return _build(
        atoms,
        structure_id,
        "custom_collinear",
        moments_by_atom_id,
        sublattices or {},
        method="custom_per_atom",
        config_id=config_id,
    )


def seeded_collinear_alternatives(
    atoms: Atoms,
    structure_id: str,
    magnitudes: Mapping[str, float],
    *,
    n: int,
    seed: int,
    up_fraction: float = 0.5,
) -> list[MagneticConfiguration]:
    """Distinct random up/down assignments on magnetic species; no full enumeration."""
    if n < 1 or not 0.0 < up_fraction < 1.0:
        raise ValueError("n must be >= 1 and up_fraction in (0, 1)")
    ids = _atom_ids(atoms)
    symbols = atoms.get_chemical_symbols()
    magnetic = [i for i, s in zip(ids, symbols, strict=True) if magnitudes.get(s, 0.0) > 0]
    if len(magnetic) < 2:
        raise ValueError("at least two magnetic atoms are needed for antiparallel patterns")
    n_up = min(max(1, round(up_fraction * len(magnetic))), len(magnetic) - 1)
    rng = numpy_rng(seed)
    sym_of = dict(zip(ids, symbols, strict=True))
    seen: set[str] = set()
    out: list[MagneticConfiguration] = []
    for _ in range(20 * n):
        if len(out) == n:
            break
        up = set(rng.choice(magnetic, size=n_up, replace=False).tolist())
        moments = {i: (1.0 if i in up else -1.0) * float(magnitudes[sym_of[i]]) for i in magnetic}
        cfg = custom_collinear(atoms, structure_id, moments)
        cfg = cfg.model_copy(update={"seed": seed, "generation_method": "seeded_random_collinear"})
        key = cfg.fingerprint()
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out
