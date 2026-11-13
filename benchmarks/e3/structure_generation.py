"""Material-specific, structure-only generators for the E3-MLIP campaign."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _ROOT / "examples" / "paper_cases" / "seeds"
_TIERS: dict[str, set[int]] = {
    "bcc_refractory": {4, 8, 16, 32, 64},
    "fcc_cantor": {4, 8, 16, 32, 64},
    "lifepo4": {4, 8, 16, 32},
    "phosphorene": {4, 16, 36, 64, 144},
    "nb_ta_o": {2, 4, 8, 16, 32},
    "zn_formate_mof": {24, 48, 96, 192},
}
_OXIDE_FORMULAS: dict[str, dict[str, int]] = {
    "Nb2O5": {"Nb": 2, "O": 5},
    "Ta2O5": {"Ta": 2, "O": 5},
    "NbTaO5": {"Nb": 1, "Ta": 1, "O": 5},
    "NbO2": {"Nb": 1, "O": 2},
    "TaO2": {"Ta": 1, "O": 2},
    "NbTaO4": {"Nb": 1, "Ta": 1, "O": 4},
}
_ALLOY_ELEMENTS = {
    "bcc_refractory": {"Nb", "Ta", "V", "Hf", "Zr", "Ti"},
    "fcc_cantor": {"Cr", "Mn", "Fe", "Co", "Ni"},
}
_FORMULA_PARSER = re.compile(r"([A-Z][a-z]?|\(|\)|\d+)")
_FAMILIES = {
    "bcc_refractory": {
        "bcc_occupation",
        "equiatomic_ordering",
        "controlled_substitution",
        "integer_ratio_ordering",
        "symmetry_distinct_occupation",
    },
    "fcc_cantor": {
        "fcc_occupation",
        "equiatomic_ordering",
        "controlled_substitution",
        "integer_ratio_ordering",
        "symmetry_distinct_occupation",
    },
    "lifepo4": {
        "olivine",
        "parent_olivine",
        "antisite_pair",
        "olivine_antisite_pair",
        "olivine_strain",
        "strain",
        "strained_olivine",
    },
    "phosphorene": {
        "black",
        "black_phosphorene",
        "black_allotrope",
        "blue",
        "blue_phosphorene",
        "blue_allotrope",
        "reconstruction",
        "vacancy_reconstruction",
        "in_plane_strain",
        "strain",
    },
    "nb_ta_o": {
        "oxide_prototype",
        "aflow_prototype",
        "pyxtal",
        "prototype",
        "oxygen_defect",
        "prototype_strain",
    },
    "zn_formate_mof": {
        "cod_parent",
        "bounded_perturbation",
        "framework_perturbation",
        "strain",
    },
}


class StructureGenerationError(ValueError):
    """Raised when an E3 structure action is invalid or cannot fill its batch."""


class GenerationExhausted(StructureGenerationError):
    """A valid seeded batch fell short; partial structures and diagnostics are retained."""

    def __init__(self, message, records, diagnostics):
        super().__init__(message)
        self.records = records
        self.diagnostics = diagnostics


def _ase():
    try:
        import ase
        import numpy as np
        from ase import Atoms
        from ase.build import bulk
        from ase.io import read, write
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise StructureGenerationError(
            "E3 structure generation requires the repository ASE dependency."
        ) from exc
    return ase, np, Atoms, bulk, read, write


def _parse_formula(formula: str) -> dict[str, int]:
    tokens = _FORMULA_PARSER.findall(formula)
    if not tokens or "".join(tokens) != formula:
        raise StructureGenerationError(f"Invalid chemical formula: {formula!r}")
    index = 0

    def formula_count() -> int:
        nonlocal index
        if index < len(tokens) and tokens[index].isdigit():
            count = int(tokens[index])
            index += 1
            if count <= 0:
                raise StructureGenerationError("Formula stoichiometries must be positive.")
            return count
        return 1

    def parse_group(nested: bool = False) -> dict[str, int]:
        nonlocal index
        result: Counter[str] = Counter()
        while index < len(tokens):
            token = tokens[index]
            if token == ")":
                if not nested:
                    raise StructureGenerationError(f"Unmatched ')' in formula {formula!r}")
                index += 1
                return dict(result)
            if token == "(":
                index += 1
                group = parse_group(nested=True)
                multiplier = formula_count()
                for symbol, count in group.items():
                    result[symbol] += count * multiplier
            elif token[0].isupper():
                index += 1
                result[token] += formula_count()
            else:
                raise StructureGenerationError(f"Unexpected token {token!r} in formula")
        if nested:
            raise StructureGenerationError(f"Unclosed '(' in formula {formula!r}")
        return dict(result)

    counts = parse_group()
    if index != len(tokens) or not counts:
        raise StructureGenerationError(f"Invalid chemical formula: {formula!r}")
    return counts


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not safe:
        raise StructureGenerationError("action_id must contain a usable filename character.")
    return safe[:80]


def _source(path: Path) -> tuple[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise StructureGenerationError(f"Required structure source is missing: {resolved}")
    return str(resolved), hashlib.sha256(resolved.read_bytes()).hexdigest()


def _fresh_atoms(atoms):
    """Copy only geometry, eliminating source labels, arrays, constraints, and calculators."""
    _, _, Atoms, _, _, _ = _ase()
    return Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.cell.array.copy(),
        pbc=[True, True, True],
    )


def _factor_triplet(total: int, cell, dimensions: int = 3) -> tuple[int, int, int]:
    lengths = [float(cell.lengths()[i]) for i in range(dimensions)]
    best: tuple[float, tuple[int, int, int]] | None = None
    factors = []
    for first in range(1, total + 1):
        if total % first:
            continue
        remainder = total // first
        for second in range(1, remainder + 1):
            if remainder % second:
                continue
            third = remainder // second
            factors.append((first, second, third))
    for triple in factors:
        if dimensions == 2 and triple[2] != 1:
            continue
        scaled = [lengths[i] * triple[i] for i in range(dimensions)]
        score = max(scaled) / min(scaled)
        candidate = (score, triple)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise StructureGenerationError(
            f"Cannot construct a {dimensions}D supercell of size {total}."
        )
    return best[1]


def _random_strain(atoms, rng, amount: float, dimensions: int = 3) -> None:
    """Apply a bounded homogeneous strain; the fractional geometry is unchanged."""
    if amount == 0:
        return
    _, np, _, _, _, _ = _ase()
    if not math.isfinite(amount) or amount < 0 or amount > 0.05:
        raise StructureGenerationError("strain must be a finite fraction between 0 and 0.05.")
    diagonal = np.ones(3)
    diagonal[:dimensions] += rng.uniform(-amount, amount, dimensions)
    shear = np.eye(3)
    if dimensions == 2:
        shear[0, 1] = shear[1, 0] = rng.uniform(-amount / 2, amount / 2)
    deformation = np.diag(diagonal) @ shear
    cell = atoms.cell.array.copy() @ deformation
    atoms.set_cell(cell, scale_atoms=True)


def _geometry_hash(atoms, *, translation_invariant: bool = False) -> str:
    _, np, _, _, _, _ = _ase()
    cell = np.round(atoms.cell.array, 8).tolist()
    scaled = atoms.get_scaled_positions(wrap=True)
    symbols = atoms.get_chemical_symbols()
    origins = scaled if translation_invariant else [np.zeros(3)]
    sites = min(
        sorted(
            (symbol, *np.round((position - origin) % 1.0, 8).tolist())
            for symbol, position in zip(symbols, scaled, strict=True)
        )
        for origin in origins
    )
    payload = json.dumps(
        {"cell": cell, "pbc": [bool(value) for value in atoms.pbc], "sites": sites},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def make_structure_matcher():
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
    except ImportError as exc:
        raise StructureGenerationError(
            "Symmetry-aware E3 deduplication requires pymatgen."
        ) from exc
    return StructureMatcher(
        ltol=0.01,
        stol=0.01,
        angle_tol=1.0,
        primitive_cell=False,
        scale=False,
        attempt_supercell=False,
    )


def structures_symmetry_equivalent(left, right, matcher=None) -> bool:
    if len(left) != len(right) or Counter(left.get_chemical_symbols()) != Counter(
        right.get_chemical_symbols()
    ):
        return False
    if matcher is None:
        matcher = make_structure_matcher()
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        raise StructureGenerationError(
            "Symmetry-aware E3 deduplication requires pymatgen's ASE adaptor."
        ) from exc
    return bool(
        matcher.fit(
            AseAtomsAdaptor.get_structure(left),
            AseAtomsAdaptor.get_structure(right),
        )
    )


def _validate_geometry(atoms, *, min_distance: float = 0.7) -> None:
    _, np, _, _, _, _ = _ase()
    if len(atoms) == 0 or not all(bool(value) for value in atoms.pbc):
        raise StructureGenerationError("Generated structures must be nonempty and fully periodic.")
    cell = atoms.cell.array
    if not np.isfinite(cell).all() or abs(float(np.linalg.det(cell))) < 1e-6:
        raise StructureGenerationError("Generated structure has a non-finite or singular cell.")
    positions = atoms.get_positions()
    if not np.isfinite(positions).all():
        raise StructureGenerationError("Generated structure has non-finite coordinates.")
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    if len(atoms) > 1 and float(np.min(distances)) < min_distance:
        raise StructureGenerationError(
            f"Generated structure contains an overlap ({float(np.min(distances)):.3f} Å)."
        )


def _space_group(atoms) -> tuple[str | None, str | None]:
    try:
        import spglib
    except ImportError:
        return None, "spglib is not installed"
    try:
        dataset = spglib.get_symmetry_dataset(
            (
                atoms.cell.array,
                atoms.get_scaled_positions(wrap=True),
                atoms.get_atomic_numbers(),
            ),
            symprec=1e-3,
        )
        if dataset is None:
            return None, "symmetry analysis did not identify a space group"
        return f"{dataset.international} ({dataset.number})", None
    except (ValueError, RuntimeError) as exc:
        return None, f"symmetry analysis failed: {exc}"


def _formula_matches(atoms, formula: str, formula_units: int) -> bool:
    expected = {key: value * formula_units for key, value in _parse_formula(formula).items()}
    return Counter(atoms.get_chemical_symbols()) == Counter(expected)


def _template(path: Path, file_format: str = "extxyz"):
    source_path, source_hash = _source(path)
    atoms = _cached_template(source_path, file_format)
    return atoms.copy(), source_path, source_hash


@lru_cache(maxsize=4)
def _cached_template(source_path: str, file_format: str):
    _, _, _, _, read, _ = _ase()
    # ASE may attach source calculators while parsing EXTXYZ; _fresh_atoms
    # immediately discards calculators and every non-geometric array.
    return _fresh_atoms(read(source_path, index=0, format=file_format))


def _alloy(
    material: str, formula: str, tier: int, family: str, strain: Any, defect: str | None, rng
):
    _, _, _, bulk, _, _ = _ase()
    composition = _parse_formula(formula)
    allowed = _ALLOY_ELEMENTS[material]
    if not 4 <= len(composition) <= len(allowed) or not set(composition) <= allowed:
        raise StructureGenerationError(
            f"{material} requires a 4-{len(allowed)}-component composition from {sorted(allowed)}."
        )
    if family == "equiatomic_ordering" and len(set(composition.values())) != 1:
        raise StructureGenerationError(
            f"{material} equiatomic_ordering requires equal integer stoichiometries."
        )
    counts = {symbol: amount * tier for symbol, amount in composition.items()}
    total = sum(counts.values())
    primitive = bulk(
        next(iter(composition)),
        crystalstructure="bcc" if material == "bcc_refractory" else "fcc",
        a=3.0 if material == "bcc_refractory" else 3.6,
        cubic=False,
    )
    repeats = _factor_triplet(total, primitive.cell, dimensions=3)
    atoms = primitive.repeat(repeats)
    symbols = [symbol for symbol, count in counts.items() for _ in range(count)]
    rng.shuffle(symbols)
    atoms.set_chemical_symbols(symbols)
    if defect is not None:
        raise StructureGenerationError(f"Alloy defect {defect!r} is not supported.")
    if strain is not None:
        _random_strain(atoms, rng, float(strain))
    return _fresh_atoms(atoms), {
        "source_file": None,
        "source_sha256": None,
        "source_description": (
            "ASE primitive periodic BCC lattice"
            if material == "bcc_refractory"
            else "ASE primitive periodic FCC lattice"
        ),
        "prototype": "bcc" if material == "bcc_refractory" else "fcc",
        "formula_units": tier,
        "operation": {
            "type": "native periodic alloy occupations",
            "strain_fraction": strain,
        },
    }


def _olivine(formula: str, tier: int, family: str, defect: str | None, strain: Any, rng):
    _, np, _, _, _, _ = _ase()
    if _parse_formula(formula) != {"Li": 1, "Fe": 1, "P": 1, "O": 4}:
        raise StructureGenerationError("The LiFePO4 generator accepts only LiFePO4.")
    base, source_path, source_hash = _template(_SOURCE_DIR / "lifepo4_olivine.extxyz")
    base_formula_units = len(base) // 7
    if len(base) % 7 or base_formula_units != 4:
        raise StructureGenerationError(
            "The LiFePO4 olivine source must contain exactly 4 formula units."
        )
    if tier % base_formula_units:
        raise StructureGenerationError(
            f"LiFePO4 tier {tier} is not commensurate with the olivine source."
        )
    if defect not in {None, "antisite_pair"}:
        raise StructureGenerationError(f"Unsupported LiFePO4 defect: {defect!r}.")
    repeats = _factor_triplet(tier // base_formula_units, base.cell)
    atoms = base.repeat(repeats)
    operation = {}
    if family in {"antisite_pair", "olivine_antisite_pair"} or defect == "antisite_pair":
        symbols = atoms.get_chemical_symbols()
        li_indices = [i for i, symbol in enumerate(symbols) if symbol == "Li"]
        fe_indices = [i for i, symbol in enumerate(symbols) if symbol == "Fe"]
        li_index = int(rng.choice(li_indices))
        fe_index = int(rng.choice(fe_indices))
        symbols[li_index], symbols[fe_index] = symbols[fe_index], symbols[li_index]
        atoms.set_chemical_symbols(symbols)
        operation = {"type": "Li-Fe antisite pair", "sites": [li_index, fe_index]}
    elif family in {"olivine", "parent_olivine", "olivine_strain", "strain", "strained_olivine"}:
        if defect is not None:
            raise StructureGenerationError(f"Unsupported LiFePO4 defect: {defect!r}.")
        operation = {"type": "source olivine parent"}
    else:
        raise StructureGenerationError(f"Unsupported LiFePO4 family/defect: {family!r}/{defect!r}.")
    if family in {"olivine_strain", "strain", "strained_olivine"} or strain is not None:
        amount = 0.01 if strain is None else float(strain)
        _random_strain(atoms, rng, amount)
        operation["strain_fraction"] = amount
        if operation["type"] == "source olivine parent":
            # Sub-percent affine strains alone collapse to the pristine parent
            # under the frozen matcher. Include small non-affine distortions.
            directions = rng.normal(size=atoms.positions.shape)
            norms = np.linalg.norm(directions, axis=1)
            directions /= np.maximum(norms[:, None], np.finfo(float).tiny)
            radii = rng.uniform(0.0, 0.04, len(atoms))
            atoms.positions += directions * radii[:, None]
            operation["type"] = "bounded strained olivine with local distortions"
            operation["maximum_local_displacement_angstrom"] = 0.04
    return _fresh_atoms(atoms), {
        "source_file": source_path,
        "source_sha256": source_hash,
        "prototype": "olivine (Pnma source)",
        "formula_units": tier,
        "operation": operation,
    }


def _phosphorene(
    formula: str,
    tier: int,
    family: str,
    strain: Any,
    defect: str | None,
    rng,
):
    _, np, _, _, _, _ = _ase()
    if _parse_formula(formula) != {"P": 1}:
        raise StructureGenerationError("The phosphorene generator accepts only elemental P.")
    if family in {"black", "black_phosphorene", "black_allotrope"}:
        allotrope, source_name = "black", "phosphorene_black_P.extxyz"
    elif family in {"blue", "blue_phosphorene", "blue_allotrope"}:
        allotrope, source_name = "blue", "phosphorene_blue_P.extxyz"
    elif family in {
        "reconstruction",
        "vacancy_reconstruction",
        "in_plane_strain",
        "strain",
    }:
        if defect not in {None, "reconstruction"}:
            raise StructureGenerationError(
                "Phosphorene supports bounded reconstruction, not atom-count-changing vacancies."
            )
        allotrope = "black"
        source_name = "phosphorene_black_P.extxyz"
    else:
        raise StructureGenerationError(f"Unsupported phosphorene family: {family!r}.")
    if defect is not None and defect != "reconstruction":
        raise StructureGenerationError(f"Unsupported phosphorene defect: {defect!r}.")
    base, source_path, source_hash = _template(_SOURCE_DIR / source_name)
    base_atoms = len(base)
    if tier % base_atoms:
        raise StructureGenerationError(
            f"Phosphorene tier {tier} atoms is not commensurate with the {allotrope} template "
            f"({base_atoms} atoms)."
        )
    repeats = _factor_triplet(tier // base_atoms, base.cell, dimensions=2)
    atoms = base.repeat(repeats)
    # Keep the source's vacuum vector exactly fixed; only in-plane cell vectors
    # participate in the area-preserving 2D strain operation.
    original_vacuum_vector = atoms.cell.array[2].copy()
    if family in {"in_plane_strain", "strain"}:
        amount = 0.02 if strain is None else float(strain)
        _random_strain(atoms, rng, amount, dimensions=2)
        operation = {"type": "bounded in-plane strain", "maximum_fraction": amount}
    elif family in {"reconstruction", "vacancy_reconstruction"} or defect == "reconstruction":
        # Collective, bounded in-plane modes avoid the few symmetry orbits
        # available when moving only one atom by a fixed distance.
        directions = rng.normal(size=(len(atoms), 3))
        directions[:, 2] = 0
        norms = np.linalg.norm(directions, axis=1)
        directions /= np.maximum(norms[:, None], np.finfo(float).tiny)
        displacements = directions * rng.uniform(0.04, 0.08, (len(atoms), 1))
        displacements -= displacements.mean(axis=0)
        largest = float(np.linalg.norm(displacements, axis=1).max())
        displacements *= min(1.0, 0.08 / max(largest, np.finfo(float).tiny))
        atoms.positions += displacements
        operation = {
            "type": "bounded collective in-plane reconstruction",
            "sites": list(range(len(atoms))),
            "max_displacement_angstrom": 0.08,
        }
        if strain is not None:
            _random_strain(atoms, rng, float(strain), dimensions=2)
            operation["strain_fraction"] = float(strain)
    elif strain is not None:
        amount = float(strain)
        _random_strain(atoms, rng, amount, dimensions=2)
        operation = {"type": "bounded in-plane strain", "maximum_fraction": amount}
    else:
        operation = {"type": f"{allotrope} phosphorene supercell"}
    atoms.set_cell(
        [atoms.cell.array[0], atoms.cell.array[1], original_vacuum_vector],
        scale_atoms=False,
    )
    return _fresh_atoms(atoms), {
        "source_file": source_path,
        "source_sha256": source_hash,
        "prototype": f"{allotrope} phosphorene 2D template",
        "formula_units": tier,
        "operation": operation,
    }


def _oxide_prototype(
    formula: str,
    tier: int,
    family: str,
    defect: str | None,
    rng,
    prototype_ids: tuple[str, ...] = (),
    space_groups: tuple[int, ...] = (),
    strain: Any = None,
):
    _, np, Atoms, _, _, _ = _ase()
    composition = _OXIDE_FORMULAS.get(formula)
    oxygen_vacancy = defect in {"oxygen_vacancy", "oxygen_defect"} or family == "oxygen_defect"
    if composition is None:
        if not (oxygen_vacancy and formula in {"Nb2O4", "Ta2O4", "NbTaO4"}):
            raise StructureGenerationError(
                f"Unsupported Nb-Ta-O composition {formula!r}; declared formulas are "
                f"{', '.join(_OXIDE_FORMULAS)} (oxygen-defect formulas require an explicit defect)."
            )
        composition = _parse_formula(formula)
    if family not in {
        "oxide_prototype",
        "aflow_prototype",
        "pyxtal",
        "prototype",
        "oxygen_defect",
        "prototype_strain",
    }:
        raise StructureGenerationError(f"Unsupported Nb-Ta-O family: {family!r}.")
    if tier not in _TIERS["nb_ta_o"]:
        raise StructureGenerationError("Unsupported Nb-Ta-O formula-unit tier.")
    counts = {symbol: amount * tier for symbol, amount in composition.items()}
    total = sum(counts.values())

    if (
        composition.get("O") == 2
        and sum(value for key, value in composition.items() if key != "O") == 1
    ):
        # One-formula fluorite primitive cell: one cation and two oxygen sites.
        lattice = np.array([[0.0, 2.8, 2.8], [2.8, 0.0, 2.8], [2.8, 2.8, 0.0]])
        atoms = Atoms(
            symbols=[next(key for key in composition if key != "O"), "O", "O"],
            scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            cell=lattice,
            pbc=True,
        )
        prototype = "fluorite primitive (AB2_cF12)"
        prototype_id = "AB2_cF12_225_a_c"
    else:
        # Explicit orthorhombic prototype basis for the declared pentoxide and
        # mixed Nb-Ta oxides; it is a deterministic bulk prototype operation,
        # not a claim that the generated cell is a measured ground state.
        metals = [key for key in composition if key != "O"]
        metal_sites = sum(composition[key] for key in metals)
        oxygens = composition.get("O", 0)
        basis = [
            (0.08, 0.10, 0.12),
            (0.58, 0.60, 0.62),
            (0.30, 0.36, 0.39),
            (0.80, 0.86, 0.89),
            (0.30, 0.86, 0.39),
            (0.80, 0.36, 0.89),
            (0.54, 0.08, 0.22),
        ]
        if metal_sites + oxygens > len(basis):
            raise StructureGenerationError(
                "This oxide prototype does not support the requested formula."
            )
        symbols = []
        for metal in metals:
            symbols.extend([metal] * composition[metal])
        symbols.extend(["O"] * oxygens)
        positions = basis[: len(symbols)]
        atoms = Atoms(
            symbols=symbols,
            scaled_positions=positions,
            cell=[[6.2, 0, 0], [0, 6.7, 0], [0, 0, 7.1]],
            pbc=True,
        )
        prototype = "orthorhombic A2B5 / mixed-metal oxide basis"
        prototype_id = "A2B5_oP7_local_prototype"
    if prototype_ids and prototype_id not in prototype_ids:
        raise StructureGenerationError(
            f"Local oxide prototype {prototype_id!r} does not match prototype_ids."
        )
    repeats = _factor_triplet(tier, atoms.cell)
    atoms = atoms.repeat(repeats)
    if formula == "NbTaO4":
        # This declared composition is also a one-oxygen-per-parent-cell
        # defect family relative to NbTaO5; the atom count follows its formula.
        prototype = "Nb-Ta-O oxygen-deficient prototype"
    if family == "prototype_strain" or strain is not None:
        amount = 0.015 if strain is None else float(strain)
        _random_strain(atoms, rng, amount)
        operation = {"type": "bounded prototype strain", "maximum_fraction": amount}
    else:
        operation = {
            "type": "declared oxide prototype",
            "operation_source": family,
            "defect": defect or ("oxygen_defect" if oxygen_vacancy else None),
        }
    strain_applied = family == "prototype_strain" or strain is not None
    group = None
    if prototype_id == "AB2_cF12_225_a_c" and not strain_applied:
        group = 225
    elif space_groups:
        symmetry, reason = _space_group(atoms)
        if symmetry is None:
            raise StructureGenerationError(f"Cannot apply oxide space_groups filter: {reason}.")
        group = int(symmetry.rsplit("(", 1)[1].rstrip(")"))
    if space_groups and group not in space_groups:
        raise StructureGenerationError(
            f"Local oxide prototype has space group {group}, outside space_groups."
        )
    if group is not None:
        operation["space_group"] = group
    if len(atoms) != total:
        raise StructureGenerationError(
            f"Oxide tier generated {len(atoms)} atoms, expected {total} for {formula}."
        )
    return _fresh_atoms(atoms), {
        "source_file": None,
        "source_sha256": None,
        "source_description": "E3 declared periodic oxide prototype basis",
        "prototype": prototype,
        "prototype_id": prototype_id,
        "formula_units": tier,
        "operation": operation,
    }


def _aflow_oxide_candidates(
    formula: str, tier: int, prototype_ids: tuple[str, ...], space_groups: tuple[int, ...]
):
    try:
        import pymatgen.analysis.prototypes as prototype_data
        from pymatgen.io.ase import AseAtomsAdaptor

        from matsim_agents.discovery.composition import parse_composition
        from matsim_agents.discovery.seeds import (
            _balanced_supercell_matrix,
            compatible_prototypes,
            decorate,
        )
    except ImportError as exc:
        raise StructureGenerationError(
            "AFLOW oxide generation requires the repository's optional pymatgen components."
        ) from exc
    source_path, source_hash = _source(
        Path(prototype_data.__file__).parent / "aflow_prototypes.json.gz"
    )
    composition = parse_composition(formula)
    if composition is None:
        raise StructureGenerationError(f"Cannot parse AFLOW oxide formula {formula!r}.")
    try:
        prototypes = compatible_prototypes(composition)
    except (ImportError, FileNotFoundError) as exc:
        raise StructureGenerationError(
            "AFLOW oxide generation could not load pymatgen's prototype encyclopedia."
        ) from exc
    if prototype_ids:
        available = {prototype.aflow for prototype in prototypes}
        missing = set(prototype_ids) - available
        if missing:
            raise StructureGenerationError(
                f"AFLOW prototype filters do not match {formula}: {', '.join(sorted(missing))}."
            )
        prototypes = [prototype for prototype in prototypes if prototype.aflow in prototype_ids]
    candidates = []
    for prototype in prototypes:
        group = prototype.structure.get_space_group_info()[1]
        if space_groups and group not in space_groups:
            continue
        for structure, decoration in decorate(prototype, composition):
            if structure.num_sites % composition.total_atoms:
                continue
            base_units = structure.num_sites // composition.total_atoms
            if tier % base_units:
                continue
            if tier > base_units:
                structure.make_supercell(_balanced_supercell_matrix(structure, tier // base_units))
            atoms = _fresh_atoms(AseAtomsAdaptor.get_atoms(structure))
            if len(atoms) != tier * composition.total_atoms:
                continue
            candidates.append(
                (
                    atoms,
                    {
                        "source_file": source_path,
                        "source_sha256": source_hash,
                        "source_description": f"AFLOW prototype {prototype.aflow}",
                        "prototype": f"AFLOW {prototype.aflow}",
                        "prototype_id": prototype.aflow,
                        "formula_units": tier,
                        "operation": {
                            "type": "AFLOW prototype decoration",
                            "decoration": decoration,
                            "space_group": group,
                        },
                    },
                )
            )
    if not candidates:
        raise StructureGenerationError(
            f"No AFLOW oxide candidates satisfy formula={formula}, tier={tier}, "
            f"prototype_ids={list(prototype_ids)}, space_groups={list(space_groups)}."
        )
    return candidates


def _pyxtal_oxide_candidates(
    formula: str,
    tier: int,
    seed: int,
    count: int,
    prototype_ids: tuple[str, ...],
    space_groups: tuple[int, ...],
):
    if prototype_ids:
        raise StructureGenerationError(
            "prototype_ids filters apply only to AFLOW oxide generation, not PyXtal proposals."
        )
    try:
        import pyxtal
        from pymatgen.io.ase import AseAtomsAdaptor

        from matsim_agents.discovery.composition import parse_composition
        from matsim_agents.discovery.seeds import random_search
    except ImportError as exc:
        raise StructureGenerationError(
            "PyXtal oxide generation requires the repository's optional pymatgen components."
        ) from exc
    source_path, source_hash = _source(Path(pyxtal.__file__))
    composition = parse_composition(formula)
    if composition is None:
        raise StructureGenerationError(f"Cannot parse PyXtal oxide formula {formula!r}.")
    try:
        proposals = random_search(
            composition,
            n=max(count * 2, 4),
            seed=seed,
            formula_units=tier,
            space_groups=list(space_groups) or None,
            seed_proposals=True,
        )
    except ImportError as exc:
        raise StructureGenerationError("PyXtal oxide generation requires pymatgen.") from exc
    candidates = []
    for structure, group in proposals:
        atoms = _fresh_atoms(AseAtomsAdaptor.get_atoms(structure))
        if len(atoms) != tier * composition.total_atoms:
            continue
        if space_groups and group not in space_groups:
            continue
        candidates.append(
            (
                atoms,
                {
                    "source_file": source_path,
                    "source_sha256": source_hash,
                    "source_description": f"PyXtal random oxide proposal in space group {group}",
                    "prototype": "PyXtal periodic oxide proposal",
                    "prototype_id": None,
                    "formula_units": tier,
                    "operation": {
                        "type": "PyXtal random bulk proposal",
                        "space_group": group,
                        "random_seed": seed,
                    },
                    "random_seed": seed,
                },
            )
        )
    if not candidates:
        raise StructureGenerationError(
            f"PyXtal produced no oxide candidate for formula={formula}, tier={tier}, "
            f"space_groups={list(space_groups)}."
        )
    return candidates


def _mof_connectivity_signature(atoms) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    symbols = atoms.get_chemical_symbols()
    distances = atoms.get_all_distances(mic=True)
    zn_o = []
    c_o = []
    c_h = []
    for index, symbol in enumerate(symbols):
        if symbol == "Zn":
            zn_o.append(
                sum(
                    1
                    for j, other in enumerate(symbols)
                    if other == "O" and distances[index, j] < 2.75
                )
            )
        elif symbol == "C":
            c_o.append(
                sum(
                    1
                    for j, other in enumerate(symbols)
                    if other == "O" and distances[index, j] < 1.68
                )
            )
            c_h.append(
                sum(
                    1
                    for j, other in enumerate(symbols)
                    if other == "H" and distances[index, j] < 1.25
                )
            )
    return tuple(sorted(zn_o)), tuple(sorted(c_o)), tuple(sorted(c_h))


def _mof(
    formula: str,
    tier: int,
    family: str,
    strain: Any,
    defect: str | None,
    rng,
    operation_label: str | None = None,
):
    _, np, _, _, _, _ = _ase()
    expected = {"Zn": 1, "C": 2, "H": 2, "O": 4}
    if _parse_formula(formula) != expected:
        raise StructureGenerationError("The Zn formate generator accepts Zn(HCOO)2 only.")
    if family not in {"cod_parent", "bounded_perturbation", "framework_perturbation", "strain"}:
        raise StructureGenerationError(f"Unsupported Zn-formate MOF family: {family!r}.")
    if defect is not None:
        raise StructureGenerationError(
            "Zn-formate defects are not supported; use bounded perturbations."
        )
    base, source_path, source_hash = _template(
        _SOURCE_DIR / "zn_formate_cod_7205949.cif", file_format="cif"
    )
    if len(base) != 216 or Counter(base.get_chemical_symbols()) != Counter(
        {"Zn": 24, "C": 48, "H": 48, "O": 96}
    ):
        raise StructureGenerationError("COD 7205949 must be the 216-atom Zn24C48H48O96 parent.")
    if tier % 24:
        raise StructureGenerationError("Zn-formate tiers must be 24, 48, 96, or 192 formula units.")
    repeats = _factor_triplet(tier // 24, base.cell)
    atoms = base.repeat(repeats)
    parent_signature = _mof_connectivity_signature(base)
    if family in {"bounded_perturbation", "framework_perturbation"}:
        amount = 0.005 if strain is None else float(strain)
        if amount > 0.01:
            raise StructureGenerationError("MOF framework strain is limited to 1%.")
        if operation_label == "proton_orientation":
            symbols = atoms.get_chemical_symbols()
            carbon_indices = [i for i, symbol in enumerate(symbols) if symbol == "C"]
            hydrogen_indices = [i for i, symbol in enumerate(symbols) if symbol == "H"]
            attached_carbons = []
            for hydrogen_index in hydrogen_indices:
                vectors = atoms.get_distances(
                    hydrogen_index, carbon_indices, mic=True, vector=True,
                )
                closest = int(np.linalg.norm(vectors, axis=1).argmin())
                carbon_index = carbon_indices[closest]
                attached_carbons.append(carbon_index)
                vector = -vectors[closest]
                axis = rng.normal(size=3)
                axis /= float((axis**2).sum() ** 0.5)
                angle = float(rng.uniform(-math.pi / 12, math.pi / 12))
                cross = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ]
                )
                rotation = (
                    np.eye(3) * math.cos(angle)
                    + np.outer(axis, axis) * (1 - math.cos(angle))
                    + cross * math.sin(angle)
                )
                atoms.positions[hydrogen_index] = atoms.positions[carbon_index] + rotation @ vector
            atoms.wrap()
            operation = {
                "type": "bounded formate proton-orientation variant",
                "hydrogen_sites": hydrogen_indices,
                "carbon_sites": attached_carbons,
                "maximum_rotation_degrees": 15,
            }
        else:
            _random_strain(atoms, rng, amount)
            displacement = rng.normal(0.0, min(0.015, amount * 2.0), size=atoms.positions.shape)
            atoms.positions += displacement
            operation = {
                "type": "bounded connectivity-preserving framework perturbation",
                "maximum_strain_fraction": amount,
                "displacement_sigma_angstrom": min(0.015, amount * 2.0),
            }
    elif family == "strain":
        amount = 0.005 if strain is None else float(strain)
        if amount > 0.01:
            raise StructureGenerationError("MOF framework strain is limited to 1%.")
        _random_strain(atoms, rng, amount)
        operation = {"type": "bounded homogeneous framework strain", "maximum_fraction": amount}
    else:
        operation = {"type": "COD 7205949 parent supercell"}
    if _mof_connectivity_signature(atoms) != tuple(
        tuple(sorted(values * (tier // 24))) for values in parent_signature
    ):
        raise StructureGenerationError("MOF perturbation changed Zn-O/formate connectivity.")
    return _fresh_atoms(atoms), {
        "source_file": source_path,
        "source_sha256": source_hash,
        "prototype": "COD 7205949 Pbcn zinc-formate framework",
        "formula_units": tier,
        "operation": operation,
    }


def _generate_one(
    material: str,
    formula: str,
    tier: int,
    family: str,
    generator: str,
    defect: str | None,
    strain: Any,
    rng,
    prototype_ids: tuple[str, ...] = (),
    space_groups: tuple[int, ...] = (),
    operation_label: str | None = None,
):
    if material in _ALLOY_ELEMENTS:
        if generator not in {"native_lattice_occupations", "class_specific"}:
            raise StructureGenerationError(
                f"{material} requires native periodic lattice occupations, not {generator!r}."
            )
        return _alloy(material, formula, tier, family, strain, defect, rng)
    if material == "lifepo4":
        if generator not in {"ase_seed_variants", "class_specific"}:
            raise StructureGenerationError("LiFePO4 requires its olivine seed-variant generator.")
        return _olivine(formula, tier, family, defect, strain, rng)
    if material == "phosphorene":
        if generator not in {"2d_template", "class_specific"}:
            raise StructureGenerationError("Phosphorene requires its 2D template generator.")
        return _phosphorene(formula, tier, family, strain, defect, rng)
    if material == "nb_ta_o":
        if generator not in {
            "aflow_prototype",
            "pyxtal",
            "aflow_vacancy",
            "pyxtal_vacancy",
            "prototype",
            "class_specific",
        }:
            raise StructureGenerationError(f"Unsupported Nb-Ta-O generator: {generator!r}.")
        if generator in {"aflow_prototype", "pyxtal"}:
            raise StructureGenerationError(
                f"{generator} proposals must be selected through the filtered oxide candidate pool."
            )
        return _oxide_prototype(
            formula,
            tier,
            family,
            defect,
            rng,
            prototype_ids,
            space_groups,
            strain,
        )
    if material == "zn_formate_mof":
        if generator not in {"cod_7205949", "class_specific", "connectivity_preserving"}:
            raise StructureGenerationError(
                "Zn-formate MOFs must use COD 7205949 connectivity-preserving variants."
            )
        return _mof(formula, tier, family, strain, defect, rng, operation_label)
    raise StructureGenerationError(f"Unknown E3 material class: {material!r}.")


def replicate_structure(
    material: str,
    base_record: dict[str, Any],
    action: dict[str, Any],
    output_dir: str | Path,
    control_family_id: int,
) -> dict[str, Any]:
    """Build a finite-size control by exact supercell replication of one base motif."""
    ase, np, _, _, _, write = _ase()
    base_path = Path(base_record["structure_path"]).expanduser().resolve()
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != base_record["structure_sha256"]:
        raise StructureGenerationError(f"Finite-size control base hash mismatch: {base_path}")
    base_generation = base_record["generation"]
    for key in ("material", "family", "family_label", "formula", "generator"):
        if base_generation.get(key) != action.get(key):
            raise StructureGenerationError(
                f"Finite-size control action changed {key}: "
                f"{base_generation.get(key)!r} != {action.get(key)!r}."
            )
    base_tier = int(base_generation["size_tier"])
    target_tier = int(action["size_tier"])
    if material not in _TIERS or target_tier not in _TIERS[material]:
        raise StructureGenerationError(f"Unsupported finite-size control tier: {target_tier}.")
    if target_tier < base_tier or target_tier % base_tier:
        raise StructureGenerationError(
            f"Finite-size control tier {target_tier} is not an integer multiple "
            f"of base tier {base_tier}."
        )

    from ase.io import read

    atoms = read(base_path, index=0, format="extxyz")
    repetitions = _factor_triplet(
        target_tier // base_tier,
        atoms.cell,
        dimensions=2 if material == "phosphorene" else 3,
    )
    atoms = atoms.repeat(repetitions)
    _validate_geometry(atoms)
    if not _formula_matches(atoms, str(action["formula"]), target_tier):
        raise StructureGenerationError(
            f"Replicated control does not match {action['formula']} × {target_tier}."
        )

    generation = dict(base_generation)
    geometry_hash = _geometry_hash(atoms, translation_invariant=material in _ALLOY_ELEMENTS)
    generation.update(
        action_id=action["action_id"],
        num_atoms=len(atoms),
        formula_units=target_tier,
        size_tier=target_tier,
        tier_unit="atoms" if material == "phosphorene" else "formula_units",
        canonical_geometry_sha256=geometry_hash,
        lattice={
            "matrix_angstrom": np.round(atoms.cell.array, 10).tolist(),
            "lengths_angstrom": np.round(atoms.cell.lengths(), 10).tolist(),
            "angles_degrees": np.round(atoms.cell.angles(), 10).tolist(),
            "pbc": [bool(value) for value in atoms.pbc],
        },
        finite_size_control={
            "control_family_id": control_family_id,
            "base_size_tier": base_tier,
            "supercell_repetitions": list(repetitions),
            "base_geometry_sha256": base_generation["canonical_geometry_sha256"],
        },
    )
    safe_action_id = re.sub(r"[^A-Za-z0-9._-]+", "-", str(action["action_id"])).strip("-")
    candidate_id = f"{safe_action_id}-{geometry_hash[:12]}"
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    structure_path = output / f"{candidate_id}.extxyz"
    atoms.info = {"generation_json": json.dumps(generation, sort_keys=True, separators=(",", ":"))}
    atoms.calc = None
    write(structure_path, atoms, format="extxyz")
    structure_hash = hashlib.sha256(structure_path.read_bytes()).hexdigest()
    return {
        **base_record,
        "candidate_id": candidate_id,
        "structure_path": str(structure_path),
        "structure_sha256": structure_hash,
        "num_atoms": len(atoms),
        "formula_units": target_tier,
        "generation": generation,
    }


def generate_structures(
    material: str, action: dict[str, Any], output_dir: str | Path,
    *, excluded_structures: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate an exact, deduplicated batch of sanitized E3-MLIP structures.

    ``size_tier`` is formula units for the bulk classes, the requested atom
    count for phosphorene, and formula units for the MOF (24-FU multiples).
    """
    if material not in _TIERS:
        raise StructureGenerationError(f"Unknown E3 material class: {material!r}.")
    if not isinstance(action, dict):
        raise StructureGenerationError("action must be a dictionary.")
    required = {"action_id", "family", "formula", "size_tier", "generator", "random_seed", "count"}
    missing = required - action.keys()
    if missing:
        raise StructureGenerationError(
            f"Action is missing required keys: {', '.join(sorted(missing))}."
        )
    action_id = str(action["action_id"])
    safe_action_id = _safe_id(action_id)
    formula = str(action["formula"])
    tier = action["size_tier"]
    count = action["count"]
    seed = action["random_seed"]
    if isinstance(tier, bool) or not isinstance(tier, int) or tier not in _TIERS[material]:
        raise StructureGenerationError(
            f"Unsupported {material} size_tier {tier!r}; allowed tiers: {sorted(_TIERS[material])}."
        )
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise StructureGenerationError("count must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StructureGenerationError("random_seed must be an integer.")
    family = str(action["family"])
    family_label = str(action.get("family_label", family))
    operation_label = action.get("operation")
    generator = str(action["generator"])
    if family not in _FAMILIES[material]:
        raise StructureGenerationError(
            f"Unsupported {material} family {family!r}; allowed families: "
            f"{', '.join(sorted(_FAMILIES[material]))}."
        )
    defect = action.get("defect")
    if defect is not None and not isinstance(defect, str):
        raise StructureGenerationError("defect must be a string when supplied.")
    strain = action.get("strain")
    prototype_ids = action.get("prototype_ids", [])
    space_groups = action.get("space_groups", [])
    if not isinstance(prototype_ids, list) or any(
        not isinstance(value, str) or not value for value in prototype_ids
    ):
        raise StructureGenerationError("prototype_ids must be a list of nonempty strings.")
    if not isinstance(space_groups, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 230
        for value in space_groups
    ):
        raise StructureGenerationError("space_groups must be a list of integers from 1 to 230.")
    if material != "nb_ta_o" and (prototype_ids or space_groups):
        raise StructureGenerationError(
            "prototype_ids and space_groups filters apply only to Nb-Ta-O."
        )
    if strain is not None:
        if isinstance(strain, bool):
            raise StructureGenerationError("strain must be a numeric fraction, not a boolean.")
        try:
            strain = float(strain)
        except (TypeError, ValueError) as exc:
            raise StructureGenerationError("strain must be a numeric fraction.") from exc
        if not math.isfinite(strain) or strain < 0 or strain > 0.05:
            raise StructureGenerationError("strain must be a finite fraction between 0 and 0.05.")

    if material == "nb_ta_o":
        if formula not in _OXIDE_FORMULAS and not (defect and formula in {"Nb2O4", "Ta2O4"}):
            raise StructureGenerationError(f"Undeclared Nb-Ta-O formula: {formula!r}.")
    elif material == "lifepo4" and _parse_formula(formula) != {"Li": 1, "Fe": 1, "P": 1, "O": 4}:
        raise StructureGenerationError("The LiFePO4 generator accepts only LiFePO4.")

    ase, np, _, _, _, write = _ase()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prototype_filter = tuple(prototype_ids)
    space_group_filter = tuple(space_groups)
    oxide_candidates = None
    oxide_formula = formula
    oxygen_vacancy_generation = defect == "oxygen_defect"
    if oxygen_vacancy_generation:
        oxide_formula = {"Nb2O4": "Nb2O5", "Ta2O4": "Ta2O5"}.get(formula, "")
        if not oxide_formula:
            raise StructureGenerationError(
                f"Charge-compensated oxygen-vacancy generation is not defined for {formula}."
            )
    if material == "nb_ta_o" and generator in {"aflow_prototype", "aflow_vacancy"}:
        oxide_candidates = _aflow_oxide_candidates(
            oxide_formula, tier, prototype_filter, space_group_filter
        )
    elif material == "nb_ta_o" and generator in {"pyxtal", "pyxtal_vacancy"}:
        oxide_candidates = _pyxtal_oxide_candidates(
            oxide_formula, tier, seed, count, prototype_filter, space_group_filter
        )
    rng_master = np.random.default_rng(seed)
    selected = []
    seen: set[str] = set()
    symmetry_matcher = make_structure_matcher()
    max_attempts = max(40, count * 30)
    geometry_rejections: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    attempts = 0
    for _ in range(max_attempts):
        attempts += 1
        candidate_seed = int(rng_master.integers(0, 2**32, dtype=np.uint64))
        rng = np.random.default_rng(candidate_seed)
        if oxide_candidates is not None:
            candidate_index = int(rng.integers(0, len(oxide_candidates)))
            atoms, source_metadata = oxide_candidates[candidate_index]
            atoms = atoms.copy()
            if oxygen_vacancy_generation:
                oxygen_indices = [
                    index
                    for index, symbol in enumerate(atoms.get_chemical_symbols())
                    if symbol == "O"
                ]
                vacancy_indices = sorted(
                    rng.choice(oxygen_indices, size=tier, replace=False).tolist(),
                    reverse=True,
                )
                vacancy_sites = [int(index) for index in vacancy_indices]
                del atoms[vacancy_indices]
                source_metadata = dict(source_metadata)
                source_metadata["formula_units"] = tier
                source_metadata["operation"] = {
                    "type": "oxygen-vacancy configuration from stoichiometric pentoxide",
                    "parent_formula": oxide_formula,
                    "removed_oxygen_sites": vacancy_sites,
                    "vacancies_per_formula_unit": 1,
                    "charge_compensation": (
                        f"charge-neutral {formula}; formal cation(IV) with O(II-)"
                    ),
                }
            if strain is not None:
                _random_strain(atoms, rng, float(strain))
                source_metadata = dict(source_metadata)
                source_metadata["bounded_strain_fraction"] = float(strain)
        else:
            atoms, source_metadata = _generate_one(
                material,
                formula,
                tier,
                family,
                generator,
                defect,
                strain,
                rng,
                prototype_filter,
                space_group_filter,
                operation_label,
            )
        try:
            _validate_geometry(atoms)
        except StructureGenerationError as exc:
            geometry_rejections[str(exc)] += 1
            rejections["invalid_geometry"] += 1
            continue
        formula_units = source_metadata["formula_units"]
        if not _formula_matches(atoms, formula, formula_units):
            raise StructureGenerationError(
                f"Generated composition does not match {formula} × {formula_units}."
            )
        geometry_hash = _geometry_hash(atoms, translation_invariant=material in _ALLOY_ELEMENTS)
        if geometry_hash in seen:
            rejections["duplicate_geometry_hash"] += 1
            continue
        if any(
            structures_symmetry_equivalent(atoms, prior, symmetry_matcher) for prior, _ in selected
        ):
            rejections["symmetry_equivalent"] += 1
            continue
        if any(
            structures_symmetry_equivalent(atoms, prior, symmetry_matcher)
            for prior in excluded_structures or []
        ):
            rejections["prior_symmetry_equivalent"] += 1
            continue
        seen.add(geometry_hash)
        space_group, space_group_reason = _space_group(atoms)
        operation = source_metadata.get("operation") or {}
        if space_group is None and operation.get("space_group") is not None:
            space_group = f"space group {operation['space_group']}"
            space_group_reason = None
        generation = {
            "generator": generator,
            "generator_module": "benchmarks.e3.structure_generation",
            "generator_version": "3",
            "generator_source_file": str(Path(__file__).resolve()),
            "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "ase_version": ase.__version__,
            "numpy_version": np.__version__,
            "action_id": action_id,
            "action_family": family,
            "family_label": family_label,
            "action_operation": operation_label,
            "random_seed": source_metadata.get("random_seed", candidate_seed),
            "candidate_selection_seed": candidate_seed,
            "batch_random_seed": seed,
            "formula": formula,
            "num_atoms": len(atoms),
            "formula_units": formula_units,
            "size_tier": tier,
            "tier_unit": "atoms" if material == "phosphorene" else "formula_units",
            "material": material,
            "family": family,
            "defect": defect,
            "strain": strain,
            "charge_compensation": action.get("charge_compensation"),
            "operation": source_metadata.get("operation"),
            "prototype_ids": prototype_ids or None,
            "space_groups": space_groups or None,
            "source_file": source_metadata.get("source_file"),
            "source_sha256": source_metadata.get("source_sha256"),
            "source_description": source_metadata.get("source_description"),
            "prototype": source_metadata.get("prototype"),
            "prototype_id": source_metadata.get("prototype_id"),
            "space_group": space_group,
            "space_group_reason": space_group_reason,
            "lattice": {
                "matrix_angstrom": np.round(atoms.cell.array, 10).tolist(),
                "lengths_angstrom": np.round(atoms.cell.lengths(), 10).tolist(),
                "angles_degrees": np.round(atoms.cell.angles(), 10).tolist(),
                "pbc": [bool(value) for value in atoms.pbc],
            },
            "canonical_geometry_sha256": geometry_hash,
        }
        selected.append((atoms, generation))
        if len(selected) == count:
            break
    records: list[dict[str, Any]] = []
    for atoms, generation in selected:
        canonical_hash = generation["canonical_geometry_sha256"]
        candidate_id = f"{safe_action_id}-{canonical_hash[:12]}"
        structure_path = output / f"{candidate_id}.extxyz"
        # The extxyz carries only a JSON provenance string; source calculators,
        # energy/force/stress results and unrelated metadata are never copied.
        atoms.info = {
            "generation_json": json.dumps(generation, sort_keys=True, separators=(",", ":"))
        }
        atoms.calc = None
        write(structure_path, atoms, format="extxyz")
        structure_hash = hashlib.sha256(structure_path.read_bytes()).hexdigest()
        records.append(
            {
                "candidate_id": candidate_id,
                "structure_path": str(structure_path.resolve()),
                "structure_sha256": structure_hash,
                "num_atoms": len(atoms),
                "formula": formula,
                "formula_units": generation["formula_units"],
                "generation": generation,
            }
        )
    diagnostics = {
        "action_id": action_id, "requested": count,
        "generation_attempts": attempts, "accepted_unique": len(records),
        "rejection_reasons": dict(rejections), "geometry_rejections": dict(geometry_rejections),
        "category": "generation_shortfall" if len(records) != count else "complete",
        "remaining_capacity": None,
    }
    (output / "generation-diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(records) != count:
        rejection_summary = "; ".join(
            f"{reason} ({rejected})" for reason, rejected in geometry_rejections.most_common(3)
        )
        raise GenerationExhausted(
            f"Requested {count} unique {material} structures for action {action_id!r}, "
            f"but only {len(selected)} were generated in {max_attempts} attempts. "
            f"Geometry rejections: {rejection_summary or 'none'}.",
            records, diagnostics,
        )
    return records
