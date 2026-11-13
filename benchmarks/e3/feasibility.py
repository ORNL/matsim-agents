"""Executable generation constraints shared by both E3 acquisition policies."""

from __future__ import annotations

import importlib.util
from collections import Counter, defaultdict
from functools import cache
from typing import Any

from benchmarks.e3 import structure_generation as generation

CATALOG_VERSION = "executable-capacities-2"


class ScientificCapacityError(ValueError):
    """The remaining admissible configurations cannot fill an acquisition."""


class InvalidActionError(ValueError):
    """An acquisition requested an unavailable or malformed action."""


@cache
def _aflow_inventory(formula: str, tier: int) -> tuple[tuple[str, int], ...]:
    candidates = generation._aflow_oxide_candidates(formula, tier, (), ())
    return tuple(
        (metadata["prototype_id"], metadata["operation"]["space_group"])
        for _, metadata in candidates
    )


def capacity_group(action: dict[str, Any]) -> tuple[str, str] | None:
    material = action.get("material")
    tier = str(action.get("size_tier"))
    if (
        material == "lifepo4"
        and action.get("family") == "olivine"
        and action.get("defect") is None
        and action.get("strain") is None
    ):
        return "lifepo4-pristine-parent", tier
    if material == "phosphorene":
        if action.get("defect") is None and action.get("strain") is None:
            return f"phosphorene-pristine-{action['family_label']}", tier
        if action.get("defect") == "reconstruction" and action.get("size_tier") == 4:
            return f"phosphorene-small-reconstruction-{action['family_label']}", tier
    if material == "zn_formate_mof" and action.get("family") == "strain":
        return "zn-formate-homogeneous-strain", tier
    return None


def _capacity_key(action: dict[str, Any]) -> str:
    group = capacity_group(action)
    if group is not None:
        return ":".join(group)
    # Family-label and operation aliases must not create additional capacity.
    fields = (
        "material",
        "formula",
        "size_tier",
        "family",
        "generator",
        "defect",
        "strain",
        "space_group_family",
    )
    key = "|".join(str(action.get(field)) for field in fields)
    if action.get("material") == "zn_formate_mof":
        key += "|" + str(action.get("operation"))
    return key


@cache
def _source_available(material: str, family: str, tier: int) -> str:
    sources = {
        "lifepo4": ("lifepo4_olivine.extxyz", 4),
        "zn_formate_mof": ("zn_formate_cod_7205949.cif", 24),
        "phosphorene": (
            "phosphorene_blue_P.extxyz"
            if family.startswith("blue")
            else "phosphorene_black_P.extxyz",
            1,
        ),
    }
    if material not in sources:
        return "native lattice generator"
    filename, units = sources[material]
    path = generation._SOURCE_DIR / filename
    generation._source(path)
    atoms, _, _ = generation._template(path, "cif" if path.suffix == ".cif" else "extxyz")
    if material == "phosphorene":
        units = len(atoms)
    if tier % units:
        raise InvalidActionError(f"tier {tier} is not commensurate with template {filename}")
    expected = {
        "lifepo4": Counter({"Li": 4, "Fe": 4, "P": 4, "O": 16}),
        "zn_formate_mof": Counter({"Zn": 24, "C": 48, "H": 48, "O": 96}),
        "phosphorene": Counter({"P": len(atoms)}),
    }
    if Counter(atoms.get_chemical_symbols()) != expected[material]:
        raise InvalidActionError(f"template composition mismatch: {path}")
    return str(path)


@cache
def _antisite_capacity(tier: int) -> int:
    import numpy as np
    import spglib

    atoms, _ = generation._olivine(
        "LiFePO4",
        tier,
        "olivine",
        None,
        None,
        np.random.default_rng(0),
    )
    fractions = atoms.get_scaled_positions(wrap=True)
    symbols = atoms.get_chemical_symbols()

    def key(position):
        return tuple(np.round(np.round(position % 1, 6) % 1, 6))

    indices = {
        (symbol, key(position)): i
        for i, (symbol, position) in enumerate(zip(symbols, fractions, strict=True))
    }
    symmetry = spglib.get_symmetry((atoms.cell.array, fractions, atoms.numbers), symprec=1e-5)
    if symmetry is None:
        raise InvalidActionError("could not establish olivine antisite symmetry capacity")
    permutations = [
        [
            indices[(symbol, key(rotation @ position + translation))]
            for symbol, position in zip(symbols, fractions, strict=True)
        ]
        for rotation, translation in zip(
            symmetry["rotations"], symmetry["translations"], strict=True
        )
    ]
    pairs = {
        (i, j)
        for i, symbol in enumerate(symbols)
        if symbol == "Li"
        for j, other in enumerate(symbols)
        if other == "Fe"
    }
    count = 0
    while pairs:
        i, j = next(iter(pairs))
        pairs.difference_update((permutation[i], permutation[j]) for permutation in permutations)
        count += 1
    return count


def action_constraints(action: dict[str, Any]) -> dict[str, Any]:
    """Describe operational limits, not a claim about all physically possible structures."""
    material, tier = action["material"], action["size_tier"]
    if material not in generation._TIERS or tier not in generation._TIERS[material]:
        raise InvalidActionError(f"unsupported material/tier: {material}/{tier}")
    if action["family"] not in generation._FAMILIES[material]:
        raise InvalidActionError(f"unsupported structural family: {action['family']}")
    method = action["generator"]
    methods = {
        "bcc_refractory": {"native_lattice_occupations"},
        "fcc_cantor": {"native_lattice_occupations"},
        "lifepo4": {"ase_seed_variants"},
        "phosphorene": {"2d_template"},
        "nb_ta_o": {"aflow_prototype", "aflow_vacancy", "pyxtal", "pyxtal_vacancy"},
        "zn_formate_mof": {"cod_7205949"},
    }
    if method not in methods[material]:
        raise InvalidActionError(f"unsupported generation method: {method}")
    composition = generation._parse_formula(action["formula"])
    expected = {
        "lifepo4": {"Li": 1, "Fe": 1, "P": 1, "O": 4},
        "phosphorene": {"P": 1},
        "zn_formate_mof": {"Zn": 1, "C": 2, "H": 2, "O": 4},
    }
    if material in expected and composition != expected[material]:
        raise InvalidActionError(f"unsupported composition: {action['formula']}")
    if (
        material in generation._ALLOY_ELEMENTS
        and not set(composition) <= generation._ALLOY_ELEMENTS[material]
    ):
        raise InvalidActionError(f"unsupported alloy composition: {action['formula']}")
    source = _source_available(material, action["family"], tier)
    prototypes = []
    if material == "nb_ta_o":
        formula = action["formula"]
        if action.get("defect") == "oxygen_defect":
            formula = {"Nb2O4": "Nb2O5", "Ta2O4": "Ta2O5"}.get(formula, "")
        if formula not in generation._OXIDE_FORMULAS:
            raise InvalidActionError(f"unsupported oxide composition: {action['formula']}")
        if method.startswith("aflow"):
            inventory = _aflow_inventory(formula, tier)
            prototypes = [
                {"prototype_id": prototype, "space_group": group}
                for prototype, group in inventory
                if (not action.get("space_groups") or group in action["space_groups"])
                and (not action.get("prototype_ids") or prototype in action["prototype_ids"])
            ]
            if not prototypes:
                raise InvalidActionError(
                    f"no installed AFLOW prototypes for {formula}, tier {tier}, "
                    f"symmetry {action.get('space_group_family')}"
                )
            source = "installed AFLOW prototype encyclopedia"
        elif importlib.util.find_spec("pyxtal") is None:
            raise InvalidActionError("PyXtal generator is not installed")
        else:
            if action.get("prototype_ids"):
                raise InvalidActionError("PyXtal does not accept AFLOW prototype ID filters")
            source = "installed PyXtal; seeded generation must pass the CPU preflight"
    singleton = capacity_group(action) is not None
    maximum = 1 if singleton else 192
    if method == "aflow_prototype" and action.get("strain") is None:
        maximum = len(prototypes)
    if material == "lifepo4" and action["family"] == "antisite_pair":
        maximum = _antisite_capacity(tier)
    constraints = {
        "strain_fraction": action.get("strain"),
        "defect": action.get("defect"),
        "deduplication": "symmetry-aware-ltol0.01-stol0.01-angle1-scale_false",
    }
    if material == "lifepo4":
        constraints["realized_operation"] = (
            "Li-Fe antisite pair"
            if action.get("defect") == "antisite_pair"
            else "bounded strained olivine with local distortions"
            if action.get("strain") is not None
            else "pristine parent (operation labels are aliases)"
        )
        if action["family"] == "olivine_strain":
            constraints["maximum_local_displacement_A"] = 0.04
    if material == "phosphorene":
        constraints.update(
            fixed_vacuum=True, reconstruction_displacement_A=0.08,
            reconstruction_modes="collective in-plane displacements, no net translation",
        )
    if material == "zn_formate_mof":
        constraints.update(
            preserve_connectivity=True, maximum_proton_rotation_degrees=15,
            proton_orientation_sites="all formate hydrogens, fixed C-H lengths",
        )
    return {
        "catalog_version": CATALOG_VERSION,
        "generator_available": True,
        "generation_method": method,
        "source": source,
        "available_prototypes": prototypes,
        "capacity_group": _capacity_key(action),
        "maximum_distinct_configurations": maximum,
        "capacity_basis": (
            "conservative operational capacity under frozen perturbation and deduplication"
            if singleton
            else "bounded acquisition ceiling; actual seeded batch must pass preflight"
        ),
        "perturbation_constraints": constraints,
    }


def executable_catalog(
    actions: list[dict[str, Any]], history: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available, unavailable = [], []
    used: dict[str, int] = defaultdict(int)
    exhausted = set()
    by_id = {action["action_id"]: action for action in actions}
    for row in history:
        action = by_id.get(row.get("action_id"), row.get("action", {}))
        if not action:
            continue
        key = _capacity_key(action)
        used[key] += int(row.get("accepted_unique", row.get("attempted", row.get("count", 0))))
        if row.get("capacity_exhausted"):
            exhausted.add(key)
    for original in actions:
        action = dict(original)
        try:
            constraints = action_constraints(action)
        except (InvalidActionError, generation.StructureGenerationError) as error:
            unavailable.append(
                {
                    "action": original,
                    "action_id": action["action_id"],
                    "material": action["material"],
                    "size_tier": action["size_tier"],
                    "generation_method": action["generator"],
                    "capacity_group": _capacity_key(action),
                    "maximum_distinct_configurations": 0,
                    "remaining_capacity": 0,
                    "generator_available": False,
                    "reason": str(error),
                    "category": "invalid_action",
                }
            )
            continue
        key = constraints["capacity_group"]
        constraints["remaining_capacity"] = (
            0
            if key in exhausted
            else max(0, constraints["maximum_distinct_configurations"] - used[key])
        )
        action["feasibility"] = constraints
        if constraints["remaining_capacity"]:
            available.append(action)
        else:
            unavailable.append(
                {
                    "action": original,
                    "action_id": action["action_id"],
                    "reason": f"capacity exhausted: {key}",
                    "category": "capacity_exhausted",
                    **constraints,
                }
            )
    return available, unavailable


def assert_capacity(
    actions: list[dict[str, Any]],
    target: int,
    minimum_by_tier: dict[str, int],
) -> None:
    groups = {}
    for action in actions:
        info = action["feasibility"]
        groups[info["capacity_group"]] = (str(action["size_tier"]), info["remaining_capacity"])
    if sum(count for _, count in groups.values()) < target:
        raise ScientificCapacityError(
            f"remaining shared action capacities cannot satisfy the round target {target}"
        )
    for tier, minimum in minimum_by_tier.items():
        remaining = sum(count for size, count in groups.values() if size == tier)
        if remaining < minimum:
            raise ScientificCapacityError(
                f"tier {tier} requires {minimum} distinct candidates but only {remaining} remain"
            )


def validate_allocations(
    allocations: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    target: int,
    minimum_by_tier: dict[str, int],
    *,
    unavailable: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    allowed = {action["action_id"]: action for action in actions}
    rejected = {action["action_id"]: action for action in unavailable or []}
    seen = set()
    used: dict[str, int] = defaultdict(int)
    tiers: dict[str, int] = defaultdict(int)
    selected = []
    for row in allocations:
        if not isinstance(row, dict) or set(row) != {"action_id", "count"}:
            raise InvalidActionError("each allocation requires exactly action_id and count")
        aid, count = row["action_id"], row["count"]
        if not isinstance(aid, str) or aid not in allowed or aid in seen:
            reason = rejected.get(aid, {}) if isinstance(aid, str) else {}
            message = f"unavailable or repeated action ID: {aid!r}"
            if reason:
                message += f": {reason['reason']}"
            if reason.get("category") == "capacity_exhausted":
                raise ScientificCapacityError(message)
            raise InvalidActionError(message)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise InvalidActionError(f"invalid count for {aid}: {count!r}")
        seen.add(aid)
        action = allowed[aid]
        info = action["feasibility"]
        group = info["capacity_group"]
        used[group] += count
        if used[group] > info["remaining_capacity"]:
            raise ScientificCapacityError(
                f"finite generation-action capacity exceeded by {aid}: shared group {group}, "
                f"requested {used[group]}, remaining {info['remaining_capacity']}"
            )
        tiers[str(action["size_tier"])] += count
        selected.append({"action": action, "count": count})
    if sum(tiers.values()) != target:
        raise InvalidActionError(f"allocation totals {sum(tiers.values())}, expected {target}")
    if any(tiers[tier] < count for tier, count in minimum_by_tier.items()):
        raise InvalidActionError("allocation does not meet minimum size-tier coverage")
    return selected
