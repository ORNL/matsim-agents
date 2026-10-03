#!/usr/bin/env python3
"""Create transparent bootstrap structures for Nb-Ta-O hull references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ase.build import bulk, molecule
from ase.io import write

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matsim_agents.discovery.composition import parse_composition  # noqa: E402
from matsim_agents.discovery.seeds import generate_seeds  # noqa: E402

DEFAULT_COMPETING_FORMULAS = [
    "NbTa",
    "NbO",
    "NbO2",
    "Nb2O3",
    "Nb2O5",
    "TaO",
    "TaO2",
    "Ta2O3",
    "Ta2O5",
    "NbTaO4",
    "NbTaO5",
]


def _curated_phases(path: Path) -> dict[str, dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    phases = raw.get("phases", raw) if isinstance(raw, dict) else None
    if not isinstance(phases, dict):
        raise ValueError("curated reference manifest must contain a phase object")
    normalized: dict[str, dict[str, object]] = {}
    for phase_id, raw_spec in phases.items():
        if not isinstance(raw_spec, dict) or "path" not in raw_spec:
            raise ValueError(f"curated phase {phase_id!r} must contain a path")
        spec = dict(raw_spec)
        structure_path = Path(str(spec["path"])).expanduser()
        if not structure_path.is_absolute():
            structure_path = (path.parent / structure_path).resolve()
        if not structure_path.is_file():
            raise ValueError(f"curated phase structure does not exist: {structure_path}")
        spec["path"] = str(structure_path)
        spec.setdefault("phase_id", str(phase_id))
        spec.setdefault("formula", str(phase_id))
        spec.setdefault("source", "curated_manifest")
        normalized[str(phase_id)] = spec
    return normalized


def _deduplicate_unary_phases(
    phases: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Deduplicate unary structures while preserving curated-first ordering."""
    from ase.io import read
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Composition, Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    matcher = StructureMatcher(primitive_cell=True, attempt_supercell=True)
    retained: dict[str, dict[str, object]] = {}
    representatives: dict[str, list[Structure]] = {}
    duplicates: list[str] = []
    for phase_id, spec in phases.items():
        composition = Composition(str(spec["formula"]))
        if len(composition.elements) != 1:
            retained[phase_id] = spec
            continue
        element = str(composition.elements[0])
        structure = AseAtomsAdaptor.get_structure(read(str(spec["path"])))
        if any(matcher.fit(existing, structure) for existing in representatives.get(element, [])):
            duplicates.append(phase_id)
            continue
        representatives.setdefault(element, []).append(structure)
        retained[phase_id] = spec
    return retained, duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend", choices=("qe", "vasp"), default="qe")
    parser.add_argument(
        "--competing-formulas",
        nargs="*",
        default=DEFAULT_COMPETING_FORMULAS,
        help="Formulas for which AFLOW prototype references are generated.",
    )
    parser.add_argument("--max-prototypes-per-formula", type=int, default=1)
    parser.add_argument(
        "--expand-unary-polymorphs",
        action="store_true",
        help="Add all compatible AFLOW unary prototypes and pyXtal unary candidates.",
    )
    parser.add_argument("--unary-random", type=int, default=0)
    parser.add_argument("--unary-random-seed", type=int, default=20261001)
    parser.add_argument("--curated-manifest", type=Path)
    parser.add_argument("--oxygen-correction-eV-per-atom", type=float, default=0.0)
    args = parser.parse_args()
    if args.max_prototypes_per_formula < 1:
        parser.error("--max-prototypes-per-formula must be positive")
    if args.unary_random < 0:
        parser.error("--unary-random must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    structures = {
        "Nb": bulk("Nb", "bcc", a=3.300),
        "Ta": bulk("Ta", "bcc", a=3.306),
        "O2": molecule("O2"),
    }
    structures["O2"].set_cell([15.0, 15.0, 15.0])
    structures["O2"].center()
    structures["O2"].set_pbc(True)

    phases: dict[str, dict[str, object]] = {}
    for formula, atoms in structures.items():
        path = (args.output_dir / f"{formula}.extxyz").resolve()
        write(path, atoms)
        phases[formula] = {
            "phase_id": formula,
            "formula": formula,
            "path": str(path),
            "relax_cell": formula != "O2",
            "source": "ase_bootstrap",
            "provenance": {"generator": "ase.build"},
        }
    if args.backend == "qe":
        phases["O2"]["settings"] = {
            "kpts": [1, 1, 1],
            "koffset": [0, 0, 0],
            "extra_system": {"nspin": 2, "starting_magnetization(1)": 1.0},
        }
    else:
        phases["O2"]["settings"] = {
            "kspacing": 10.0,
            "kgamma": True,
            "ispin": 2,
            "extra_incar": {"MAGMOM": "2*1.0"},
        }
    phases["O2"]["energy_correction_eV_per_atom"] = args.oxygen_correction_eV_per_atom

    if args.expand_unary_polymorphs:
        unary_root = args.output_dir / "unary_candidates"
        for element_index, element in enumerate(("Nb", "Ta", "O")):
            composition = parse_composition(element)
            if composition is None:  # pragma: no cover - fixed valid symbols
                raise ValueError(f"could not parse unary element {element!r}")
            candidates = generate_seeds(
                composition,
                str(unary_root / element),
                n_random=args.unary_random,
                fmt="extxyz",
                random_seed=args.unary_random_seed + element_index,
            )
            for index, candidate in enumerate(candidates):
                source_label = "aflow" if candidate.source == "prototype" else "pyxtal"
                phase_id = f"{element}-{source_label}-{index:04d}"
                phases[phase_id] = {
                    "phase_id": phase_id,
                    "formula": element,
                    "path": str(Path(candidate.structure_path).resolve()),
                    "relax_cell": True,
                    "source": f"{source_label}_unary",
                    "provenance": {
                        "candidate_id": candidate.candidate_id or "unknown",
                        "prototype_id": candidate.prototype_id or "none",
                        "space_group": str(candidate.space_group or "unknown"),
                        "random_seed": str(candidate.random_seed or "none"),
                    },
                }

    phases, duplicate_unary_phases = _deduplicate_unary_phases(phases)

    prototype_root = args.output_dir / "competing_phases"
    missing_formulas: list[str] = []
    for formula in args.competing_formulas:
        composition = parse_composition(formula)
        if composition is None:
            raise ValueError(f"could not parse competing formula {formula!r}")
        candidates = generate_seeds(
            composition,
            str(prototype_root / formula),
            n_random=0,
            fmt="extxyz",
        )
        prototypes = sorted(
            (candidate for candidate in candidates if candidate.source == "prototype"),
            key=lambda candidate: (
                candidate.prototype_id or "",
                candidate.decoration_mapping or "",
                candidate.structure_path,
            ),
        )[: args.max_prototypes_per_formula]
        if not prototypes:
            missing_formulas.append(formula)
            continue
        for index, candidate in enumerate(prototypes):
            phase_id = f"{formula}-aflow-{index:02d}"
            phases[phase_id] = {
                "phase_id": phase_id,
                "formula": formula,
                "path": str(Path(candidate.structure_path).resolve()),
                "relax_cell": True,
                "source": "aflow_prototype",
                "provenance": {
                    "prototype_id": candidate.prototype_id or "unknown",
                    "space_group": str(candidate.space_group or "unknown"),
                    "decoration_mapping": candidate.decoration_mapping or "unknown",
                },
            }

    if args.curated_manifest is not None:
        phases.update(_curated_phases(args.curated_manifest.resolve()))

    manifest = {
        "schema_version": 2,
        "phases": phases,
        "completeness": {
            "required_formulas": list(args.competing_formulas),
            "require_binary_subsystems": True,
            "require_ternary_competitor": True,
        },
        "generation": {
            "missing_aflow_formulas": missing_formulas,
            "max_prototypes_per_formula": args.max_prototypes_per_formula,
            "oxygen_correction_eV_per_atom": args.oxygen_correction_eV_per_atom,
            "expand_unary_polymorphs": args.expand_unary_polymorphs,
            "unary_random": args.unary_random,
            "unary_random_seed": args.unary_random_seed,
            "duplicate_unary_phases_removed": duplicate_unary_phases,
        },
    }
    manifest_path = args.output_dir / "reference_structures.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())