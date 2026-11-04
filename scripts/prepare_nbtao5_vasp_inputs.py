from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from nbtao5_kpoints import sha256, write_kpoints
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "research/nb-ta-o/pilot"
OUTPUT_DIR = PILOT_DIR / "vasp_inputs"
COMPONENTS = {
    "Nb_pv": "086061c276946e9fc55bb7633d9ce78b32a420dec762592491c07c1c47ded114",
    "Ta_pv": "ffb05e92ac3f68ca84c26af10a80f2fbb0cf79e16fcc96fcae98089455bbbaaa",
    "O": "8a74b9a1f5fdb3d0c3e0183c7873177abdbef07d407b310b7edcd9ed0a3eea64",
}
PROTOCOL_ID = "nbtao5-fixed-kmesh-v1"
RUNTIME_MANIFEST = json.loads((PILOT_DIR / "vasp_runtime_manifest.json").read_text())
PAW_ENMAX_EV = {
    name: float(record["enmax_eV"])
    for name, record in RUNTIME_MANIFEST["proposed_components"].items()
}
MAX_ENMAX_EV = max(PAW_ENMAX_EV.values())
ENCUT_EV = math.ceil(max(600.0, 1.3 * MAX_ENMAX_EV))
BASELINE = {
    "encut_eV": int(ENCUT_EV),
    "kpoint_spacing_A^-1": 0.15,
    "ediff_eV": 1e-6,
    "ediffg_eV_A": -0.01,
}


def incar_text(
    case_id: str,
    species: tuple[str, ...],
    settings: dict[str, float | int],
    relax: bool,
) -> str:
    metallic = case_id.startswith(("Nb-bcc", "Ta-bcc"))
    oxygen_molecule = case_id.startswith("O2-")
    tags = [
        f"SYSTEM = Nb-Ta-O {case_id} {'relax' if relax else 'static'}",
        "PREC = Accurate",
        f"ENCUT = {int(settings['encut_eV'])}",
        f"EDIFF = {settings['ediff_eV']:.1E}",
        "ALGO = Normal",
        "NELM = 200",
        "ISPIN = 2" if oxygen_molecule else "ISPIN = 1",
        "ISMEAR = 1" if metallic else "ISMEAR = 0",
        "SIGMA = 0.20" if metallic else "SIGMA = 0.05",
        "LREAL = .FALSE.",
        "LASPH = .TRUE.",
        "ISYM = 2" if metallic else "ISYM = 0",
        "LWAVE = .FALSE.",
        "LCHARG = .FALSE.",
    ]
    if oxygen_molecule:
        tags.append("MAGMOM = 2*1.0")
    if relax:
        tags.extend(
            [
                f"EDIFFG = {settings['ediffg_eV_A']:.3f}",
                "NSW = 100" if oxygen_molecule else "NSW = 200",
                "IBRION = 2",
                "ISIF = 2" if oxygen_molecule else "ISIF = 7" if metallic else "ISIF = 3",
            ]
        )
    else:
        tags.extend(["NSW = 0", "IBRION = -1"])
    if not species:
        raise ValueError(f"No species listed for {case_id}")
    return "\n".join(tags) + "\n"


def potcar_order(species: tuple[str, ...]) -> str:
    component_by_species = {"Nb": "Nb_pv", "Ta": "Ta_pv", "O": "O"}
    lines = []
    for symbol in species:
        component = component_by_species[symbol]
        lines.append(f"{component} {COMPONENTS[component]}")
    return "\n".join(
        [
            "# element-order follows POSCAR; source files are never copied into this repository",
            *lines,
            "",
        ]
    )


def stage_case(
    directory: Path,
    case_id: str,
    structure_path: Path,
    species: tuple[str, ...],
    settings: dict[str, float | int],
    category: str,
    gamma_only: bool = False,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    poscar_path = directory / "POSCAR"
    shutil.copyfile(structure_path, poscar_path)
    structure = Structure.from_file(poscar_path)
    actual_species = tuple(str(item) for item in structure.composition.elements)
    if set(actual_species) != set(species):
        raise ValueError(f"Species mismatch for {case_id}: {actual_species} != {species}")
    kpoint_record = write_kpoints(
        poscar_path,
        directory / "KPOINTS",
        directory / "KPOINTS.json",
        None if gamma_only else float(settings["kpoint_spacing_A^-1"]),
        gamma_only=gamma_only,
    )
    (directory / "INCAR").write_text(
        incar_text(case_id, species, settings, relax=True), encoding="ascii"
    )
    (directory / "INCAR.static").write_text(
        incar_text(case_id, species, settings, relax=False), encoding="ascii"
    )
    (directory / "POTCAR.order").write_text(potcar_order(species), encoding="ascii")
    input_hashes = {
        name: sha256(directory / name)
        for name in (
            "POSCAR",
            "INCAR",
            "INCAR.static",
            "KPOINTS",
            "KPOINTS.json",
            "POTCAR.order",
        )
    }
    record = {
        "protocol_id": PROTOCOL_ID,
        "case_id": case_id,
        "category": category,
        "status": "input_candidate_not_authorized_for_execution",
        "species_order": list(species),
        "species_counts": {
            str(element): int(amount)
            for element, amount in structure.composition.get_el_amt_dict().items()
        },
        "settings": settings,
        "gamma_only": gamma_only,
        "kpoints": kpoint_record,
        "acceptance_criteria": {
            "electronic_convergence_required": True,
            "ionic_convergence_required_for_relaxation": True,
            "max_force_eV_A": 0.01,
            "max_abs_principal_stress_kbar": None if gamma_only else 2.0,
        },
        "input_hashes": input_hashes,
        "potcar_required_components": [
            line.split()[0]
            for line in potcar_order(species).splitlines()
            if line and not line.startswith("#")
        ],
    }
    (directory / "case.json").write_text(json.dumps(record, indent=2) + "\n", encoding="ascii")
    return record


def crystal_cases(phase_id: str) -> list[tuple[str, dict[str, float | int]]]:
    del phase_id
    return [("baseline", dict(BASELINE))]


def oxygen_structure(box_A: float) -> Structure:
    path = PILOT_DIR / "elemental_candidates/O2-triplet.vasp"
    source = Structure.from_file(path)
    source_box = source.lattice.a
    cart_coords = source.cart_coords + (box_A - source_box) / 2.0
    return Structure(
        Lattice.cubic(box_A),
        [site.species for site in source],
        cart_coords,
        coords_are_cartesian=True,
    )


def main() -> None:
    ordering_manifest = json.loads((PILOT_DIR / "ordering_manifest.json").read_text())
    elemental_manifest = json.loads((PILOT_DIR / "elemental_reference_inventory.json").read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    selected = sorted(
        (entry for entry in ordering_manifest["orderings"] if entry["pilot_selected"]),
        key=lambda entry: entry["pilot_rank"],
    )
    for entry in selected:
        source = ROOT / entry["poscar"]["path"]
        settings = dict(BASELINE)
        record = stage_case(
            OUTPUT_DIR / "production-candidates" / entry["ordering_id"],
            entry["ordering_id"],
            source,
            ("Nb", "Ta", "O"),
            settings,
            "production_candidate",
        )
        record["qualification_gate"] = "blocked_until_protocol_qualified_and_explicit_authorization"
        case_json = OUTPUT_DIR / "production-candidates" / entry["ordering_id"] / "case.json"
        case_record = json.loads(case_json.read_text())
        case_record["qualification_gate"] = record["qualification_gate"]
        case_json.write_text(json.dumps(case_record, indent=2) + "\n", encoding="ascii")
        cases.append(record)

    selected_ids = {entry["ordering_id"] for entry in selected}
    for entry in ordering_manifest["orderings"]:
        if entry["ordering_id"] in selected_ids:
            continue
        record = stage_case(
            OUTPUT_DIR / "ordering-family-pending-approval" / entry["ordering_id"],
            entry["ordering_id"],
            ROOT / entry["poscar"]["path"],
            ("Nb", "Ta", "O"),
            dict(BASELINE),
            "ordering_family_extension_pending_approval",
        )
        record["qualification_gate"] = (
            "blocked_until_structure_review_numerical_qualification_and_explicit_authorization"
        )
        case_json = (
            OUTPUT_DIR / "ordering-family-pending-approval" / entry["ordering_id"] / "case.json"
        )
        case_record = json.loads(case_json.read_text())
        case_record["qualification_gate"] = record["qualification_gate"]
        case_json.write_text(json.dumps(case_record, indent=2) + "\n", encoding="ascii")
        cases.append(record)

    reference_specs = (
        ("Nb-bcc", ("Nb",)),
        ("Ta-bcc", ("Ta",)),
        ("NbTaO5-ordering-01", ("Nb", "Ta", "O")),
    )
    for phase_id, species in reference_specs:
        if phase_id == "NbTaO5-ordering-01":
            source = ROOT / next(
                entry["poscar"]["path"]
                for entry in ordering_manifest["orderings"]
                if entry["ordering_id"] == "NbTaO5_ordering_01"
            )
        else:
            source = ROOT / elemental_manifest["phases"][phase_id]["structure_path"]
        for label, settings in crystal_cases(phase_id):
            case_id = f"{phase_id}-{label}"
            cases.append(
                stage_case(
                    OUTPUT_DIR / "qualification" / phase_id / label,
                    case_id,
                    source,
                    species,
                    settings,
                    "crystalline_numerical_qualification",
                )
            )

    for label, box_A in [("baseline", 15.0)]:
        settings = dict(BASELINE)
        source_dir = OUTPUT_DIR / "qualification" / "O2-triplet" / label
        source_dir.mkdir(parents=True, exist_ok=True)
        oxygen_path = source_dir / "O2.POSCAR"

        Poscar(
            oxygen_structure(box_A), comment=f"O2 triplet candidate; {box_A:g} A box"
        ).write_file(oxygen_path)
        oxygen_settings = {
            key: value for key, value in settings.items() if key != "kpoint_spacing_A^-1"
        }
        cases.append(
            stage_case(
                source_dir,
                f"O2-triplet-{label}",
                oxygen_path,
                ("O",),
                oxygen_settings,
                "oxygen_reference_qualification",
                gamma_only=True,
            )
        )

    smoke_case_id = "NbTaO5-ordering-01-baseline"
    reference_case_ids = ["Nb-bcc-baseline", "Ta-bcc-baseline"]
    oxygen_case_ids = [
        case["case_id"] for case in cases if case["category"] == "oxygen_reference_qualification"
    ]

    def qualification_stage(
        stage_id: str,
        workflow_ids: list[str],
        depends_on: list[str],
        pass_conditions: list[str],
    ) -> dict[str, object]:
        return {
            "stage_id": stage_id,
            "workflow_ids": workflow_ids,
            "depends_on": depends_on,
            "max_nodes_per_workflow": 1,
            "max_walltime_per_workflow_minutes": 120,
            "max_concurrent_workflows": 1,
            "node_hours_cap": len(workflow_ids) * 2.0,
            "pass_conditions": pass_conditions,
            "progression_requires_review_and_explicit_stage_authorization": True,
        }

    stages = [
        qualification_stage(
            "Q1-runtime-and-workflow-smoke",
            [smoke_case_id],
            [],
            [
                "Approved executable and homogeneous PAW component hashes "
                "match the frozen manifest.",
                "Relaxation and static VASP runs complete with electronic convergence.",
                "The relaxation terminates normally and reaches its requested ionic "
                "force criterion.",
                "Extracted energies, forces, stress, outputs, and provenance are "
                "internally consistent.",
            ],
        ),
        qualification_stage(
            "Q2-elemental-baselines",
            reference_case_ids,
            ["Q1-runtime-and-workflow-smoke"],
            [
                "Both Nb and Ta baseline workflows complete with electronic and ionic convergence.",
                "Final forces and residual stress pass the frozen crystalline criteria.",
            ],
        ),
        qualification_stage(
            "Q3-oxygen-reference",
            oxygen_case_ids,
            [
                "Q1-runtime-and-workflow-smoke",
                "Q2-elemental-baselines",
            ],
            [
                "O2 remains triplet and passes electronic and ionic-force criteria "
                "in its fixed cell.",
            ],
        ),
    ]

    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "inputs_staged_execution_not_authorized",
        "method": {
            "encut_rule": "max(600 eV, 1.3 * maximum selected PAW ENMAX)",
            "maximum_paw_enmax_eV": MAX_ENMAX_EV,
            "paw_enmax_eV": PAW_ENMAX_EV,
            "encut_eV": BASELINE["encut_eV"],
            "prec": "Accurate",
            "ediff_eV": BASELINE["ediff_eV"],
            "ediffg_eV_A": BASELINE["ediffg_eV_A"],
            "periodic_kpoint_policy": (
                "Explicit Gamma-centered regular meshes; target reciprocal spacing 0.15 A^-1."
            ),
            "periodic_kpoint_mesh_rule": (
                "N_i=max(1,ceil(|B_i|/spacing)) using 2*pi reciprocal-vector lengths; "
                "the ceiling keeps each interval at or below 0.15 A^-1. This explicit-mesh "
                "rule does not attempt to reproduce VASP KSPACING integer rounding."
            ),
            "static_kpoint_policy": (
                "Recompute the explicit mesh from the relaxed CONTCAR cell "
                "before static calculation."
            ),
            "molecular_kpoint_policy": (
                "O2 uses a separate Gamma-only 1x1x1 KPOINTS in a fixed 15 A cell."
            ),
            "forbidden_incar_tags": ["KSPACING", "KGAMMA"],
            "convergence_sweeps": (
                "Removed; fixed settings are not represented as independently convergence-tested."
            ),
        },
        "resource_limit": {
            "platform": "Aurora",
            "max_nodes_per_workflow": 1,
            "max_walltime_per_workflow_minutes": 120,
            "max_concurrent_workflows": 1,
            "qualification_workflows": 4,
            "qualification_max_node_hours": 8.0,
            "walltime_scope": (
                "120 minutes per PBS workflow, including its relaxation and static VASP invocations"
            ),
            "walltime_basis": (
                "Raised from 30 min: job 9039526 ran 18 ionic steps (~80 s each, 28-atom "
                "NbTaO5, 1 node) without converging before its 25-min soft stop."
            ),
            "queue": "capacity",
            "soft_stop_seconds_before_walltime": 300,
            "pilot_ordering_workflows": 5,
            "pilot_ordering_max_node_hours": 10.0,
        },
        "qualification_case_counts": {
            "Nb_bcc": 1,
            "Ta_bcc": 1,
            "NbTaO5_ordering_01": 1,
            "O2_triplet": 1,
            "total_workflows": 4,
            "vasp_invocations_if_all_complete": 8,
        },
        "qualification_stages": stages,
        "production_pilot_candidates": [entry["ordering_id"] for entry in selected],
        "additional_orderings_prepared_pending_approval": [
            entry["ordering_id"]
            for entry in ordering_manifest["orderings"]
            if entry["ordering_id"] not in selected_ids
        ],
        "cases": cases,
        "potcar_assembly": (
            "Use deployments/aurora/jobs/assemble-nbtao5-potcar.sh in a run directory "
            "outside the Git worktree after PAW approval. No POTCAR is staged here."
        ),
    }
    path = OUTPUT_DIR / "inputs_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(
        f"Staged {len(cases)} input workflows "
        f"({manifest['qualification_case_counts']['total_workflows']} qualification); "
        "no POTCAR, VASP, or scheduler action."
    )
    print(f"Manifest: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
