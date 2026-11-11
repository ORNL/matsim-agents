"""Stage scientifically qualified, unsubmitted E4 inputs for the six-class scaling plan.

Restart-ready case directories (including assembled POTCARs) are written outside
the repository; source POSCARs and a provenance manifest are written to
``research/paper/e4/aurora-dft-scaling/six-class-inputs-v1``.  Nothing here
submits jobs or authorizes execution.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.spacegroup import crystal

REPO = Path(__file__).resolve().parents[3]
RESEARCH_ROOT = REPO / "research/paper/e4/aurora-dft-scaling/six-class-inputs-v1"
MANIFEST = RESEARCH_ROOT / "manifest.json"
STRUCTURES = RESEARCH_ROOT / "structures"
DEFAULT_INPUT_ROOT = REPO.parent / "e4-six-class-inputs-v1"
PAW_ROOT = REPO / "external/vasp6/potcar"
FIRST_BATCH = REPO / "deployments/aurora/jobs/e4-first-batch/batch-manifest-hydragnn.json"
STAGE1 = REPO / "research/paper/e4/aurora-dft-scaling/stage1-inputs-v2"
LFP_INVENTORY = REPO / "research/paper/e4/aurora-dft-scaling/lifepo4-inventory-v1"
NBTAO_ROOT = REPO / "research/nb-ta-o/pilot"
MOF_ARCHIVE = REPO.parent / "e2-dft-znformate-anhydrous-v1-20261104"
NBTAO_E2_RUN = REPO.parent / "nbtao5-q1-fixed-kmesh-v1-20261104TJHFN69"
MOF_E2_POTCAR_SHA256 = "f8ebdae2fad119cadaf23c83ac1dcea9f9161361c642606c2baf02980bec9dd7"

PROTOCOLS = {
    "nb_ta_o": "nbtao5-fixed-kmesh-v1",
    "zn_formate_mof": "znformate-pbe-d3bj-fixed-volume-eos-v1",
    "phosphorene": "phosphorene-pbe-2d-inplane-relax-v1",
    "fcc_hea": "fcc-pbe-verified-v1",
    "lifepo4": "lifepo4-pbe-verified-v1",
}
TASK_WALLTIME_HOURS = {
    "nb_ta_o": 5,
    "zn_formate_mof": 12,
    "phosphorene": 12,
    "fcc_hea": 24,
    "lifepo4": 24,
}
NBTAO_PRODUCTION = ("01", "09", "14", "12", "06")
NBTAO_EXTENSION = ("02", "03", "04", "05", "07", "08", "10", "11", "13")
MOF_FRAMEWORKS = {
    "ALPHA": "zn_formate_alpha_Zn3HCOO6_COD4515120",
    "PBCN": "zn_formate_lonsdaleite_Pbcn_COD7205949",
}
# Interleaved so each weak-scaling node receives one alpha and one Pbcn task and
# the first eight (strong fixed workload) cover both frameworks symmetrically.
MOF_STRAINS = (-0.02, -0.01, 0.01, 0.02, -0.015, -0.005, 0.005, 0.015)
PHOS_SUPERCELLS = ((3, 3), (3, 4), (4, 4), (4, 5), (2, 3))
FCC_NEW_REALIZATIONS = (2, 3, 4)
LFP_EXTENSION = {
    "E4-LFP-0004": 1,
    "E4-LFP-0005": 1,
    "E4-LFP-0150": 2,
    "E4-LFP-0200": 2,
    "E4-LFP-0250": 2,
    "E4-LFP-0300": 2,
    "E4-LFP-0507": 3,
    "E4-LFP-0508": 3,
}
# Brown & Rundqvist, Acta Cryst. 19, 684 (1965): black P, Cmce (No. 64), 8f site.
BLACK_P = {"a": 3.3136, "b": 10.478, "c": 4.3763, "y": 0.10168, "z": 0.08056}
PHOS_VACUUM_CELL_A = 20.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def rel(path: Path) -> str:
    return os.path.relpath(Path(path).resolve(), REPO)


def incar_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        active = re.split(r"[#!]", line, maxsplit=1)[0]
        for field in active.split(";"):
            if "=" in field:
                key, value = field.split("=", 1)
                values[key.strip().upper()] = value.strip()
    return values


def check_start(
    start: str, ranks: int, expected: dict[str, str], precision: str = "Accurate"
) -> None:
    """Mirror the per-calculation wrapper's INCAR preconditions."""
    if not re.search(rf"(?m)^PREC *= *{precision}$", start):
        raise ValueError(f"START must contain an exact 'PREC = {precision}' line")
    values = incar_values(start)
    kpar = values.get("KPAR", "")
    if not kpar.isdigit() or ranks % int(kpar):
        raise ValueError(f"KPAR={kpar or 'missing'} incompatible with {ranks} MPI ranks")
    ncore = values.get("NCORE")
    if ncore is not None and (not ncore.isdigit() or (ranks // int(kpar)) % int(ncore)):
        raise ValueError(f"NCORE={ncore} incompatible with {ranks} ranks and KPAR={kpar}")
    for key, wanted in expected.items():
        actual = values.get(key)
        if actual is None:
            raise ValueError(f"START is missing {key}")
        try:
            same = float(actual.replace("D", "E")) == float(wanted)
        except ValueError:
            same = actual.upper() == wanted.upper()
        if not same:
            raise ValueError(f"START {key}={actual}; expected {wanted}")


def poscar_species(text: str) -> list[tuple[str, int]]:
    lines = text.splitlines()
    return list(zip(lines[5].split(), map(int, lines[6].split()), strict=True))


def poscar_text(atoms: Atoms, label: str) -> str:
    """VASP5 direct POSCAR preserving the given species grouping."""
    symbols = atoms.get_chemical_symbols()
    order = list(dict.fromkeys(symbols))
    index = sorted(range(len(atoms)), key=lambda i: order.index(symbols[i]))
    ordered = atoms[index]
    counts = [symbols.count(symbol) for symbol in order]
    lines = [label, "1.0"]
    lines += ["  " + " ".join(f"{value:21.16f}" for value in row) for row in ordered.cell]
    lines += [" ".join(order), " ".join(map(str, counts)), "Direct"]
    lines += [
        " ".join(f"{value:19.16f}" for value in row)
        for row in np.mod(ordered.get_scaled_positions(wrap=False), 1.0)
    ]
    return "\n".join(lines) + "\n"


def paw_components(species: list[tuple[str, int]], labels: dict[str, str]):
    components, payload = [], bytearray()
    for symbol, _ in species:
        label = labels[symbol]
        path = PAW_ROOT / label / "POTCAR"
        data = path.read_bytes()
        title = re.search(rb"(?m)^\s*TITEL\s*=\s*PAW_PBE\s+(\S+)", data)
        if title is None or title.group(1).decode() != label:
            raise ValueError(f"PAW identity mismatch for {symbol}: {path}")
        components.append(
            {
                "element": symbol,
                "label": label,
                "sha256": sha256_bytes(data),
                "source": f"external/vasp6/potcar/{label}/POTCAR",
            }
        )
        payload.extend(data)
    return components, bytes(payload)


def bonded_graph_components(atoms: Atoms, cutoffs: dict[tuple[str, str], float]) -> int:
    i, j, d = neighbor_list("ijd", atoms, max(cutoffs.values()))
    symbols = atoms.get_chemical_symbols()
    parent = list(range(len(atoms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, dist in zip(i, j, d, strict=True):
        key = tuple(sorted((symbols[a], symbols[b])))
        if dist <= cutoffs.get(key, 0.0):
            parent[find(a)] = find(b)
    return len({find(x) for x in range(len(atoms))})


def validate_zn_formate(atoms: Atoms) -> dict[str, Any]:
    """Chemical validity of an anhydrous Zn(HCOO)2 framework."""
    symbols = np.array(atoms.get_chemical_symbols())
    counts = {s: int((symbols == s).sum()) for s in ("Zn", "C", "H", "O")}
    if set(symbols) != {"Zn", "C", "H", "O"}:
        raise ValueError("Zn-formate contains unexpected species")
    zn = counts["Zn"]
    if counts != {"Zn": zn, "C": 2 * zn, "H": 2 * zn, "O": 4 * zn}:
        raise ValueError(f"composition {counts} is not Zn(HCOO)2")
    i, j, d = neighbor_list("ijd", atoms, 2.6)
    if d.min() < 0.9:
        raise ValueError(f"unphysical interatomic distance {d.min():.3f} A")

    def neighbours(center: str, other: str, cutoff: float) -> np.ndarray:
        mask = (symbols[i] == center) & (symbols[j] == other) & (d <= cutoff)
        return np.bincount(i[mask], minlength=len(atoms))[symbols == center]

    zn_o = neighbours("Zn", "O", 2.45)
    c_o, c_h, h_c, o_c = (
        neighbours("C", "O", 1.40),
        neighbours("C", "H", 1.20),
        neighbours("H", "C", 1.20),
        neighbours("O", "C", 1.40),
    )
    o_zn = neighbours("O", "Zn", 2.45)
    if not (np.all(c_o == 2) and np.all(c_h == 1) and np.all(h_c == 1) and np.all(o_c == 1)):
        raise ValueError("formate linkers are not intact HCOO- groups")
    if not np.all(zn_o == 6):
        raise ValueError(f"Zn coordination numbers {sorted(set(zn_o))}; expected octahedral 6")
    if not np.all(o_zn >= 1):
        raise ValueError("a formate O atom is not bound to Zn")
    components = bonded_graph_components(
        atoms, {("O", "Zn"): 2.45, ("C", "O"): 1.40, ("C", "H"): 1.20}
    )
    if components != 1:
        raise ValueError(f"framework splits into {components} disconnected components")
    zn_o_d = d[(symbols[i] == "Zn") & (symbols[j] == "O") & (d <= 2.45)]
    return {
        "formula_units_ZnHCOO2": zn,
        "zn_o_coordination": 6,
        "zn_o_distance_range_A": [round(float(zn_o_d.min()), 4), round(float(zn_o_d.max()), 4)],
        "intact_formate_linkers": counts["C"],
        "periodic_bond_graph_components": components,
        "minimum_interatomic_distance_A": round(float(d.min()), 4),
    }


def phosphorene_unit_cell() -> Atoms:
    """Puckered monolayer cut from bulk black P; x armchair, y zigzag, z vacuum."""
    bp = BLACK_P
    bulk = crystal(
        "P",
        [(0.0, bp["y"], bp["z"])],
        spacegroup=64,
        cellpar=[bp["a"], bp["b"], bp["c"], 90, 90, 90],
    )
    frac = bulk.get_scaled_positions()
    layer = [k for k in range(len(bulk)) if (frac[k, 1] + 0.25) % 1.0 < 0.5]
    if len(layer) != 4:
        raise AssertionError("black-P conventional cell must contain two 4-atom layers")
    normal = (((frac[layer, 1] + 0.25) % 1.0) - 0.25) * bp["b"]
    xyz = np.column_stack([bulk.positions[layer, 2], bulk.positions[layer, 0], normal])
    xyz[:, 2] += PHOS_VACUUM_CELL_A / 2 - xyz[:, 2].mean()
    atoms = Atoms(
        "P4",
        positions=xyz,
        cell=[bp["c"], bp["a"], PHOS_VACUUM_CELL_A],
        pbc=[True, True, False],
    )
    atoms.wrap()
    return atoms


def phosphorene_variant(nx: int, ny: int, vacancies: int) -> tuple[Atoms, dict[str, Any]]:
    atoms = phosphorene_unit_cell().repeat((nx, ny, 1))
    removed: list[int] = []
    if vacancies:
        center = np.array([atoms.cell[0, 0] / 2, atoms.cell[1, 1] / 2, PHOS_VACUUM_CELL_A / 2])
        first = int(np.argmin(np.linalg.norm(atoms.positions - center, axis=1)))
        removed = [first]
        if vacancies == 2:
            # Nearest in-plane (same sub-layer) bonded neighbour: the 2.224 A zigzag bond.
            i, j, d = neighbor_list("ijd", atoms, 2.5)
            same_layer = [
                (dist, int(b))
                for a, b, dist in zip(i, j, d, strict=True)
                if a == first and abs(atoms.positions[b, 2] - atoms.positions[first, 2]) < 0.1
            ]
            removed.append(min(same_layer)[1])
        del atoms[sorted(removed, reverse=True)]
    return atoms, {"supercell": [nx, ny, 1], "vacancies": vacancies, "removed_sites": removed}


def validate_phosphorene(atoms: Atoms, vacancies: int) -> dict[str, Any]:
    i, j, d = neighbor_list("ijd", atoms, 2.5)
    cn = np.bincount(i, minlength=len(atoms))
    if d.min() < 2.15 or d.max() > 2.30:
        raise ValueError(f"P-P bonds {d.min():.3f}-{d.max():.3f} A outside puckered range")
    expected_undercoordinated = {0: 0, 1: 3, 2: 4}[vacancies]
    if int((cn == 2).sum()) != expected_undercoordinated or not np.all((cn == 2) | (cn == 3)):
        raise ValueError(f"coordination histogram {np.bincount(cn).tolist()} is not puckered")
    thickness = float(np.ptp(atoms.positions[:, 2]))
    vacuum = float(atoms.cell[2, 2]) - thickness
    if not np.allclose(atoms.cell[2], [0, 0, PHOS_VACUUM_CELL_A]) or vacuum < 15:
        raise ValueError("vacuum axis or vacuum gap changed")
    return {
        "coordination_histogram": np.bincount(cn).tolist(),
        "bond_length_range_A": [round(float(d.min()), 4), round(float(d.max()), 4)],
        "puckering_thickness_A": round(thickness, 4),
        "vacuum_gap_A": round(vacuum, 4),
        "in_plane_cell_A": [round(float(atoms.cell[0, 0]), 4), round(float(atoms.cell[1, 1]), 4)],
    }


def phosphorene_start(natoms: int) -> str:
    return (
        "SYSTEM = E4 phosphorene\n"
        "PREC = Accurate\nENCUT = 520\nEDIFF = 1.0E-06\nALGO = Normal\nNELM = 200\n"
        "LREAL = .FALSE.\nLASPH = .TRUE.\nISPIN = 2\n"
        f"MAGMOM = {natoms}*0.5\n"
        "ISMEAR = 0\nSIGMA = 0.05\nIBRION = 2\nISIF = 3\n"
        "LATTICE_CONSTRAINTS = .TRUE. .TRUE. .FALSE.\n"
        "NSW = 200\nEDIFFG = -0.01\nLWAVE = .TRUE.\nLCHARG = .FALSE.\nLORBIT = 11\n"
        "KPAR = 1\nNCORE = 12\n"
    )


def gamma_kpoints(mesh: tuple[int, int, int], comment: str) -> str:
    return f"{comment}\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"


def set_private(path: Path) -> None:
    path.chmod(stat.S_IRWXU)
    for item in path.rglob("*"):
        item.chmod(stat.S_IRWXU if item.is_dir() else stat.S_IRUSR | stat.S_IWUSR)


def stage_case(
    root: Path,
    *,
    material_class: str,
    calculation_id: str,
    source_poscar: Path,
    start: str,
    kpoints: str,
    paw_labels: dict[str, str],
    ranks: int,
    expected_start: dict[str, str],
    ediffg: str,
    extra: dict[str, Any],
    protocol_id: str | None = None,
    walltime_hours: float | None = None,
    stage_nsw_max: int = 200,
    precision: str = "Accurate",
) -> dict[str, Any]:
    check_start(start, ranks, expected_start, precision)
    poscar = source_poscar.read_text()
    species = poscar_species(poscar)
    components, potcar = paw_components(species, paw_labels)
    case_dir = root / "inputs" / calculation_id
    case_dir.mkdir(parents=True)
    (case_dir / "POSCAR").write_text(poscar)
    (case_dir / "START").write_text(start)
    (case_dir / "KPOINTS").write_text(kpoints)
    (case_dir / "POTCAR").write_bytes(potcar)
    (case_dir / "POTCAR.order").write_text(
        "".join(f"{c['label']} {c['sha256']}\n" for c in components)
    )
    hashes = {name: sha256(case_dir / name) for name in ("START", "POSCAR", "KPOINTS", "POTCAR")}
    (case_dir / "inputs.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    )
    bundle = sha256_bytes(
        "".join(
            f"{name}:{hashes[name]}\n" for name in ("START", "POSCAR", "KPOINTS", "POTCAR")
        ).encode()
    )
    record = {
        "material_class": material_class,
        "calculation_id": calculation_id,
        "source_calculation_id": extra.pop("source_calculation_id", calculation_id),
        "composition": dict(species),
        "atom_count": sum(count for _, count in species),
        "case_dir": rel(case_dir),
        "source_poscar": rel(source_poscar),
        "source_poscar_sha256": sha256(source_poscar),
        "structure_id": extra.pop("structure_id", sha256(source_poscar)),
        "protocol_id": protocol_id or PROTOCOLS[material_class],
        "input_hashes": hashes,
        "prepared_input_bundle_sha256": bundle,
        "paw_components": components,
        "potcar_sha256": hashes["POTCAR"],
        "mpi_ranks": ranks,
        "omp_threads_per_rank": 8,
        "ediffg_eV_per_A": ediffg,
        "stage_nsw_max": stage_nsw_max,
        "walltime_seconds": int((walltime_hours or TASK_WALLTIME_HOURS[material_class]) * 3600),
        **extra,
    }
    if hashes["POSCAR"] != record["source_poscar_sha256"]:
        raise AssertionError(f"{calculation_id}: staged POSCAR differs from source")
    (case_dir / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    set_private(case_dir)
    return record


def stage_nb_ta_o(root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((NBTAO_ROOT / "vasp_inputs/inputs_manifest.json").read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    executed = sorted(
        glob.glob(str(NBTAO_E2_RUN / "cases/NbTaO5-ordering-01-baseline/runs/9040072*/relax"))
    )
    if not executed:
        raise FileNotFoundError("E2 NbTaO5 ordering-01 qualification run is missing")
    executed_incar = incar_values((Path(executed[0]) / "INCAR").read_text())
    outcar = (Path(executed[0]) / "OUTCAR").read_text(errors="replace")
    if "reached required accuracy" not in outcar or executed_incar.get("KPAR") != "12":
        raise ValueError("E2 NbTaO5 reference run is not converged with KPAR=12")
    records = []
    for number in NBTAO_PRODUCTION + NBTAO_EXTENSION:
        case_id = f"NbTaO5_ordering_{number}"
        case = cases[case_id]
        subdir = (
            "production-candidates"
            if number in NBTAO_PRODUCTION
            else ("ordering-family-pending-approval")
        )
        inputs = NBTAO_ROOT / "vasp_inputs" / subdir / case_id
        source = NBTAO_ROOT / "poscars" / f"{case_id}.POSCAR"
        for name, path in (
            ("POSCAR", source),
            ("INCAR", inputs / "INCAR"),
            ("KPOINTS", inputs / "KPOINTS"),
        ):
            if sha256(path) != case["input_hashes"][name]:
                raise ValueError(f"{case_id}: {name} differs from the frozen E2 manifest")
        incar = (inputs / "INCAR").read_text()
        if "KPAR" in incar_values(incar):
            raise ValueError(f"{case_id}: unexpected KPAR in frozen INCAR")
        science = {k: v for k, v in incar_values(incar).items() if k not in {"SYSTEM", "LWAVE"}}
        reference = {
            k: v for k, v in executed_incar.items() if k not in {"SYSTEM", "LWAVE", "KPAR"}
        }
        if science != reference:
            raise ValueError(f"{case_id}: INCAR differs from the converged E2 reference run")
        order = [line.split() for line in (inputs / "POTCAR.order").read_text().splitlines()]
        expected_paw = {fields[0]: fields[1] for fields in order if len(fields) == 2}
        start = incar.rstrip("\n") + "\nKPAR = 12\n"
        record = stage_case(
            root,
            material_class="nb_ta_o",
            calculation_id=f"E4-{case_id}",
            source_poscar=source,
            start=start,
            kpoints=(inputs / "KPOINTS").read_text(),
            paw_labels={"Nb": "Nb_pv", "Ta": "Ta_pv", "O": "O"},
            ranks=12,
            expected_start={
                "ENCUT": "600",
                "EDIFF": "1e-6",
                "ISPIN": "1",
                "ISMEAR": "0",
                "IBRION": "2",
                "ISIF": "3",
                "KPAR": "12",
            },
            ediffg="-0.01",
            extra={
                "source_calculation_id": case_id,
                "structure_family": "28-atom monoclinic NbTaO5; symmetry-distinct Nb/Ta ordering",
                "structure_source": f"E2 NbTaO5 {case['category']} (parent COD 1528678 Nb2O5)",
                "variation": {"ordering_id": case_id, "e2_category": case["category"]},
                "e4_adapter": {
                    "added_tags": {"KPAR": "12"},
                    "justification": "parallelization only; identical to converged E2 run",
                    "reference_run": rel(Path(executed[0])),
                },
                "source_hashes": {
                    rel(inputs / "INCAR"): sha256(inputs / "INCAR"),
                    rel(inputs / "KPOINTS"): sha256(inputs / "KPOINTS"),
                },
            },
        )
        if {c["label"]: c["sha256"] for c in record["paw_components"]} != expected_paw:
            raise ValueError(f"{case_id}: PAW components differ from frozen POTCAR.order")
        if record["composition"] != {"Nb": 4, "Ta": 4, "O": 20}:
            raise ValueError(f"{case_id}: not charge-balanced NbTaO5")
        records.append(record)
    return records


def stage_zn_formate(root: Path) -> list[dict[str, Any]]:
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    sources = {}
    for tag, case in MOF_FRAMEWORKS.items():
        case_dir = MOF_ARCHIVE / "cases" / case
        status = json.loads((case_dir / "status.json").read_text())
        verify = json.loads((case_dir / "verify-relax.json").read_text())
        if status.get("state") != "accepted" or verify.get("qualified") is not True:
            raise ValueError(f"{case}: E2 relaxation is not qualified")
        if sha256(case_dir / "POTCAR") != MOF_E2_POTCAR_SHA256:
            raise ValueError(f"{case}: archived POTCAR identity changed")
        relaxed = read(case_dir / "relax/CONTCAR", format="vasp")
        static = read(case_dir / "static/POSCAR", format="vasp")
        if not np.allclose(relaxed.positions, static.positions, atol=1e-6):
            raise ValueError(f"{case}: static geometry is not the relaxed CONTCAR")
        sources[tag] = (case_dir, relaxed, validate_zn_formate(relaxed))
    records = []
    ordered = [(tag, strain) for strain in MOF_STRAINS for tag in ("ALPHA", "PBCN")]
    for index, (tag, strain) in enumerate(ordered, start=1):
        case_dir, relaxed, _ = sources[tag]
        atoms = relaxed.copy()
        atoms.set_cell(np.asarray(relaxed.cell) * (1.0 + strain), scale_atoms=True)
        chemistry = validate_zn_formate(atoms)
        calc_id = f"E4-ZNF-{index:04d}"
        source = STRUCTURES / f"{calc_id}.POSCAR"
        source.write_text(
            poscar_text(atoms, f"{calc_id} {MOF_FRAMEWORKS[tag]} isotropic strain {strain:+.3f}")
        )
        start = (case_dir / "INCAR.relax").read_text()
        start = re.sub(r"(?m)^SYSTEM *=.*$", f"SYSTEM = {calc_id} fixed-volume EOS point", start)
        start = re.sub(r"(?m)^ISIF *= *3$", "ISIF = 4", start)
        record = stage_case(
            root,
            material_class="zn_formate_mof",
            calculation_id=calc_id,
            source_poscar=source,
            start=start,
            kpoints=(case_dir / "KPOINTS").read_text(),
            paw_labels={"Zn": "Zn", "H": "H", "C": "C", "O": "O"},
            ranks=12,
            expected_start={
                "ENCUT": "520",
                "IVDW": "12",
                "ISPIN": "1",
                "ISMEAR": "0",
                "ISIF": "4",
                "KPAR": "12",
                "LASPH": ".TRUE.",
            },
            ediffg="-0.02",
            extra={
                "structure_family": f"anhydrous Zn(HCOO)2 framework ({tag.lower()})",
                "structure_source": f"E2 qualified PBE-D3(BJ) relaxed CONTCAR of {case_dir.name}",
                "variation": {
                    "framework": MOF_FRAMEWORKS[tag],
                    "isotropic_linear_strain": strain,
                    "volume_ratio": round((1 + strain) ** 3, 6),
                    "purpose": "fixed-volume equation-of-state point (bulk modulus)",
                },
                "chemistry_validation": chemistry,
                "source_hashes": {
                    rel(case_dir / "relax/CONTCAR"): sha256(case_dir / "relax/CONTCAR"),
                    rel(case_dir / "relax/OUTCAR"): sha256(case_dir / "relax/OUTCAR"),
                    rel(case_dir / "INCAR.relax"): sha256(case_dir / "INCAR.relax"),
                    rel(case_dir / "KPOINTS"): sha256(case_dir / "KPOINTS"),
                    rel(case_dir / "POTCAR"): MOF_E2_POTCAR_SHA256,
                },
            },
        )
        if record["potcar_sha256"] != MOF_E2_POTCAR_SHA256:
            raise ValueError(f"{calc_id}: assembled POTCAR differs from the E2 MOF POTCAR")
        records.append(record)
    return records


def stage_phosphorene(root: Path) -> list[dict[str, Any]]:
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    plan = [(nx, ny, v) for nx, ny in PHOS_SUPERCELLS[:4] for v in (0, 1)]
    plan += [(2, 3, 0), (2, 3, 1)] + [(nx, ny, 2) for nx, ny in PHOS_SUPERCELLS] + [(1, 1, 0)]
    records = []
    for index, (nx, ny, vacancies) in enumerate(plan, start=1):
        atoms, variation = phosphorene_variant(nx, ny, vacancies)
        geometry = validate_phosphorene(atoms, vacancies)
        calc_id = f"E4-PHOS-{index:04d}"
        source = STRUCTURES / f"{calc_id}.POSCAR"
        label = {0: "pristine", 1: "monovacancy", 2: "in-plane divacancy"}[vacancies]
        source.write_text(poscar_text(atoms, f"{calc_id} phosphorene {nx}x{ny} {label}"))
        mesh = (max(1, math.ceil(8 / nx)), max(1, math.ceil(11 / ny)), 1)
        records.append(
            stage_case(
                root,
                material_class="phosphorene",
                calculation_id=calc_id,
                source_poscar=source,
                start=phosphorene_start(len(atoms)),
                kpoints=gamma_kpoints(mesh, f"Gamma mesh scaled from E2 8x11x1 for {nx}x{ny}"),
                paw_labels={"P": "P"},
                ranks=12,
                expected_start={
                    "ENCUT": "520",
                    "ISIF": "3",
                    "ISPIN": "2",
                    "KPAR": "1",
                    "NCORE": "12",
                    "LATTICE_CONSTRAINTS": ".TRUE. .TRUE. .FALSE.",
                },
                ediffg="-0.01",
                extra={
                    "structure_family": "puckered monolayer phosphorene with 20 A vacuum axis",
                    "structure_source": (
                        "rebuilt from bulk black P (Brown & Rundqvist 1965, Cmce) per the E2 "
                        "follow-up; archived E2 P4/P36/P35 geometries are not bonded phosphorene"
                    ),
                    "variation": {**variation, "defect": label, "kpoint_mesh": list(mesh)},
                    "geometry_validation": geometry,
                },
            )
        )
    return records


def stage_fcc(root: Path) -> list[dict[str, Any]]:
    from benchmarks.e4.aurora_dft.planning import build_candidate_inventory
    from benchmarks.e4.aurora_dft.prepare_stage1 import (
        _fcc_parent,
        _lattice_occupations,
        _write_poscar,
        canonical_occupation_hash,
        lattice_site_permutations,
    )

    STRUCTURES.mkdir(parents=True, exist_ok=True)
    stage = [json.loads(line) for line in (STAGE1 / "manifest.jsonl").read_text().splitlines()]
    frozen = {r["calculation_id"]: r for r in stage if r["material_family"] == "fcc_hea"}
    source = frozen["E4-FCC-0001"]["protocol"]["source"]
    parent_path = Path(source["parent_path"])
    if sha256(parent_path) != source["parent_poscar_sha256"]:
        raise ValueError("archived FCC Cantor parent changed")
    parent = _fcc_parent(read(parent_path))
    permutations = lattice_site_permutations(parent, preserve_species=False)
    batch = {c["calculation_id"]: c for c in json.loads(FIRST_BATCH.read_text())["cases"]}
    inventory = {
        r["calculation_id"]: r
        for r in build_candidate_inventory()
        if r["material_family"] == "fcc_hea"
    }
    known = {r["canonical_structure_id"] for r in frozen.values()}
    records = []
    for composition_index in range(1, 5):
        template_id = f"E4-FCC-{5 * (composition_index - 1) + 1:04d}"
        template = REPO / batch[template_id]["case_dir"]
        for realization in FCC_NEW_REALIZATIONS:
            calc_id = f"E4-FCC-{5 * (composition_index - 1) + realization:04d}"
            candidate = inventory[calc_id]
            if candidate["composition"] != inventory[template_id]["composition"]:
                raise AssertionError(f"{calc_id}: composition differs from frozen template")
            structure = _lattice_occupations(
                parent, candidate["composition"], candidate["random_seed"]
            )
            canonical = canonical_occupation_hash(structure, permutations)
            if canonical in known:
                raise ValueError(f"{calc_id}: duplicate symmetry-canonical occupation")
            known.add(canonical)
            path = STRUCTURES / f"{calc_id}.POSCAR"
            _write_poscar(structure, path, calc_id)
            if poscar_species(path.read_text()) != poscar_species(
                (template / "POSCAR").read_text()
            ):
                raise AssertionError(f"{calc_id}: species order differs from frozen MAGMOM")
            records.append(
                stage_case(
                    root,
                    material_class="fcc_hea",
                    calculation_id=f"E4-SCALE-FCC-{calc_id.rsplit('-', 1)[-1]}",
                    source_poscar=path,
                    start=(template / "START").read_text(),
                    kpoints=(template / "KPOINTS").read_text(),
                    paw_labels={
                        c["element"]: c["label"] for c in batch[template_id]["paw_components"]
                    },
                    ranks=8,
                    expected_start={"ENCUT": "480", "ISPIN": "2", "KPAR": "8", "ISIF": "3"},
                    ediffg="-0.02",
                    extra={
                        "source_calculation_id": calc_id,
                        "structure_id": canonical,
                        "structure_family": "128-site FCC Cantor alloy",
                        "structure_source": (
                            "frozen stage-1 generator: archived E2 Cantor parent, inventory seed"
                        ),
                        "composition_index": composition_index,
                        "disorder_realization": realization,
                        "random_seed": candidate["random_seed"],
                        "variation": {
                            "composition_index": composition_index,
                            "disorder_realization": realization,
                            "random_seed": candidate["random_seed"],
                            "start_template": template_id,
                        },
                    },
                )
            )
    return records


def stage_lifepo4(root: Path) -> list[dict[str, Any]]:
    inventory = {
        r["calculation_id"]: r
        for r in map(json.loads, (LFP_INVENTORY / "manifest.jsonl").read_text().splitlines())
    }
    batch = {c["calculation_id"]: c for c in json.loads(FIRST_BATCH.read_text())["cases"]}
    frozen = {c["structure_id"] for c in batch.values() if c["material_family"] == "lifepo4"}
    template = REPO / batch["E4-LFP-0002"]["case_dir"]
    records = []
    for calc_id, pairs in LFP_EXTENSION.items():
        entry = inventory[calc_id]
        if entry["antisite_exchange_pairs"] != pairs or entry["canonical_structure_id"] in frozen:
            raise ValueError(f"{calc_id}: inventory selection mismatch or duplicate")
        source = LFP_INVENTORY / entry["poscar"]
        if sha256(source) != entry["structure_sha256"]:
            raise ValueError(f"{calc_id}: inventory POSCAR hash mismatch")
        if poscar_species(source.read_text()) != poscar_species((template / "POSCAR").read_text()):
            raise AssertionError(f"{calc_id}: species order differs from frozen MAGMOM")
        records.append(
            stage_case(
                root,
                material_class="lifepo4",
                calculation_id=f"E4-SCALE-LFP-{calc_id.rsplit('-', 1)[-1]}",
                source_poscar=source,
                start=(template / "START").read_text(),
                kpoints=(template / "KPOINTS").read_text(),
                paw_labels={
                    c["element"]: c["label"] for c in batch["E4-LFP-0002"]["paw_components"]
                },
                ranks=12,
                expected_start={
                    "ENCUT": "520",
                    "ISPIN": "2",
                    "KPAR": "1",
                    "NCORE": "12",
                    "ISIF": "3",
                },
                ediffg="-0.01",
                extra={
                    "source_calculation_id": calc_id,
                    "structure_id": entry["canonical_structure_id"],
                    "structure_family": "112-atom LiFePO4 olivine antisite supercell",
                    "structure_source": "lifepo4-inventory-v1 canonical structure",
                    "antisite_exchange_pairs": pairs,
                    "variation": {"antisite_exchange_pairs": pairs},
                },
            )
        )
    return records


def stage_all(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty input root: {root}")
    if STRUCTURES.exists() and any(STRUCTURES.iterdir()):
        raise FileExistsError(f"refusing to overwrite frozen source POSCARs: {STRUCTURES}")
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    classes = {
        "nb_ta_o": stage_nb_ta_o(root),
        "zn_formate_mof": stage_zn_formate(root),
        "phosphorene": stage_phosphorene(root),
        "fcc_hea": stage_fcc(root),
        "lifepo4": stage_lifepo4(root),
    }
    for name, records in classes.items():
        hashes = [r["source_poscar_sha256"] for r in records]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"{name}: duplicate staged POSCARs")
    manifest = {
        "schema_version": 1,
        "manifest_id": "E4-six-class-qualified-inputs-v1",
        "execution_authorized": False,
        "scheduler_submissions": 0,
        "input_root": rel(root),
        "protocols": PROTOCOLS,
        "task_walltime_hours": TASK_WALLTIME_HOURS,
        "classes": classes,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


# Full production campaign: 256 unique inputs per class (weak scaling to 128 nodes at
# 2 tasks/node) and a 128-task fixed strong workload (no idle node at 128 nodes).
FULL_RESEARCH_ROOT = REPO / "research/paper/e4/aurora-dft-scaling/six-class-inputs-v2"
FULL_MANIFEST = FULL_RESEARCH_ROOT / "manifest.json"
FULL_STRUCTURES = FULL_RESEARCH_ROOT / "structures"
DEFAULT_FULL_INPUT_ROOT = REPO.parent / "e4-six-class-inputs-v2"
BCC_WEAK_MANIFEST = (
    REPO / "research/paper/e4/aurora-dft-scaling/priority-scaling/weak-input-manifest.json"
)
BCC_AUTHORIZED_STRONG = (
    "E4-BCC-0001",
    "E4-BCC-0561",
    "E4-BCC-1121",
    "E4-BCC-1201",
    "E4-BCC-1501",
    "E4-BCC-1701",
    "E4-BCC-1801",
    "E4-BCC-1805",
)
LFP_FROZEN = (
    "E4-LFP-0002",
    "E4-LFP-0003",
    "E4-LFP-0010",
    "E4-LFP-0025",
    "E4-LFP-0050",
    "E4-LFP-0100",
    "E4-LFP-0505",
    "E4-LFP-0506",
)
UNIQUE_INPUTS = 256
STRONG_TASKS = 128
FULL_CLASSES = {
    "bcc_refractory": {
        "protocol_id": "bcc-pbe-low-normal-v1-approved-ediffg-0.01",
        "task_cap_hours": 24,
        "expected_task_hours": 6.5,
        "runtime_basis": "measured E4 BCC tasks on Aurora: 3.7-7.4 h, 68-137 ionic steps "
        "(9 converged, 1 still running at 7.4 h); authorized 24-h task cap",
    },
    "fcc_hea": {
        "protocol_id": "fcc-pbe-verified-v1",
        "task_cap_hours": 48,
        "expected_task_hours": 21.0,
        "runtime_basis": "E2 108-atom Cantor relaxations (1,420-1,690 s/step, 25-37 steps) "
        "scaled to 128 atoms",
    },
    "lifepo4": {
        "protocol_id": "lifepo4-pbe-verified-v1",
        "task_cap_hours": 72,
        "expected_task_hours": 43.0,
        "runtime_basis": "archived 28-atom LiFePO4 relaxation (222 s/step) extrapolated to "
        "112 atoms and the 2x4x8 mesh; unmeasured at E4 size",
    },
    "phosphorene": {
        "protocol_id": "phosphorene-pbe-2d-strain-fixed-cell-v1",
        "task_cap_hours": 12,
        "expected_task_hours": 4.7,
        "runtime_basis": "archived E2 P36 timing (38 s/step) scaled to 80 atoms, spin-polarized",
    },
    "zn_formate_mof": {
        "protocol_id": "znformate-pbe-d3bj-strain-stress-v1",
        "task_cap_hours": 10,
        "expected_task_hours": 1.8,
        "runtime_basis": "E2 relaxations: alpha 99 s/step, Pbcn 245 s/step; 15-40 steps from "
        "the relaxed framework under <=2% strain",
    },
    "nb_ta_o": {
        "protocol_id": "nbtao5-fixed-kmesh-v1",
        "task_cap_hours": 10,
        "expected_task_hours": 3.5,
        "runtime_basis": "E2 ordering-01 (72 s/step, 50 steps) scaled to the 56-atom cell "
        "and 4x8x5 mesh",
    },
}


def kpoint_mesh(atoms: Atoms, spacing: float = 0.15) -> tuple[int, int, int]:
    """E2 NbTaO5 policy: N_i = max(1, ceil(|B_i| / spacing)), |B_i| including 2*pi."""
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(atoms.cell)).T
    return tuple(max(1, math.ceil(np.linalg.norm(b) / spacing)) for b in reciprocal)


def ordered_poscar_text(atoms: Atoms, label: str, order: list[str]) -> str:
    symbols = atoms.get_chemical_symbols()
    index = sorted(range(len(atoms)), key=lambda i: (order.index(symbols[i]), i))
    return poscar_text(atoms[index], label)


def _write_full_structure(calc_id: str, text: str) -> Path:
    FULL_STRUCTURES.mkdir(parents=True, exist_ok=True)
    path = FULL_STRUCTURES / f"{calc_id}.POSCAR"
    path.write_text(text)
    return path


def _full_kwargs(name: str) -> dict[str, Any]:
    spec = FULL_CLASSES[name]
    return {
        "material_class": name,
        "protocol_id": spec["protocol_id"],
        "walltime_hours": spec["task_cap_hours"],
    }


def full_bcc(root: Path) -> list[dict[str, Any]]:
    from benchmarks.e4.aurora_dft.planning import build_candidate_inventory
    from benchmarks.e4.aurora_dft.prepare_stage1 import (
        PAW_BY_ELEMENT,
        _lattice_occupations,
        _write_poscar,
        canonical_occupation_hash,
        lattice_site_permutations,
    )

    batch = {c["calculation_id"]: c for c in json.loads(FIRST_BATCH.read_text())["cases"]}
    weak = json.loads(BCC_WEAK_MANIFEST.read_text())["cases"]
    authorized = [batch[calc_id] for calc_id in BCC_AUTHORIZED_STRONG] + weak
    template = REPO / authorized[0]["case_dir"]
    if len({sha256(REPO / c["case_dir"] / "START") for c in authorized}) != 1:
        raise ValueError("authorized BCC inputs do not share one START template")
    start = (template / "START").read_text()
    kpoints = (template / "KPOINTS").read_text()
    parent = read(STAGE1 / "bcc_refractory/E4-BCC-0001/POSCAR", format="vasp")
    permutations = lattice_site_permutations(parent, preserve_species=False)
    inventory = [r for r in build_candidate_inventory() if r["material_family"] == "bcc_refractory"]
    by_key = {
        (tuple(r["elements"]), r["composition_index"], r["disorder_realization"]): r
        for r in inventory
    }
    probe = next(r for r in inventory if r["calculation_id"] == "E4-BCC-0002")
    probe_path = FULL_STRUCTURES / ".probe.POSCAR"
    FULL_STRUCTURES.mkdir(parents=True, exist_ok=True)
    _write_poscar(
        _lattice_occupations(parent, probe["composition"], probe["random_seed"]),
        probe_path,
        "E4-BCC-0002",
    )
    reproduced = sha256(probe_path) == next(
        c["source_poscar_sha256"] for c in weak if c["calculation_id"] == "E4-BCC-0002"
    )
    probe_path.unlink()
    if not reproduced:
        raise ValueError("BCC generator does not reproduce the frozen weak-scaling POSCAR")

    def stage(calc_id: str, source: Path, extra: dict[str, Any]) -> dict[str, Any]:
        species = poscar_species(source.read_text())
        return stage_case(
            root,
            calculation_id=f"E4-SCALE-BCC-{calc_id.rsplit('-', 1)[-1]}",
            source_poscar=source,
            start=start,
            kpoints=kpoints,
            paw_labels={symbol: PAW_BY_ELEMENT[symbol] for symbol, _ in species},
            ranks=8,
            expected_start={"ENCUT": "350", "ISPIN": "1", "ISIF": "3", "KPAR": "8"},
            ediffg="-0.01",
            extra={
                "source_calculation_id": calc_id,
                "structure_family": "128-site BCC refractory alloy",
                **extra,
            },
            stage_nsw_max=240,
            precision="PRECISION",
            **_full_kwargs("bcc_refractory"),
        )

    records = []
    for case in authorized:
        source = REPO / case["source_poscar"]
        if sha256(source) != case["source_poscar_sha256"]:
            raise ValueError(f"{case['calculation_id']}: authorized POSCAR changed")
        records.append(
            stage(
                case["calculation_id"],
                source,
                {
                    "structure_id": case.get("structure_id") or case["source_poscar_sha256"],
                    "structure_source": "authorized E4 BCC scaling input (unchanged)",
                    "variation": {"authorized_input": True},
                },
            )
        )
    subsets = list(dict.fromkeys(tuple(r["elements"]) for r in inventory))
    used = {case["calculation_id"] for case in authorized}
    known: set[tuple[str, str]] = set()
    composition_index = 1
    while len(records) < UNIQUE_INPUTS:
        for subset in subsets:
            candidate = by_key.get((subset, composition_index, 1))
            if candidate is None or candidate["calculation_id"] in used:
                continue
            structure = _lattice_occupations(
                parent, candidate["composition"], candidate["random_seed"]
            )
            canonical = canonical_occupation_hash(structure, permutations)
            key = (json.dumps(candidate["composition"], sort_keys=True), canonical)
            if key in known:
                raise ValueError(f"{candidate['calculation_id']}: duplicate occupation")
            known.add(key)
            path = FULL_STRUCTURES / f"{candidate['calculation_id']}.POSCAR"
            _write_poscar(structure, path, candidate["calculation_id"])
            records.append(
                stage(
                    candidate["calculation_id"],
                    path,
                    {
                        "structure_id": canonical,
                        "structure_source": "frozen BCC weak-scaling generator; inventory seed",
                        "composition_index": candidate["composition_index"],
                        "disorder_realization": 1,
                        "random_seed": candidate["random_seed"],
                        "variation": {
                            "elements": list(subset),
                            "composition_index": candidate["composition_index"],
                            "random_seed": candidate["random_seed"],
                        },
                    },
                )
            )
            if len(records) == UNIQUE_INPUTS:
                break
        composition_index += 1
    return records


def full_fcc(root: Path) -> list[dict[str, Any]]:
    from benchmarks.e4.aurora_dft.planning import build_candidate_inventory
    from benchmarks.e4.aurora_dft.prepare_stage1 import (
        PAW_BY_ELEMENT,
        _fcc_parent,
        _incar,
        _lattice_occupations,
        _write_poscar,
        canonical_occupation_hash,
        lattice_site_permutations,
    )

    stage1 = {
        r["calculation_id"]: r
        for r in map(json.loads, (STAGE1 / "manifest.jsonl").read_text().splitlines())
        if r["material_family"] == "fcc_hea"
    }
    source = stage1["E4-FCC-0001"]["protocol"]["source"]
    parent_path = Path(source["parent_path"])
    if sha256(parent_path) != source["parent_poscar_sha256"]:
        raise ValueError("archived FCC Cantor parent changed")
    parent = _fcc_parent(read(parent_path))
    permutations = lattice_site_permutations(parent, preserve_species=False)
    batch = {c["calculation_id"]: c for c in json.loads(FIRST_BATCH.read_text())["cases"]}
    template = REPO / batch["E4-FCC-0001"]["case_dir"]
    inventory = {
        (r["composition_index"], r["disorder_realization"]): r
        for r in build_candidate_inventory()
        if r["material_family"] == "fcc_hea"
    }
    records, known = [], set()
    for composition_index in range(1, UNIQUE_INPUTS // 4 + 1):
        for realization in range(1, 5):
            candidate = inventory[(composition_index, realization)]
            calc_id = candidate["calculation_id"]
            structure = _lattice_occupations(
                parent, candidate["composition"], candidate["random_seed"]
            )
            canonical = canonical_occupation_hash(structure, permutations)
            if (composition_index, canonical) in known:
                raise ValueError(f"{calc_id}: duplicate canonical occupation")
            known.add((composition_index, canonical))
            path = FULL_STRUCTURES / f"{calc_id}.POSCAR"
            FULL_STRUCTURES.mkdir(parents=True, exist_ok=True)
            _write_poscar(structure, path, calc_id)
            if calc_id in stage1 and sha256(path) != stage1[calc_id]["structure_sha256"]:
                raise ValueError(f"{calc_id}: generator no longer reproduces frozen POSCAR")
            species = poscar_species(path.read_text())
            start = _incar("fcc_hea", species, "relax")
            if (
                calc_id in batch
                and start != (REPO / batch[calc_id]["case_dir"] / "START").read_text()
            ):
                raise ValueError(f"{calc_id}: START differs from frozen first-batch input")
            records.append(
                stage_case(
                    root,
                    calculation_id=f"E4-SCALE-FCC-{calc_id.rsplit('-', 1)[-1]}",
                    source_poscar=path,
                    start=start,
                    kpoints=(template / "KPOINTS").read_text(),
                    paw_labels={symbol: PAW_BY_ELEMENT[symbol] for symbol, _ in species},
                    ranks=8,
                    expected_start={"ENCUT": "480", "ISPIN": "2", "KPAR": "8", "ISIF": "3"},
                    ediffg="-0.02",
                    extra={
                        "source_calculation_id": calc_id,
                        "structure_id": canonical,
                        "structure_family": "128-site FCC Cantor alloy",
                        "structure_source": "frozen stage-1 generator on the E2 Cantor parent",
                        "composition_index": composition_index,
                        "disorder_realization": realization,
                        "random_seed": candidate["random_seed"],
                        "variation": {
                            "composition_index": composition_index,
                            "disorder_realization": realization,
                            "random_seed": candidate["random_seed"],
                        },
                    },
                    **_full_kwargs("fcc_hea"),
                )
            )
    return records


def full_lifepo4(root: Path) -> list[dict[str, Any]]:
    inventory = [
        json.loads(line) for line in (LFP_INVENTORY / "manifest.jsonl").read_text().splitlines()
    ]
    by_id = {r["calculation_id"]: r for r in inventory}
    order = list(LFP_FROZEN) + list(LFP_EXTENSION)
    order += [r["calculation_id"] for r in inventory if r["calculation_id"] not in order]
    batch = {c["calculation_id"]: c for c in json.loads(FIRST_BATCH.read_text())["cases"]}
    template = REPO / batch["E4-LFP-0002"]["case_dir"]
    species_template = poscar_species((template / "POSCAR").read_text())
    records, canonical = [], set()
    for calc_id in order[:UNIQUE_INPUTS]:
        entry = by_id[calc_id]
        source = LFP_INVENTORY / entry["poscar"]
        if sha256(source) != entry["structure_sha256"]:
            raise ValueError(f"{calc_id}: inventory POSCAR hash mismatch")
        if entry["canonical_structure_id"] in canonical:
            raise ValueError(f"{calc_id}: duplicate canonical structure")
        canonical.add(entry["canonical_structure_id"])
        if poscar_species(source.read_text()) != species_template:
            raise ValueError(f"{calc_id}: species order differs from the frozen MAGMOM")
        records.append(
            stage_case(
                root,
                calculation_id=f"E4-SCALE-LFP-{calc_id.rsplit('-', 1)[-1]}",
                source_poscar=source,
                start=(template / "START").read_text(),
                kpoints=(template / "KPOINTS").read_text(),
                paw_labels={
                    c["element"]: c["label"] for c in batch["E4-LFP-0002"]["paw_components"]
                },
                ranks=12,
                expected_start={"ENCUT": "520", "ISPIN": "2", "KPAR": "1", "NCORE": "12"},
                ediffg="-0.01",
                extra={
                    "source_calculation_id": calc_id,
                    "structure_id": entry["canonical_structure_id"],
                    "structure_family": "112-atom LiFePO4 olivine antisite supercell",
                    "structure_source": "lifepo4-inventory-v1 canonical structure",
                    "antisite_exchange_pairs": entry["antisite_exchange_pairs"],
                    "variation": {"antisite_exchange_pairs": entry["antisite_exchange_pairs"]},
                },
                **_full_kwargs("lifepo4"),
            )
        )
    return records


def mof_strain_specs() -> list[tuple[str, float]]:
    """8 volumetric + 6 Voigt components x 2 signs x 10 magnitudes = 128 strain states."""
    specs = [("volumetric", s * m) for m in (0.005, 0.01, 0.015, 0.02) for s in (-1, 1)]
    specs += [
        (f"voigt{component}", s * m / 1000)
        for m in range(1, 11)
        for component in range(1, 7)
        for s in (-1, 1)
    ]
    return specs


def strain_tensor(kind: str, value: float) -> np.ndarray:
    if kind == "volumetric":
        return value * np.eye(3)
    component = int(kind[-1])
    eps = np.zeros((3, 3))
    pairs = {1: (0, 0), 2: (1, 1), 3: (2, 2), 4: (1, 2), 5: (0, 2), 6: (0, 1)}
    i, j = pairs[component]
    # Voigt shear components are engineering strains.
    eps[i, j] = eps[j, i] = value if i == j else value / 2
    return eps


def full_zn_formate(root: Path) -> list[dict[str, Any]]:
    frameworks = {}
    for tag, case in MOF_FRAMEWORKS.items():
        case_dir = MOF_ARCHIVE / "cases" / case
        verify = json.loads((case_dir / "verify-relax.json").read_text())
        status = json.loads((case_dir / "status.json").read_text())
        if verify.get("qualified") is not True or status.get("state") != "accepted":
            raise ValueError(f"{case}: E2 relaxation is not qualified")
        if sha256(case_dir / "POTCAR") != MOF_E2_POTCAR_SHA256:
            raise ValueError(f"{case}: archived POTCAR identity changed")
        relaxed = read(case_dir / "relax/CONTCAR", format="vasp")
        validate_zn_formate(relaxed)
        frameworks[tag] = (case_dir, relaxed)
    records = []
    for index, ((kind, value), tag) in enumerate(
        ((spec, tag) for spec in mof_strain_specs() for tag in ("ALPHA", "PBCN")), start=1
    ):
        case_dir, relaxed = frameworks[tag]
        atoms = relaxed.copy()
        deformation = np.eye(3) + strain_tensor(kind, value)
        atoms.set_cell(np.asarray(relaxed.cell) @ deformation.T, scale_atoms=True)
        chemistry = validate_zn_formate(atoms)
        calc_id = f"E4-ZNFS-{index:04d}"
        source = _write_full_structure(
            calc_id,
            poscar_text(atoms, f"{calc_id} {MOF_FRAMEWORKS[tag]} {kind} strain {value:+.4f}"),
        )
        start = (case_dir / "INCAR.relax").read_text()
        start = re.sub(r"(?m)^SYSTEM *=.*$", f"SYSTEM = {calc_id} strain-stress point", start)
        start = re.sub(r"(?m)^ISIF *= *3$", "ISIF = 2", start)
        records.append(
            stage_case(
                root,
                calculation_id=calc_id,
                source_poscar=source,
                start=start,
                kpoints=(case_dir / "KPOINTS").read_text(),
                paw_labels={"Zn": "Zn", "H": "H", "C": "C", "O": "O"},
                ranks=12,
                expected_start={
                    "ENCUT": "520",
                    "IVDW": "12",
                    "ISPIN": "1",
                    "ISIF": "2",
                    "KPAR": "12",
                    "LASPH": ".TRUE.",
                },
                ediffg="-0.02",
                extra={
                    "structure_family": f"anhydrous Zn(HCOO)2 framework ({tag.lower()})",
                    "structure_source": (
                        f"E2 qualified PBE-D3(BJ) relaxed CONTCAR of {case_dir.name}"
                    ),
                    "variation": {
                        "framework": MOF_FRAMEWORKS[tag],
                        "strain_kind": kind,
                        "strain_value": value,
                        "purpose": "relaxed-ion stress-strain sampling (elastic tensor, EOS)",
                    },
                    "chemistry_validation": chemistry,
                    "source_hashes": {
                        rel(case_dir / "relax/CONTCAR"): sha256(case_dir / "relax/CONTCAR"),
                        rel(case_dir / "INCAR.relax"): sha256(case_dir / "INCAR.relax"),
                    },
                },
                **_full_kwargs("zn_formate_mof"),
            )
        )
        if records[-1]["potcar_sha256"] != MOF_E2_POTCAR_SHA256:
            raise ValueError(f"{calc_id}: assembled POTCAR differs from the E2 MOF POTCAR")
    return records


def phosphorene_defect(nx: int, ny: int, kind: str) -> tuple[Atoms, list[int]]:
    atoms = phosphorene_unit_cell().repeat((nx, ny, 1))
    if kind == "pristine":
        return atoms, []
    center = np.array([atoms.cell[0, 0] / 2, atoms.cell[1, 1] / 2, PHOS_VACUUM_CELL_A / 2])
    first = int(np.argmin(np.linalg.norm(atoms.positions - center, axis=1)))
    removed = [first]
    if kind != "monovacancy":
        i, j, d = neighbor_list("ijd", atoms, 2.5)
        same_layer = kind == "divacancy_in_plane"
        partner = min(
            (dist, int(b))
            for a, b, dist in zip(i, j, d, strict=True)
            if a == first
            and (abs(atoms.positions[b, 2] - atoms.positions[first, 2]) < 0.1) == same_layer
        )[1]
        removed.append(partner)
    del atoms[sorted(removed, reverse=True)]
    return atoms, removed


def validate_strained_phosphorene(atoms: Atoms, vacancies: int) -> dict[str, Any]:
    i, j, d = neighbor_list("ijd", atoms, 2.6)
    cn = np.bincount(i, minlength=len(atoms))
    if d.min() < 2.0 or d.max() > 2.6:
        raise ValueError("strained P-P bonds left the 2.0-2.6 A range")
    expected = {0: 0, 1: 3, 2: 4}[vacancies]
    if int((cn == 2).sum()) != expected or not np.all((cn == 2) | (cn == 3)):
        raise ValueError(f"coordination histogram {np.bincount(cn).tolist()}")
    if not np.allclose(atoms.cell[2], [0, 0, PHOS_VACUUM_CELL_A]):
        raise ValueError("vacuum axis changed")
    return {
        "coordination_histogram": np.bincount(cn).tolist(),
        "bond_length_range_A": [round(float(d.min()), 4), round(float(d.max()), 4)],
        "vacuum_gap_A": round(PHOS_VACUUM_CELL_A - float(np.ptp(atoms.positions[:, 2])), 4),
        "in_plane_cell_A": [round(float(atoms.cell[0, 0]), 4), round(float(atoms.cell[1, 1]), 4)],
    }


def full_phosphorene(root: Path) -> list[dict[str, Any]]:
    nx, ny = 4, 5
    kinds = {"pristine": 0, "monovacancy": 1, "divacancy_in_plane": 2, "divacancy_interlayer": 2}
    grid = [(ex / 100, ey / 100) for ex in range(-2, 13, 2) for ey in range(-6, 9, 2)]
    # Nearest the PBE monolayer equilibrium (armchair +5.6%, zigzag ~0%) first, so the
    # 128-task strong workload brackets equilibrium for every defect type.
    grid.sort(key=lambda e: (abs(e[0] - 0.056) + abs(e[1]), e))
    bases = {kind: phosphorene_defect(nx, ny, kind) for kind in kinds}
    records = []
    for index, ((ex, ey), kind) in enumerate(
        ((strain, kind) for strain in grid for kind in kinds), start=1
    ):
        base, removed = bases[kind]
        atoms = base.copy()
        atoms.set_cell(np.asarray(base.cell) @ np.diag([1 + ex, 1 + ey, 1.0]), scale_atoms=True)
        geometry = validate_strained_phosphorene(atoms, kinds[kind])
        calc_id = f"E4-PHOSS-{index:04d}"
        source = _write_full_structure(
            calc_id,
            poscar_text(atoms, f"{calc_id} phosphorene 4x5 {kind} ex={ex:+.2f} ey={ey:+.2f}"),
        )
        start = (
            f"SYSTEM = {calc_id}\n"
            "PREC = Accurate\nENCUT = 520\nEDIFF = 1.0E-06\nALGO = Normal\nNELM = 200\n"
            "LREAL = .FALSE.\nLASPH = .TRUE.\nISPIN = 2\n"
            f"MAGMOM = {len(atoms)}*0.5\n"
            "ISMEAR = 0\nSIGMA = 0.05\nIBRION = 2\nISIF = 2\n"
            "NSW = 200\nEDIFFG = -0.01\nLWAVE = .TRUE.\nLCHARG = .FALSE.\nLORBIT = 11\n"
            "KPAR = 1\nNCORE = 12\n"
        )
        records.append(
            stage_case(
                root,
                calculation_id=calc_id,
                source_poscar=source,
                start=start,
                kpoints=gamma_kpoints((2, 3, 1), "Gamma mesh scaled from E2 8x11x1 for 4x5"),
                paw_labels={"P": "P"},
                ranks=12,
                expected_start={"ENCUT": "520", "ISIF": "2", "ISPIN": "2", "KPAR": "1"},
                ediffg="-0.01",
                extra={
                    "structure_family": "4x5 puckered phosphorene, 20 A vacuum axis",
                    "structure_source": "rebuilt from bulk black P (Cmce); archived E2 "
                    "geometries are unbonded",
                    "variation": {
                        "defect": kind,
                        "removed_sites": removed,
                        "armchair_strain": ex,
                        "zigzag_strain": ey,
                    },
                    "geometry_validation": geometry,
                },
                **_full_kwargs("phosphorene"),
            )
        )
    return records


def nbtao_doubled_orderings() -> tuple[Atoms, list[int], list[tuple[int, ...]], list[int]]:
    """Symmetry-distinct equimolar Nb/Ta orderings of the 56-atom (1x1x2) NbTaO5 cell."""
    import itertools

    import spglib

    parent = read(NBTAO_ROOT / "poscars/NbTaO5_ordering_01.POSCAR", format="vasp").repeat((1, 1, 2))
    symbols = parent.get_chemical_symbols()
    cations = [k for k, s in enumerate(symbols) if s in ("Nb", "Ta")]
    types = np.array([1 if s in ("Nb", "Ta") else 2 for s in symbols])
    positions = np.mod(parent.get_scaled_positions(), 1.0)
    symmetry = spglib.get_symmetry((np.asarray(parent.cell), positions, types), symprec=1e-3)

    def key(x: np.ndarray) -> tuple[int, ...]:
        return tuple(np.mod(np.rint(np.mod(x, 1.0) * 1e5), 100000).astype(int))

    site = {key(x): k for k, x in enumerate(positions)}
    local = {atom: k for k, atom in enumerate(cations)}
    perms = []
    for rotation, translation in zip(symmetry["rotations"], symmetry["translations"], strict=True):
        image = [site.get(key(rotation @ x + translation)) for x in positions]
        if None not in image:
            perms.append(np.argsort([local[image[atom]] for atom in cations]))

    def canonical(occupation: np.ndarray) -> tuple[int, ...]:
        return min(tuple(occupation[p]) for p in perms)

    e2 = []
    for number in NBTAO_PRODUCTION + NBTAO_EXTENSION:
        doubled = read(
            NBTAO_ROOT / f"poscars/NbTaO5_ordering_{number}.POSCAR", format="vasp"
        ).repeat((1, 1, 2))
        occupation = np.zeros(len(cations), dtype=int)
        for k, x in enumerate(np.mod(doubled.get_scaled_positions(), 1.0)):
            if doubled[k].symbol == "Ta":
                occupation[local[site[key(x)]]] = 1
        e2.append(canonical(occupation))
    if len(set(e2)) != len(e2):
        raise ValueError("doubled E2 orderings are not distinct")
    classes = {
        canonical(np.isin(np.arange(len(cations)), combo).astype(int))
        for combo in itertools.combinations(range(len(cations)), len(cations) // 2)
    }
    ordered = e2 + sorted(classes - set(e2))
    return parent, cations, ordered, [int(n) for n in NBTAO_PRODUCTION + NBTAO_EXTENSION]


def full_nb_ta_o(root: Path) -> list[dict[str, Any]]:
    parent, cations, orderings, e2_numbers = nbtao_doubled_orderings()
    mesh = kpoint_mesh(parent)
    reference = NBTAO_ROOT / "vasp_inputs/production-candidates/NbTaO5_ordering_01/INCAR"
    frozen = json.loads((NBTAO_ROOT / "vasp_inputs/inputs_manifest.json").read_text())
    frozen_hash = next(
        c["input_hashes"]["INCAR"] for c in frozen["cases"] if c["case_id"] == "NbTaO5_ordering_01"
    )
    if sha256(reference) != frozen_hash:
        raise ValueError("NbTaO5 reference INCAR differs from the frozen E2 manifest")
    base_incar = reference.read_text()
    records = []
    for index, occupation in enumerate(orderings[:UNIQUE_INPUTS], start=1):
        atoms = parent.copy()
        symbols = atoms.get_chemical_symbols()
        for k, atom in enumerate(cations):
            symbols[atom] = "Ta" if occupation[k] else "Nb"
        atoms.set_chemical_symbols(symbols)
        if atoms.get_chemical_formula(mode="metal") != "Nb8Ta8O40":
            raise ValueError("ordering is not equimolar NbTaO5")
        calc_id = f"E4-NBTAO56-{index:04d}"
        origin = (
            f"doubled E2 NbTaO5_ordering_{e2_numbers[index - 1]:02d}"
            if index <= len(e2_numbers)
            else "symmetry-distinct 1x1x2 ordering"
        )
        source = _write_full_structure(
            calc_id, ordered_poscar_text(atoms, f"{calc_id}; {origin}", ["Nb", "Ta", "O"])
        )
        start = re.sub(r"(?m)^SYSTEM *=.*$", f"SYSTEM = {calc_id} relax", base_incar)
        start = start.rstrip("\n") + "\nKPAR = 12\n"
        records.append(
            stage_case(
                root,
                calculation_id=calc_id,
                source_poscar=source,
                start=start,
                kpoints=gamma_kpoints(mesh, "Gamma-centered mesh; target spacing 0.15 1/A"),
                paw_labels={"Nb": "Nb_pv", "Ta": "Ta_pv", "O": "O"},
                ranks=12,
                expected_start={"ENCUT": "600", "ISPIN": "1", "ISIF": "3", "KPAR": "12"},
                ediffg="-0.01",
                extra={
                    "structure_family": "56-atom (1x1x2) NbTaO5; symmetry-distinct Nb/Ta ordering",
                    "structure_source": f"{origin}; parent COD 1528678 Nb2O5 framework",
                    "variation": {
                        "ordering": "".join(map(str, occupation)),
                        "e2_parent_ordering": origin,
                        "kpoint_mesh": list(mesh),
                    },
                },
                **_full_kwargs("nb_ta_o"),
            )
        )
    return records


def stage_full(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty input root: {root}")
    if FULL_STRUCTURES.exists() and any(FULL_STRUCTURES.iterdir()):
        raise FileExistsError(f"refusing to overwrite frozen POSCARs: {FULL_STRUCTURES}")
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    stagers = {
        "bcc_refractory": full_bcc,
        "fcc_hea": full_fcc,
        "lifepo4": full_lifepo4,
        "phosphorene": full_phosphorene,
        "zn_formate_mof": full_zn_formate,
        "nb_ta_o": full_nb_ta_o,
    }
    classes = {}
    for name, stager in stagers.items():
        records = stager(root)
        hashes = [r["source_poscar_sha256"] for r in records]
        if len(records) != UNIQUE_INPUTS or len(set(hashes)) != UNIQUE_INPUTS:
            raise ValueError(f"{name}: expected {UNIQUE_INPUTS} unique inputs")
        classes[name] = records
    manifest = {
        "schema_version": 1,
        "manifest_id": "E4-six-class-production-inputs-v2",
        "execution_authorized": False,
        "scheduler_submissions": 0,
        "input_root": rel(root),
        "unique_inputs_per_class": UNIQUE_INPUTS,
        "strong_fixed_tasks_per_class": STRONG_TASKS,
        "class_settings": FULL_CLASSES,
        "classes": classes,
    }
    FULL_MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--full", action="store_true", help="stage the 6x256 production set")
    args = parser.parse_args(argv)
    if args.full:
        manifest = stage_full(args.root or DEFAULT_FULL_INPUT_ROOT)
        target = FULL_MANIFEST
    else:
        manifest = stage_all(args.root or DEFAULT_INPUT_ROOT)
        target = MANIFEST
    for name, records in manifest["classes"].items():
        print(f"{name}: {len(records)} staged inputs")
    print(f"Manifest: {target}; no jobs submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
