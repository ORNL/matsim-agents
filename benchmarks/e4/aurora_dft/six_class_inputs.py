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


def check_start(start: str, ranks: int, expected: dict[str, str]) -> None:
    """Mirror the per-calculation wrapper's INCAR preconditions."""
    if not re.search(r"(?m)^PREC *= *Accurate$", start):
        raise ValueError("START must contain an exact 'PREC = Accurate' line")
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
) -> dict[str, Any]:
    check_start(start, ranks, expected_start)
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
        "protocol_id": PROTOCOLS[material_class],
        "input_hashes": hashes,
        "prepared_input_bundle_sha256": bundle,
        "paw_components": components,
        "potcar_sha256": hashes["POTCAR"],
        "mpi_ranks": ranks,
        "omp_threads_per_rank": 8,
        "ediffg_eV_per_A": ediffg,
        "stage_nsw_max": 200,
        "walltime_seconds": TASK_WALLTIME_HOURS[material_class] * 3600,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_INPUT_ROOT)
    args = parser.parse_args(argv)
    manifest = stage_all(args.root)
    for name, records in manifest["classes"].items():
        print(f"{name}: {len(records)} staged inputs")
    print(f"Manifest: {MANIFEST}; no jobs submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
