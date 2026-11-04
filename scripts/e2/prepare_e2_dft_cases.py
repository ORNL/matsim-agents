#!/usr/bin/env python3
"""Stage E2 VASP cases for Cantor FCC, NbTaVHfZrTi BCC, and Zn(HCOO)2.2H2O.

Each case directory receives POSCAR, INCAR.relax, INCAR.static, KPOINTS,
POTCAR (assembled from the approved PAW library), and case.json recording the
exact structure source and hash, settings, acceptance criterion, budget, and
runnable command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PAW = {
    "Co": "Co", "Cr": "Cr_pv", "Fe": "Fe_pv", "Mn": "Mn_pv", "Ni": "Ni_pv",
    "Hf": "Hf_pv", "Nb": "Nb_pv", "Ta": "Ta_pv", "Ti": "Ti_pv", "V": "V_pv", "Zr": "Zr_sv",
    "Zn": "Zn", "C": "C", "H": "H", "O": "O",
}
# Ferromagnetic collinear initialization; the converged magnetic state is a result.
FCC_MAGMOM = {"Co": 2.0, "Cr": 3.0, "Fe": 3.0, "Mn": 3.0, "Ni": 1.0}
ARCHIVE_CALC = Path("dft_reference/vasp/calculations")

COMMON = {
    "PREC": "Accurate", "EDIFF": "1.0E-06", "ALGO": "Normal", "NELM": "200", "NELMIN": "4",
    "LREAL": ".FALSE.", "LASPH": ".TRUE.", "LORBIT": "11", "LCHARG": ".FALSE.",
}
FAMILIES = {
    "fcc": {
        "INCAR": {**COMMON, "ENCUT": "480", "ISPIN": "2", "ISMEAR": "1", "SIGMA": "0.10"},
        "kmesh": (2, 2, 2), "ranks": 8,
        "basis": "ENCUT=1.3*max ENMAX (Ni_pv 368 eV); 2x2x2 Gamma = 864 KPPRA; MP1 smearing 0.1 eV",
    },
    "bcc": {
        "INCAR": {**COMMON, "ENCUT": "345", "ISPIN": "1", "ISMEAR": "1", "SIGMA": "0.10"},
        "kmesh": (2, 2, 2), "ranks": 8,
        "basis": "ENCUT=1.3*max ENMAX (V_pv 264 eV); 2x2x2 Gamma = 1024 KPPRA; non-magnetic refractory HEA",
    },
    "mof": {
        "INCAR": {**COMMON, "ENCUT": "520", "ISPIN": "1", "ISMEAR": "0", "SIGMA": "0.05", "IVDW": "12"},
        "kmesh": (3, 4, 3), "ranks": 12,
        "basis": "ENCUT=1.3*ENMAX(C,O)=520 eV; PBE-D3(BJ) IVDW=12; 3x4x3 Gamma (~0.25 1/A spacing)",
    },
}
RELAX = {"IBRION": "2", "ISIF": "3", "NSW": "200", "EDIFFG": "-0.02", "LWAVE": ".TRUE."}
STATIC = {"IBRION": "-1", "NSW": "0", "LWAVE": ".FALSE."}
ACCEPT = {
    "relax": "VASP 'reached required accuracy' (all |F| <= 0.02 eV/A) and final ionic step SCF-converged",
    "static": "electronic SCF converged at the relaxed geometry (vasprun converged_electronic)",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def species_counts(poscar: Path) -> list[tuple[str, int]]:
    lines = poscar.read_text().splitlines()
    return list(zip(lines[5].split(), map(int, lines[6].split())))


def write_incar(path: Path, title: str, tags: dict[str, str]) -> None:
    path.write_text(f"SYSTEM = {title}\n" + "".join(f"{k} = {v}\n" for k, v in tags.items()))


def stage(case_dir: Path, name: str, family: str, poscar_src: Path, source: dict, paw_root: Path,
          walltime: str, repo: Path, kmesh: tuple[int, int, int] | None = None) -> dict:
    fam = FAMILIES[family]
    case_dir.mkdir(parents=True, exist_ok=False)
    poscar = case_dir / "POSCAR"
    poscar.write_bytes(poscar_src.read_bytes())
    counts = species_counts(poscar)
    incar = dict(fam["INCAR"], KPAR=str(fam["ranks"]))
    if family == "fcc":
        incar["MAGMOM"] = " ".join(f"{n}*{FCC_MAGMOM[s]:.1f}" for s, n in counts)
    write_incar(case_dir / "INCAR.relax", f"{name} relax", {**incar, **RELAX})
    write_incar(case_dir / "INCAR.static", f"{name} static", {**incar, **STATIC})
    k = kmesh or fam["kmesh"]
    (case_dir / "KPOINTS").write_text(f"Gamma-centred\n0\nGamma\n{k[0]} {k[1]} {k[2]}\n0 0 0\n")
    potcar = b"".join((paw_root / PAW[s] / "POTCAR").read_bytes() for s, _ in counts)
    (case_dir / "POTCAR").write_bytes(potcar)
    record = {
        "case": name, "family": family,
        "structure": {**source, "poscar_sha256": sha(poscar), "species_counts": counts,
                      "natoms": sum(n for _, n in counts)},
        "backend": {"code": "VASP 6.6.1 vasp_std (Aurora Intel GPU build)",
                    "paw": {s: PAW[s] for s, _ in counts}, "potcar_sha256": sha(case_dir / "POTCAR"),
                    "xc": "PBE" + ("-D3(BJ)" if family == "mof" else "")},
        "settings": {"relax": {**incar, **RELAX}, "static": {**incar, **STATIC},
                     "kpoints": {"mode": "Gamma", "mesh": list(k)}, "basis": fam["basis"],
                     "ranks_per_node": fam["ranks"], "omp_threads": 8,
                     "parallelization_note": "KPAR = ranks: one k-point group per tile; band splitting hangs Davidson on this build"},
        "acceptance": ACCEPT,
        "facility": {"system": "ALCF Aurora", "account": "HydraGNN", "queue": "capacity",
                     "nodes": 1, "walltime_per_segment": walltime},
        "command": f"bash {repo}/deployments/aurora/launchers/run-e2-dft-case-aurora.sh {case_dir}",
        "outputs": {"relax": "relax/{OUTCAR,vasprun.xml,CONTCAR}", "static": "static/{OUTCAR,vasprun.xml}",
                    "summary": "summary.json", "status": "status.json"},
    }
    (case_dir / "case.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--walltime", default="12:00:00")
    ap.add_argument("--anhydrous", nargs=3, action="append", metavar=("NAME", "POSCAR", "SOURCE_JSON"),
                    help="stage only anhydrous Zn formate frameworks (repeatable)")
    ap.add_argument("--archive", type=Path, help="matsim-agents-dft-reference checkout")
    ap.add_argument("--archive-commit")
    ap.add_argument("--zn-cif-poscar", type=Path)
    ap.add_argument("--zn-cif-sha256")
    args = ap.parse_args()
    paw_root = (args.repo / "external/vasp6/potcar").resolve()
    records = []
    if args.anhydrous:
        for name, poscar, source in args.anhydrous:
            src = json.loads(source)
            kmesh = tuple(src.pop("kmesh"))
            records.append(stage(args.out / "cases" / name, name, "mof", Path(poscar), src, paw_root,
                                 args.walltime, args.repo, kmesh))
        (args.out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"staged {len(records)} cases under {args.out / 'cases'}")
        return
    for family, prefix in (("fcc", "fcc_hea_CrMnFeCoNi_Cantor_108_"), ("bcc", "bcc_hea_NbTaVHfZrTi_equiatomic_128_")):
        for tag in ("ideal", "random1", "random2", "random3"):
            name = prefix + tag
            src = args.archive / ARCHIVE_CALC / name / "POSCAR"
            records.append(stage(args.out / "cases" / name, name, family, src,
                                 {"source": f"matsim-agents-dft-reference@{args.archive_commit}:{ARCHIVE_CALC / name / 'POSCAR'}",
                                  "source_sha256": sha(src),
                                  "note": "original unrelaxed archive POSCAR; archived runs never completed an ionic step"},
                                 paw_root, args.walltime, args.repo))
    name = "zn_formate_dihydrate_COD7102783"
    records.append(stage(args.out / "cases" / name, name, "mof", args.zn_cif_poscar,
                         {"source": "COD 7102783 (Zn(HCOO)2.2H2O, P2_1/c, 213 K, MoKa, R=0.0223)",
                          "cif_sha256": args.zn_cif_sha256,
                          "note": "experimental X-ray H positions (short O-H/C-H) are relaxed by DFT; "
                                  "anhydrous alpha/beta polymorphs absent from COD; shipped seeds are H-free placeholders"},
                         paw_root, "06:00:00", args.repo))
    (args.out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"staged {len(records)} cases under {args.out / 'cases'}")


if __name__ == "__main__":
    main()
