"""K-point convergence test for the qe-1024 campaign on a smaller random-alloy cell.

``prepare`` writes one SCF input per mesh; ``analyze`` compares energy, forces and pressure
against the densest mesh. Meshes are chosen so the k-spacing equals that of the full
8x8x8 cell at 2x2x2 / 3x3x3 / 4x4x4 (mesh scales with L_full / L_test). The test cell is 4x4x4
(128 atoms).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input
from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.workflows.full_supercell_campaign import FullSupercellCampaign

RY_TO_EV = 13.605693122994
RY_BOHR_TO_EV_ANG = RY_TO_EV / 0.529177210903
TEST_REPEATS = 4
FULL_MESHES = (2, 3, 4)

_ENERGY = re.compile(r"^!\s*total energy\s*=\s*(-?\d+\.\d+)\s*Ry", re.M)
_FORCE = re.compile(r"atom\s+\d+\s+type\s+\d+\s+force =\s+(\S+)\s+(\S+)\s+(\S+)")
_PRESSURE = re.compile(r"total\s+stress.*?P=\s*(-?\d+\.\d+)")
_WALL = re.compile(r"PWSCF\s*:.*?CPU\s+.*?\s+(\S+)\s+WALL")


def _test_meshes(cfg: FullSupercellCampaign) -> dict[int, int]:
    full = cfg.parent_supercell.conventional_repeats[0]
    return {m: m * full // TEST_REPEATS for m in FULL_MESHES}


def prepare(campaign: Path, out: Path) -> dict:
    cfg = FullSupercellCampaign.from_yaml(campaign)
    parent = cfg.parent_supercell
    scale = math_prod(parent.conventional_repeats) // TEST_REPEATS**3
    counts = {k: v // scale for k, v in parent.composition.items()}
    if sum(counts.values()) != 2 * TEST_REPEATS**3 or any(
        v * scale != parent.composition[k] for k, v in counts.items()
    ):
        raise ValueError(f"composition is not divisible by {scale}: {counts}")
    atoms = generate_alloy(
        AlloySpec(
            counts=counts,
            repeats=(TEST_REPEATS,) * 3,
            lattice_parameter=parent.lattice_parameter,
            ordering="random",
            seed=parent.seed,
        )
    )
    q = cfg.qe
    meshes = _test_meshes(cfg)
    manifest = {"natoms": len(atoms), "composition": counts, "cases": {}}
    for full_mesh, mesh in meshes.items():
        d = out / f"k{mesh}"
        d.mkdir(parents=True, exist_ok=True)
        settings = QESettings(
            pseudo_dir=q.pseudo_dir,
            calculation="scf",
            ecutwfc_ry=q.ecutwfc_ry,
            ecutrho_ry=q.ecutrho_ry,
            occupations="smearing",
            smearing=q.smearing,
            degauss_ry=q.degauss_ry,
            kpts=(mesh, mesh, mesh),
            conv_thr_ry=q.conv_thr_ry,
            mixing_beta=q.mixing_beta,
            pseudopotentials=dict(q.pseudopotentials),
        )
        write_pw_input(atoms, settings, str(d / "pw.in"), prefix=f"k{mesh}", outdir="./tmp")
        manifest["cases"][f"k{mesh}"] = {"mesh": mesh, "equivalent_full_cell_mesh": full_mesh}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def math_prod(values) -> int:
    return int(np.prod(values))


def _parse(path: Path) -> dict | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    energies = _ENERGY.findall(text)
    if not energies or "convergence has been achieved" not in text:
        return None
    block = text.split("Forces acting on atoms", 1)
    forces = None
    if len(block) == 2:
        lines = block[1].split("The non-local contrib", 1)[0]
        forces = np.array([[float(x) for x in m] for m in _FORCE.findall(lines)])
    pressure = _PRESSURE.search(text)
    wall = _WALL.search(text)
    return {
        "energy_ry": float(energies[-1]),
        "forces": forces,
        "pressure_kbar": float(pressure.group(1)) if pressure else None,
        "wall": wall.group(1) if wall else None,
    }


def analyze(out: Path) -> list[dict]:
    manifest = json.loads((out / "manifest.json").read_text())
    n = manifest["natoms"]
    results = {name: _parse(out / name / "pw.out") for name in manifest["cases"]}
    done = {k: v for k, v in results.items() if v is not None}
    if not done:
        raise SystemExit("no converged pw.out found")
    ref_name = max(done, key=lambda k: manifest["cases"][k]["mesh"])
    ref = done[ref_name]
    rows = []
    for name, r in results.items():
        info = manifest["cases"][name]
        row = {
            "case": name,
            "mesh": info["mesh"],
            "full_cell_mesh": info["equivalent_full_cell_mesh"],
        }
        if r is None:
            row["status"] = "NOT_CONVERGED_OR_MISSING"
        else:
            row["status"] = "ok"
            row["dE_meV_per_atom_vs_ref"] = (r["energy_ry"] - ref["energy_ry"]) * RY_TO_EV * 1e3 / n
            if r["forces"] is not None and ref["forces"] is not None:
                diff = (r["forces"] - ref["forces"]) * RY_BOHR_TO_EV_ANG
                row["max_dF_meV_per_A"] = float(np.abs(diff).max() * 1e3)
                row["rms_dF_meV_per_A"] = float(np.sqrt((diff**2).mean()) * 1e3)
            if r["pressure_kbar"] is not None and ref["pressure_kbar"] is not None:
                row["dP_kbar_vs_ref"] = r["pressure_kbar"] - ref["pressure_kbar"]
            row["wall"] = r["wall"]
        rows.append(row)
    summary = {"reference": ref_name, "natoms": n, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--campaign", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("analyze")
    b.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "prepare":
        print(json.dumps(prepare(args.campaign, args.out), indent=2))
    else:
        for row in analyze(args.out):
            print(row)


if __name__ == "__main__":
    main()
