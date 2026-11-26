"""ASE vs LAMMPS ML-IAP energy/force parity gate and checkpoint-hash export record."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from matsim_agents.backends.external.lammps.mliap.bridge import (
    CalculatorMLIAP,
    UncertaintyGate,
)
from matsim_agents.core.hashing import sha256_file


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    n_atoms: int
    energy_ase_eV: float
    energy_lammps_eV: float
    energy_abs_diff_eV: float
    max_force_abs_diff_eV_per_A: float
    energy_tol_eV: float
    force_tol_eV_per_A: float


def lammps_data_text(atoms: Atoms, species: list[str]) -> str:
    """LAMMPS data file text (metal units, triclinic-safe) for `atoms`."""
    cell = atoms.cell.array
    a, b, c = cell
    if not (abs(a[1]) < 1e-12 and abs(a[2]) < 1e-12 and abs(b[2]) < 1e-12):
        raise ValueError("cell must be lower-triangular; rotate with ase.build.niggli/rotate")
    lines = [
        "matsim-agents parity",
        f"{len(atoms)} atoms",
        f"{len(species)} atom types",
        f"0 {a[0]:.16g} xlo xhi",
        f"0 {b[1]:.16g} ylo yhi",
        f"0 {c[2]:.16g} zlo zhi",
        f"{b[0]:.16g} {c[0]:.16g} {c[1]:.16g} xy xz yz",
        "",
        "Atoms # atomic",
        "",
    ]
    for k, (z, p) in enumerate(zip(atoms.numbers, atoms.positions, strict=True), start=1):
        sym = next(s for s in species if atomic_numbers[s] == int(z))
        lines.append(f"{k} {species.index(sym) + 1} {p[0]:.16g} {p[1]:.16g} {p[2]:.16g}")
    return "\n".join(lines) + "\n"


def run_lammps_once(atoms: Atoms, calculator, species: list[str], gate=None):
    """Evaluate `atoms` once through LAMMPS ML-IAP; returns (energy, forces by atom id)."""
    if not all(atoms.pbc):
        raise ValueError(
            "parity evaluation needs a fully periodic structure (pbc=True on all axes)"
        )
    import lammps
    import lammps.mliap

    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    lammps.mliap.activate_mliappy(lmp)
    obj = CalculatorMLIAP(calculator, species, lmp=lmp, gate=gate)
    lammps.mliap.load_unified(obj)
    with tempfile.TemporaryDirectory(prefix="matsim-parity-") as tmp:
        data_path = Path(tmp) / "parity.data"
        data_path.write_text(lammps_data_text(atoms, species))
        lmp.commands_string(
            "units metal\natom_style atomic\nboundary p p p\n"
            f'read_data "{data_path}"\n'
            + "".join(f"mass {i + 1} {_mass(s)}\n" for i, s in enumerate(species))
            + "pair_style mliap unified EXISTS 0\n"
            f"pair_coeff * * {' '.join(species)}\n"
            "thermo_style custom step pe\nrun 0\n"
        )
        energy = float(lmp.get_thermo("pe"))
        tags = np.array(lmp.numpy.extract_atom("id")[: len(atoms)], dtype=int)
        f = np.array(lmp.numpy.extract_atom("f")[: len(atoms)], dtype=float)
    out = np.zeros_like(f)
    out[tags - 1] = f
    return energy, out


def _mass(symbol: str) -> float:
    from ase.data import atomic_masses

    return float(atomic_masses[atomic_numbers[symbol]])


def parity_check(
    atoms: Atoms,
    calculator,
    species: list[str],
    *,
    energy_tol_eV: float = 1e-4,
    force_tol_eV_per_A: float = 1e-4,
) -> ParityResult:
    ref = atoms.copy()
    ref.calc = calculator
    e_ase = float(ref.get_potential_energy())
    f_ase = np.asarray(ref.get_forces(), dtype=float)
    e_lmp, f_lmp = run_lammps_once(atoms, calculator, species)
    de = abs(e_ase - e_lmp)
    df = float(np.max(np.abs(f_ase - f_lmp)))
    return ParityResult(
        passed=bool(de <= energy_tol_eV and df <= force_tol_eV_per_A),
        n_atoms=len(atoms),
        energy_ase_eV=e_ase,
        energy_lammps_eV=e_lmp,
        energy_abs_diff_eV=de,
        max_force_abs_diff_eV_per_A=df,
        energy_tol_eV=energy_tol_eV,
        force_tol_eV_per_A=force_tol_eV_per_A,
    )


def write_export_record(
    path: Path,
    *,
    checkpoint: Path,
    species: list[str],
    short_range: dict | None,
    parity: ParityResult | None,
) -> dict:
    """Sidecar JSON tying a LAMMPS run to the exact checkpoint bytes and ZBL choice."""
    record = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "species": species,
        "short_range": short_range,
        "parity": asdict(parity) if parity else None,
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


__all__ = [
    "ParityResult",
    "UncertaintyGate",
    "lammps_data_text",
    "parity_check",
    "run_lammps_once",
    "write_export_record",
]
