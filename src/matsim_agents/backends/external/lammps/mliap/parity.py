"""ASE vs LAMMPS ML-IAP energy/force parity gate and checkpoint-hash export record."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase import units as ase_units
from ase.data import atomic_numbers
from ase.geometry import find_mic
from ase.md.verlet import VelocityVerlet

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
    if atoms.has("momenta"):
        velocities = atoms.get_velocities()
        # ASE velocity unit -> Angstrom/ps (LAMMPS metal units)
        scale = 1000.0 * ase_units.fs
        lines += ["", "Velocities", ""]
        lines += [
            f"{k} {v[0] * scale:.16g} {v[1] * scale:.16g} {v[2] * scale:.16g}"
            for k, v in enumerate(velocities, start=1)
        ]
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
        out = _gather(lmp, "f", len(atoms))
    return energy, out


def _gather(lmp, name: str, natoms: int) -> np.ndarray:
    """Per-atom vectors of all ranks, ordered by atom id (the data file ids are 1..N)."""
    return np.array(lmp.gather_atoms(name, 1, 3), dtype=float).reshape(natoms, 3)


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


@dataclass(frozen=True)
class MDStabilityResult:
    passed: bool
    n_atoms: int
    steps: int
    timestep_fs: float
    finite: bool
    atoms_conserved: bool
    max_total_energy_deviation_eV_per_atom: float
    energy_tol_eV_per_atom: float
    max_position_diff_vs_ase_A: float
    position_tol_A: float
    max_displacement_A: float


def run_lammps_md(
    atoms: Atoms,
    calculator,
    species: list[str],
    *,
    steps: int,
    timestep_fs: float,
    gate=None,
):
    """NVE through LAMMPS ML-IAP from the velocities carried by `atoms`.

    Returns (positions by atom id, total energies at steps 0..steps, final atom count).
    """
    if steps < 1 or timestep_fs <= 0:
        raise ValueError("steps must be at least 1 and timestep_fs positive")
    if not all(atoms.pbc):
        raise ValueError("MD needs a fully periodic structure (pbc=True on all axes)")
    import lammps
    import lammps.mliap

    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    lammps.mliap.activate_mliappy(lmp)
    obj = CalculatorMLIAP(calculator, species, lmp=lmp, gate=gate)
    lammps.mliap.load_unified(obj)
    with tempfile.TemporaryDirectory(prefix="matsim-md-") as tmp:
        data_path = Path(tmp) / "md.data"
        data_path.write_text(lammps_data_text(atoms, species))
        lmp.commands_string(
            "units metal\natom_style atomic\nboundary p p p\n"
            f'read_data "{data_path}"\n'
            + "".join(f"mass {i + 1} {_mass(s)}\n" for i, s in enumerate(species))
            + "pair_style mliap unified EXISTS 0\n"
            f"pair_coeff * * {' '.join(species)}\n"
            f"timestep {timestep_fs / 1000.0:.16g}\n"
            "fix integrate all nve\n"
            "thermo_style custom step pe ke etotal\n"
        )
        energies = []
        lmp.command("run 0")
        energies.append(float(lmp.get_thermo("etotal")))
        for _ in range(steps):
            lmp.command("run 1 pre no post no")
            energies.append(float(lmp.get_thermo("etotal")))
        natoms = int(lmp.get_natoms())
        positions = _gather(lmp, "x", natoms)
    return positions, np.array(energies), natoms


def md_stability_check(
    atoms: Atoms,
    calculator,
    species: list[str],
    *,
    steps: int = 50,
    timestep_fs: float = 1.0,
    energy_tol_eV_per_atom: float = 1e-3,
    position_tol_A: float = 1e-5,
) -> MDStabilityResult:
    """NVE stability of the ML-IAP bridge: finite values, conserved atoms and energy, and the
    LAMMPS trajectory against an ASE velocity-Verlet run from the same state.

    `atoms` must carry velocities (for example from ``MaxwellBoltzmannDistribution``).
    """
    if not atoms.has("momenta"):
        raise ValueError("atoms must carry velocities; set them before the stability check")
    positions, energies, natoms = run_lammps_md(
        atoms, calculator, species, steps=steps, timestep_fs=timestep_fs
    )
    reference = atoms.copy()
    reference.calc = calculator
    VelocityVerlet(reference, timestep=timestep_fs * ase_units.fs).run(steps)
    delta, _ = find_mic(positions - reference.positions, atoms.cell, pbc=True)
    max_diff = float(np.max(np.abs(delta)))
    moved, _ = find_mic(reference.positions - atoms.positions, atoms.cell, pbc=True)
    finite = bool(np.isfinite(positions).all() and np.isfinite(energies).all())
    conserved = natoms == len(atoms)
    deviation = float(np.max(np.abs(energies - energies[0])) / len(atoms)) if finite else np.inf
    return MDStabilityResult(
        passed=bool(
            finite
            and conserved
            and deviation <= energy_tol_eV_per_atom
            and max_diff <= position_tol_A
        ),
        n_atoms=len(atoms),
        steps=steps,
        timestep_fs=timestep_fs,
        finite=finite,
        atoms_conserved=conserved,
        max_total_energy_deviation_eV_per_atom=deviation,
        energy_tol_eV_per_atom=energy_tol_eV_per_atom,
        max_position_diff_vs_ase_A=max_diff,
        position_tol_A=position_tol_A,
        max_displacement_A=float(np.max(np.linalg.norm(moved, axis=1))),
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
    "MDStabilityResult",
    "ParityResult",
    "UncertaintyGate",
    "lammps_data_text",
    "md_stability_check",
    "parity_check",
    "run_lammps_md",
    "run_lammps_once",
    "write_export_record",
]
