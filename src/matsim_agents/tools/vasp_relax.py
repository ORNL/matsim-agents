"""VASP geometry-optimisation helpers.

Mirrors :mod:`matsim_agents.tools.qe_relax` for VASP. There is intentionally
no hard dependency on a VASP install: the Python side only writes input
files (POSCAR / INCAR / KPOINTS / POTCAR), shells out to a user-provided
launcher command, and parses ``vasprun.xml`` (or, as a fallback, ``OUTCAR``)
for the trajectory and final energetics.

This module is the natural pairing for :mod:`qe_relax` when the user wants a
DFT relaxation outside of the active-learning loop. The AL loop itself only
ever does single-point SCF labelling and uses
:mod:`active_learning.backends.vasp` directly — that pathway is unchanged.

Element-aware defaults:
    * ``ENCUT`` is set to ``1.3 × max(ENMAX over POTCARs)`` if a ``potcar_dir``
      is given (PREC=Accurate convention) and falls back to 520 eV otherwise.
    * For systems containing any 3d/4d/5d transition metal, alkali, or
      alkaline-earth element, smearing defaults to ``ISMEAR=1, SIGMA=0.1``
      (Methfessel–Paxton). Otherwise: ``ISMEAR=0, SIGMA=0.05`` (Gaussian).
    * ``KSPACING`` defaults to 0.20 Å⁻¹ for metals, 0.30 Å⁻¹ otherwise.

All defaults can be overridden via :class:`VASPSettings`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reuse the writers we already maintain for the AL backend.
from matsim_agents.active_learning.vasp_io import (
    concat_potcar,
    write_kpoints_auto,
    write_poscar,
)

# Same metallic-element table as qe_relax (kept private to avoid a circular
# import — qe_relax shouldn't depend on vasp_relax and vice versa).
_METALLIC: set[str] = {
    "Li", "Na", "K", "Rb", "Cs",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "La",
}  # fmt: skip

_DEFAULT_ENCUT_EV = 520.0  # safe floor when POTCARs are unavailable

# --------------------------------------------------------------------------- #
# Settings dataclass                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class VASPSettings:
    """All knobs needed to write a self-contained VASP INCAR.

    Most fields are optional; if left as ``None`` they are filled in by
    :func:`recommend_settings` from the structure's composition.

    The ``calculation`` field maps to VASP's ``IBRION/ISIF/NSW`` triple:

    ===============  =======  ======  ===========================================
    calculation      IBRION   ISIF    Meaning
    ===============  =======  ======  ===========================================
    ``scf``           -1       2      Single-point SCF (no ionic / cell motion).
    ``relax``          2       2      Relax atomic positions only (CG).
    ``vc-relax``       2       3      Relax atoms *and* cell (volume + shape).
    ``vc-relax-shape`` 2       4      Relax atoms + cell shape, fixed volume.
    ===============  =======  ======  ===========================================
    """

    calculation: str = "vc-relax"
    encut_ev: float | None = None  # auto from POTCAR's max(ENMAX)*1.3 if None
    prec: str = "Accurate"  # "Normal" / "Accurate"
    ediff: float = 1.0e-6  # electronic SCF convergence (eV)
    ediffg_eV_per_A: float = -0.02  # ionic convergence: max |force| ≤ |ediffg|
    nsw: int = 100  # max ionic steps
    isym: int | None = None  # let VASP decide
    ismear: int | None = None  # auto from composition if None
    sigma: float | None = None  # auto from composition if None
    kspacing: float | None = None  # auto from composition if None
    kgamma: bool = True  # Γ-centered mesh
    kpoints_template: str | None = None  # if set, copied verbatim and KSPACING ignored
    ispin: int = 1  # 1=non-spin, 2=collinear
    lreal: str = "Auto"  # "Auto"/".TRUE."/".FALSE."
    lwave: bool = False  # don't keep WAVECAR by default (huge)
    lcharg: bool = False  # don't keep CHGCAR
    algo: str = "Normal"  # SCF algorithm
    nelm: int = 100  # max electronic SCF iterations
    nelmin: int = 4
    extra_incar: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Composition-aware recommender                                               #
# --------------------------------------------------------------------------- #


_RE_ENMAX = re.compile(r"ENMAX\s*=\s*([-+]?\d+\.\d+)")


def _max_enmax_from_potcar_dir(symbols: list[str], potcar_dir: str | Path) -> float | None:
    """Scan per-element POTCAR files for ENMAX and return the maximum."""
    potcar_dir = Path(potcar_dir)
    if not potcar_dir.is_dir():
        return None
    enmaxs: list[float] = []
    for sym in symbols:
        for cand in (
            potcar_dir / sym / "POTCAR",
            potcar_dir / sym,
            potcar_dir / f"POTCAR.{sym}",
        ):
            if cand.is_file():
                try:
                    text = cand.read_text(errors="replace")
                except OSError:
                    continue
                m = _RE_ENMAX.search(text)
                if m:
                    enmaxs.append(float(m.group(1)))
                break
    if not enmaxs:
        return None
    return max(enmaxs)


def recommend_settings(
    atoms,
    potcar_dir: str | Path | None = None,
    **overrides: Any,
) -> VASPSettings:
    """Produce composition-aware default settings for an ``Atoms`` object.

    ``overrides`` are forwarded verbatim to :class:`VASPSettings` (so the
    caller can write ``recommend_settings(atoms, calculation='scf')``).
    """
    symbols = sorted(set(atoms.get_chemical_symbols()))
    metallic = any(s in _METALLIC for s in symbols)

    encut: float | None = None
    if potcar_dir is not None:
        max_enmax = _max_enmax_from_potcar_dir(symbols, potcar_dir)
        if max_enmax is not None:
            encut = round(max_enmax * 1.3, 1)
    if encut is None:
        encut = _DEFAULT_ENCUT_EV

    if metallic:
        ismear, sigma, kspacing = 1, 0.10, 0.20
    else:
        ismear, sigma, kspacing = 0, 0.05, 0.30

    base = VASPSettings(
        encut_ev=encut,
        ismear=ismear,
        sigma=sigma,
        kspacing=kspacing,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# --------------------------------------------------------------------------- #
# INCAR writer                                                                #
# --------------------------------------------------------------------------- #


_CALC_MODE_MAP: dict[str, tuple[int, int]] = {
    # name              -> (IBRION, ISIF)
    "scf":              (-1, 2),
    "relax":            (2, 2),
    "vc-relax":         (2, 3),
    "vc-relax-shape":   (2, 4),
}  # fmt: skip


def _bool_to_vasp(b: bool) -> str:
    return ".TRUE." if b else ".FALSE."


def write_incar(s: VASPSettings, out_path: str | Path) -> str:
    """Render an INCAR from a :class:`VASPSettings` instance.

    Returns the absolute path written. The output is a self-contained INCAR
    ready to feed ``vasp_std``.
    """
    if s.calculation not in _CALC_MODE_MAP:
        raise ValueError(
            f"Unknown calculation mode {s.calculation!r}. Choose from {sorted(_CALC_MODE_MAP)}."
        )
    if s.encut_ev is None:
        raise ValueError("VASPSettings.encut_ev is required (use recommend_settings).")

    ibrion, isif = _CALC_MODE_MAP[s.calculation]
    nsw = 0 if s.calculation == "scf" else s.nsw

    lines: list[str] = [
        f"# matsim-agents vasp_relax: {s.calculation}",
        "SYSTEM = matsim-agents",
        "",
        "# --- electronic ---",
        f"PREC   = {s.prec}",
        f"ENCUT  = {s.encut_ev}",
        f"EDIFF  = {s.ediff:.1e}",
        f"ALGO   = {s.algo}",
        f"NELM   = {s.nelm}",
        f"NELMIN = {s.nelmin}",
        f"ISPIN  = {s.ispin}",
        f"LREAL  = {s.lreal}",
        f"LWAVE  = {_bool_to_vasp(s.lwave)}",
        f"LCHARG = {_bool_to_vasp(s.lcharg)}",
        "",
        "# --- ionic ---",
        f"IBRION = {ibrion}",
        f"ISIF   = {isif}",
        f"NSW    = {nsw}",
        f"EDIFFG = {s.ediffg_eV_per_A}",
        "",
        "# --- smearing ---",
        f"ISMEAR = {s.ismear if s.ismear is not None else 0}",
        f"SIGMA  = {s.sigma if s.sigma is not None else 0.05}",
    ]
    if s.isym is not None:
        lines += ["", f"ISYM   = {s.isym}"]
    if s.kpoints_template is None and s.kspacing is not None:
        lines += [
            "",
            "# --- k-points (KSPACING in 1/Å, Γ-centered) ---",
            f"KSPACING = {s.kspacing}",
            f"KGAMMA   = {_bool_to_vasp(s.kgamma)}",
        ]
    if s.extra_incar:
        lines += ["", "# --- user overrides ---"]
        lines += [f"{k} = {v}" for k, v in s.extra_incar.items()]
    lines.append("")

    out_path = os.path.abspath(str(out_path))
    Path(out_path).write_text("\n".join(lines))
    return out_path


# --------------------------------------------------------------------------- #
# Workdir preparation                                                         #
# --------------------------------------------------------------------------- #


def prepare_relax_workdir(
    atoms,
    work_dir: str | Path,
    settings: VASPSettings,
    potcar_dir: str | Path,
) -> str:
    """Populate ``work_dir`` with POSCAR + INCAR + POTCAR (+KPOINTS).

    KPOINTS handling:
        * If ``settings.kpoints_template`` is set, that file is copied verbatim.
        * Otherwise ``settings.kspacing`` drives KSPACING in the INCAR and no
          KPOINTS file is written. (Modern VASP recommendation.)
        * If neither is set, an ``Auto 30`` KPOINTS file is written as a
          conservative fallback.
    """
    work_dir = os.path.abspath(str(work_dir))
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    write_poscar(atoms, os.path.join(work_dir, "POSCAR"))
    write_incar(settings, os.path.join(work_dir, "INCAR"))

    if settings.kpoints_template is not None:
        shutil.copyfile(str(settings.kpoints_template), os.path.join(work_dir, "KPOINTS"))
    elif settings.kspacing is None:
        write_kpoints_auto(os.path.join(work_dir, "KPOINTS"), kspacing_per_A=0.30)

    # POTCAR ordering must follow POSCAR's unique-symbol order.
    poscar = Path(work_dir) / "POSCAR"
    symbols_line = poscar.read_text().splitlines()[5]
    concat_potcar(symbols_line.split(), potcar_dir, os.path.join(work_dir, "POTCAR"))

    return work_dir


# --------------------------------------------------------------------------- #
# Parser                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class VASPRelaxResult:
    """Summary of a VASP geometry-optimisation run."""

    work_dir: str
    return_code: int
    converged: bool
    job_done: bool
    n_ionic_steps: int
    scf_iterations_per_step: list[int]
    energies_eV: list[float]  # one per ionic step
    final_energy_eV: float | None
    final_max_force_eV_per_A: float | None
    final_atoms: Any | None  # ASE Atoms or None
    wall_time_sec: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_dir": self.work_dir,
            "return_code": self.return_code,
            "converged": self.converged,
            "job_done": self.job_done,
            "n_ionic_steps": self.n_ionic_steps,
            "scf_iterations_per_step": list(self.scf_iterations_per_step),
            "energies_eV": list(self.energies_eV),
            "final_energy_eV": self.final_energy_eV,
            "final_max_force_eV_per_A": self.final_max_force_eV_per_A,
            "wall_time_sec": self.wall_time_sec,
        }


_RE_REACHED_ACCURACY = re.compile(r"reached required accuracy", re.IGNORECASE)
_RE_OUTCAR_WALL = re.compile(r"Elapsed time \(sec\):\s*([0-9.]+)")
_RE_OUTCAR_NSCF = re.compile(r"-+ Iteration\s+(\d+)\(\s*(\d+)\)")


def parse_relax_outputs(work_dir: str | Path, return_code: int = 0) -> VASPRelaxResult:
    """Parse a VASP relax/vc-relax run from ``work_dir``.

    Strategy: prefer ``vasprun.xml`` (gives full trajectory + energies +
    forces in one ASE call). Fall back to ``OUTCAR`` for walltime and
    convergence detection. ``CONTCAR`` is used as a last-resort source for
    the final geometry.
    """
    import numpy as np  # local: keep module import cheap

    work_dir = os.path.abspath(str(work_dir))
    vasprun = os.path.join(work_dir, "vasprun.xml")
    outcar = os.path.join(work_dir, "OUTCAR")
    contcar = os.path.join(work_dir, "CONTCAR")

    energies: list[float] = []
    scf_per_step: list[int] = []
    final_atoms: Any | None = None
    final_force: float | None = None

    # ---- vasprun.xml: full trajectory ---------------------------------------
    if os.path.isfile(vasprun) and os.path.getsize(vasprun) > 0:
        try:
            from ase.io import read as ase_read

            traj = ase_read(vasprun, index=":", format="vasp-xml")
            if not isinstance(traj, list):
                traj = [traj]
            for frame in traj:
                if frame.calc is not None and "energy" in frame.calc.results:
                    energies.append(float(frame.calc.results["energy"]))
            if traj:
                final_atoms = traj[-1]
                if final_atoms.calc is not None and "forces" in final_atoms.calc.results:
                    f = np.asarray(final_atoms.calc.results["forces"], dtype=np.float64)
                    final_force = float(np.linalg.norm(f, axis=1).max())
        except Exception:  # noqa: BLE001 — many possible XML errors
            pass

    # ---- OUTCAR: SCF iter counts + walltime + convergence flag --------------
    job_done = False
    converged_flag = False
    wall_sec: float | None = None
    if os.path.isfile(outcar):
        try:
            text = Path(outcar).read_text(errors="replace")
        except OSError:
            text = ""
        if text:
            # Group (ionic, electronic) -> count electronic iters per ionic step.
            counts: dict[int, int] = {}
            for m in _RE_OUTCAR_NSCF.finditer(text):
                ionic = int(m.group(1))
                counts[ionic] = max(counts.get(ionic, 0), int(m.group(2)))
            scf_per_step = [counts[k] for k in sorted(counts)]
            converged_flag = bool(_RE_REACHED_ACCURACY.search(text))
            job_done = "General timing and accounting" in text
            mw = _RE_OUTCAR_WALL.search(text)
            if mw:
                try:
                    wall_sec = float(mw.group(1))
                except ValueError:
                    wall_sec = None

    # ---- CONTCAR: last-resort final geometry --------------------------------
    if final_atoms is None and os.path.isfile(contcar):
        try:
            from ase.io import read as ase_read

            final_atoms = ase_read(contcar, format="vasp")
        except Exception:  # noqa: BLE001
            pass

    # If vasprun didn't supply per-step SCF counts but OUTCAR did, prefer that.
    n_ionic = len(energies) if energies else len(scf_per_step)
    if n_ionic == 0:
        n_ionic = 1  # one (failed) attempt

    final_energy = energies[-1] if energies else None
    converged = return_code == 0 and (converged_flag or job_done) and final_energy is not None

    return VASPRelaxResult(
        work_dir=work_dir,
        return_code=return_code,
        converged=converged,
        job_done=job_done,
        n_ionic_steps=n_ionic,
        scf_iterations_per_step=scf_per_step or [0] * n_ionic,
        energies_eV=energies,
        final_energy_eV=final_energy,
        final_max_force_eV_per_A=final_force,
        final_atoms=final_atoms,
        wall_time_sec=wall_sec,
    )


# --------------------------------------------------------------------------- #
# Subprocess runner                                                           #
# --------------------------------------------------------------------------- #


def run_vasp(
    work_dir: str,
    launcher_cmd: list[str] | str,
    stdout_name: str = "vasp.out",
    timeout_sec: int | None = None,
) -> VASPRelaxResult:
    """Run VASP via the user-provided launcher and parse the relax outputs.

    ``launcher_cmd`` may be a list (e.g.
    ``["bash", "run-vasp-gpu-frontier.sh"]``) or a single string. VASP reads
    its inputs from the current working directory, so unlike QE the input
    file path is *not* appended to argv. The wrapper is fully responsible
    for invoking ``vasp_std`` with the right MPI launcher.
    """
    work_dir = os.path.abspath(work_dir)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    stdout_path = os.path.join(work_dir, stdout_name)

    argv = [launcher_cmd] if isinstance(launcher_cmd, str) else list(launcher_cmd)

    t0 = time.time()
    with open(stdout_path, "w") as out:
        try:
            proc = subprocess.run(
                argv,
                cwd=work_dir,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    elapsed = time.time() - t0

    result = parse_relax_outputs(work_dir, return_code=rc)
    if result.wall_time_sec is None:
        result.wall_time_sec = elapsed
    return result


def find_default_launcher() -> str | None:
    """Return the env-overridable default VASP launcher, or ``None``."""
    val = os.environ.get("MATSIM_VASP_LAUNCHER")
    if val and shutil.which(val.split()[0] if " " in val else val):
        return val
    return None
