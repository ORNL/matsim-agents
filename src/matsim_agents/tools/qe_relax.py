"""Quantum ESPRESSO ``pw.x`` geometry-optimisation helpers.

This module is intentionally lightweight and has no hard dependency on a QE
install: the Python side only generates a ``pw.x`` input deck, shells out to
a user-provided launcher command, and parses the resulting stdout. ``ase`` is
used for I/O and to inspect the input ``Atoms`` object.

The element-aware defaults (plane-wave cutoffs, smearing, k-mesh) are loosely
based on the SSSP-PBE-efficiency (1.3) recommendations. They are meant to be
*reasonable*, not authoritative — production studies should pick parameters
deliberately. All defaults can be overridden through ``QESettings``.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Element-aware defaults                                                      #
# --------------------------------------------------------------------------- #

# Recommended ecutwfc per element (Ry) — rounded SSSP-efficiency-1.3 values.
# Missing entries fall back to ``_DEFAULT_ECUTWFC``.
_ECUTWFC_RY: dict[str, int] = {
    "H": 60, "Li": 40, "Be": 60, "B": 35, "C": 60, "N": 60, "O": 50, "F": 50,
    "Na": 40, "Mg": 30, "Al": 30, "Si": 30, "P": 30, "S": 35, "Cl": 40,
    "K": 60, "Ca": 30, "Sc": 40, "Ti": 40, "V": 40, "Cr": 40, "Mn": 90,
    "Fe": 90, "Co": 45, "Ni": 45, "Cu": 55, "Zn": 40,
    "Ga": 70, "Ge": 40, "As": 35, "Se": 30, "Br": 30,
    "Rb": 30, "Sr": 30, "Y": 35, "Zr": 30, "Nb": 40, "Mo": 35, "Tc": 30,
    "Ru": 35, "Rh": 35, "Pd": 45, "Ag": 50, "Cd": 60,
    "In": 50, "Sn": 70, "Sb": 40, "Te": 30, "I": 35,
    "Cs": 30, "Ba": 30, "La": 40,
    "Hf": 50, "Ta": 45, "W": 30, "Re": 30, "Os": 40, "Ir": 50, "Pt": 35,
    "Au": 45, "Hg": 50, "Tl": 50, "Pb": 40, "Bi": 45,
}

_DEFAULT_ECUTWFC = 50  # Ry — safe floor when an element is missing from table

# Elements that are metallic (or routinely benefit from smearing). When any of
# these appears in the input, occupations='smearing' with a small Gaussian
# degauss is used.
_METALLIC: set[str] = {
    # Alkalis & alkaline earths (besides Be/Mg which are also metallic enough).
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    # 3d / 4d / 5d transition metals.
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    # Post-transition / metalloid metals.
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "La",
}

# Approximate atomic masses (g/mol) for ATOMIC_SPECIES — pw.x is forgiving
# about the exact value, but it must be present.
_ATOMIC_MASS: dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990,
    "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Sc": 44.956,
    "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Rb": 85.468,
    "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Tc": 98.0, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87,
    "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60,
    "I": 126.90, "Xe": 131.29, "Cs": 132.91, "Ba": 137.33, "La": 138.91,
    "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23,
    "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Tl": 204.38,
    "Pb": 207.2, "Bi": 208.98,
}


# --------------------------------------------------------------------------- #
# Settings                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class QESettings:
    """All knobs needed to write a ``pw.x`` input deck.

    Most fields are optional; if left as ``None`` they are filled in by
    :func:`recommend_settings` from the structure's composition.
    """

    pseudo_dir: str
    calculation: str = "vc-relax"          # "relax" (atoms only) or "vc-relax"
    ecutwfc_ry: float | None = None
    ecutrho_ry: float | None = None
    occupations: str | None = None         # "smearing" | "fixed" | "tetrahedra"
    smearing: str = "gaussian"
    degauss_ry: float = 0.01
    kpts: tuple[int, int, int] | None = None
    koffset: tuple[int, int, int] = (0, 0, 0)
    is_2d: bool = False                    # if True, k-grid in vacuum dir = 1
    forc_conv_thr_ry_au: float = 1.0e-3
    etot_conv_thr_ry: float = 1.0e-4
    conv_thr_ry: float = 1.0e-8
    nstep: int = 100
    mixing_beta: float = 0.4
    pseudopotentials: dict[str, str] | None = None  # {symbol: filename}
    extra_control: dict[str, Any] = field(default_factory=dict)
    extra_system: dict[str, Any] = field(default_factory=dict)
    extra_electrons: dict[str, Any] = field(default_factory=dict)
    extra_ions: dict[str, Any] = field(default_factory=dict)
    extra_cell: dict[str, Any] = field(default_factory=dict)


def recommend_settings(atoms, pseudo_dir: str, **overrides) -> QESettings:
    """Produce composition-aware default settings for an ``Atoms`` object.

    The user can override any field via ``overrides`` keyword arguments;
    they are forwarded verbatim to ``QESettings``.
    """
    symbols = sorted(set(atoms.get_chemical_symbols()))

    ecutwfc = max((_ECUTWFC_RY.get(s, _DEFAULT_ECUTWFC) for s in symbols),
                  default=_DEFAULT_ECUTWFC)
    ecutrho = ecutwfc * 8.0

    metallic = any(s in _METALLIC for s in symbols)
    occupations = "smearing" if metallic else "fixed"

    cell_lengths = atoms.cell.lengths()
    target = 30.0 if metallic else 20.0  # ~ kpoints * length [Å]
    kpts = tuple(max(1, int(math.ceil(target / float(L)))) for L in cell_lengths)

    # Detect 2-D / slab-like cells: largest dimension noticeably bigger than
    # the others. If so, flatten the k-mesh in the vacuum direction.
    is_2d = bool(overrides.pop("is_2d", False))
    if not is_2d:
        max_idx = int(np.argmax(cell_lengths))
        max_L = float(cell_lengths[max_idx])
        others = float(min(cell_lengths[i] for i in range(3) if i != max_idx))
        if max_L > 1.8 * others:
            is_2d = True
    if is_2d:
        max_idx = int(np.argmax(cell_lengths))
        kpts = tuple(1 if i == max_idx else kpts[i] for i in range(3))

    pseudos = _autodetect_pseudos(symbols, pseudo_dir)

    base = QESettings(
        pseudo_dir=pseudo_dir,
        ecutwfc_ry=ecutwfc,
        ecutrho_ry=ecutrho,
        occupations=occupations,
        kpts=kpts,
        is_2d=is_2d,
        pseudopotentials=pseudos,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _autodetect_pseudos(symbols: list[str], pseudo_dir: str) -> dict[str, str]:
    """Pick a pseudopotential file per element by glob-matching ``pseudo_dir``.

    Looks for files starting with ``<Symbol>.`` or ``<Symbol>_`` (case-
    insensitive). Raises ``FileNotFoundError`` if any element has no match —
    callers that want to defer this to runtime can pass ``pseudopotentials=``
    explicitly to ``QESettings``.
    """
    if not os.path.isdir(pseudo_dir):
        # Defer: caller may supply pseudopotentials= manually. Return empty.
        return {}
    out: dict[str, str] = {}
    files = os.listdir(pseudo_dir)
    for sym in symbols:
        prefix_dot = sym.lower() + "."
        prefix_us = sym.lower() + "_"
        prefix_dash = sym.lower() + "-"
        candidates = [
            f for f in files
            if f.lower().startswith((prefix_dot, prefix_us, prefix_dash))
            and f.lower().endswith((".upf", ".upf.gz"))
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No pseudopotential file found for {sym!r} in {pseudo_dir} "
                f"(looked for {sym}.*.UPF). Pass pseudopotentials= to override."
            )
        candidates.sort(key=len)
        out[sym] = candidates[0]
    return out


# --------------------------------------------------------------------------- #
# Input deck                                                                  #
# --------------------------------------------------------------------------- #


def write_pw_input(
    atoms,
    settings: QESettings,
    input_path: str,
    prefix: str = "pwscf",
    outdir: str = "./tmp",
    title: str | None = None,
) -> str:
    """Write a ``pw.x`` input file for ``atoms`` using ``settings``.

    Returns the absolute path of the file that was written.
    """
    settings = _fill_defaults(atoms, settings)
    symbols_unique = sorted(set(atoms.get_chemical_symbols()))

    if not settings.pseudopotentials:
        raise ValueError(
            "QESettings.pseudopotentials is empty. Either point pseudo_dir at "
            "a directory with .UPF files (auto-detection) or pass an explicit "
            "{symbol: filename} dictionary."
        )
    missing = [s for s in symbols_unique if s not in settings.pseudopotentials]
    if missing:
        raise ValueError(f"Missing pseudopotentials for: {missing}")

    nat = len(atoms)
    ntyp = len(symbols_unique)

    control = {
        "calculation": f"'{settings.calculation}'",
        "prefix": f"'{prefix}'",
        "pseudo_dir": f"'{os.path.abspath(settings.pseudo_dir)}'",
        "outdir": f"'{outdir}'",
        "tprnfor": ".true.",
        "tstress": ".true.",
        "verbosity": "'high'",
        "etot_conv_thr": _fortran(settings.etot_conv_thr_ry),
        "forc_conv_thr": _fortran(settings.forc_conv_thr_ry_au),
        "nstep": str(settings.nstep),
    }
    control.update({k: _fortran(v) for k, v in settings.extra_control.items()})

    system = {
        "ibrav": "0",
        "nat": str(nat),
        "ntyp": str(ntyp),
        "ecutwfc": _fortran(settings.ecutwfc_ry),
        "ecutrho": _fortran(settings.ecutrho_ry),
        "occupations": f"'{settings.occupations}'",
    }
    if settings.occupations == "smearing":
        system["smearing"] = f"'{settings.smearing}'"
        system["degauss"] = _fortran(settings.degauss_ry)
    system.update({k: _fortran(v) for k, v in settings.extra_system.items()})

    electrons = {
        "conv_thr": _fortran(settings.conv_thr_ry),
        "mixing_beta": _fortran(settings.mixing_beta),
        "electron_maxstep": "200",
    }
    electrons.update({k: _fortran(v) for k, v in settings.extra_electrons.items()})

    ions = {"ion_dynamics": "'bfgs'"}
    ions.update({k: _fortran(v) for k, v in settings.extra_ions.items()})

    cell = {"cell_dynamics": "'bfgs'"} if settings.calculation == "vc-relax" else {}
    cell.update({k: _fortran(v) for k, v in settings.extra_cell.items()})

    parts: list[str] = []
    if title:
        parts.append(f"! {title}")
    parts.append(_namelist("CONTROL", control))
    parts.append(_namelist("SYSTEM", system))
    parts.append(_namelist("ELECTRONS", electrons))
    parts.append(_namelist("IONS", ions))
    if cell:
        parts.append(_namelist("CELL", cell))

    parts.append("ATOMIC_SPECIES")
    for sym in symbols_unique:
        mass = _ATOMIC_MASS.get(sym, 1.0)
        parts.append(f"  {sym}  {mass:.4f}  {settings.pseudopotentials[sym]}")
    parts.append("")

    parts.append("CELL_PARAMETERS angstrom")
    for v in atoms.cell.array:
        parts.append(f"  {v[0]:18.10f} {v[1]:18.10f} {v[2]:18.10f}")
    parts.append("")

    parts.append("ATOMIC_POSITIONS angstrom")
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True):
        parts.append(f"  {sym}  {pos[0]:18.10f} {pos[1]:18.10f} {pos[2]:18.10f}")
    parts.append("")

    nk = settings.kpts or (1, 1, 1)
    ko = settings.koffset
    parts.append("K_POINTS automatic")
    parts.append(f"  {nk[0]} {nk[1]} {nk[2]}  {ko[0]} {ko[1]} {ko[2]}")
    parts.append("")

    Path(input_path).parent.mkdir(parents=True, exist_ok=True)
    Path(input_path).write_text("\n".join(parts))
    return os.path.abspath(input_path)


def _fill_defaults(atoms, s: QESettings) -> QESettings:
    """Backfill any None fields in ``s`` from element-aware defaults."""
    if (s.ecutwfc_ry is None or s.ecutrho_ry is None or s.occupations is None
            or s.kpts is None or not s.pseudopotentials):
        rec = recommend_settings(atoms, s.pseudo_dir)
        if s.ecutwfc_ry is None:
            s.ecutwfc_ry = rec.ecutwfc_ry
        if s.ecutrho_ry is None:
            s.ecutrho_ry = rec.ecutrho_ry
        if s.occupations is None:
            s.occupations = rec.occupations
        if s.kpts is None:
            s.kpts = rec.kpts
        if not s.pseudopotentials:
            s.pseudopotentials = rec.pseudopotentials
    return s


def _namelist(name: str, kv: dict[str, str]) -> str:
    lines = [f"&{name}"]
    width = max((len(k) for k in kv), default=1)
    for k, v in kv.items():
        lines.append(f"  {k.ljust(width)} = {v}")
    lines.append("/")
    return "\n".join(lines)


def _fortran(v: Any) -> str:
    if isinstance(v, bool):
        return ".true." if v else ".false."
    if isinstance(v, float):
        return f"{v:.8e}".replace("e", "d")
    if isinstance(v, (list, tuple)):
        return " ".join(_fortran(x) for x in v)
    return str(v)


# --------------------------------------------------------------------------- #
# Output parsing                                                              #
# --------------------------------------------------------------------------- #


_RY_TO_EV = 13.605693122994


@dataclass
class PWResult:
    """Summary of a ``pw.x`` geometry-optimisation stdout."""

    stdout_path: str
    bfgs_steps: int
    scf_iterations_total: int
    scf_iterations_per_step: list[int]
    final_energy_ry: float
    final_energy_ev: float
    final_max_force_ry_au: float | None
    converged: bool
    job_done: bool
    wall_time_sec: float | None
    return_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "stdout_path": self.stdout_path,
            "bfgs_steps": self.bfgs_steps,
            "scf_iterations_total": self.scf_iterations_total,
            "scf_iterations_per_step": list(self.scf_iterations_per_step),
            "final_energy_ry": self.final_energy_ry,
            "final_energy_ev": self.final_energy_ev,
            "final_max_force_ry_au": self.final_max_force_ry_au,
            "converged": self.converged,
            "job_done": self.job_done,
            "wall_time_sec": self.wall_time_sec,
            "return_code": self.return_code,
        }


_RE_TOTAL_ENERGY = re.compile(r"!\s*total energy\s*=\s*(-?\d+\.\d+)\s*Ry")
_RE_ITER = re.compile(r"iteration #\s*(\d+)\s+ecut")
_RE_BFGS_STEP = re.compile(r"number of scf cycles\s*=\s*(\d+)")
_RE_BFGS_CONV = re.compile(r"bfgs converged in\s*(\d+)\s*scf cycles")
_RE_FORCE_MAX = re.compile(r"Total force\s*=\s*(-?\d+\.\d+)\s*Total SCF correction")
_RE_WALL = re.compile(r"PWSCF\s*:.*?CPU\s+(?:[\d:.h ms]+)\s+WALL")
_RE_WALL_SEC = re.compile(r"PWSCF\s*:.*?(?P<w>[\d.]+s)\s+WALL", re.IGNORECASE)
_RE_JOB_DONE = re.compile(r"JOB DONE\.")


def parse_pw_stdout(stdout_path: str, return_code: int = 0) -> PWResult:
    """Parse a ``pw.x`` stdout produced by a ``relax`` / ``vc-relax`` run."""
    text = Path(stdout_path).read_text(errors="replace")

    # SCF iterations grouped by BFGS step. pw.x prints "iteration #" inside
    # each electronic SCF cycle; the boundary between geometry steps is the
    # "number of scf cycles" line (printed once at the start of each ionic
    # iteration except possibly the first). We split on those boundaries.
    iter_matches = list(_RE_ITER.finditer(text))
    bfgs_boundaries = [m.start() for m in _RE_BFGS_STEP.finditer(text)]

    scf_per_step: list[int] = []
    if not bfgs_boundaries:
        scf_per_step.append(len(iter_matches))
    else:
        # First ionic step covers iterations before the first boundary.
        first = sum(1 for m in iter_matches if m.start() < bfgs_boundaries[0])
        scf_per_step.append(first)
        for i, b in enumerate(bfgs_boundaries):
            nxt = bfgs_boundaries[i + 1] if i + 1 < len(bfgs_boundaries) else len(text)
            scf_per_step.append(sum(1 for m in iter_matches if b <= m.start() < nxt))

    scf_total = sum(scf_per_step)
    bfgs_steps = max(1, len(scf_per_step))

    energies_ry = [float(m.group(1)) for m in _RE_TOTAL_ENERGY.finditer(text)]
    final_e_ry = energies_ry[-1] if energies_ry else float("nan")

    forces = [float(m.group(1)) for m in _RE_FORCE_MAX.finditer(text)]
    final_force = forces[-1] if forces else None

    bfgs_done = _RE_BFGS_CONV.search(text)
    if bfgs_done:
        bfgs_steps = int(bfgs_done.group(1))

    job_done = bool(_RE_JOB_DONE.search(text))
    converged = bool(bfgs_done) or job_done

    wall = _RE_WALL_SEC.search(text)
    wall_sec: float | None
    if wall:
        try:
            wall_sec = float(wall.group("w").rstrip("s"))
        except ValueError:
            wall_sec = None
    else:
        wall_sec = None

    return PWResult(
        stdout_path=os.path.abspath(stdout_path),
        bfgs_steps=bfgs_steps,
        scf_iterations_total=scf_total,
        scf_iterations_per_step=scf_per_step,
        final_energy_ry=final_e_ry,
        final_energy_ev=final_e_ry * _RY_TO_EV if not math.isnan(final_e_ry) else float("nan"),
        final_max_force_ry_au=final_force,
        converged=converged,
        job_done=job_done,
        wall_time_sec=wall_sec,
        return_code=return_code,
    )


# --------------------------------------------------------------------------- #
# Subprocess runner                                                           #
# --------------------------------------------------------------------------- #


def run_pw(
    input_path: str,
    work_dir: str,
    launcher_cmd: list[str] | str,
    stdout_name: str = "pw.out",
    timeout_sec: int | None = None,
) -> PWResult:
    """Run ``pw.x`` via the user-provided launcher and parse the output.

    ``launcher_cmd`` may be a list (e.g. ``["bash", "run-pw-gpu-frontier.sh"]``)
    or a single string treated as one argv element. The QE input file path is
    appended as the final argument. Set the env var
    ``MATSIM_QE_LAUNCHER_APPEND_FLAG`` to e.g. ``-in`` if your wrapper expects
    a flag before the input path.
    """
    work_dir = os.path.abspath(work_dir)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    stdout_path = os.path.join(work_dir, stdout_name)

    if isinstance(launcher_cmd, str):
        argv = [launcher_cmd]
    else:
        argv = list(launcher_cmd)

    flag = os.environ.get("MATSIM_QE_LAUNCHER_APPEND_FLAG", "").strip()
    if flag:
        argv += [flag, input_path]
    else:
        argv += [input_path]

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

    result = parse_pw_stdout(stdout_path, return_code=rc)
    if result.wall_time_sec is None:
        result.wall_time_sec = elapsed
    return result


def find_default_launcher() -> str | None:
    """Return the env-overridable default QE launcher, or ``None``."""
    val = os.environ.get("MATSIM_QE_LAUNCHER")
    if val and shutil.which(val.split()[0] if " " in val else val):
        return val
    return None
