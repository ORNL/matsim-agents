"""Read/write VASP I/O via ASE, with composition-aware POTCAR concatenation.

We deliberately use ASE's ``ase.io.vasp`` for POSCAR + ``ase.io.vasp_parsers``
(or ``vasprun.xml`` parser) for outputs rather than rolling our own parser.
ASE's POTCAR support requires ``$VASP_PP_PATH`` pointing at the parent of the
``potpaw_PBE/`` directory; we work around that by concatenating per-element
POTCAR files manually so the user only needs to point ``potcar_dir`` at a
flat directory of element-named POTCAR files.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ASE is a hard dep of matsim-agents (see pyproject.toml).
from ase import Atoms
from ase.io import read as ase_read
from ase.io.vasp import write_vasp

# --------------------------------------------------------------------------- #
# Result dataclass                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class VASPResult:
    """Outcome of one VASP single-point (or relax) calculation."""

    work_dir: str
    return_code: int
    converged: bool
    energy_eV: float | None
    forces_eV_per_A: Any | None  # numpy array (N, 3); Any avoids a hard numpy import
    stress_eV_per_A3: Any | None  # numpy array, shape (3, 3) or (6,)
    n_atoms: int | None
    wall_time_sec: float | None
    final_atoms: Atoms | None  # parsed from CONTCAR or vasprun.xml
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #


def write_poscar(atoms: Atoms, path: str | Path, sort: bool = True) -> str:
    """Write a POSCAR for ``atoms``. Returns the absolute path written."""
    path = os.path.abspath(str(path))
    write_vasp(path, atoms, direct=True, sort=sort, vasp5=True)
    return path


def write_incar(
    template_path: str | Path,
    out_path: str | Path,
    extra: dict[str, Any] | None = None,
    **fmt_kwargs: Any,
) -> str:
    """Render an INCAR from a template file.

    The template is read as text and ``str.format(**fmt_kwargs)`` is applied,
    then any ``extra`` key=value pairs are appended (one per line). This lets
    the caller hard-code most of the INCAR while overriding e.g. NSW or NCORE
    per job.
    """
    template_path = Path(template_path)
    out_path = os.path.abspath(str(out_path))
    text = template_path.read_text()
    if fmt_kwargs:
        text = text.format(**fmt_kwargs)
    if extra:
        appended = "\n".join(f"{k} = {v}" for k, v in extra.items())
        text = text.rstrip() + "\n\n# --- per-job overrides ---\n" + appended + "\n"
    Path(out_path).write_text(text)
    return out_path


def write_kpoints_auto(out_path: str | Path, kspacing_per_A: float = 0.3) -> str:
    """Write a minimal KPOINTS file using the Auto scheme (length parameter).

    For most AL workflows we recommend setting KSPACING in the INCAR instead
    (no KPOINTS file needed). This helper exists for templates that prefer an
    explicit file. ``kspacing_per_A`` is mapped to VASP's KSPACING via the
    ``Auto N`` line where N ≈ 2π / kspacing.
    """
    out_path = os.path.abspath(str(out_path))
    n = max(1, int(round(2 * 3.141592653589793 / kspacing_per_A)))
    Path(out_path).write_text(f"K-Points\n0\nAuto\n{n}\n")
    return out_path


def concat_potcar(
    symbols_in_poscar_order: list[str],
    potcar_dir: str | Path,
    out_path: str | Path,
) -> str:
    """Concatenate per-element POTCAR files in the order they appear in POSCAR.

    Looks for ``potcar_dir/<Symbol>/POTCAR`` first, then ``potcar_dir/<Symbol>``,
    then ``potcar_dir/POTCAR.<Symbol>``. Raises ``FileNotFoundError`` if any
    element is missing — this fails fast so a typo in ``potcar_dir`` never
    silently produces a wrong calculation.
    """
    potcar_dir = Path(potcar_dir)
    out_path = os.path.abspath(str(out_path))
    if not potcar_dir.is_dir():
        raise FileNotFoundError(f"potcar_dir not a directory: {potcar_dir}")

    # Deduplicate while preserving order; POSCAR's POTCAR ordering matches the
    # *unique* symbols in the order they first appear.
    seen: list[str] = []
    for s in symbols_in_poscar_order:
        if s not in seen:
            seen.append(s)

    # Preferred PAW variant when the bare element directory is absent.
    # Keys are element symbols; values are the subdirectory names to try next.
    _PREFERRED_VARIANTS: dict[str, list[str]] = {
        "Ba": ["Ba_sv"],
        "Ca": ["Ca_sv", "Ca_pv"],
        "Sr": ["Sr_sv"],
        "K": ["K_sv", "K_pv"],
        "Rb": ["Rb_sv", "Rb_pv"],
        "Cs": ["Cs_sv"],
        "Na": ["Na_pv"],
        "Li": ["Li_sv"],
        "Pb": ["Pb_d"],
        "Bi": ["Bi_d"],
        "In": ["In_d"],
        "Tl": ["Tl_d"],
        "Hf": ["Hf_pv"],
        "Zr": ["Zr_sv"],
        "Y": ["Y_sv"],
        "Sc": ["Sc_sv"],
        "La": ["La"],
        "Ce": ["Ce"],
    }

    pieces: list[bytes] = []
    for sym in seen:
        # Build candidate list: plain element first, then preferred variants,
        # then legacy flat-file conventions.
        variant_dirs = [sym] + _PREFERRED_VARIANTS.get(sym, [])
        candidates = []
        for vd in variant_dirs:
            candidates += [
                potcar_dir / vd / "POTCAR",
                potcar_dir / vd,
            ]
        candidates += [potcar_dir / f"POTCAR.{sym}"]
        match = next((c for c in candidates if c.is_file()), None)
        if match is None:
            raise FileNotFoundError(
                f"No POTCAR found for element {sym!r}. Tried: "
                + ", ".join(str(c) for c in candidates)
            )
        pieces.append(match.read_bytes())

    Path(out_path).write_bytes(b"".join(pieces))
    return out_path


def prepare_vasp_workdir(
    atoms: Atoms,
    work_dir: str | Path,
    incar_template: str | Path,
    potcar_dir: str | Path,
    kpoints_template: str | Path | None = None,
    kspacing_per_A: float | None = None,
    extra_incar: dict[str, Any] | None = None,
) -> str:
    """Create ``work_dir`` and populate it with POSCAR, INCAR, KPOINTS, POTCAR.

    Returns the absolute work_dir path. ``kpoints_template`` takes precedence;
    if both it and ``kspacing_per_A`` are None, no KPOINTS file is written and
    the INCAR is expected to set ``KSPACING``.
    """
    work_dir = os.path.abspath(str(work_dir))
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    write_poscar(atoms, os.path.join(work_dir, "POSCAR"))

    write_incar(
        incar_template,
        os.path.join(work_dir, "INCAR"),
        extra=extra_incar,
        natoms=len(atoms),
    )

    if kpoints_template is not None:
        shutil.copyfile(str(kpoints_template), os.path.join(work_dir, "KPOINTS"))
    elif kspacing_per_A is not None:
        write_kpoints_auto(os.path.join(work_dir, "KPOINTS"), kspacing_per_A)

    # POTCAR ordering must match POSCAR's unique-symbol order.
    poscar = Path(work_dir) / "POSCAR"
    symbols_line = poscar.read_text().splitlines()[5]
    symbols = symbols_line.split()
    concat_potcar(symbols, potcar_dir, os.path.join(work_dir, "POTCAR"))

    return work_dir


# --------------------------------------------------------------------------- #
# Outputs                                                                     #
# --------------------------------------------------------------------------- #


_OSZICAR_E_RE = re.compile(r"E0\s*=\s*([-+]?\d+\.\d+E[-+]?\d+|[-+]?\d+\.\d+)")


def _parse_walltime_outcar(outcar_path: str) -> float | None:
    """Best-effort wall-time extraction from OUTCAR's ``Total CPU time used``."""
    try:
        with open(outcar_path) as f:
            for line in f:
                if "Elapsed time" in line:
                    parts = line.strip().split(":")
                    if len(parts) >= 2:
                        try:
                            return float(parts[-1].strip())
                        except ValueError:
                            return None
    except OSError:
        pass
    return None


def parse_vasp_workdir(work_dir: str | Path, return_code: int = 0) -> VASPResult:
    """Parse VASP outputs from ``work_dir``.

    Strategy: prefer ``vasprun.xml`` (gives Atoms + energy + forces + stress
    in one shot via ASE). Fall back to ``OUTCAR`` if vasprun is missing or
    truncated. Convergence is reported as True iff the output files parsed
    cleanly *and* return_code == 0.
    """
    import numpy as np  # local to keep module import cheap

    work_dir = os.path.abspath(str(work_dir))
    vasprun = os.path.join(work_dir, "vasprun.xml")
    outcar = os.path.join(work_dir, "OUTCAR")
    contcar = os.path.join(work_dir, "CONTCAR")

    energy: float | None = None
    forces: np.ndarray | None = None
    stress: np.ndarray | None = None
    final_atoms: Atoms | None = None
    converged = False
    notes: list[str] = []

    if os.path.isfile(vasprun) and os.path.getsize(vasprun) > 0:
        try:
            final_atoms = ase_read(vasprun, index=-1, format="vasp-xml")
            calc = final_atoms.calc
            if calc is not None:
                energy = float(calc.results.get("energy", calc.results.get("free_energy", 0.0)))
                if "forces" in calc.results:
                    forces = np.asarray(calc.results["forces"], dtype=np.float64)
                if "stress" in calc.results:
                    stress = np.asarray(calc.results["stress"], dtype=np.float64)
            converged = (return_code == 0) and (energy is not None) and (forces is not None)
        except Exception as exc:  # noqa: BLE001 — many possible XML errors
            notes.append(f"vasprun.xml parse failed: {exc}")

    if energy is None and os.path.isfile(outcar):
        # Last-resort: scrape OSZICAR / OUTCAR for the final energy
        oszicar = os.path.join(work_dir, "OSZICAR")
        if os.path.isfile(oszicar):
            with open(oszicar) as f:
                for m in _OSZICAR_E_RE.finditer(f.read()):
                    with contextlib.suppress(ValueError):
                        energy = float(m.group(1))
        # Try ASE's OUTCAR reader too (slower)
        try:
            final_atoms = final_atoms or ase_read(outcar, index=-1, format="vasp-out")
            if final_atoms is not None and final_atoms.calc is not None:
                forces = forces or np.asarray(
                    final_atoms.calc.results.get("forces", []), dtype=np.float64
                )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"OUTCAR parse failed: {exc}")

    if final_atoms is None and os.path.isfile(contcar):
        try:
            final_atoms = ase_read(contcar, format="vasp")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"CONTCAR parse failed: {exc}")

    return VASPResult(
        work_dir=work_dir,
        return_code=return_code,
        converged=converged,
        energy_eV=energy,
        forces_eV_per_A=forces,
        stress_eV_per_A3=stress,
        n_atoms=len(final_atoms) if final_atoms is not None else None,
        wall_time_sec=_parse_walltime_outcar(outcar) if os.path.isfile(outcar) else None,
        final_atoms=final_atoms,
        notes="; ".join(notes) if notes else None,
    )
