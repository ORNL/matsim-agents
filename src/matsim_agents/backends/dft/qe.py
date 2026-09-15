"""Quantum ESPRESSO ``pw.x`` backend (single-point SCF for AL labelling).

Reuses :mod:`matsim_agents.backends.dft.qe_relax` to write the input deck (with
``calculation='scf'`` so we get one SCF per call, no geometry steps) and
parses the resulting ``pw.out`` with ASE's ``espresso-out`` reader, which
gives us ``Atoms`` plus a SinglePointCalculator carrying energy, forces, and
stress.

When ``QEBackendConfig.pw_template`` is set, the namelist section is taken
verbatim from that file (analogous to VASP's INCAR template) and only the
structure-dependent cards are appended programmatically.

Energy / force / stress units returned to the caller match those of the VASP
backend (eV, eV/Å, eV/Å³) so the rest of the AL loop never has to special-
case the backend.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ase.io import read as ase_read

from matsim_agents.active_learning.dft_backend import DFTJobSpec, DFTResult
from matsim_agents.backends.dft.qe_relax import (
    QESettings,
    recommend_settings,
    write_pw_input,
)

if TYPE_CHECKING:
    from matsim_agents.active_learning.config import QEBackendConfig

log = logging.getLogger(__name__)


class QEBackend:
    """DFTBackend implementation for Quantum ESPRESSO ``pw.x``."""

    name = "qe"

    def __init__(self, cfg: QEBackendConfig) -> None:
        self.cfg = cfg
        self.nodes_per_job = cfg.nodes_per_job
        self.ranks_per_node = cfg.ranks_per_node
        self.threads_per_rank = cfg.threads_per_rank
        self.timeout_sec = cfg.timeout_sec

    # ----- input prep -------------------------------------------------------

    def _settings_for(self, atoms) -> QESettings:
        """Build a per-job QESettings from cfg + composition-aware defaults."""
        cfg = self.cfg
        # Auto-fill ecutwfc/ecutrho/kpts/pseudos from element table, then
        # override with whatever the user pinned in the YAML.
        s = recommend_settings(atoms, str(cfg.pseudo_dir))
        s.calculation = "scf"  # AL only needs a single SCF; never relax here
        if cfg.ecutwfc_ry is not None:
            s.ecutwfc_ry = cfg.ecutwfc_ry
        if cfg.ecutrho_ry is not None:
            s.ecutrho_ry = cfg.ecutrho_ry
        if cfg.kpts is not None:
            s.kpts = tuple(cfg.kpts)  # type: ignore[assignment]
        if cfg.koffset is not None:
            s.koffset = tuple(cfg.koffset)  # type: ignore[assignment]
        if cfg.occupations is not None:
            s.occupations = cfg.occupations
        s.smearing = cfg.smearing
        s.degauss_ry = cfg.degauss_ry
        if cfg.pseudopotentials:
            s.pseudopotentials = dict(cfg.pseudopotentials)
        if cfg.extra_control:
            s.extra_control = {**s.extra_control, **cfg.extra_control}
        if cfg.extra_system:
            s.extra_system = {**s.extra_system, **cfg.extra_system}
        if cfg.extra_electrons:
            s.extra_electrons = {**s.extra_electrons, **cfg.extra_electrons}
        return s

    # ----- run --------------------------------------------------------------

    def run_one(self, spec: DFTJobSpec) -> DFTResult:
        cfg = self.cfg
        os.makedirs(spec.work_dir, exist_ok=True)

        input_path = os.path.join(spec.work_dir, "pw.in")
        if cfg.pw_template is not None:
            _write_pw_from_template(
                template_path=cfg.pw_template,
                atoms=spec.atoms,
                pseudo_dir=str(cfg.pseudo_dir),
                kpts=tuple(cfg.kpts) if cfg.kpts is not None else (1, 1, 1),
                koffset=tuple(cfg.koffset) if cfg.koffset is not None else (0, 0, 0),
                pseudopotentials=dict(cfg.pseudopotentials) if cfg.pseudopotentials else None,
                input_path=input_path,
                prefix="pwscf",
                outdir="./tmp",
            )
        else:
            settings = self._settings_for(spec.atoms)
            write_pw_input(spec.atoms, settings, input_path, prefix="pwscf", outdir="./tmp")

        # Wrapper contract:
        #   <work_dir> <pw_bin> <input_file> <nodes> <ranks_per_node> <threads_per_rank>
        argv = [
            "bash",
            str(cfg.pw_wrapper),
            spec.work_dir,
            str(cfg.pw_bin),
            input_path,
            str(cfg.nodes_per_job),
            str(cfg.ranks_per_node),
            str(cfg.threads_per_rank),
        ]

        stdout_path = os.path.join(spec.work_dir, "pw.out")
        t0 = time.time()
        with open(stdout_path, "w") as logf:
            try:
                env = os.environ.copy()
                if spec.assigned_nodes:
                    env["MATSIM_DFT_ASSIGNED_NODES"] = ",".join(spec.assigned_nodes)
                proc = subprocess.run(
                    argv,
                    cwd=spec.work_dir,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=cfg.timeout_sec,
                    check=False,
                    env=env,
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = 124
        elapsed = time.time() - t0

        return _parse_pw_out(
            stdout_path=stdout_path,
            work_dir=spec.work_dir,
            return_code=rc,
            elapsed_sec=elapsed,
            backend_name=self.name,
        )


# --------------------------------------------------------------------------- #
# Output parsing (eV / eV·Å⁻¹ / eV·Å⁻³)                                       #
# --------------------------------------------------------------------------- #


def _parse_pw_out(
    stdout_path: str,
    work_dir: str,
    return_code: int,
    elapsed_sec: float,
    backend_name: str,
) -> DFTResult:
    """Use ASE's espresso-out reader to extract energy/forces/stress."""
    import numpy as np

    energy: float | None = None
    forces: Any | None = None
    stress: Any | None = None
    final_atoms = None
    converged = False
    notes_list: list[str] = []
    n_atoms: int | None = None

    if not os.path.isfile(stdout_path) or os.path.getsize(stdout_path) == 0:
        notes_list.append("pw.out missing or empty")
    else:
        try:
            final_atoms = ase_read(stdout_path, index=-1, format="espresso-out")
            n_atoms = len(final_atoms)
            calc = final_atoms.calc
            if calc is not None:
                results = calc.results
                if "energy" in results:
                    energy = float(results["energy"])
                elif "free_energy" in results:
                    energy = float(results["free_energy"])
                if "forces" in results:
                    forces = np.asarray(results["forces"], dtype=np.float64)
                if "stress" in results:
                    stress = np.asarray(results["stress"], dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 — ASE raises many subclasses
            notes_list.append(f"espresso-out parse failed: {exc}")

        # Final convergence sanity check: pw.x prints "JOB DONE." on success.
        try:
            with open(stdout_path) as f:
                tail = f.read()[-4096:]
            converged = (
                return_code == 0
                and energy is not None
                and forces is not None
                and "JOB DONE." in tail
            )
            if "convergence NOT achieved" in tail:
                converged = False
                notes_list.append("SCF did not converge")
        except OSError:
            pass

    return DFTResult(
        backend=backend_name,
        work_dir=os.path.abspath(work_dir),
        return_code=return_code,
        converged=converged,
        energy_eV=energy,
        forces_eV_per_A=forces,
        stress_eV_per_A3=stress,
        n_atoms=n_atoms,
        wall_time_sec=elapsed_sec,
        final_atoms=final_atoms,
        notes="; ".join(notes_list) if notes_list else None,
    )


# --------------------------------------------------------------------------- #
# Template-based pw.in writer (mirror of vasp_io.write_incar template path)   #
# --------------------------------------------------------------------------- #


def _write_pw_from_template(
    template_path: str | Path,
    atoms,
    pseudo_dir: str,
    kpts: tuple[int, int, int],
    koffset: tuple[int, int, int],
    pseudopotentials: dict[str, str] | None,
    input_path: str,
    prefix: str,
    outdir: str,
) -> str:
    """Render a pw.in by appending structure-cards to a namelist template.

    The template file may contain any of these ``str.format`` placeholders:
    ``{nat}``, ``{ntyp}``, ``{pseudo_dir}``, ``{prefix}``, ``{outdir}``,
    ``{natoms}`` (alias of ``nat``).

    After substitution the function appends ``ATOMIC_SPECIES``,
    ``CELL_PARAMETERS angstrom``, ``ATOMIC_POSITIONS angstrom`` and
    ``K_POINTS automatic`` derived from ``atoms``, ``kpts`` and
    ``koffset``. Pseudopotential filenames are taken from
    ``pseudopotentials`` when given, else auto-detected by scanning
    ``pseudo_dir`` for the first file matching ``<symbol>.*UPF``
    (case-insensitive).
    """
    # Approximate atomic masses (g/mol). Reused from qe_relax to avoid
    # leaking that table into the public API of this module.
    from matsim_agents.backends.dft.qe_relax import _ATOMIC_MASS  # noqa: PLC2701

    template_path = Path(template_path)
    text = template_path.read_text()

    symbols_unique = sorted(set(atoms.get_chemical_symbols()))
    nat = len(atoms)
    ntyp = len(symbols_unique)

    # Auto-detect UPF filenames if the user did not pin them.
    if pseudopotentials is None:
        pseudopotentials = {}
        pdir = Path(pseudo_dir)
        for sym in symbols_unique:
            matches = sorted(
                p.name
                for p in pdir.iterdir()
                if p.is_file()
                and p.name.lower().endswith(".upf")
                and p.name.split(".")[0].lower() == sym.lower()
            )
            if not matches:
                raise FileNotFoundError(
                    f"No UPF pseudopotential found for {sym!r} in {pdir}. "
                    "Provide qe.pseudopotentials = {{Sym: filename}} explicitly."
                )
            pseudopotentials[sym] = matches[0]

    fmt_kwargs = dict(
        nat=nat,
        natoms=nat,
        ntyp=ntyp,
        pseudo_dir=os.path.abspath(pseudo_dir),
        prefix=prefix,
        outdir=outdir,
    )
    try:
        text = text.format(**fmt_kwargs)
    except KeyError as exc:
        raise ValueError(
            f"pw_template {template_path} references unknown placeholder {{{exc.args[0]}}}. "
            f"Allowed: {sorted(fmt_kwargs)}."
        ) from exc

    cards: list[str] = [text.rstrip(), ""]
    cards.append("ATOMIC_SPECIES")
    for sym in symbols_unique:
        mass = _ATOMIC_MASS.get(sym, 1.0)
        cards.append(f"  {sym}  {mass:.4f}  {pseudopotentials[sym]}")
    cards.append("")
    cards.append("CELL_PARAMETERS angstrom")
    for v in atoms.cell.array:
        cards.append(f"  {v[0]:18.10f} {v[1]:18.10f} {v[2]:18.10f}")
    cards.append("")
    cards.append("ATOMIC_POSITIONS angstrom")
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True):
        cards.append(f"  {sym}  {pos[0]:18.10f} {pos[1]:18.10f} {pos[2]:18.10f}")
    cards.append("")
    cards.append("K_POINTS automatic")
    cards.append(f"  {kpts[0]} {kpts[1]} {kpts[2]}  {koffset[0]} {koffset[1]} {koffset[2]}")
    cards.append("")

    Path(input_path).parent.mkdir(parents=True, exist_ok=True)
    Path(input_path).write_text("\n".join(cards))
    return os.path.abspath(input_path)
