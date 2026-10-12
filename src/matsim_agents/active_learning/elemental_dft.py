"""Approved, cached DFT references for declared elemental phases."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from ase.data import atomic_numbers
from ase.io import read, write
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.active_learning.config import DFTConfig
from matsim_agents.active_learning.dataset_governance import sha256_file
from matsim_agents.active_learning.dft_backend import DFTJobSpec, make_backend
from matsim_agents.active_learning.dft_protocol import dft_method_signature
from matsim_agents.discovery.energy_references import load_elemental_reference_manifest

log = logging.getLogger(__name__)


class ElementalPhaseSpec(BaseModel):
    """An approved elemental geometry policy and explicit magnetic settings."""

    model_config = ConfigDict(extra="forbid")
    phase_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    element: str
    structure_path: Path
    kind: Literal["bulk", "molecule"]
    geometry_status: Literal["verified_converged", "requires_relaxation"]
    magnetic_state: str = Field(min_length=1)
    relax_cell: bool | None = None
    relaxation_force_tolerance_eV_per_A: float = Field(default=0.02, gt=0)
    relaxation_max_steps: int = Field(default=200, gt=0)
    qe_magnetic_settings: dict[str, int | float] = Field(default_factory=dict)
    vasp_magnetic_settings: dict[str, str] = Field(default_factory=dict)
    kpts: tuple[int, int, int] | None = None

    @model_validator(mode="after")
    def _validate_inputs(self) -> ElementalPhaseSpec:
        if self.element not in atomic_numbers or self.element == "X":
            raise ValueError(f"Invalid elemental reference species: {self.element}")
        qe_keys = {"nspin", "tot_magnetization", "starting_magnetization(1)"}
        vasp_keys = {"ISPIN", "MAGMOM", "NUPDOWN"}
        if set(self.qe_magnetic_settings) - qe_keys:
            raise ValueError("Only declared QE spin/magnetization overrides are allowed")
        if set(self.vasp_magnetic_settings) - vasp_keys:
            raise ValueError("Only declared VASP spin/magnetization overrides are allowed")
        if not all(np.isfinite(value) for value in self.qe_magnetic_settings.values()):
            raise ValueError("QE magnetic settings must be finite")
        if self.kpts is not None and any(value < 1 for value in self.kpts):
            raise ValueError("Reference k-point dimensions must be positive")
        if self.relax_cell and self.kind == "molecule":
            raise ValueError("Molecular reference relaxation must preserve its vacuum cell")
        if not np.isfinite(self.relaxation_force_tolerance_eV_per_A):
            raise ValueError("Reference relaxation force tolerance must be finite")
        if not self.magnetic_state.strip():
            raise ValueError("An explicit magnetic state is required")
        return self


class ElementalReferencePlan(BaseModel):
    """Candidate phases for a declared-phase, zero-pressure energy comparison."""

    model_config = ConfigDict(extra="forbid")
    phases: list[ElementalPhaseSpec] = Field(min_length=1)
    protocol: Literal["declared_phases_zero_pressure_v2"] = "declared_phases_zero_pressure_v2"

    @model_validator(mode="after")
    def _unique_phase_ids(self) -> ElementalReferencePlan:
        ids = [phase.phase_id for phase in self.phases]
        if len(ids) != len(set(ids)):
            raise ValueError("Elemental reference phase IDs must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ElementalReferencePlan:
        plan = cls.model_validate(yaml.safe_load(path.read_text()))
        for phase in plan.phases:
            if not phase.structure_path.is_absolute():
                phase.structure_path = (path.resolve().parent / phase.structure_path).resolve()
        return plan


class CachedElementalCalculation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cache_key: str
    geometry_sha256: str
    method_signature: str
    n_atoms: int = Field(gt=0)
    energy_eV: float = Field(allow_inf_nan=False)
    work_dir: str


class CachedElementalRelaxation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cache_key: str
    input_geometry_sha256: str
    relaxed_geometry_sha256: str
    method_signature: str
    n_atoms: int = Field(gt=0)
    converged: bool
    final_max_force_eV_per_A: float = Field(ge=0, allow_inf_nan=False)
    ionic_steps: int = Field(gt=0)
    work_dir: str


def _phase_config(dft: DFTConfig, phase: ElementalPhaseSpec) -> DFTConfig:
    cfg = dft.model_copy(deep=True)
    if cfg.backend == "qe":
        assert cfg.qe is not None
        if phase.vasp_magnetic_settings:
            raise ValueError(f"{phase.phase_id}: VASP overrides supplied for QE")
        if phase.geometry_status == "requires_relaxation" and cfg.qe.pw_template is not None:
            raise ValueError(
                f"{phase.phase_id}: QE template-based inputs cannot be safely converted "
                "to a relaxation; use generated QE inputs or attest the geometry as verified"
            )
        protected = {"calculation", "forc_conv_thr", "nstep", "pseudo_dir"}
        conflicting = protected.intersection(key.lower() for key in cfg.qe.extra_control)
        if phase.geometry_status == "requires_relaxation" and conflicting:
            raise ValueError(
                f"{phase.phase_id}: QE relaxation control overrides conflict with "
                f"{sorted(conflicting)}"
            )
        if phase.geometry_status == "requires_relaxation" and _relaxes_cell(phase):
            pressure = next(
                (value for key, value in cfg.qe.extra_control.items() if key.lower() == "press"),
                0,
            )
            if not _is_zero_pressure(pressure):
                raise ValueError(
                    f"{phase.phase_id}: elemental cell relaxation requires zero external pressure"
                )
        if cfg.qe.pw_template is not None and phase.qe_magnetic_settings:
            raise ValueError("QE template ignores magnetic overrides; use generated inputs")
        if cfg.qe.pw_template is None and any(
            value is None
            for value in (cfg.qe.ecutwfc_ry, cfg.qe.ecutrho_ry, cfg.qe.kpts, cfg.qe.occupations)
        ):
            raise ValueError("Reference QE protocol must pin cutoffs, k-points, and occupations")
        cfg.qe.extra_system.update(phase.qe_magnetic_settings)
        if phase.kpts is not None:
            cfg.qe.kpts = phase.kpts
    else:
        assert cfg.vasp is not None
        if phase.qe_magnetic_settings:
            raise ValueError(f"{phase.phase_id}: QE overrides supplied for VASP")
        if phase.kpts is not None:
            raise ValueError("VASP k-points must be declared in its shared DFT configuration")
        cfg.vasp.extra_incar.update(phase.vasp_magnetic_settings)
        if phase.geometry_status == "requires_relaxation" and _relaxes_cell(phase):
            extra_pressure = next(
                (value for key, value in cfg.vasp.extra_incar.items() if key.lower() == "pstress"),
                0,
            )
            if not _is_zero_pressure(extra_pressure):
                raise ValueError(
                    f"{phase.phase_id}: elemental cell relaxation requires zero external pressure"
                )
            template = cfg.vasp.incar_template.read_text()
            for match in re.finditer(r"(?im)^\s*PSTRESS\s*=\s*([^#!\n]+)", template):
                pressure_text = match.group(1).split("#", 1)[0].split("!", 1)[0].strip()
                if not _is_zero_pressure(pressure_text):
                    raise ValueError(f"{phase.phase_id}: VASP INCAR template sets nonzero PSTRESS")
    return cfg


def _relaxes_cell(phase: ElementalPhaseSpec) -> bool:
    return phase.kind == "bulk" and (phase.relax_cell if phase.relax_cell is not None else True)


def _is_zero_pressure(value: object) -> bool:
    try:
        pressure = float(str(value).strip().strip("'\"").replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return False
    return np.isfinite(pressure) and abs(pressure) <= 1e-12


def _relaxation_key(phase: ElementalPhaseSpec, input_digest: str, method_signature: str) -> str:
    identity = {
        "schema_version": 1,
        "input_geometry_sha256": input_digest,
        "method_signature": method_signature,
        "phase": phase.model_dump(
            mode="json",
            exclude={"structure_path", "phase_id", "geometry_status"},
        ),
        "backend": "qe-or-vasp-relaxation",
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _run_reference_relaxation(
    phase: ElementalPhaseSpec, cfg: DFTConfig, atoms, work_dir: Path
) -> tuple[object, float, int]:
    """Relax positions (and optionally a bulk cell) using the reference DFT method."""
    work_dir.mkdir(parents=True)
    if cfg.backend == "qe":
        from matsim_agents.backends.dft.qe_relax import (
            parse_pw_stdout,
            recommend_settings,
            write_pw_input,
        )

        assert cfg.qe is not None
        qe = cfg.qe
        do_cell = _relaxes_cell(phase)
        settings = recommend_settings(
            atoms,
            str(qe.pseudo_dir),
            calculation="vc-relax" if do_cell else "relax",
            ecutwfc_ry=qe.ecutwfc_ry,
            ecutrho_ry=qe.ecutrho_ry,
            occupations=qe.occupations,
            smearing=qe.smearing,
            degauss_ry=qe.degauss_ry,
            kpts=qe.kpts,
            koffset=qe.koffset or (0, 0, 0),
            nstep=phase.relaxation_max_steps,
            forc_conv_thr_ry_au=phase.relaxation_force_tolerance_eV_per_A / 25.711043,
            pseudopotentials=qe.pseudopotentials,
            extra_control=dict(qe.extra_control),
            extra_system=dict(qe.extra_system),
            extra_electrons=dict(qe.extra_electrons),
            extra_cell={} if not do_cell else {"cell_dofree": "all"},
        )
        input_path = work_dir / "pw.in"
        write_pw_input(atoms, settings, str(input_path), outdir=str(work_dir / "tmp"))
        log_path = work_dir / "pw.out"
        argv = [
            "bash",
            str(qe.pw_wrapper),
            str(work_dir),
            str(qe.pw_bin),
            str(input_path),
            str(qe.nodes_per_job),
            str(qe.ranks_per_node),
            str(qe.threads_per_rank),
        ]
        try:
            with log_path.open("w") as log_file:
                proc = subprocess.run(
                    argv,
                    cwd=work_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=qe.timeout_sec,
                    check=False,
                    env=os.environ.copy(),
                )
            result = parse_pw_stdout(str(log_path), return_code=proc.returncode)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{phase.phase_id}: QE relaxation timed out in {work_dir}") from exc
        if not result.converged or result.return_code != 0:
            raise RuntimeError(f"{phase.phase_id}: QE relaxation did not converge; see {work_dir}")
        if (
            result.final_max_force_ry_au is None
            or not np.isfinite(result.final_max_force_ry_au)
            or result.final_max_force_ry_au * 25.711043 > phase.relaxation_force_tolerance_eV_per_A
        ):
            raise RuntimeError(
                f"{phase.phase_id}: QE relaxation did not meet its force threshold; see {work_dir}"
            )
        try:
            relaxed = read(log_path, index=-1, format="espresso-out")
        except (OSError, ValueError, IndexError) as exc:
            raise RuntimeError(
                f"{phase.phase_id}: QE relaxation has no readable final geometry; see {work_dir}"
            ) from exc
        if not np.isfinite(result.final_energy_ev):
            raise RuntimeError(f"{phase.phase_id}: QE relaxation energy is nonfinite")
        return relaxed, float(result.final_max_force_ry_au * 25.711043), result.bfgs_steps

    from matsim_agents.active_learning.vasp_io import prepare_vasp_workdir
    from matsim_agents.backends.dft.vasp_relax import parse_relax_outputs

    assert cfg.vasp is not None
    vasp = cfg.vasp
    do_cell = _relaxes_cell(phase)
    relax_incar = {
        **(vasp.extra_incar or {}),
        **phase.vasp_magnetic_settings,
        "IBRION": 2,
        "ISIF": 3 if do_cell else 2,
        "NSW": phase.relaxation_max_steps,
        "EDIFFG": -phase.relaxation_force_tolerance_eV_per_A,
    }
    prepare_vasp_workdir(
        atoms,
        str(work_dir),
        incar_template=vasp.incar_template,
        potcar_dir=vasp.potcar_dir,
        kpoints_template=vasp.kpoints_template,
        extra_incar=relax_incar,
    )
    argv = [
        "bash",
        str(vasp.vasp_wrapper),
        str(work_dir),
        str(vasp.vasp_bin),
        str(vasp.nodes_per_job),
        str(vasp.ranks_per_node),
        str(vasp.threads_per_rank),
    ]
    log_path = work_dir / "vasp.out"
    try:
        with log_path.open("w") as log_file:
            proc = subprocess.run(
                argv,
                cwd=work_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=vasp.timeout_sec,
                check=False,
                env=os.environ.copy(),
            )
        result = parse_relax_outputs(work_dir, return_code=proc.returncode)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{phase.phase_id}: VASP relaxation timed out in {work_dir}") from exc
    if (
        not result.converged
        or result.return_code != 0
        or result.final_atoms is None
        or result.final_energy_eV is None
        or not np.isfinite(result.final_energy_eV)
        or result.final_max_force_eV_per_A is None
        or not np.isfinite(result.final_max_force_eV_per_A)
        or result.final_max_force_eV_per_A > phase.relaxation_force_tolerance_eV_per_A
    ):
        raise RuntimeError(
            f"{phase.phase_id}: VASP relaxation did not converge to its force threshold; "
            f"see {work_dir}"
        )
    return result.final_atoms, result.final_max_force_eV_per_A, result.n_ionic_steps


def prepare_elemental_references(
    plan: ElementalReferencePlan,
    dft: DFTConfig,
    *,
    required_elements: set[str],
    cache_dir: Path,
    output_dir: Path,
    phases_approved: bool = False,
    dft_approved: bool = False,
    max_dft_calculations: int | None = None,
) -> Path:
    """Select lowest energy/atom among declared phases, never an implicit search.

    A phase explicitly attested as ``verified_converged`` gets a fixed-geometry
    single point. Other phases are relaxed first, must meet the requested force
    threshold, and then receive a production-protocol single point. This only
    compares declared phases; it is not an implicit structure or magnetic-state
    search.
    """
    if not phases_approved:
        raise ValueError("Elemental reference phase selection requires explicit approval")
    if not required_elements:
        raise ValueError("Elemental references require a nonempty element set")
    missing = required_elements - {phase.element for phase in plan.phases}
    if missing:
        raise ValueError(f"Approved reference plan lacks coverage for {sorted(missing)}")
    if output_dir.exists():
        raise ValueError(f"Elemental reference snapshot already exists: {output_dir}")
    if max_dft_calculations is not None and max_dft_calculations < 0:
        raise ValueError("Reference DFT calculation cap must be nonnegative")
    compound_signature = dft_method_signature(dft, required_elements)

    prepared = []
    for phase in sorted(plan.phases, key=lambda item: item.phase_id):
        if phase.element not in required_elements:
            continue
        geometry_bytes = phase.structure_path.read_bytes()
        frames = list(read(phase.structure_path, index=":"))
        if len(frames) != 1 or not len(frames[0]):
            raise ValueError(f"{phase.phase_id}: reference must contain exactly one nonempty frame")
        atoms = frames[0]
        if sha256_file(phase.structure_path) != hashlib.sha256(geometry_bytes).hexdigest():
            raise ValueError(f"{phase.phase_id}: reference geometry changed during preflight")
        if set(atoms.get_chemical_symbols()) != {phase.element}:
            raise ValueError(f"{phase.phase_id}: reference must be pure {phase.element}")
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
            raise ValueError(f"{phase.phase_id}: invalid reference geometry")
        if phase.kind == "bulk" and not atoms.pbc.all():
            raise ValueError(f"{phase.phase_id}: bulk reference must be periodic")
        cfg = _phase_config(dft, phase)
        signature = dft_method_signature(cfg, {phase.element})
        input_digest = hashlib.sha256(geometry_bytes).hexdigest()
        relax_key = (
            _relaxation_key(phase, input_digest, signature)
            if phase.geometry_status == "requires_relaxation"
            else None
        )
        serialized = io.StringIO()
        snapshot_atoms = atoms.copy()
        snapshot_atoms.calc = None
        write(serialized, snapshot_atoms, format="extxyz")
        relaxed_bytes = serialized.getvalue().encode()
        relaxation = None
        if relax_key is not None:
            relax_entry = cache_dir / "relaxations" / relax_key
            result_path = relax_entry / "result.json"
            geometry_path = relax_entry / "geometry.extxyz"
            if result_path.exists() or geometry_path.exists():
                if not result_path.is_file() or not geometry_path.is_file():
                    raise ValueError(f"{phase.phase_id}: incomplete cached elemental relaxation")
                relaxation = CachedElementalRelaxation.model_validate_json(result_path.read_text())
                relaxed_bytes = geometry_path.read_bytes()
                if (
                    relaxation.cache_key != relax_key
                    or relaxation.input_geometry_sha256 != input_digest
                    or relaxation.method_signature != signature
                    or relaxation.n_atoms != len(atoms)
                    or not relaxation.converged
                    or relaxation.final_max_force_eV_per_A
                    > phase.relaxation_force_tolerance_eV_per_A
                    or sha256_file(geometry_path) != relaxation.relaxed_geometry_sha256
                ):
                    raise ValueError(f"{phase.phase_id}: invalid cached elemental relaxation")
                relaxed_atoms = read(geometry_path)
                if (
                    relaxed_atoms.get_chemical_symbols() != atoms.get_chemical_symbols()
                    or not np.isfinite(relaxed_atoms.positions).all()
                    or not np.isfinite(relaxed_atoms.cell.array).all()
                    or (phase.kind == "bulk" and not relaxed_atoms.pbc.all())
                    or (
                        phase.kind == "molecule"
                        and not np.allclose(
                            relaxed_atoms.cell.array, atoms.cell.array, atol=1e-8, rtol=0
                        )
                    )
                ):
                    raise ValueError(f"{phase.phase_id}: invalid cached relaxed geometry")
                atoms_for_static = relaxed_atoms
            else:
                atoms_for_static = atoms
        else:
            atoms_for_static = atoms
        static_digest = hashlib.sha256(relaxed_bytes).hexdigest()
        static_identity = {
            "schema_version": 3,
            "geometry_sha256": static_digest,
            "method_signature": signature,
            "phase": {
                "element": phase.element,
                "kind": phase.kind,
                "magnetic_state": phase.magnetic_state,
                "qe_magnetic_settings": phase.qe_magnetic_settings,
                "vasp_magnetic_settings": phase.vasp_magnetic_settings,
                "kpts": phase.kpts,
            },
            "protocol": plan.protocol,
        }
        static_key = hashlib.sha256(
            json.dumps(static_identity, sort_keys=True).encode()
        ).hexdigest()
        if relax_key is None or relaxation is not None:
            static_entry = cache_dir / "calculations" / static_key
            static_result_path = static_entry / "result.json"
            static_geometry_path = static_entry / "geometry.extxyz"
            if static_result_path.exists() or static_geometry_path.exists():
                if not static_result_path.is_file() or not static_geometry_path.is_file():
                    raise ValueError(f"{phase.phase_id}: incomplete cached elemental calculation")
                cached = CachedElementalCalculation.model_validate_json(
                    static_result_path.read_text()
                )
                if (
                    cached.cache_key != static_key
                    or cached.geometry_sha256 != static_digest
                    or cached.method_signature != signature
                    or cached.n_atoms != len(atoms_for_static)
                    or sha256_file(static_geometry_path) != static_digest
                ):
                    raise ValueError(f"{phase.phase_id}: invalid cached elemental calculation")
        prepared.append(
            (
                phase,
                atoms,
                geometry_bytes,
                cfg,
                signature,
                input_digest,
                relax_key,
                relaxation,
                relaxed_bytes,
                atoms_for_static,
                static_digest,
                static_key,
            )
        )

    # Preflight every phase before any DFT work, including the full approval gate.
    missing_relaxations = {
        relax_key: phase.phase_id
        for phase, _, _, _, _, _, relax_key, relaxation, *_ in prepared
        if relax_key is not None and relaxation is None
    }
    missing_static_keys = {
        static_key: phase.phase_id
        for (
            phase,
            _,
            _,
            _,
            _,
            _,
            relax_key,
            relaxation,
            _,
            _,
            _,
            static_key,
        ) in prepared
        if (relax_key is None or relaxation is not None)
        and not (cache_dir / "calculations" / static_key / "result.json").is_file()
    }
    calculation_count = len(missing_static_keys) + 2 * len(missing_relaxations)
    if not dft_approved and calculation_count:
        raise ValueError(
            "Missing reference DFT calculations require approval: "
            f"{sorted(set(missing_static_keys.values()) | set(missing_relaxations.values()))}"
        )
    if max_dft_calculations is not None and calculation_count > max_dft_calculations:
        raise ValueError(
            f"Reference plan needs {calculation_count} DFT calculations; "
            f"cap is {max_dft_calculations}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    dft_attempts = 0
    for (
        phase,
        atoms,
        _geometry_bytes,
        cfg,
        signature,
        input_digest,
        relax_key,
        cached_relaxation,
        relaxed_bytes,
        atoms_for_static,
        static_digest,
        static_key,
    ) in prepared:
        relaxation = cached_relaxation
        if relax_key is not None:
            relax_entry = cache_dir / "relaxations" / relax_key
            relax_entry.mkdir(parents=True, exist_ok=True)
            with (relax_entry / "calculation.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                relax_result_path = relax_entry / "result.json"
                relax_geometry_path = relax_entry / "geometry.extxyz"
                if relax_result_path.exists():
                    relaxation = CachedElementalRelaxation.model_validate_json(
                        relax_result_path.read_text()
                    )
                    relaxed_bytes = relax_geometry_path.read_bytes()
                    atoms_for_static = read(relax_geometry_path)
                    if (
                        relaxation.cache_key != relax_key
                        or relaxation.input_geometry_sha256 != input_digest
                        or relaxation.method_signature != signature
                        or sha256_file(relax_geometry_path) != relaxation.relaxed_geometry_sha256
                    ):
                        raise ValueError(f"{phase.phase_id}: invalid cached elemental relaxation")
                    log.info(
                        "Reusing relaxed elemental geometry %s (%s)", phase.phase_id, relax_key
                    )
                else:
                    if not dft_approved:
                        raise ValueError(
                            f"{phase.phase_id}: missing relaxation requires DFT approval"
                        )
                    if max_dft_calculations is not None and dft_attempts >= max_dft_calculations:
                        raise ValueError("Reference DFT calculation cap reached before launch")
                    dft_attempts += 1
                    work_dir = relax_entry / f"relax-{uuid.uuid4().hex}"
                    log.info("Relaxing elemental reference %s (%s)", phase.phase_id, relax_key)
                    relaxed_atoms, final_force, ionic_steps = _run_reference_relaxation(
                        phase, cfg, atoms, work_dir
                    )
                    if (
                        relaxed_atoms.get_chemical_symbols() != atoms.get_chemical_symbols()
                        or not np.isfinite(relaxed_atoms.positions).all()
                        or not np.isfinite(relaxed_atoms.cell.array).all()
                    ):
                        raise RuntimeError(
                            f"{phase.phase_id}: relaxation returned invalid final geometry"
                        )
                    if phase.kind == "molecule" and not np.allclose(
                        relaxed_atoms.cell.array, atoms.cell.array, atol=1e-8, rtol=0
                    ):
                        raise RuntimeError(
                            f"{phase.phase_id}: molecular relaxation changed its vacuum cell"
                        )
                    relaxed_atoms.calc = None
                    geometry_tmp = relax_entry / "geometry.tmp.extxyz"
                    write(geometry_tmp, relaxed_atoms, format="extxyz")
                    relaxed_bytes = geometry_tmp.read_bytes()
                    relaxed_digest = hashlib.sha256(relaxed_bytes).hexdigest()
                    geometry_tmp.replace(relax_geometry_path)
                    relaxation = CachedElementalRelaxation(
                        cache_key=relax_key,
                        input_geometry_sha256=input_digest,
                        relaxed_geometry_sha256=relaxed_digest,
                        method_signature=signature,
                        n_atoms=len(atoms),
                        converged=True,
                        final_max_force_eV_per_A=final_force,
                        ionic_steps=ionic_steps,
                        work_dir=str(work_dir),
                    )
                    temporary = relax_entry / "result.tmp"
                    temporary.write_text(relaxation.model_dump_json(indent=2) + "\n")
                    temporary.replace(relax_result_path)
                    atoms_for_static = relaxed_atoms
            static_digest = hashlib.sha256(relaxed_bytes).hexdigest()
            static_identity = {
                "schema_version": 3,
                "geometry_sha256": static_digest,
                "method_signature": signature,
                "phase": {
                    "element": phase.element,
                    "kind": phase.kind,
                    "magnetic_state": phase.magnetic_state,
                    "qe_magnetic_settings": phase.qe_magnetic_settings,
                    "vasp_magnetic_settings": phase.vasp_magnetic_settings,
                    "kpts": phase.kpts,
                },
                "protocol": plan.protocol,
            }
            static_key = hashlib.sha256(
                json.dumps(static_identity, sort_keys=True).encode()
            ).hexdigest()

        entry = cache_dir / "calculations" / static_key
        entry.mkdir(parents=True, exist_ok=True)
        with (entry / "calculation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result_path = entry / "result.json"
            if result_path.exists():
                record = CachedElementalCalculation.model_validate_json(result_path.read_text())
                if (
                    record.cache_key != static_key
                    or record.geometry_sha256 != static_digest
                    or record.method_signature != signature
                    or record.n_atoms != len(atoms_for_static)
                    or sha256_file(entry / "geometry.extxyz") != static_digest
                ):
                    raise ValueError(f"{phase.phase_id}: invalid cached elemental calculation")
                log.info("Reusing elemental DFT reference %s (%s)", phase.phase_id, static_key)
            else:
                if not dft_approved:
                    raise ValueError(f"{phase.phase_id}: missing DFT calculation requires approval")
                if max_dft_calculations is not None and dft_attempts >= max_dft_calculations:
                    raise ValueError("Reference DFT calculation cap reached before launch")
                dft_attempts += 1
                work_dir = entry / f"dft-{uuid.uuid4().hex}"
                log.info("Calculating elemental DFT reference %s (%s)", phase.phase_id, static_key)
                result = make_backend(cfg).run_one(
                    DFTJobSpec(phase.phase_id, atoms_for_static, str(work_dir))
                )
                if (
                    not result.converged
                    or result.return_code != 0
                    or result.backend != cfg.backend
                    or result.energy_eV is None
                    or not np.isfinite(result.energy_eV)
                    or result.n_atoms != len(atoms_for_static)
                ):
                    raise RuntimeError(
                        f"{phase.phase_id}: elemental DFT failed: {result.notes}; see {work_dir}"
                    )
                if dft_method_signature(cfg, {phase.element}) != signature:
                    raise ValueError("DFT protocol changed during elemental calculation")
                if result.final_atoms is not None and (
                    result.final_atoms.get_chemical_symbols()
                    != atoms_for_static.get_chemical_symbols()
                    or not np.allclose(
                        result.final_atoms.positions,
                        atoms_for_static.positions,
                        atol=1e-6,
                        rtol=0,
                    )
                    or not np.allclose(
                        result.final_atoms.cell.array,
                        atoms_for_static.cell.array,
                        atol=1e-6,
                        rtol=0,
                    )
                ):
                    raise ValueError(f"{phase.phase_id}: single-point changed reference geometry")
                record = CachedElementalCalculation(
                    cache_key=static_key,
                    geometry_sha256=static_digest,
                    method_signature=signature,
                    n_atoms=len(atoms_for_static),
                    energy_eV=result.energy_eV,
                    work_dir=str(work_dir),
                )
                (entry / "geometry.extxyz").write_bytes(relaxed_bytes)
                temporary = entry / "result.tmp"
                temporary.write_text(record.model_dump_json(indent=2) + "\n")
                temporary.replace(result_path)
        records.append((phase, relaxed_bytes, record, relaxation, input_digest))

    if dft_method_signature(dft, required_elements) != compound_signature:
        raise ValueError("Compound DFT protocol changed during reference preparation")
    for phase, _, geometry_bytes, cfg, signature, *_ in prepared:
        if (
            phase.structure_path.read_bytes() != geometry_bytes
            or dft_method_signature(cfg, {phase.element}) != signature
        ):
            raise ValueError(f"{phase.phase_id}: reference inputs changed during preparation")
    output_dir.mkdir(parents=True)
    selected = {}
    audit = []
    for element in sorted(required_elements):
        candidates = [item for item in records if item[0].element == element]
        winner = min(
            candidates, key=lambda item: (item[2].energy_eV / item[2].n_atoms, item[0].phase_id)
        )
        phase, geometry_bytes, record, relaxation, input_digest = winner
        geometry = output_dir / f"{element}.extxyz"
        geometry.write_bytes(geometry_bytes)
        selected[element] = {
            "structure_path": geometry.name,
            "structure_sha256": record.geometry_sha256,
            "input_geometry_sha256": input_digest,
            "geometry_status": phase.geometry_status,
            "energy_eV": record.energy_eV,
        }
        for candidate, _, result, phase_relaxation, phase_input_digest in candidates:
            audit.append(
                {
                    "phase": candidate.model_dump(mode="json"),
                    "calculation": result.model_dump(mode="json"),
                    "energy_eV_per_atom": result.energy_eV / result.n_atoms,
                    "input_geometry_sha256": phase_input_digest,
                    "geometry_status": candidate.geometry_status,
                    "relaxation": (
                        phase_relaxation.model_dump(mode="json")
                        if phase_relaxation is not None
                        else None
                    ),
                    "selected": candidate.phase_id == phase.phase_id,
                }
            )
    manifest_path = output_dir / "elemental-references.json"
    manifest_path.write_text(
        json.dumps(
            {
                "backend": dft.backend,
                "method_signature": compound_signature,
                "references": selected,
                "selection": {
                    "protocol": plan.protocol,
                    "phases_approved": phases_approved,
                    "dft_approved": dft_approved,
                    "dft_attempts": dft_attempts,
                    "scope": "lowest_among_declared_phases_after_required_relaxations",
                    "phases": audit,
                },
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    load_elemental_reference_manifest(manifest_path, required_elements=required_elements)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-plan", type=Path, required=True)
    parser.add_argument("--dft-config", type=Path, required=True, help="YAML DFTConfig block")
    parser.add_argument("--elements", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approve-reference-phases", action="store_true")
    parser.add_argument("--approve-dft", action="store_true")
    parser.add_argument("--max-dft-calculations", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    dft = DFTConfig.model_validate(yaml.safe_load(args.dft_config.read_text()))
    manifest = prepare_elemental_references(
        ElementalReferencePlan.from_yaml(args.phase_plan),
        dft,
        required_elements=set(args.elements),
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        phases_approved=args.approve_reference_phases,
        dft_approved=args.approve_dft,
        max_dft_calculations=args.max_dft_calculations,
    )
    print(manifest)


if __name__ == "__main__":
    main()
