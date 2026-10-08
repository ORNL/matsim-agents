"""Approved, cached DFT single-points for declared elemental reference phases."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from ase.data import atomic_numbers
from ase.io import read
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.active_learning.config import DFTConfig
from matsim_agents.active_learning.dataset_governance import sha256_file
from matsim_agents.active_learning.dft_backend import DFTJobSpec, make_backend
from matsim_agents.active_learning.dft_protocol import dft_method_signature
from matsim_agents.discovery.energy_references import load_elemental_reference_manifest

log = logging.getLogger(__name__)


class ElementalPhaseSpec(BaseModel):
    """An approved fixed geometry and explicit magnetic input settings."""

    model_config = ConfigDict(extra="forbid")
    phase_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    element: str
    structure_path: Path
    kind: Literal["bulk", "molecule"]
    magnetic_state: str = Field(min_length=1)
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
        if not self.magnetic_state.strip():
            raise ValueError("An explicit magnetic state is required")
        return self


class ElementalReferencePlan(BaseModel):
    """Candidate phases for a fixed-geometry, zero-pressure energy comparison."""

    model_config = ConfigDict(extra="forbid")
    phases: list[ElementalPhaseSpec] = Field(min_length=1)
    protocol: Literal["fixed_geometry_zero_pressure"] = "fixed_geometry_zero_pressure"

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


def _phase_config(dft: DFTConfig, phase: ElementalPhaseSpec) -> DFTConfig:
    cfg = dft.model_copy(deep=True)
    if cfg.backend == "qe":
        assert cfg.qe is not None
        if phase.vasp_magnetic_settings:
            raise ValueError(f"{phase.phase_id}: VASP overrides supplied for QE")
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
    return cfg


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

    Geometry optimization is not performed: approved input geometries must
    already represent the intended phases. Spin and reference sampling
    exceptions are explicit, recorded separately from the compound protocol.
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
        digest = hashlib.sha256(geometry_bytes).hexdigest()
        identity = {
            "schema_version": 1,
            "geometry_sha256": digest,
            "method_signature": signature,
            "phase": phase.model_dump(mode="json", exclude={"structure_path", "phase_id"}),
            "protocol": plan.protocol,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        prepared.append((phase, atoms, geometry_bytes, cfg, signature, digest, key))

    # Preflight every phase before any DFT work, including the full approval gate.
    missing_keys = {
        key: phase.phase_id
        for phase, _, _, _, _, _, key in prepared
        if not (cache_dir / key / "result.json").is_file()
    }
    if not dft_approved and missing_keys:
        raise ValueError(
            f"Missing reference DFT calculations require approval: {list(missing_keys.values())}"
        )
    if max_dft_calculations is not None and len(missing_keys) > max_dft_calculations:
        raise ValueError(
            f"Reference plan needs {len(missing_keys)} DFT calculations; "
            f"cap is {max_dft_calculations}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    dft_attempts = 0
    for phase, atoms, geometry_bytes, cfg, signature, digest, key in prepared:
        entry = cache_dir / key
        entry.mkdir(exist_ok=True)
        with (entry / "calculation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result_path = entry / "result.json"
            if result_path.exists():
                record = CachedElementalCalculation.model_validate_json(result_path.read_text())
                if (
                    record.cache_key != key
                    or record.geometry_sha256 != digest
                    or record.method_signature != signature
                    or record.n_atoms != len(atoms)
                    or sha256_file(entry / "geometry.extxyz") != digest
                ):
                    raise ValueError(f"{phase.phase_id}: invalid cached elemental calculation")
                log.info("Reusing elemental DFT reference %s (%s)", phase.phase_id, key)
            else:
                if not dft_approved:
                    raise ValueError(f"{phase.phase_id}: missing DFT calculation requires approval")
                if max_dft_calculations is not None and dft_attempts >= max_dft_calculations:
                    raise ValueError("Reference DFT calculation cap reached before launch")
                dft_attempts += 1
                work_dir = entry / f"dft-{uuid.uuid4().hex}"
                log.info("Calculating elemental DFT reference %s (%s)", phase.phase_id, key)
                result = make_backend(cfg).run_one(DFTJobSpec(phase.phase_id, atoms, str(work_dir)))
                if (
                    not result.converged
                    or result.return_code != 0
                    or result.backend != cfg.backend
                    or result.energy_eV is None
                    or not np.isfinite(result.energy_eV)
                ):
                    raise RuntimeError(
                        f"{phase.phase_id}: elemental DFT failed: {result.notes}; see {work_dir}"
                    )
                if dft_method_signature(cfg, {phase.element}) != signature:
                    raise ValueError("DFT protocol changed during elemental calculation")
                if result.final_atoms is not None and (
                    result.final_atoms.get_chemical_symbols() != atoms.get_chemical_symbols()
                    or not np.allclose(
                        result.final_atoms.positions, atoms.positions, atol=1e-6, rtol=0
                    )
                    or not np.allclose(
                        result.final_atoms.cell.array, atoms.cell.array, atol=1e-6, rtol=0
                    )
                ):
                    raise ValueError(f"{phase.phase_id}: single-point changed reference geometry")
                record = CachedElementalCalculation(
                    cache_key=key,
                    geometry_sha256=digest,
                    method_signature=signature,
                    n_atoms=len(atoms),
                    energy_eV=result.energy_eV,
                    work_dir=str(work_dir),
                )
                (entry / "geometry.extxyz").write_bytes(geometry_bytes)
                temporary = entry / "result.tmp"
                temporary.write_text(record.model_dump_json(indent=2) + "\n")
                temporary.replace(result_path)
        records.append((phase, geometry_bytes, record))

    if dft_method_signature(dft, required_elements) != compound_signature:
        raise ValueError("Compound DFT protocol changed during reference preparation")
    for phase, _, geometry_bytes, cfg, signature, _, _ in prepared:
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
        phase, geometry_bytes, record = winner
        geometry = output_dir / f"{element}.extxyz"
        geometry.write_bytes(geometry_bytes)
        selected[element] = {
            "structure_path": geometry.name,
            "structure_sha256": record.geometry_sha256,
            "energy_eV": record.energy_eV,
        }
        for candidate, _, result in candidates:
            audit.append(
                {
                    "phase": candidate.model_dump(mode="json"),
                    "calculation": result.model_dump(mode="json"),
                    "energy_eV_per_atom": result.energy_eV / result.n_atoms,
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
                    "scope": "lowest_among_declared_fixed_geometries",
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
