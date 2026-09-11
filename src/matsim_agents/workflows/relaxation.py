"""Unified MLIP, DFT, and MLIP-to-DFT relaxation workflow.

This module composes existing backend implementations.  It owns scientific
policy, run storage, validation, and failure records; numerical kernels remain
in :mod:`matsim_agents.backends` and can still be used independently.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from matsim_agents.backends.mlip.relaxation import RelaxStructureInput
from matsim_agents.backends.mlip.relaxation import _run as run_mlip
from matsim_agents.execution.contracts import (
    ApprovalPolicy,
    EvidenceLevel,
    ProvenanceRecord,
    ValidationRecord,
    WorkflowStatus,
)
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.orchestration.state import RelaxationResult


class RelaxationMode(StrEnum):
    MLIP = "mlip"
    DFT = "dft"
    MLIP_DFT = "mlip-dft"


class GeometryControls(BaseModel):
    relax_atoms: bool = True
    relax_cell: bool = False
    fixed_atom_indices: list[int] = Field(default_factory=list)
    charge: float = 0.0
    spin: float = 0.0
    pressure_GPa: float | None = None
    preserve_symmetry: bool = False

    @model_validator(mode="after")
    def _has_degrees_of_freedom(self) -> GeometryControls:
        if not self.relax_atoms and not self.relax_cell:
            raise ValueError("a relaxation must enable relax_atoms and/or relax_cell")
        return self


class DFTBackendConfig(BaseModel):
    backend: str = "qe"
    launcher: str | list[str] | None = None
    pseudo_dir: str | None = None
    potcar_dir: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    timeout_sec: int | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _valid_backend(self) -> DFTBackendConfig:
        if self.backend not in {"qe", "vasp"}:
            raise ValueError("DFT relaxation backend must be 'qe' or 'vasp'")
        if self.backend == "qe" and not self.pseudo_dir:
            raise ValueError("QE relaxation requires pseudo_dir")
        if self.backend == "vasp" and not self.potcar_dir:
            raise ValueError("VASP relaxation requires potcar_dir")
        return self


class ScientificRelaxationConfig(BaseModel):
    mode: RelaxationMode
    structure_path: str
    output_root: str = "./runs"
    geometry: GeometryControls = Field(default_factory=GeometryControls)
    mlip: dict[str, Any] = Field(default_factory=dict)
    dft: DFTBackendConfig | None = None
    approvals: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    dft_approved: bool = False
    max_steps: int = Field(200, ge=1)
    force_tolerance_eV_per_A: float = Field(0.02, gt=0)
    parent_run_id: str | None = None

    @model_validator(mode="after")
    def _mode_requirements(self) -> ScientificRelaxationConfig:
        if self.mode in {RelaxationMode.DFT, RelaxationMode.MLIP_DFT} and self.dft is None:
            raise ValueError(f"mode={self.mode.value!r} requires dft configuration")
        return self


class RelaxationStageResult(BaseModel):
    stage: str
    backend: str
    evidence_level: EvidenceLevel
    input_structure_path: str
    optimized_structure_path: str | None = None
    energy_eV: float | None = None
    max_force_eV_per_A: float | None = None
    steps: int = 0
    converged: bool = False
    failure_reason: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)


class ScientificRelaxationResult(BaseModel):
    run_id: str
    run_directory: str
    mode: RelaxationMode
    status: WorkflowStatus
    stages: list[RelaxationStageResult]
    final_structure_path: str | None = None
    validations: list[ValidationRecord] = Field(default_factory=list)
    failure_reason: str | None = None


def _apply_constraints(path: str, destination: Path, controls: GeometryControls) -> str:
    from ase.constraints import FixAtoms
    from ase.io import read, write

    atoms = read(path)
    invalid = [i for i in controls.fixed_atom_indices if i < 0 or i >= len(atoms)]
    if invalid:
        raise ValueError(f"fixed atom indices out of range: {invalid}")
    if controls.fixed_atom_indices:
        atoms.set_constraint(FixAtoms(indices=controls.fixed_atom_indices))
    destination.parent.mkdir(parents=True, exist_ok=True)
    write(destination, atoms)
    return str(destination)


def _mlip_stage(
    input_path: str,
    run: ScientificRunDirectory,
    cfg: ScientificRelaxationConfig,
    runner: Callable[[RelaxStructureInput], RelaxationResult],
) -> RelaxationStageResult:
    kwargs = dict(cfg.mlip)
    args = RelaxStructureInput(
        structure_path=input_path,
        output_dir=str(run.path / "calculations" / "mlip"),
        maxiter=cfg.max_steps,
        fmax=cfg.force_tolerance_eV_per_A,
        relax_cell=cfg.geometry.relax_cell,
        charge=cfg.geometry.charge,
        spin=cfg.geometry.spin,
        **kwargs,
    )
    result = runner(args)
    return RelaxationStageResult(
        stage="mlip_relaxation",
        backend=str(kwargs.get("mlip_backend", "hydragnn")),
        evidence_level=EvidenceLevel.MLIP_RELAXATION,
        input_structure_path=input_path,
        optimized_structure_path=result.optimized_structure_path,
        energy_eV=result.final_energy_eV,
        max_force_eV_per_A=result.final_max_force_eV_per_A,
        steps=result.num_steps,
        converged=result.converged,
        failure_reason=None
        if result.converged
        else "MLIP relaxation did not meet convergence criteria",
        artifacts={"trajectory": result.trajectory_path, "log": result.log_csv_path},
    )


def _dft_stage(
    input_path: str, run: ScientificRunDirectory, cfg: ScientificRelaxationConfig
) -> RelaxationStageResult:
    from ase.io import read, write

    dft = cfg.dft
    assert dft is not None
    work_dir = run.path / "calculations" / dft.backend
    atoms = read(input_path)
    calculation = "vc-relax" if cfg.geometry.relax_cell else "relax"
    if dft.backend == "vasp":
        from matsim_agents.backends.dft.vasp_relax import (
            prepare_relax_workdir,
            recommend_settings,
            run_vasp,
        )

        settings = recommend_settings(
            atoms,
            dft.potcar_dir,
            calculation=calculation,
            nsw=cfg.max_steps,
            ediffg_eV_per_A=-cfg.force_tolerance_eV_per_A,
            **dft.settings,
        )
        prepare_relax_workdir(atoms, work_dir, settings, dft.potcar_dir or "")
        result = run_vasp(work_dir, dft.launcher or "vasp_std", timeout_sec=dft.timeout_sec)
        optimized = None
        if result.final_atoms is not None:
            optimized = str(run.path / "structures" / "dft_optimized.vasp")
            write(optimized, result.final_atoms)
        return RelaxationStageResult(
            stage="dft_relaxation",
            backend="vasp",
            evidence_level=(
                EvidenceLevel.CONVERGED_DFT if result.converged else EvidenceLevel.LOW_FIDELITY_DFT
            ),
            input_structure_path=input_path,
            optimized_structure_path=optimized,
            energy_eV=result.final_energy_eV,
            max_force_eV_per_A=result.final_max_force_eV_per_A,
            steps=result.n_ionic_steps,
            converged=result.converged,
            failure_reason=None
            if result.converged
            else f"VASP did not converge (return code {result.return_code})",
            artifacts={"work_dir": str(work_dir)},
        )

    from matsim_agents.backends.dft.qe_relax import QESettings, run_pw, write_pw_input

    settings = QESettings(
        calculation=calculation,
        pseudo_dir=dft.pseudo_dir or "",
        nstep=cfg.max_steps,
        forc_conv_thr_ry_au=cfg.force_tolerance_eV_per_A / 25.71104309541616,
        **dft.settings,
    )
    input_file = write_pw_input(atoms, settings, str(work_dir / "pw.in"))
    result = run_pw(input_file, str(work_dir), dft.launcher or "pw.x", timeout_sec=dft.timeout_sec)
    optimized = None
    try:
        final_atoms = read(result.stdout_path, index=-1, format="espresso-out")
        optimized = str(run.path / "structures" / "dft_optimized.extxyz")
        write(optimized, final_atoms)
    except Exception:  # output remains available for diagnosis
        pass
    force = (
        result.final_max_force_ry_au * 25.71104309541616
        if result.final_max_force_ry_au is not None
        else None
    )
    return RelaxationStageResult(
        stage="dft_relaxation",
        backend="qe",
        evidence_level=(
            EvidenceLevel.CONVERGED_DFT if result.converged else EvidenceLevel.LOW_FIDELITY_DFT
        ),
        input_structure_path=input_path,
        optimized_structure_path=optimized,
        energy_eV=result.final_energy_ev,
        max_force_eV_per_A=force,
        steps=result.bfgs_steps,
        converged=result.converged,
        failure_reason=None
        if result.converged
        else f"QE did not converge (return code {result.return_code})",
        artifacts={"input": input_file, "stdout": result.stdout_path},
    )


def run_relaxation(
    cfg: ScientificRelaxationConfig,
    *,
    mlip_runner: Callable[[RelaxStructureInput], RelaxationResult] = run_mlip,
) -> ScientificRelaxationResult:
    """Execute a composable relaxation and always persist a terminal result."""

    evidence = (
        EvidenceLevel.MLIP_RELAXATION
        if cfg.mode == RelaxationMode.MLIP
        else EvidenceLevel.CONVERGED_DFT
    )
    provenance = ProvenanceRecord(
        workflow="structure_relaxation",
        evidence_level=evidence,
        parent_run_id=cfg.parent_run_id,
        numerical_settings=cfg.model_dump(mode="json", exclude={"approvals"}),
        units={"energy": "eV", "force": "eV/angstrom", "pressure": "GPa"},
    )
    run = ScientificRunDirectory.create(
        cfg.output_root,
        workflow="structure_relaxation",
        request={"structure_path": cfg.structure_path, "mode": cfg.mode.value},
        resolved_config=cfg.model_dump(mode="json"),
        provenance=provenance,
    )
    stages: list[RelaxationStageResult] = []
    validations: list[ValidationRecord] = []
    constrained = run.path / "structures" / f"input{Path(cfg.structure_path).suffix or '.extxyz'}"
    try:
        current = _apply_constraints(cfg.structure_path, constrained, cfg.geometry)
        if cfg.mode in {RelaxationMode.MLIP, RelaxationMode.MLIP_DFT}:
            stage = _mlip_stage(current, run, cfg, mlip_runner)
            stages.append(stage)
            run.append_event("stage_complete", stage.model_dump(mode="json"))
            if stage.optimized_structure_path:
                current = stage.optimized_structure_path
        if cfg.mode in {RelaxationMode.DFT, RelaxationMode.MLIP_DFT}:
            if cfg.approvals.before_dft and not cfg.dft_approved:
                raise PermissionError("DFT execution requires explicit dft_approved=true")
            stage = _dft_stage(current, run, cfg)
            stages.append(stage)
            run.append_event("stage_complete", stage.model_dump(mode="json"))
            if stage.optimized_structure_path:
                current = stage.optimized_structure_path
        final_stage = stages[-1]
        validations.append(
            ValidationRecord(
                stage="numerical",
                name="geometry_convergence",
                passed=final_stage.converged,
                message=(
                    "converged"
                    if final_stage.converged
                    else final_stage.failure_reason or "not converged"
                ),
                metrics={
                    "max_force_eV_per_A": final_stage.max_force_eV_per_A,
                    "steps": final_stage.steps,
                },
            )
        )
        status = WorkflowStatus.COMPLETE if final_stage.converged else WorkflowStatus.PARTIAL
        result = ScientificRelaxationResult(
            run_id=run.run_id,
            run_directory=str(run.path),
            mode=cfg.mode,
            status=status,
            stages=stages,
            final_structure_path=current,
            validations=validations,
            failure_reason=final_stage.failure_reason,
        )
    except Exception as exc:  # noqa: BLE001 - failures are first-class artifacts
        result = ScientificRelaxationResult(
            run_id=run.run_id,
            run_directory=str(run.path),
            mode=cfg.mode,
            status=WorkflowStatus.FAILED,
            stages=stages,
            validations=validations,
            failure_reason=str(exc),
        )
    run.write_json("results.json", result)
    run.append_event(
        "run_finished", {"status": result.status, "failure_reason": result.failure_reason}
    )
    return result


__all__ = [
    "DFTBackendConfig",
    "GeometryControls",
    "RelaxationMode",
    "RelaxationStageResult",
    "ScientificRelaxationConfig",
    "ScientificRelaxationResult",
    "run_relaxation",
]
