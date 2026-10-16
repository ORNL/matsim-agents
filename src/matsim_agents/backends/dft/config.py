"""Configuration models for DFT single-point labelling backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class VASPConfig(BaseModel):
    """How to run VASP single-point calculations."""

    vasp_bin: Path = Field(..., description="Path to vasp_std (or vasp_gam/vasp_ncl) on Frontier.")
    vasp_wrapper: Path = Field(
        ...,
        description=(
            "Bash wrapper that does `module reset && module load PrgEnv-cray ...` "
            "and execs `srun vasp_std`. Provided at "
            "deployments/frontier/launchers/_vasp-step-frontier.sh."
        ),
    )
    incar_template: Path = Field(..., description="INCAR template with Python str.format() slots.")
    kpoints_template: Path | None = Field(
        None, description="Optional KPOINTS template; otherwise auto KSPACING in INCAR."
    )
    potcar_dir: Path = Field(..., description="Directory containing per-element POTCAR files.")
    nodes_per_job: int = 1
    ranks_per_node: int = 8  # 8 GCDs per MI250X node
    threads_per_rank: int = 7
    timeout_sec: int = 7200
    extra_incar: dict[str, str] = Field(default_factory=dict)


class QEBackendConfig(BaseModel):
    """How to run Quantum ESPRESSO ``pw.x`` single-point SCF calculations.

    The Python side fills in element-aware defaults (plane-wave cutoffs,
    smearing, k-mesh) via :func:`matsim_agents.backends.dft.qe_relax.recommend_settings`
    so most fields are optional. Pin them explicitly for production runs.
    """

    pw_bin: Path = Field(..., description="Path to pw.x on Frontier (e.g. install-gpu/bin/pw.x).")
    pw_wrapper: Path = Field(
        ...,
        description=(
            "Bash wrapper that does `module reset && module load PrgEnv-cray ...` "
            "and execs `srun pw.x -in <input>`. Provided at "
            "deployments/frontier/launchers/_qe-step-frontier.sh."
        ),
    )
    pseudo_dir: Path = Field(
        ...,
        description=(
            "Directory containing per-element UPF pseudopotentials. The QE "
            "backend auto-detects ``<symbol>.*.UPF`` per element."
        ),
    )
    pw_template: Path | None = Field(
        None,
        description=(
            "Optional pw.in namelist template (analogue of VASP's INCAR "
            "template). When set, the backend reads this file, runs "
            "``str.format(nat=, ntyp=, pseudo_dir=, prefix=, outdir=)`` on "
            "it, and APPENDS the auto-generated structure cards "
            "(ATOMIC_SPECIES, CELL_PARAMETERS, ATOMIC_POSITIONS, K_POINTS). "
            "When None, the backend generates the full pw.in programmatically "
            "via recommend_settings(). The two paths are mutually exclusive: "
            "the namelist-level pins below (ecutwfc_ry, occupations, "
            "extra_control, ...) are IGNORED when pw_template is set — put "
            "those values in the template directly."
        ),
    )
    # Optional explicit overrides (None ⇒ auto from element table).
    ecutwfc_ry: float | None = None
    ecutrho_ry: float | None = None
    kpts: tuple[int, int, int] | None = None
    koffset: tuple[int, int, int] | None = None
    occupations: Literal["smearing", "fixed", "tetrahedra"] | None = None
    smearing: str = "gaussian"
    degauss_ry: float = 0.01
    pseudopotentials: dict[str, str] | None = Field(
        None,
        description="Optional explicit {symbol: filename} mapping (else auto-detected).",
    )
    extra_control: dict[str, Any] = Field(default_factory=dict)
    extra_system: dict[str, Any] = Field(default_factory=dict)
    extra_electrons: dict[str, Any] = Field(default_factory=dict)
    nodes_per_job: int = 1
    ranks_per_node: int = 8
    threads_per_rank: int = 7
    timeout_sec: int = 7200


class DFTConfig(BaseModel):
    """Selects which DFT backend labels candidate structures.

    Exactly one of ``vasp`` / ``qe`` must be populated, matching ``backend``.
    """

    backend: Literal["vasp", "qe"] = "vasp"
    vasp: VASPConfig | None = None
    qe: QEBackendConfig | None = None
    max_concurrent_jobs: int | None = Field(
        None,
        ge=1,
        description=(
            "Optional concurrency cap. The scheduler-neutral allocation planner "
            "otherwise uses every complete nodes_per_job partition."
        ),
    )

    @model_validator(mode="after")
    def _check_backend_block(self) -> DFTConfig:
        if self.backend == "vasp" and self.vasp is None:
            raise ValueError("dft.backend='vasp' requires a dft.vasp block.")
        if self.backend == "qe" and self.qe is None:
            raise ValueError("dft.backend='qe' requires a dft.qe block.")
        return self
