"""Validated configuration of the hierarchical W-Ta-Cr-V-Hf-Mo transfer campaign.

Every section forbids unknown keys and every switch is honoured by the planner, so a typo or a
stale option fails loudly instead of being ignored. The default is a dry run: nothing is
submitted unless ``execution.dry_run`` is false AND an approver and a bounded node-hour
allowance are written down.
"""

from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from matsim_agents.campaign.acquisition import CandidateAcquisitionPolicy
from matsim_agents.campaign.qe_plan import MethodSignature, ResourceLimits, ResourceRequest
from matsim_agents.campaign.registry import TARGET_ELEMENTS
from matsim_agents.campaign.stages import StageThresholds
from matsim_agents.campaign.transfer import standard_experiments

TARGET_SITES = 1024
TARGET_COMPOSITION = {"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}
TARGET_REFERENCE = "Lupo Pasini et al., Comput. Mater. Sci. 257 (2025) 113908"
KNOWN_EXPERIMENTS = tuple(e.name for e in standard_experiments())


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CampaignSection(_Section):
    campaign_id: str = Field(min_length=1)
    description: str = ""
    seed: StrictInt = 0


class TargetSection(_Section):
    composition: dict[str, StrictInt] = Field(default_factory=lambda: dict(TARGET_COMPOSITION))
    supercell: tuple[StrictInt, StrictInt, StrictInt] = (8, 8, 8)
    parent_lattice: Literal["bcc"] = "bcc"
    reference: str = TARGET_REFERENCE

    @property
    def n_sites(self) -> int:
        return 2 * prod(self.supercell)  # BCC conventional cell holds two atoms

    @model_validator(mode="after")
    def _is_the_1024_site_target(self) -> TargetSection:
        if self.n_sites != TARGET_SITES:
            raise ValueError(
                f"target must have {TARGET_SITES} sites; supercell gives {self.n_sites}"
            )
        if set(self.composition) != set(TARGET_ELEMENTS):
            raise ValueError(f"target elements must be exactly {sorted(TARGET_ELEMENTS)}")
        if any(n <= 0 for n in self.composition.values()):
            raise ValueError("every target element needs a positive atom count")
        if sum(self.composition.values()) != TARGET_SITES:
            raise ValueError(f"target composition must sum to {TARGET_SITES} atoms")
        return self


class DatasetSource(_Section):
    """A historical dataset: a JSON list of ``DatasetEntry`` plus the method that produced it."""

    name: str = Field(min_length=1)
    path: Path
    method: MethodSignature | None = None


class DataSection(_Section):
    datasets: list[DatasetSource] = Field(default_factory=list)
    transfer_data: dict[str, Path] = Field(
        default_factory=dict, description="property -> .npz of labelled transfer rows"
    )
    candidates: Path | None = Field(None, description="JSON list of proposed calculations")
    reserved_validation_ids: list[str] = Field(
        default_factory=list,
        description="candidate ids, structure hashes or calculation ids that are never selected",
    )

    @model_validator(mode="after")
    def _unique(self) -> DataSection:
        names = [d.name for d in self.datasets]
        if len(set(names)) != len(names):
            raise ValueError("dataset names must be unique")
        return self


class ChemicalHierarchySection(_Section):
    elements: tuple[str, ...] = TARGET_ELEMENTS
    orders: tuple[StrictInt, ...] = (2, 3, 4, 5, 6)

    @model_validator(mode="after")
    def _valid(self) -> ChemicalHierarchySection:
        if set(self.elements) != set(TARGET_ELEMENTS):
            raise ValueError(f"hierarchy elements must be {sorted(TARGET_ELEMENTS)}")
        if sorted(set(self.orders)) != sorted(self.orders) or not set(self.orders) <= {
            2,
            3,
            4,
            5,
            6,
        }:
            raise ValueError("orders must be distinct values in 2..6")
        if 6 not in self.orders:
            raise ValueError("the hierarchy must end at the six-component target (order 6)")
        return self


class SamplingSection(_Section):
    acquisition: Literal["policy", "deterministic_baseline"] = "policy"
    n_orderings: StrictInt = Field(3, ge=1)
    policy: CandidateAcquisitionPolicy = Field(default_factory=CandidateAcquisitionPolicy)


class TransferEvaluationSection(_Section):
    experiments: tuple[str, ...] = KNOWN_EXPERIMENTS
    required_properties: tuple[str, ...] = ("energy",)
    magnetic_property: str = "magnetic_moment"
    defect_property: str = "defect_formation_energy"
    n_new: tuple[StrictInt, ...] = (20,)
    seeds: tuple[StrictInt, ...] = (0, 1, 2, 3, 4)
    min_seeds: StrictInt = Field(3, ge=2)
    min_eval_rows: StrictInt = Field(20, ge=1)
    thresholds: StageThresholds = Field(default_factory=StageThresholds)

    @model_validator(mode="after")
    def _known(self) -> TransferEvaluationSection:
        unknown = set(self.experiments) - set(KNOWN_EXPERIMENTS)
        if unknown:
            raise ValueError(f"unknown experiments {sorted(unknown)}; known: {KNOWN_EXPERIMENTS}")
        if not self.required_properties:
            raise ValueError("at least one required property is needed")
        return self


class QESection(_Section):
    method: MethodSignature
    resources: ResourceRequest
    limits: ResourceLimits
    input_ready: bool = False
    conv_thr_ry: float = Field(1e-6, gt=0.0)
    magnetic_treatment: str = "collinear"

    @model_validator(mode="after")
    def _pseudopotentials(self) -> QESection:
        missing = set(TARGET_ELEMENTS) - set(self.method.pseudopotentials)
        if missing:
            raise ValueError(f"method.pseudopotentials lacks {sorted(missing)}")
        return self


class BudgetSection(_Section):
    total_node_hours: float = Field(gt=0.0)
    validation_reserve_fraction: float = Field(0.15, gt=0.0, lt=1.0)
    allocation: dict[str, float] = Field(default_factory=dict)


class ExecutionSection(_Section):
    dry_run: bool = True
    restartable: bool = True
    prevent_duplicate_jobs: Literal[True] = True
    max_jobs_per_dispatch: StrictInt = Field(1, ge=1)
    approved_by: str | None = None
    approved_node_hours: float | None = Field(None, gt=0.0)
    qos: str = "debug"
    account: str = "LRN110"
    job_script: str = "qe_1024_job.sbatch"

    @model_validator(mode="after")
    def _live_runs_need_approval(self) -> ExecutionSection:
        if not self.dry_run and not (
            self.approved_by and self.approved_by.strip() and self.approved_node_hours
        ):
            raise ValueError("dry_run=false needs approved_by and a bounded approved_node_hours")
        return self


class CampaignConfig(_Section):
    campaign: CampaignSection
    target: TargetSection = Field(default_factory=TargetSection)
    data: DataSection = Field(default_factory=DataSection)
    chemical_hierarchy: ChemicalHierarchySection = Field(default_factory=ChemicalHierarchySection)
    sampling: SamplingSection = Field(default_factory=SamplingSection)
    transfer_evaluation: TransferEvaluationSection = Field(
        default_factory=TransferEvaluationSection
    )
    qe: QESection
    budget: BudgetSection
    execution: ExecutionSection = Field(default_factory=ExecutionSection)

    @model_validator(mode="after")
    def _approval_within_budget(self) -> CampaignConfig:
        approved = self.execution.approved_node_hours
        if approved is not None and approved > self.budget.total_node_hours:
            raise ValueError("approved_node_hours exceeds budget.total_node_hours")
        return self


def load_campaign_config(path: str | Path) -> CampaignConfig:
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("the campaign config must be a mapping")
    return CampaignConfig(**data)
