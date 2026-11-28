"""End-to-end cascade investigation from one configuration.

One run chains the complete production workflow: qualification-gate resolution (recorded,
never silently bypassed), an adaptive cascade ensemble, surviving-defect characterization,
ensemble statistics with an explicit stopping decision, evidence-ledger records and a
deterministic hypothesis revision, closed by a JSON report. ``production`` classification is
refused below ``CASCADE_QUALIFIED``; everything else is labelled ``exploratory_unqualified``
in every record it produces. The PKA energy is imposed, so no result here is a physical
damage prediction.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from matsim_agents.core.contracts import EvidenceLevel
from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.core.yamlio import load_yaml_mapping
from matsim_agents.execution.ledger import (
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    HypothesisRecord,
    HypothesisStatus,
    LedgerQuantity,
    MethodRecord,
)
from matsim_agents.science.cascade.qualification import (
    Check,
    Domain,
    Level,
    QualificationRecord,
    Query,
    Resolution,
    resolve_level,
)
from matsim_agents.workflows.radiation import cascade_study

Z_95 = 1.959963984540054
Classification = Literal["production", "exploratory"]
EXPLORATORY_LABEL = "exploratory_unqualified"
PRODUCTION_LABEL = "qualified_production"


class GateRefusal(PermissionError):
    """Production was requested without the qualification evidence to allow it."""


class MaterialSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    element: str = Field(min_length=1)
    lattice_parameter_A: float = Field(gt=0, allow_inf_nan=False)
    n_cells: StrictInt = Field(ge=2, description="Cubic conventional-cell repeats per axis.")
    counts: dict[str, StrictInt] | None = Field(
        default=None, description="Random substitutional alloy occupation; None means pure."
    )


class PKASpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str | None = None
    energy_eV: float = Field(gt=0, allow_inf_nan=False)
    direction: str = "iso"
    temperature_K: float = Field(gt=0, allow_inf_nan=False)


class PhaseDurations(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ballistic_fs: float = Field(default=200.0, gt=0)
    thermal_spike_fs: float = Field(default=2000.0, gt=0)
    cooling_fs: float = Field(default=5000.0, gt=0)


class EnsembleSpec(BaseModel):
    """Adaptive sample size: N ~ (1.96 sd / (w |mean|))^2 for relative half-width ``w``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_realizations: StrictInt = Field(ge=2)
    max_realizations: StrictInt = Field(ge=2)
    relative_ci_half_width: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered(self) -> EnsembleSpec:
        if self.max_realizations < self.initial_realizations:
            raise ValueError("max_realizations must be >= initial_realizations")
        return self


class PotentialSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["emt"] = Field(description="Calculator backend; emt is verification-only.")
    checkpoint: str = Field(min_length=1, description="Identity the qualification record names.")
    short_range: Literal["none", "zbl"] = "none"
    precision: str = "float64"
    qualification_record: Path | None = None


class HypothesisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)


class IntegratedCascadeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    campaign_id: str = Field(min_length=1)
    classification: Classification
    seed: StrictInt = 0
    material: MaterialSpec
    pka: PKASpec
    phases: PhaseDurations = PhaseDurations()
    ensemble: EnsembleSpec
    potential: PotentialSpec
    hypothesis: HypothesisSpec

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> IntegratedCascadeConfig:
        cfg = cls.model_validate(load_yaml_mapping(path))
        record = cfg.potential.qualification_record
        if record is not None and not record.is_absolute():
            cfg = cfg.model_copy(
                update={
                    "potential": cfg.potential.model_copy(
                        update={"qualification_record": (Path(path).parent / record).resolve()}
                    )
                }
            )
        return cfg


def load_qualification_record(path: Path) -> QualificationRecord:
    raw = json.loads(Path(path).read_text())
    domain = raw["domain"]
    return QualificationRecord(
        checkpoint=raw["checkpoint"],
        domain=Domain(
            elements=frozenset(domain["elements"]),
            temperature_K=tuple(domain["temperature_K"]),
            recoil_energy_eV=tuple(domain["recoil_energy_eV"]),
            n_atoms=tuple(domain["n_atoms"]),
            precision=domain["precision"],
            short_range_model=domain["short_range_model"],
        ),
        checks=tuple(
            Check(
                name=c["name"],
                passed=bool(c["passed"]),
                criterion=c["criterion"],
                value=c.get("value"),
                dataset=c["dataset"],
            )
            for c in raw["checks"]
        ),
    )


def _n_atoms(material: MaterialSpec) -> int:
    sites = {"bcc": 2, "fcc": 4}[
        {"Fe": "bcc", "W": "bcc", "Cr": "bcc", "Cu": "fcc", "Ni": "fcc", "Al": "fcc"}[
            material.element
        ]
    ]
    return sites * material.n_cells**3


def resolve_gate(cfg: IntegratedCascadeConfig) -> Resolution:
    """Resolve the qualification level of this exact query; no record means UNASSESSED."""
    path = cfg.potential.qualification_record
    if path is None or not Path(path).is_file():
        return Resolution(
            Level.UNASSESSED,
            (f"no qualification record for checkpoint {cfg.potential.checkpoint!r}",),
            {},
        )
    record = load_qualification_record(Path(path))
    elements = set(cfg.material.counts or {cfg.material.element: 1})
    query = Query(
        elements=frozenset(elements | {cfg.pka.symbol or cfg.material.element}),
        temperature_K=cfg.pka.temperature_K,
        recoil_energy_eV=cfg.pka.energy_eV,
        n_atoms=_n_atoms(cfg.material),
        precision=cfg.potential.precision,
        short_range_model=cfg.potential.short_range,
    )
    return resolve_level(record, query, checkpoint=cfg.potential.checkpoint)


def classify(cfg: IntegratedCascadeConfig, resolution: Resolution) -> str:
    """The run label; production below CASCADE_QUALIFIED is refused, not downgraded."""
    if resolution.level >= Level.CASCADE_QUALIFIED:
        return PRODUCTION_LABEL
    if cfg.classification == "production":
        missing = "; ".join(resolution.reasons or ("no evidence",))
        raise GateRefusal(
            f"production requires CASCADE_QUALIFIED, model is {resolution.level.name}: {missing}"
        )
    return EXPLORATORY_LABEL


def required_realizations(values: list[float], relative_half_width: float) -> int | None:
    """N for a 95% CI of +-w|mean|; None when the mean is zero (formula inapplicable)."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean == 0.0:
        return None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return max(2, math.ceil((Z_95 * math.sqrt(var) / (relative_half_width * abs(mean))) ** 2))


def build_calculator(spec: PotentialSpec):
    if spec.backend == "emt":
        from ase.calculators.emt import EMT

        base = EMT()
        native = False
        justification = (
            "EMT verification potential; software integration check only, no cascade physics"
        )
    else:  # pragma: no cover - the Literal forbids it
        raise ValueError(f"unsupported backend {spec.backend!r}")
    from matsim_agents.science.cascade.zbl import build_short_range_calculator

    calculator, model = build_short_range_calculator(
        base,
        native_short_range=native,
        additive_zbl=spec.short_range == "zbl",
        justification=justification,
    )
    return calculator, model


def _plan(cfg: IntegratedCascadeConfig) -> list[dict]:
    """All potential tasks up front so seeds never depend on the stopping decision."""
    m = cfg.material
    counts = dict(m.counts) if m.counts else None
    tasks = cascade_study.plan_cascades(
        "integrated",
        m.element,
        m.lattice_parameter_A,
        [m.n_cells],
        cfg.pka.energy_eV,
        cfg.ensemble.max_realizations,
        cfg.potential.short_range,
        counts_for=(lambda n, c=counts: c) if counts else None,
        pka_symbol=cfg.pka.symbol,
        temperature_K=cfg.pka.temperature_K,
        base_seed=cfg.seed,
    )
    durations = (cfg.phases.ballistic_fs, cfg.phases.thermal_spike_fs, cfg.phases.cooling_fs)
    for task in tasks:
        task["direction"] = cfg.pka.direction
        task["phase_durations_fs"] = list(durations)
    return tasks


def run_investigation(cfg: IntegratedCascadeConfig, out_dir: str | os.PathLike) -> dict[str, Any]:
    """Execute the full chain and return the report (also written to ``report.json``)."""
    out = Path(out_dir)
    tasks_dir = out / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    resolution = resolve_gate(cfg)
    gate = {
        "level": resolution.level.name,
        "reasons": list(resolution.reasons),
        "missing": {k: list(v) for k, v in resolution.missing.items()},
    }
    atomic_write_text(out / "gate.json", json.dumps(gate, indent=2))
    label = classify(cfg, resolution)  # raises GateRefusal; gate.json records why

    calculator, short_range_model = build_calculator(cfg.potential)
    calculators = {cfg.potential.short_range: calculator}
    tasks = _plan(cfg)

    records: list[dict] = []
    stopping = "budget_exhausted"
    required: int | None = None
    for task in tasks:
        if len(records) >= cfg.ensemble.initial_realizations:
            usable_so_far = [r for r in records if r["numerically_stable"] and r["boundary_ok"]]
            required = required_realizations(
                [float(r["n_frenkel_pairs"]) for r in usable_so_far],
                cfg.ensemble.relative_ci_half_width,
            )
            if required is not None and len(usable_so_far) >= required:
                stopping = "target_met"
                break
        task["qualification"] = label
        records.append(cascade_study.run_task(task, calculators, tasks_dir))

    usable = [r for r in records if r["numerically_stable"] and r["boundary_ok"]]
    excluded = [
        {
            "id": r["id"],
            "reasons": (["numerically unstable"] if not r["numerically_stable"] else [])
            + r["boundary_findings"],
        }
        for r in records
        if r not in usable
    ]
    values = [float(r["n_frenkel_pairs"]) for r in usable]
    mean, sem = cascade_study.mean_sem(values) if values else (float("nan"), float("nan"))
    if required is None and values:
        required = required_realizations(values, cfg.ensemble.relative_ci_half_width)

    def finite(x: float) -> float | None:
        return float(x) if math.isfinite(x) else None

    statistics = {
        "observable": "surviving_frenkel_pairs",
        "n_run": len(records),
        "n_usable": len(usable),
        "n_excluded": len(excluded),
        "mean": finite(mean),
        "sem": finite(sem),
        "required_realizations_95": required,
        "stopping": stopping if values else "no_usable_realizations",
        "statement": (
            "boundary-contaminated and unstable runs are excluded from the statistics and "
            "retained as qualification evidence"
        ),
    }

    subject = f"{cfg.campaign_id}:{cfg.material.element}-{cfg.pka.energy_eV:g}eV"
    method = MethodRecord(
        backend=cfg.potential.backend,
        model=cfg.potential.checkpoint,
        settings={
            "short_range": short_range_model.policy,
            "phase_durations_fs": [
                cfg.phases.ballistic_fs,
                cfg.phases.thermal_spike_fs,
                cfg.phases.cooling_fs,
            ],
            "temperature_K": cfg.pka.temperature_K,
            "n_cells": cfg.material.n_cells,
            "qualification": label,
        },
    )
    ledger = EvidenceLedger(out / "ledger.jsonl")
    assumptions = [
        f"classification: {label}",
        "PKA energy imposed by configuration; no nuclear data",
        *short_range_model.assumptions,
    ]
    if values:
        evidence = EvidenceRecord(
            kind="property",
            subject=subject,
            campaign_id=cfg.campaign_id,
            method=method,
            evidence_level=EvidenceLevel.MLIP_PREDICTION,
            outcome=EvidenceOutcome.MEASURED,
            values=[
                LedgerQuantity(name="surviving_frenkel_pairs_mean", value=mean, unit="count"),
                LedgerQuantity(name="n_realizations", value=float(len(usable)), unit="count"),
                *(
                    [LedgerQuantity(name="surviving_frenkel_pairs_sem", value=sem, unit="count")]
                    if math.isfinite(sem)
                    else []
                ),
            ],
            uncertainties={"standard_error": sem} if math.isfinite(sem) else {},
            physical_assumptions=assumptions,
        )
    else:
        evidence = EvidenceRecord(
            kind="property",
            subject=subject,
            campaign_id=cfg.campaign_id,
            method=method,
            evidence_level=EvidenceLevel.MLIP_PREDICTION,
            outcome=EvidenceOutcome.FAILED_TO_COMPUTE,
            failure_reason="no numerically stable, boundary-clean realization",
            physical_assumptions=assumptions,
        )
    stored = ledger.append(evidence)

    status = HypothesisStatus.UNRESOLVED if values else HypothesisStatus.PROPOSED
    proposed = [
        f"{level}: obtain {', '.join(names)}" for level, names in resolution.missing.items()
    ] or ["none: the qualification domain covers this query"]
    revision = ledger.append(
        HypothesisRecord(
            hypothesis_id=cfg.hypothesis.hypothesis_id,
            campaign_id=cfg.campaign_id,
            claim_type="property",
            scientific_claim=cfg.hypothesis.claim,
            falsifiable_prediction=(
                f"an independent qualified ensemble gives a surviving-Frenkel-pair mean "
                f"outside {mean:.3g} +- {Z_95 * sem:.3g}"
                if values and math.isfinite(sem)
                else None
            ),
            status=status,
            evidence_ids=[stored.evidence_id],
            support_level=ledger.support_level([stored.evidence_id]),
            scientific_rationale=(
                f"Deterministic: {label} ensemble of {len(usable)} usable / {len(records)} run "
                f"cascades; stopping={statistics['stopping']}. Not a physical damage prediction."
            ),
            proposed_tests=proposed,
        )
    )

    report = {
        "campaign_id": cfg.campaign_id,
        "classification": label,
        "gate": gate,
        "potential": {
            "backend": cfg.potential.backend,
            "checkpoint": cfg.potential.checkpoint,
            "short_range": short_range_model.policy,
        },
        "statistics": statistics,
        "excluded": excluded,
        "evidence_id": stored.evidence_id,
        "hypothesis_revision_id": revision.revision_id,
        "hypothesis_status": status.value,
        "caveats": assumptions,
    }
    atomic_write_text(out / "report.json", json.dumps(report, indent=2, sort_keys=True))
    return report


__all__ = [
    "EXPLORATORY_LABEL",
    "PRODUCTION_LABEL",
    "GateRefusal",
    "IntegratedCascadeConfig",
    "build_calculator",
    "classify",
    "load_qualification_record",
    "required_realizations",
    "resolve_gate",
    "run_investigation",
]
