"""End-to-end cascade investigation from one configuration.

One run chains qualification-gate resolution, an adaptive cascade ensemble, surviving-defect
characterization, ensemble statistics, evidence-ledger records and a hypothesis revision.
Only exploratory execution is supported. The PKA energy is imposed and the protocol has not
been qualified, so no result here is a physical damage prediction.
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
from matsim_agents.core.hashing import sha256_file, sha256_json
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
from matsim_agents.science.irradiation.contracts import PKASpectrum
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
    energy_eV: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    spectrum_file: Path | None = None
    direction: str = "iso"
    temperature_K: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _source(self) -> PKASpec:
        if (self.energy_eV is None) == (self.spectrum_file is None):
            raise ValueError("specify exactly one of energy_eV or spectrum_file")
        if self.spectrum_file is not None and self.symbol:
            raise ValueError("spectrum_file forbids symbol")
        return self


class DisplacementSpectrum(BaseModel):
    """A saved spectrum explicitly identified as one non-transmuting channel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel: Literal["elastic", "inelastic"]
    spectrum: PKASpectrum


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

    backend: Literal["emt", "eam_fs"] = Field(description="ASE calculator backend.")
    checkpoint: str = Field(min_length=1, description="Identity the qualification record names.")
    potential_file: Path | None = None
    short_range: Literal["none", "zbl"] = "none"
    precision: str = "float64"
    qualification_record: Path | None = None

    @model_validator(mode="after")
    def _potential_identity(self) -> PotentialSpec:
        if self.backend == "eam_fs":
            if self.potential_file is None:
                raise ValueError("eam_fs requires potential_file")
            if len(self.checkpoint) != 71 or not self.checkpoint.startswith("sha256:"):
                raise ValueError("eam_fs checkpoint must be sha256:<64 lowercase hex digits>")
            if any(c not in "0123456789abcdef" for c in self.checkpoint[7:]):
                raise ValueError("eam_fs checkpoint must be sha256:<64 lowercase hex digits>")
            if self.short_range != "none":
                raise ValueError("eam_fs supports only the original potential without added ZBL")
            if self.precision != "float64":
                raise ValueError("eam_fs requires precision float64")
        elif self.potential_file is not None:
            raise ValueError("potential_file is only supported for eam_fs")
        return self


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
        updates = {}
        for field in ("qualification_record", "potential_file"):
            value = getattr(cfg.potential, field)
            if value is not None and not value.is_absolute():
                updates[field] = (Path(path).parent / value).resolve()
        if updates:
            cfg = cfg.model_copy(update={"potential": cfg.potential.model_copy(update=updates)})
        spectrum = cfg.pka.spectrum_file
        if spectrum is not None and not spectrum.is_absolute():
            cfg = cfg.model_copy(
                update={
                    "pka": cfg.pka.model_copy(
                        update={"spectrum_file": (Path(path).parent / spectrum).resolve()}
                    )
                }
            )
        return cfg


def load_qualification_record(path: Path) -> QualificationRecord:
    raw = json.loads(Path(path).read_text())
    domain = raw["domain"]
    for check in raw["checks"]:
        if not isinstance(check["passed"], bool):
            raise ValueError(
                f"{path}: qualification check {check['name']!r} must have a boolean 'passed'"
            )
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
                passed=c["passed"],
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
    if cfg.pka.spectrum_file is not None:
        return Resolution(
            Level.UNASSESSED,
            ("spectrum-sampled events have not been individually qualified",),
            {},
        )
    path = cfg.potential.qualification_record
    if path is None or not Path(path).is_file():
        return Resolution(
            Level.UNASSESSED,
            (f"no qualification record for checkpoint {cfg.potential.checkpoint!r}",),
            {},
        )
    record = load_qualification_record(Path(path))
    assert cfg.pka.energy_eV is not None
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
    """Only exploratory execution is supported, regardless of record assertions."""
    if cfg.classification == "production":
        if resolution.level < Level.CASCADE_QUALIFIED:
            missing = "; ".join(resolution.reasons or ("no evidence",))
            raise GateRefusal(
                f"production requires CASCADE_QUALIFIED, "
                f"model is {resolution.level.name}: {missing}"
            )
        raise GateRefusal("this integration is exploratory only; production is not supported")
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
    elif spec.backend == "eam_fs":
        from ase.calculators.eam import EAM

        assert spec.potential_file is not None
        actual = f"sha256:{sha256_file(spec.potential_file)}"
        if actual != spec.checkpoint:
            raise ValueError(
                f"potential file checksum mismatch: expected {spec.checkpoint}, got {actual}"
            )
        base = EAM(potential=str(spec.potential_file), form="fs")
        native = False
        justification = "Original EAM/FS potential, without an added short-range correction."
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

    spectrum = None
    spectrum_sha256 = None
    spectrum_channel = None
    if cfg.pka.spectrum_file is not None:
        spectrum_sha256 = sha256_file(cfg.pka.spectrum_file)
        source = DisplacementSpectrum.model_validate_json(cfg.pka.spectrum_file.read_text())
        spectrum = source.spectrum
        spectrum_channel = source.channel
        if spectrum.provenance.numerical_validation.value == "STRUCTURALLY_INVALID":
            raise ValueError("spectrum provenance reports structurally invalid data")
        counts = cfg.material.counts or {cfg.material.element: _n_atoms(cfg.material)}
        total = sum(counts.values())
        composition = {element: count / total for element, count in counts.items()}
        if set(composition) != set(spectrum.scenario.composition) or any(
            not math.isclose(fraction, spectrum.scenario.composition[element], abs_tol=1e-6)
            for element, fraction in composition.items()
        ):
            raise ValueError("spectrum composition does not match the simulated material")
        for entry in spectrum.spectra:
            if (
                entry.parent is None
                or entry.parent.mass_number is None
                or entry.parent != entry.species
            ):
                raise ValueError(
                    f"spectrum species {entry.species.key} does not identify an unchanged isotope"
                )
            if entry.species.element not in counts or entry.species.element in {"H", "He"}:
                raise ValueError(
                    f"spectrum species {entry.species.key} is not a supported lattice recoil"
                )

    calculator, short_range_model = build_calculator(cfg.potential)
    if cfg.potential.backend == "eam_fs":
        elements = set(cfg.material.counts or {cfg.material.element: 1})
        elements.add(cfg.pka.symbol or cfg.material.element)
        missing = elements - set(calculator.elements)
        if missing:
            raise ValueError(f"potential does not cover elements: {', '.join(sorted(missing))}")
    calculators = {cfg.potential.short_range: calculator}
    tasks = _plan(cfg)
    fingerprint = (
        sha256_json(cfg.model_dump(mode="json", exclude={"pka": {"spectrum_file"}}))
        if spectrum is None
        else sha256_json(
            {"config": cfg.model_dump(mode="json"), "spectrum_sha256": spectrum_sha256}
        )
    )

    records: list[dict] = []
    stopping = "budget_exhausted"
    required: int | None = None
    for task in tasks:
        task["qualification"] = label
        task["execution_fingerprint"] = fingerprint
        if spectrum is None:
            records.append(cascade_study.run_task(task, calculators, tasks_dir))
        else:
            task["spectrum_sha256"] = spectrum_sha256
            task["spectrum_channel"] = spectrum_channel
            records.append(cascade_study.run_task(task, calculators, tasks_dir, spectrum=spectrum))
        if len(records) >= cfg.ensemble.initial_realizations:
            usable_so_far = [r for r in records if r["numerically_stable"] and r["boundary_ok"]]
            required = required_realizations(
                [float(r["n_frenkel_pairs"]) for r in usable_so_far],
                cfg.ensemble.relative_ci_half_width,
            )
            if required is not None and len(usable_so_far) >= required:
                stopping = "target_met"
                break

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

    recoil_label = (
        f"{cfg.pka.energy_eV:g}eV"
        if cfg.pka.energy_eV is not None
        else f"spectrum-{spectrum_sha256}"
    )
    subject = f"{cfg.campaign_id}:{cfg.material.element}-{recoil_label}"
    method = MethodRecord(
        backend=cfg.potential.backend,
        model=cfg.potential.checkpoint,
        model_sha256=(cfg.potential.checkpoint[7:] if cfg.potential.backend == "eam_fs" else None),
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
            "spectrum_sha256": spectrum_sha256,
            "spectrum_channel": spectrum_channel,
        },
    )
    ledger = EvidenceLedger(out / "ledger.jsonl")
    assumptions = [
        f"classification: {label}",
        (
            "PKA energy imposed by configuration; no nuclear data"
            if spectrum is None
            else (
                "PKA spectrum and reaction channel supplied by the user; "
                "not independently validated"
            )
        ),
        *(
            [
                "isotope-specific masses and isotope site assignments are not modelled",
                "excluded trajectories can bias spectrum-averaged defect statistics",
            ]
            if spectrum is not None
            else []
        ),
        "initial velocities are sampled without equilibrium MD; electronic stopping is off",
        *short_range_model.assumptions,
    ]
    input_hashes = {"pka_spectrum": spectrum_sha256} if spectrum_sha256 is not None else {}
    if values:
        evidence = EvidenceRecord(
            kind="property",
            subject=subject,
            campaign_id=cfg.campaign_id,
            method=method,
            evidence_level=EvidenceLevel.EMPIRICAL_POTENTIAL_PREDICTION,
            input_sha256=input_hashes,
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
            evidence_level=EvidenceLevel.EMPIRICAL_POTENTIAL_PREDICTION,
            input_sha256=input_hashes,
            outcome=EvidenceOutcome.FAILED_TO_COMPUTE,
            failure_reason="no numerically stable, boundary-clean realization",
            physical_assumptions=assumptions,
        )
    stored = ledger.append(evidence)

    status = HypothesisStatus.UNRESOLVED if values else HypothesisStatus.PROPOSED
    proposed = [
        f"{level}: obtain {', '.join(names)}" for level, names in resolution.missing.items()
    ] or (
        ["none: the qualification domain covers this query"]
        if resolution.level >= Level.CASCADE_QUALIFIED
        else ["obtain a qualification record for this exact potential and domain"]
    )
    revision = ledger.append(
        HypothesisRecord(
            hypothesis_id=cfg.hypothesis.hypothesis_id,
            campaign_id=cfg.campaign_id,
            claim_type="property",
            scientific_claim=cfg.hypothesis.claim,
            falsifiable_prediction=None,
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
            "potential_file": (
                str(cfg.potential.potential_file) if cfg.potential.potential_file else None
            ),
            "short_range": short_range_model.policy,
        },
        "pka_source": (
            {"type": "imposed", "energy_eV": cfg.pka.energy_eV}
            if spectrum is None
            else {
                "type": "spectrum",
                "sha256": spectrum_sha256,
                "scenario_id": spectrum.scenario.scenario_id,
                "channel": spectrum_channel,
                "rate_basis": spectrum.rate_basis,
                "rate_unit": spectrum.rate_unit,
                "nuclear_data_identity": spectrum.provenance.nuclear_data_identity,
                "numerical_validation": spectrum.provenance.numerical_validation.value,
            }
        ),
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
