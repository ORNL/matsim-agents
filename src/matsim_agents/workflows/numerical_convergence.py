"""Checkpointed numerical-convergence workflow for ``pw.x``.

Stages, each restartable: ``plan`` (grid and input decks), ``submit`` (scheduler dispatch),
``collect`` (parse finished outputs into a ``KeyedResultLog``), ``analyse`` (compare with the
reference point, select the cheapest qualified one) and ``report`` (JSON and Markdown with
provenance). A scheduler ``COMPLETED`` state is never read as success; only the parsed
``pw.x`` output counts.

Nothing here knows a composition, element, account or path: every such value comes from the
YAML configuration. Results from synthetic or software-test outputs must be labelled
``software verification only`` by the caller; this module reports what the outputs contain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ase.io import read
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.dft.pseudo_validation import (
    load_manifest,
    validate_pseudopotentials,
)
from matsim_agents.backends.dft.qe_convergence import (
    AcceptanceCriteria,
    PointObservables,
    ScanOutcome,
    ScanPoint,
    analyse_scan,
    apply_derived,
    parse_observables,
)
from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input
from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.core.hashing import sha256_file
from matsim_agents.core.restart import KeyedResultLog
from matsim_agents.core.yamlio import load_yaml_mapping
from matsim_agents.workflows.active_learning.campaign_init import kpts_for_spacing

ScanParameter = Literal["cutoff", "kpoints", "smearing", "conv_thr"]
SOFTWARE_ONLY = "software verification only"
_RE_VERSION = re.compile(r"Program PWSCF\s+v\.(\S+)")


class CaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,48}$")
    structure: str  # path read by ASE
    kind: str = "bulk"  # bulk | vacancy | substitution | interstitial | gas | neb_image | ...
    magnetic: bool = False
    starting_magnetization: dict[str, float] = Field(default_factory=dict)


class ScanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter: ScanParameter
    # cutoff: [[ecutwfc, ecutrho], ...]; kpoints: spacings (1/A); smearing: degauss (Ry);
    # conv_thr: thresholds (Ry). Ordered from least to most demanding.
    values: list[Any] = Field(min_length=3)


class BaseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pseudo_dir: str
    pseudopotentials: dict[str, str]
    pseudo_manifest: str | None = None
    functional: str = "PBE"
    ecutwfc_ry: float
    ecutrho_ry: float
    kspacing_inv_A: float
    smearing: str = "mv"
    degauss_ry: float = 0.01
    conv_thr_ry: float = 1.0e-8


class Resources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    launcher: list[str] = Field(min_length=1)  # may use {input} and {output}
    time_limit: str = "00:30:00"
    account: str
    qos: str = "debug"
    partition: str = "batch"
    gpus: int = 0


class ConvergenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: str
    out_dir: str
    base: BaseSettings
    cases: list[CaseConfig] = Field(min_length=1)
    scans: list[ScanConfig] = Field(min_length=1)
    criteria: dict[str, float] = Field(default_factory=dict)
    derived: list[dict[str, Any]] = Field(default_factory=list)
    resources: Resources
    label: str = ""  # "software verification only" for tests; empty for real data

    @model_validator(mode="after")
    def _unique(self) -> ConvergenceConfig:
        names = [c.name for c in self.cases]
        if len(set(names)) != len(names):
            raise ValueError("case names must be unique")
        parameters = [s.parameter for s in self.scans]
        if len(set(parameters)) != len(parameters):
            raise ValueError("each scan parameter may appear once")
        return self

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class PlannedRun:
    key: str
    scan: str
    value: Any
    case: str
    natoms: int
    directory: Path
    input_path: Path
    output_path: Path
    kpts: tuple[int, int, int]
    ecutwfc_ry: float
    ecutrho_ry: float
    below_recommendation: bool


def load_config(path: str | Path) -> ConvergenceConfig:
    return ConvergenceConfig.model_validate(load_yaml_mapping(path))


def _settings_for(
    config: ConvergenceConfig, scan: ScanConfig, value: Any, case_atoms: Any
) -> tuple[QESettings, tuple[int, int, int]]:
    base = config.base
    wfc, rho, spacing, degauss, conv = (
        base.ecutwfc_ry,
        base.ecutrho_ry,
        base.kspacing_inv_A,
        base.degauss_ry,
        base.conv_thr_ry,
    )
    if scan.parameter == "cutoff":
        wfc, rho = float(value[0]), float(value[1])
    elif scan.parameter == "kpoints":
        spacing = float(value)
    elif scan.parameter == "smearing":
        degauss = float(value)
    else:
        conv = float(value)
    kpts = tuple(kpts_for_spacing(case_atoms, spacing))
    manifest = load_manifest(base.pseudo_manifest) if base.pseudo_manifest else None
    minimum_ok = validate_pseudopotentials(
        base.pseudopotentials,
        base.pseudo_dir,
        ecutwfc_ry=wfc,
        ecutrho_ry=rho,
        manifest=manifest,
        functional=base.functional,
    ).problems
    settings = QESettings(
        pseudo_dir=base.pseudo_dir,
        calculation="scf",
        ecutwfc_ry=wfc,
        ecutrho_ry=rho,
        occupations="smearing",
        smearing=base.smearing,
        degauss_ry=degauss,
        kpts=kpts,
        conv_thr_ry=conv,
        pseudopotentials=dict(base.pseudopotentials),
        pseudo_manifest=manifest,
        functional=base.functional,
        # A cutoff scan deliberately includes points below the recommendation. They are
        # diagnostic points; the selected point is checked again before any production use.
        cutoff_override_reason=(
            "convergence scan point below the pseudopotential recommendation"
            if minimum_ok and scan.parameter == "cutoff"
            else None
        ),
        extra_control={"tprnfor": ".true.", "tstress": ".true."},
    )
    return settings, kpts


def plan(config: ConvergenceConfig) -> list[PlannedRun]:
    """Write one input deck per (scan, value, case) and return the runs. Deterministic."""
    out = Path(config.out_dir)
    runs: list[PlannedRun] = []
    structures = {c.name: read(c.structure) for c in config.cases}
    for scan in config.scans:
        for index, value in enumerate(scan.values):
            for case in config.cases:
                atoms = structures[case.name]
                settings, kpts = _settings_for(config, scan, value, atoms)
                directory = out / "runs" / scan.parameter / f"p{index:02d}" / case.name
                directory.mkdir(parents=True, exist_ok=True)
                input_path = directory / "pw.in"
                if case.magnetic:
                    species = sorted(set(atoms.get_chemical_symbols()))
                    settings.extra_system.update(
                        {
                            f"starting_magnetization({i})": case.starting_magnetization[s]
                            for i, s in enumerate(species, start=1)
                        }
                    )
                    settings.extra_system["nspin"] = 2
                write_pw_input(
                    atoms,
                    settings,
                    str(input_path),
                    prefix=case.name,
                    outdir=str(directory / "tmp"),
                )
                text = input_path.read_text(encoding="utf-8")
                key = hashlib.sha256(f"{scan.parameter}|{case.name}|{text}".encode()).hexdigest()
                below = settings.cutoff_override_reason is not None
                runs.append(
                    PlannedRun(
                        key=key,
                        scan=scan.parameter,
                        value=value,
                        case=case.name,
                        natoms=len(atoms),
                        directory=directory,
                        input_path=input_path,
                        output_path=directory / "pw.out",
                        kpts=kpts,
                        ecutwfc_ry=float(settings.ecutwfc_ry),
                        ecutrho_ry=float(settings.ecutrho_ry),
                        below_recommendation=below,
                    )
                )
    atomic_write_text(
        out / "plan.json",
        json.dumps(
            {
                "protocol": config.protocol,
                "config_hash": config.config_hash,
                "runs": [
                    {
                        "key": r.key,
                        "scan": r.scan,
                        "value": r.value,
                        "case": r.case,
                        "natoms": r.natoms,
                        "kpts": list(r.kpts),
                        "ecutwfc_ry": r.ecutwfc_ry,
                        "ecutrho_ry": r.ecutrho_ry,
                        "below_recommendation": r.below_recommendation,
                        "input": str(r.input_path),
                    }
                    for r in runs
                ],
            },
            indent=2,
        ),
    )
    return runs


def estimate_resources(runs: Sequence[PlannedRun]) -> dict[str, Any]:
    """Relative cost proxy, ordering only: atoms * ecutwfc^1.5 * k-points. Not core-hours."""
    cost = sum(r.natoms * r.ecutwfc_ry**1.5 * math.prod(r.kpts) for r in runs)
    return {"runs": len(runs), "relative_cost_proxy": cost, "unit": "arbitrary; not core-hours"}


Submitter = Callable[[PlannedRun], str]


def slurm_submitter(config: ConvergenceConfig, manifest: str | Path) -> Submitter:
    """Dispatch each run as one scheduler job through ``execution.slurm_jobs``."""
    from matsim_agents.execution.slurm_jobs import JobSpec, submit_job

    res = config.resources

    def submit(run: PlannedRun) -> str:
        command = tuple(
            part.format(input=str(run.input_path), output=str(run.output_path))
            for part in res.launcher
        )
        spec = JobSpec(
            name=f"qe-{run.scan[:8]}-{run.key[:10]}",
            command=command,
            time_limit=res.time_limit,
            gpus=res.gpus,
            account=res.account,
            qos=res.qos,
            partition=res.partition,
            cwd=str(run.directory),
        )
        entry, _ = submit_job(spec, manifest)
        return entry.job_id

    return submit


def submit(
    runs: Sequence[PlannedRun],
    log: KeyedResultLog,
    submitter: Submitter,
    *,
    submitted: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Submit runs that have neither a collected result nor an earlier submission."""
    done = set(log.load())
    sent = dict(submitted or {})
    for run in runs:
        if run.key in done or run.key in sent or run.output_path.is_file():
            continue
        sent[run.key] = submitter(run)
    return sent


def collect(runs: Sequence[PlannedRun], log: KeyedResultLog) -> dict[str, Any]:
    """Parse every finished output once; record unusable outputs as such."""
    known = log.load()
    versions: dict[str, str] = {}
    pending = 0
    for run in runs:
        if run.key in known:
            continue
        if not run.output_path.is_file():
            pending += 1
            continue
        observed = parse_observables(run.output_path, run.natoms)
        match = _RE_VERSION.search(run.output_path.read_text(errors="replace"))
        if match:
            versions[run.key] = match.group(1)
        log.save(run.key, {"observables": observed.to_dict(), "qe_version": versions.get(run.key)})
    return {"pending": pending, "versions": sorted(set(versions.values()))}


def analyse(
    config: ConvergenceConfig, runs: Sequence[PlannedRun], log: KeyedResultLog
) -> list[ScanOutcome]:
    results = log.load()
    criteria = AcceptanceCriteria.from_mapping(config.criteria)
    kinds = {c.name: c.kind for c in config.cases}
    magnetic = [c.name for c in config.cases if c.magnetic]
    outcomes: list[ScanOutcome] = []
    for scan in config.scans:
        points: list[ScanPoint] = []
        for index, value in enumerate(scan.values):
            cases: dict[str, PointObservables] = {}
            cost = 0.0
            for run in runs:
                if run.scan != scan.parameter or run.value != value:
                    continue
                stored = results.get(run.key)
                if stored is None:
                    continue
                obs = PointObservables.from_dict(stored["observables"])
                cases[run.case] = obs
                cost += obs.wall_time_sec or 0.0
            apply_derived(cases, config.derived)
            points.append(ScanPoint(value=value, cases=cases, cost=cost or float(index)))
        outcomes.append(analyse_scan(scan.parameter, points, criteria, kinds, magnetic))
    return outcomes


def overall_status(outcomes: Sequence[ScanOutcome]) -> str:
    if all(o.status == "qualified" for o in outcomes):
        return "qualified"
    if any(o.status == "incomplete" for o in outcomes):
        return "incomplete"
    return "not_converged"


def report(
    config: ConvergenceConfig,
    outcomes: Sequence[ScanOutcome],
    collected: Mapping[str, Any],
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write ``report.json`` and ``report.md``; the status is that of the parsed outputs."""
    out = Path(out_dir or config.out_dir)
    base = config.base
    manifest_sha = sha256_file(base.pseudo_manifest) if base.pseudo_manifest else None
    data = {
        "protocol": config.protocol,
        "status": overall_status(outcomes),
        "label": config.label,
        "config_hash": config.config_hash,
        "pseudo_manifest_sha256": manifest_sha,
        "functional": base.functional,
        "qe_versions": collected.get("versions", []),
        "python": platform.python_version(),
        "criteria": AcceptanceCriteria.from_mapping(config.criteria).__dict__,
        "scans": [o.to_dict() for o in outcomes],
    }
    atomic_write_text(out / "report.json", json.dumps(data, indent=2, default=str))
    lines = [f"# Convergence report: {config.protocol}", "", f"Status: **{data['status']}**"]
    if config.label:
        lines.append(f"Label: {config.label}")
    lines += ["", "| scan | reference | selected | status |", "|---|---|---|---|"]
    lines += [
        f"| {o.parameter} | {o.reference_value} | {o.selected_value} | {o.status} |"
        for o in outcomes
    ]
    atomic_write_text(out / "report.md", "\n".join(lines) + "\n")
    return data


def run_stage(
    config: ConvergenceConfig, stage: str, submitter: Submitter | None = None
) -> dict[str, Any]:
    """Run one stage (or ``all`` without dispatch) and return its summary."""
    out = Path(config.out_dir)
    log = KeyedResultLog(out / "results.jsonl")
    runs = plan(config)
    if stage == "plan":
        return estimate_resources(runs)
    if stage == "submit":
        if submitter is None:
            raise ValueError("submit needs a submitter")
        sent = submit(runs, log, submitter)
        atomic_write_text(out / "submitted.json", json.dumps(sent, indent=2))
        return {"submitted": len(sent)}
    collected = collect(runs, log)
    if stage == "collect":
        return collected
    outcomes = analyse(config, runs, log)
    return report(config, outcomes, collected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("stage", choices=["plan", "submit", "collect", "analyse"])
    parser.add_argument("--job-manifest", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    submitter = None
    if args.stage == "submit":
        if not args.job_manifest:
            parser.error("submit needs --job-manifest")
        submitter = slurm_submitter(config, args.job_manifest)
    result = run_stage(config, args.stage, submitter)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status", "ok") != "not_converged" else 1


if __name__ == "__main__":
    sys.exit(main())
