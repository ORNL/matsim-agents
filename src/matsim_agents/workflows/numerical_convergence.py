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
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase.io import read
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matsim_agents.backends.dft.pseudo_validation import (
    PseudopotentialError,
    load_manifest,
    require_valid_pseudopotentials,
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
from matsim_agents.science.magnetism.generators import bcc_afm
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
    # Set with ``magnetic``: two-sublattice (corner / body-centre) antiferromagnet of a cubic BCC
    # supercell of this lattice parameter. The sign of ``starting_magnetization`` alternates.
    afm_lattice_A: float | None = None


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
    max_outstanding: int = Field(default=2, ge=1)
    retry_launcher: list[str] | None = None  # used once for runs that hit the time limit
    retry_gpus: int = 0


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
    pseudo_sha256: dict[str, str]


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
    )
    return settings, kpts


def plan(config: ConvergenceConfig) -> list[PlannedRun]:
    """Write one input deck per (scan, value, case) and return the runs. Deterministic."""
    out = Path(config.out_dir)
    plan_path = out / "plan.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("identity_version") != 2:
            raise ValueError(
                f"{plan_path} is a legacy plan without pseudopotential identities; "
                "preserve it and use a new out_dir for new runs"
            )
    runs: list[PlannedRun] = []
    structures = {c.name: read(c.structure) for c in config.cases}
    for scan in config.scans:
        for value in scan.values:
            for case in config.cases:
                atoms = structures[case.name]
                settings, kpts = _settings_for(config, scan, value, atoms)
                symbols = sorted(set(atoms.get_chemical_symbols()))
                pseudo_hashes = {
                    symbol: sha256_file(
                        Path(config.base.pseudo_dir) / config.base.pseudopotentials[symbol]
                    )
                    for symbol in symbols
                }
                identity = {
                    "scan": scan.parameter,
                    "value": value,
                    "case": case.model_dump(mode="json"),
                    "structure_sha256": sha256_file(case.structure),
                    "base": config.base.model_dump(mode="json"),
                    "pseudo_sha256": pseudo_hashes,
                }
                identity_hash = hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest()
                directory = out / "runs" / scan.parameter / case.name / identity_hash
                directory.mkdir(parents=True, exist_ok=True)
                input_path = directory / "pw.in"
                magnetic_config = None
                if case.magnetic and case.afm_lattice_A is not None:
                    atoms = atoms.copy()
                    atoms.set_array("atom_id", np.arange(1, len(atoms) + 1))
                    afm = bcc_afm(atoms, case.name, case.afm_lattice_A, case.starting_magnetization)
                    magnetic_config = afm.model_copy(
                        update={"qe_starting_magnetization": afm.initial_moments_muB}
                    )
                elif case.magnetic:
                    species = sorted(set(atoms.get_chemical_symbols()))
                    settings.extra_system.update(
                        {
                            f"starting_magnetization({i})": case.starting_magnetization.get(s, 0.0)
                            for i, s in enumerate(species, start=1)
                        }
                    )
                    settings.extra_system["nspin"] = 2
                with tempfile.TemporaryDirectory(prefix="matsim-qe-input-") as temp_dir:
                    candidate = Path(temp_dir) / "pw.in"
                    write_pw_input(
                        atoms,
                        settings,
                        str(candidate),
                        prefix=case.name,
                        outdir=str(directory / "tmp"),
                        magnetic_config=magnetic_config,
                    )
                    text = candidate.read_text(encoding="utf-8")
                if input_path.exists() and input_path.read_text(encoding="utf-8") != text:
                    raise ValueError(
                        f"{input_path} differs from its planned identity; refusing overwrite"
                    )
                if not input_path.exists():
                    atomic_write_text(input_path, text)
                key = hashlib.sha256(f"{identity_hash}|{text}".encode()).hexdigest()
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
                        pseudo_sha256=pseudo_hashes,
                    )
                )
    atomic_write_text(
        out / "plan.json",
        json.dumps(
            {
                "protocol": config.protocol,
                "identity_version": 2,
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
                        "pseudo_sha256": r.pseudo_sha256,
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
        entry, _ = submit_job(spec, manifest, max_outstanding_gpu_jobs=res.max_outstanding)
        return entry.job_id

    return submit


def retry_timed_out(
    config: ConvergenceConfig,
    runs: Sequence[PlannedRun],
    manifest: str | Path,
    runner: Any = None,
) -> dict[str, str]:
    """Resubmit each run whose job hit the time limit once, with the retry launcher and GPUs."""
    from matsim_agents.execution.slurm_jobs import (
        JobSpec,
        load_manifest,
        retry_job,
        subprocess_runner,
    )

    runner = runner or subprocess_runner
    res = config.resources
    if res.retry_launcher is None:
        raise ValueError("resources.retry_launcher is not configured")
    path = Path(config.out_dir) / "submitted.json"
    sent: dict[str, str] = json.loads(path.read_text())
    by_job = {e.job_id: e for e in load_manifest(manifest)}
    by_key = {r.key: r for r in runs}
    retried: dict[str, str] = {}
    for key, job_id in sorted(sent.items()):
        entry = by_job.get(job_id)
        run = by_key.get(key)
        if entry is None or run is None or entry.state != "TIMEOUT" or entry.retried:
            continue
        command = tuple(
            part.format(input=str(run.input_path), output=str(run.output_path))
            for part in res.retry_launcher
        )
        spec = entry.spec.model_copy(update={"command": command, "gpus": res.retry_gpus})
        new = retry_job(
            entry.spec_hash,
            manifest,
            diagnosis="the single-GPU run reached its time limit; rerun on more GPUs",
            spec=JobSpec.model_validate(spec.model_dump()),
            max_outstanding_gpu_jobs=res.max_outstanding,
            runner=runner,
        )
        retried[key] = new.job_id
        sent[key] = new.job_id
        atomic_write_text(path, json.dumps(sent, indent=2))
    return retried


def submit(
    runs: Sequence[PlannedRun],
    log: KeyedResultLog,
    submitter: Submitter,
    *,
    submitted: Mapping[str, str] | None = None,
    persist: Callable[[Mapping[str, str]], None] | None = None,
) -> dict[str, str]:
    """Submit runs that have neither a collected result nor an earlier submission."""
    done = set(log.load())
    sent = dict(submitted or {})
    for run in runs:
        if run.key in done or run.key in sent or run.output_path.is_file():
            continue
        sent[run.key] = submitter(run)
        if persist is not None:
            persist(sent)
    return sent


def collect(
    runs: Sequence[PlannedRun], log: KeyedResultLog, ended: Collection[str] = ()
) -> dict[str, Any]:
    """Parse each finished output once. An output counts as finished when it ends in JOB DONE or
    its run key is in ``ended`` (the scheduler stopped the job); a partial output stays pending."""
    known = log.load()
    run_keys = {run.key for run in runs}
    versions: dict[str, str] = {
        key: result["qe_version"]
        for key, result in known.items()
        if key in run_keys and result.get("qe_version")
    }
    pending = 0
    for run in runs:
        previous = known.get(run.key)
        if previous is not None and PointObservables.from_dict(previous["observables"]).usable:
            continue
        if not run.output_path.is_file():
            pending += 1
            continue
        if run.key not in ended and "JOB DONE" not in run.output_path.read_text(errors="replace"):
            pending += 1
            continue
        output_sha = sha256_file(run.output_path)
        if previous is not None and previous.get("output_sha256") == output_sha:
            continue
        observed = parse_observables(run.output_path, run.natoms)
        match = _RE_VERSION.search(run.output_path.read_text(errors="replace"))
        versions.pop(run.key, None)
        if match:
            versions[run.key] = match.group(1)
        log.save(
            run.key,
            {
                "observables": observed.to_dict(),
                "qe_version": versions.get(run.key),
                "output_sha256": output_sha,
            },
        )
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
            diagnostic = any(
                r.below_recommendation
                for r in runs
                if r.scan == scan.parameter and r.value == value
            )
            points.append(
                ScanPoint(
                    value=value,
                    cases=cases,
                    cost=cost or float(index),
                    expected_cases=frozenset(c.name for c in config.cases),
                    diagnostic_only=diagnostic,
                )
            )
        outcome = analyse_scan(scan.parameter, points, criteria, kinds, magnetic)
        if scan.parameter == "cutoff" and outcome.selected_value is not None:
            try:
                for run in runs:
                    if run.scan != scan.parameter or run.value != outcome.selected_value:
                        continue
                    for symbol, digest in run.pseudo_sha256.items():
                        path = Path(config.base.pseudo_dir) / config.base.pseudopotentials[symbol]
                        if sha256_file(path) != digest:
                            raise PseudopotentialError(
                                f"{symbol}: pseudopotential changed since planning"
                            )
                    require_valid_pseudopotentials(
                        {s: config.base.pseudopotentials[s] for s in run.pseudo_sha256},
                        config.base.pseudo_dir,
                        ecutwfc_ry=run.ecutwfc_ry,
                        ecutrho_ry=run.ecutrho_ry,
                        manifest=load_manifest(config.base.pseudo_manifest)
                        if config.base.pseudo_manifest
                        else None,
                        functional=config.base.functional,
                    )
            except (PseudopotentialError, OSError) as exc:
                outcome.status = "incomplete"
                outcome.selected_value = None
                outcome.comparisons[-1].reasons.append(f"selected cutoff validation failed: {exc}")
        outcomes.append(outcome)
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


def ended_run_keys(config: ConvergenceConfig, manifest: str | Path) -> set[str]:
    """Run keys whose scheduler job has stopped, from one refresh of the job manifest."""
    from matsim_agents.execution.slurm_jobs import refresh_manifest

    path = Path(config.out_dir) / "submitted.json"
    if not path.exists():
        return set()
    submitted = json.loads(path.read_text())
    stopped = {e.job_id for e in refresh_manifest(manifest) if e.terminal}
    return {key for key, job_id in submitted.items() if job_id in stopped}


def run_stage(
    config: ConvergenceConfig,
    stage: str,
    submitter: Submitter | None = None,
    ended: Collection[str] = (),
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
        from matsim_agents.execution.slurm_jobs import OutstandingLimitError

        submitted_path = out / "submitted.json"
        previous = json.loads(submitted_path.read_text()) if submitted_path.exists() else {}

        def persist(sent: Mapping[str, str]) -> None:
            atomic_write_text(submitted_path, json.dumps(sent, indent=2))

        try:
            sent = submit(runs, log, submitter, submitted=previous, persist=persist)
        except OutstandingLimitError as exc:
            current = (
                json.loads(submitted_path.read_text()) if submitted_path.exists() else previous
            )
            return {"submitted": len(current), "status": "capacity_reached", "reason": str(exc)}
        persist(sent)
        return {"submitted": len(sent)}
    collected = collect(runs, log, ended)
    if stage == "collect":
        return collected
    outcomes = analyse(config, runs, log)
    return report(config, outcomes, collected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("stage", choices=["plan", "submit", "retry", "collect", "analyse"])
    parser.add_argument("--job-manifest", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    submitter = None
    if args.stage == "submit":
        if not args.job_manifest:
            parser.error("submit needs --job-manifest")
        submitter = slurm_submitter(config, args.job_manifest)
    if args.stage == "retry":
        if not args.job_manifest:
            parser.error("retry needs --job-manifest")
        ended_run_keys(config, args.job_manifest)  # refresh states first
        done = retry_timed_out(config, plan(config), args.job_manifest)
        print(json.dumps({"retried": len(done)}, indent=2))
        return 0
    ended = ended_run_keys(config, args.job_manifest) if args.job_manifest else ()
    result = run_stage(config, args.stage, submitter, ended)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status", "ok") != "not_converged" else 1


if __name__ == "__main__":
    sys.exit(main())
