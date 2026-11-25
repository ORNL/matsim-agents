"""Orchestrate: validate -> stage -> (dry run | execute) -> collect -> parse -> validate.

Dry run is the default. Nothing here downloads, compiles, or installs SPECTRA-PKA, and no
number is produced unless the external program produced it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from matsim_agents.backends.external.spectra_pka.availability import (
    Availability,
    Status,
    check_availability,
)
from matsim_agents.backends.external.spectra_pka.config import (
    CampaignSpec,
    SpectraPKAConfig,
)
from matsim_agents.backends.external.spectra_pka.input_writer import (
    INPUT_NAME,
    StagedInputs,
    stage_inputs,
)
from matsim_agents.backends.external.spectra_pka.parser import (
    ParsedOutput,
    ParseError,
    parse_output_file,
    to_pka_spectrum,
)
from matsim_agents.backends.external.spectra_pka.runner import (
    RUN_RECORD,
    STDERR_NAME,
    STDOUT_NAME,
    RunOutcome,
    run_local,
    submit_slurm,
)
from matsim_agents.backends.external.spectra_pka.validation import (
    ValidationReport,
    validate_output,
)
from matsim_agents.core.hashing import sha256_file, sha256_json
from matsim_agents.science.irradiation.contracts import (
    IrradiationScenario,
    PKASourceProvenance,
    PKASpectrum,
    ValidationStatus,
)

SOURCE_NAME = "SPECTRA-PKA (external)"
CACHE_RECORD = "result_cache.json"
ARCHIVE_DIR = "archive"


class AdapterError(RuntimeError):
    """The adapter refused to proceed; the message says why."""


@dataclass(frozen=True)
class PreparedRun:
    run_dir: Path
    staged: StagedInputs
    command: tuple[str, ...]
    availability: Availability


@dataclass(frozen=True)
class RunResult:
    # DRY_RUN | SUBMITTED | EXECUTION_FAILED | PARSE_FAILED
    # | STRUCTURALLY_INVALID | STRUCTURALLY_VALID
    status: str
    run_dir: Path
    command: tuple[str, ...]
    outcome: RunOutcome | None = None
    validation: ValidationReport | None = None
    spectrum: PKASpectrum | None = None
    message: str = ""
    # True when no process was launched; ``outcome`` is then that of the run that made the bytes.
    cached: bool = False


def scenario_from_campaign(campaign: CampaignSpec) -> IrradiationScenario:
    return IrradiationScenario(
        scenario_id=campaign.scenario_id,
        parent_alloy_id=campaign.parent_alloy_id,
        composition=dict(campaign.composition),
        isotopic_composition=campaign.isotopic_composition(),
        neutron_spectrum_identity=campaign.flux.identity,
        flux_normalization=campaign.flux.normalization,
    )


def run_directory(config: SpectraPKAConfig, campaign: CampaignSpec) -> Path:
    return config.workdir / campaign.scenario_id


def prepare(config: SpectraPKAConfig, campaign: CampaignSpec) -> PreparedRun:
    """Stage inputs. Refuses unless the executable exists and the data are present."""
    if not config.enabled:
        raise AdapterError("external_tools.spectra_pka.enabled is false")
    availability = check_availability(config, campaign, invoke=False)
    if availability.status is Status.NOT_INSTALLED:
        raise AdapterError("SPECTRA-PKA is not installed at the configured executable path")
    if availability.status is Status.INPUTS_MISSING:
        bad = [c.detail for c in availability.checks if c.ok is False]
        raise AdapterError("inputs are missing: " + "; ".join(bad))
    assert config.executable is not None
    run_dir = run_directory(config, campaign)
    staged = stage_inputs(campaign, run_dir)
    command = (str(config.executable), INPUT_NAME)
    return PreparedRun(run_dir, staged, command, availability)


def collect(
    config: SpectraPKAConfig,
    campaign: CampaignSpec,
    run_dir: Path,
    *,
    availability: Availability | None = None,
    staged_hashes: dict[str, str] | None = None,
    input_sha256: str | None = None,
    outcome: RunOutcome | None = None,
) -> RunResult:
    """Parse and validate the raw outputs already in ``run_dir``. Raw files stay untouched."""
    out_path = run_dir / f"{campaign.results_stub}.out"
    idx_path = run_dir / f"{campaign.results_stub}.indexes"
    command = (str(config.executable or ""), INPUT_NAME)
    if not out_path.is_file():
        return RunResult("EXECUTION_FAILED", run_dir, command, outcome, message="no .out file")
    try:
        parsed: ParsedOutput = parse_output_file(out_path, idx_path)
    except ParseError as exc:
        return RunResult("PARSE_FAILED", run_dir, command, outcome, message=str(exc))
    marker = outcome.end_marker_seen if outcome is not None else None
    report = validate_output(parsed, campaign, end_marker_seen=marker)
    spectrum = None
    if report.ok:
        provenance = PKASourceProvenance(
            source=SOURCE_NAME,
            executable_sha256=availability.executable_sha256 if availability else None,
            executable_path=str(config.executable) if config.executable else None,
            nuclear_data_identity=campaign.nuclear_data_identity,
            input_sha256={INPUT_NAME: input_sha256 or "", **(staged_hashes or {})},
            output_sha256={
                out_path.name: parsed.output_sha256,
                **({idx_path.name: parsed.indexes_sha256} if parsed.indexes_sha256 else {}),
            },
            execution=outcome.as_dict() if outcome else {},
            raw_output_dir=str(run_dir),
            numerical_validation=ValidationStatus.NOT_VALIDATED,
            notes=(
                "structurally valid output; not compared with any independent reference",
                "PKA production is not surviving-defect production",
            ),
        )
        try:
            spectrum = to_pka_spectrum(parsed, scenario_from_campaign(campaign), provenance)
        except ParseError as exc:
            return RunResult("PARSE_FAILED", run_dir, command, outcome, report, message=str(exc))
    return RunResult(report.status.value, run_dir, command, outcome, report, spectrum, message="")


def cache_key(availability: Availability, staged: StagedInputs, results_stub: str) -> str:
    """Identity of a result: executable, input text, every staged data file, output stub."""
    return sha256_json(
        {
            "executable_sha256": availability.executable_sha256,
            "input_sha256": staged.input_sha256,
            "data_sha256": staged.data_sha256,
            "results_stub": results_stub,
        }
    )


def _output_hashes(run_dir: Path, results_stub: str) -> dict[str, str]:
    names = (f"{results_stub}.out", f"{results_stub}.indexes")
    return {name: sha256_file(run_dir / name) for name in names if (run_dir / name).is_file()}


def _cache_hit(run_dir: Path, key: str, results_stub: str) -> RunOutcome | None:
    """The outcome of the run that produced the cached bytes, or None when there is no hit."""
    record = run_dir / CACHE_RECORD
    if not (record.is_file() and (run_dir / f"{results_stub}.out").is_file()):
        return None
    try:
        data = json.loads(record.read_text())
        # Structure alone cannot show that the preserved bytes came from the cached execution.
        if data.get("key") != key or data.get("outputs") != _output_hashes(run_dir, results_stub):
            return None
        return RunOutcome.from_dict(data["outcome"])
    except (ValueError, KeyError, TypeError):
        return None


def _archive_previous_run(run_dir: Path, results_stub: str) -> Path | None:
    """Move the previous run's artifacts aside so none is collected as new and none is lost."""
    names = [
        f"{results_stub}{suffix}" for suffix in (".out", ".indexes", ".log", ".input.snapshot")
    ]
    names += [STDOUT_NAME, STDERR_NAME, RUN_RECORD, CACHE_RECORD]
    present = [n for n in names if (run_dir / n).exists()]
    if not present:
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = run_dir / ARCHIVE_DIR / stamp
    target.mkdir(parents=True)
    for name in present:
        shutil.move(str(run_dir / name), str(target / name))
    return target


def run(
    config: SpectraPKAConfig,
    campaign: CampaignSpec,
    *,
    execute: bool = False,
    slurm_manifest: Path | None = None,
    use_cache: bool = True,
) -> RunResult:
    """Stage inputs, then execute only if ``execute`` is true. Default is a dry run.

    A local run whose cache key matches a previous structurally valid run is not repeated:
    the preserved raw output is parsed again instead, with the original execution outcome.
    Artifacts of an earlier run are moved to ``archive/<utc stamp>/``, never deleted.
    """
    prepared = prepare(config, campaign)
    if not execute:
        return RunResult(
            "DRY_RUN",
            prepared.run_dir,
            prepared.command,
            message="inputs staged; nothing executed (pass execute=True to run)",
        )
    assert config.executable is not None
    if config.execution.mode == "slurm":
        outcome = submit_slurm(
            config.executable,
            INPUT_NAME,
            prepared.run_dir,
            config.execution,
            name=f"spka-{campaign.scenario_id}",
            manifest=slurm_manifest or config.workdir / "jobs.json",
        )
        status = "SUBMITTED" if outcome.job_id else "EXECUTION_FAILED"
        return RunResult(
            status, prepared.run_dir, prepared.command, outcome, message=outcome.stderr_tail
        )
    key = cache_key(prepared.availability, prepared.staged, campaign.results_stub)
    original = _cache_hit(prepared.run_dir, key, campaign.results_stub) if use_cache else None
    if original is not None:
        hit = collect(
            config,
            campaign,
            prepared.run_dir,
            availability=prepared.availability,
            staged_hashes=prepared.staged.data_sha256,
            input_sha256=prepared.staged.input_sha256,
            outcome=original,
        )
        if hit.status == "STRUCTURALLY_VALID":
            return RunResult(
                hit.status,
                hit.run_dir,
                hit.command,
                hit.outcome,
                hit.validation,
                hit.spectrum,
                message=f"cache hit {key[:12]}; not re-executed",
                cached=True,
            )
    # Only artifacts of this invocation may be collected and cached.
    _archive_previous_run(prepared.run_dir, campaign.results_stub)
    # a real probe before a real run
    availability = check_availability(config, campaign, invoke=True)
    if availability.status is Status.EXECUTION_FAILED:
        raise AdapterError("the executable failed its invocation probe")
    outcome = run_local(
        config.executable,
        INPUT_NAME,
        prepared.run_dir,
        campaign.results_stub,
        config.execution.timeout_seconds,
    )
    shutil.copyfile(
        prepared.run_dir / INPUT_NAME, prepared.run_dir / f"{campaign.results_stub}.input.snapshot"
    )
    if not outcome.process_succeeded:
        return RunResult(
            "EXECUTION_FAILED",
            prepared.run_dir,
            prepared.command,
            outcome,
            message=outcome.stderr_tail or outcome.stdout_tail[-500:],
        )
    result = collect(
        config,
        campaign,
        prepared.run_dir,
        availability=availability,
        staged_hashes=prepared.staged.data_sha256,
        input_sha256=prepared.staged.input_sha256,
        outcome=outcome,
    )
    if result.status == "STRUCTURALLY_VALID":
        outputs = _output_hashes(prepared.run_dir, campaign.results_stub)
        (prepared.run_dir / CACHE_RECORD).write_text(
            json.dumps({"key": key, "outputs": outputs, "outcome": outcome.as_dict()}) + "\n"
        )
    return result


def status(config: SpectraPKAConfig, campaign: CampaignSpec) -> dict:
    run_dir = run_directory(config, campaign)
    record = run_dir / RUN_RECORD
    return {
        "run_dir": str(run_dir),
        "staged": (run_dir / INPUT_NAME).is_file(),
        "run_record": json.loads(record.read_text()) if record.is_file() else None,
        "output_present": (run_dir / f"{campaign.results_stub}.out").is_file(),
    }


__all__ = [
    "AdapterError",
    "PreparedRun",
    "RunResult",
    "collect",
    "prepare",
    "run",
    "scenario_from_campaign",
    "status",
]
