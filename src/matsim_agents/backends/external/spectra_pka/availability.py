"""Availability checks for an externally installed SPECTRA-PKA.

Statuses describe tool readiness only. ``READY`` does not mean any result is scientifically
qualified. Each check is reported separately and nothing here invents a ``--version`` flag:
the executable is identified by its SHA-256 and its install-root metadata.

The only invocation used is ``<exe> <input-file>``, the form the upstream README documents.
The reachability probe passes a deliberately missing input file; the upstream program reports
``error opening parameter input file`` and stops, which shows that the binary starts and is
the expected program without running any physics.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from matsim_agents.backends.external.spectra_pka.config import (
    CampaignSpec,
    SpectraPKAConfig,
)
from matsim_agents.core.hashing import sha256_file

PROBE_INPUT = "matsim_agents_probe_missing.input"
PROBE_MARKER = "error opening parameter input file"
SUCCESS_MARKER = "END as expected"
SMOKE_RECORD = "smoke_test.json"
PROBE_TIMEOUT_SECONDS = 30


class Status(StrEnum):
    NOT_INSTALLED = "NOT_INSTALLED"
    INSTALLED = "INSTALLED"
    INPUTS_MISSING = "INPUTS_MISSING"
    READY = "READY"
    EXECUTION_FAILED = "EXECUTION_FAILED"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool | None  # None = not assessed
    detail: str = ""


@dataclass(frozen=True)
class Availability:
    status: Status
    checks: tuple[Check, ...]
    executable_sha256: str | None = None
    install_metadata: dict[str, str] = field(default_factory=dict)
    smoke_test: str = "not_run"  # not_run | passed | failed | stale

    def check(self, name: str) -> Check:
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "executable_sha256": self.executable_sha256,
            "install_metadata": dict(self.install_metadata),
            "smoke_test": self.smoke_test,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
            "meaning": "tool readiness only; not scientific qualification",
        }


def locate_executable(config: SpectraPKAConfig) -> Path | None:
    """The configured executable only. There is no search of ``PATH`` and no download."""
    return config.executable


def install_metadata(config: SpectraPKAConfig) -> dict[str, str]:
    """Facts that identify the install without running it (no invented version flag)."""
    meta: dict[str, str] = {}
    root = config.installation_root
    if root is None and config.executable is not None:
        root = config.executable.parent
    if root is not None and root.is_dir():
        meta["installation_root"] = str(root)
        for name in ("VERSION", "version.txt", "LICENSE", "COPYING"):
            candidate = root / name
            if candidate.is_file():
                meta[f"file:{name}"] = sha256_file(candidate)
        git_head = root / ".git" / "HEAD"
        if git_head.is_file():
            meta["git_head"] = git_head.read_text().strip()[:200]
    return meta


def probe_invocation(executable: Path, *, timeout: int = PROBE_TIMEOUT_SECONDS) -> Check:
    """Start the binary once with a missing input file, in an empty scratch directory."""
    with tempfile.TemporaryDirectory(prefix="spectra-pka-probe-") as scratch:
        try:
            proc = subprocess.run(  # noqa: S603 - argument list, no shell, explicit cwd
                [str(executable), PROBE_INPUT],
                cwd=scratch,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Check("invocation", False, f"no exit within {timeout}s")
        except OSError as exc:
            return Check("invocation", False, f"cannot execute: {exc}")
    text = (proc.stdout or "") + (proc.stderr or "")
    if PROBE_MARKER in text:
        return Check("invocation", True, "starts and reports a missing input file as expected")
    return Check(
        "invocation",
        False,
        "started but did not behave like SPECTRA-PKA (expected the message "
        f"{PROBE_MARKER!r}); exit code {proc.returncode}",
    )


def read_smoke_record(workdir: Path) -> dict | None:
    path = workdir / SMOKE_RECORD
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_smoke_record(
    workdir: Path, *, executable_sha256: str, passed: bool, evidence: str
) -> Path:
    """Record the outcome of a real smoke run. Written only by a run that finished, never by a
    mock-based test of this repository."""
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / SMOKE_RECORD
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "executable_sha256": executable_sha256,
                "evidence": evidence,
                "note": "process-level smoke result; says nothing about physical accuracy",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def _smoke_state(workdir: Path, exe_hash: str | None) -> str:
    record = read_smoke_record(workdir)
    if record is None:
        return "not_run"
    if record.get("executable_sha256") != exe_hash:
        return "stale"
    return "passed" if record.get("passed") is True else "failed"


def _data_checks(config: SpectraPKAConfig, campaign: CampaignSpec | None) -> list[Check]:
    checks: list[Check] = []
    root = config.nuclear_data_root
    if root is None:
        checks.append(Check("nuclear_data", False, "nuclear_data_root is not configured"))
    elif not root.is_dir():
        checks.append(Check("nuclear_data", False, f"{root} is not a directory"))
    else:
        checks.append(Check("nuclear_data", True, str(root)))
    if campaign is None:
        checks.append(Check("data_coverage", None, "no campaign given"))
        checks.append(Check("spectrum_input", None, "no campaign given"))
        return checks
    missing = [
        f"{ch.parent_ele}{ch.parent_num}: {ch.pka_file}"
        for ch in campaign.channels
        if not ch.pka_file.is_file()
    ]
    checks.append(
        Check(
            "data_coverage",
            not missing,
            "all requested isotopes have a PKA matrix file"
            if not missing
            else "missing PKA files for " + "; ".join(missing),
        )
    )
    flux = campaign.flux.file
    checks.append(
        Check(
            "spectrum_input",
            flux.is_file(),
            str(flux) if flux.is_file() else f"{flux} not found",
        )
    )
    return checks


def _environment_check(config: SpectraPKAConfig) -> Check:
    workdir = config.workdir
    probe = workdir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usable = probe.is_dir() and os.access(probe, os.W_OK | os.X_OK)
    return Check(
        "execution_environment",
        usable,
        f"{workdir} can be created and written" if usable else f"cannot write under {probe}",
    )


def check_availability(
    config: SpectraPKAConfig,
    campaign: CampaignSpec | None = None,
    *,
    invoke: bool = True,
) -> Availability:
    """Run every check separately and derive a readiness status."""
    exe = locate_executable(config)
    checks: list[Check] = []
    if exe is None:
        checks.append(Check("executable_exists", False, "executable is not configured"))
    else:
        exists = exe.is_file()
        checks.append(Check("executable_exists", exists, str(exe) if exists else f"{exe} missing"))
        if exists:
            checks.append(Check("executable_permission", os.access(exe, os.X_OK), "x bit set"))
    exe_ok = all(c.ok for c in checks) and bool(checks)
    exe_hash = sha256_file(exe) if exe is not None and exe.is_file() else None
    meta = install_metadata(config) if exe_ok else {}
    if not exe_ok:
        return Availability(Status.NOT_INSTALLED, tuple(checks), exe_hash, meta, "not_run")

    invocation = probe_invocation(exe) if invoke else Check("invocation", None, "probe skipped")
    checks.append(invocation)
    checks.extend(_data_checks(config, campaign))
    checks.append(_environment_check(config))
    smoke = _smoke_state(config.workdir, exe_hash)
    checks.append(
        Check(
            "numerical_smoke_test",
            {"passed": True, "failed": False}.get(smoke),
            smoke,
        )
    )

    if invocation.ok is False or smoke == "failed":
        status = Status.EXECUTION_FAILED
    else:
        by_name = {c.name: c for c in checks}
        inputs_assessed = campaign is not None
        data_ok = all(by_name[n].ok for n in ("nuclear_data", "data_coverage", "spectrum_input"))
        if inputs_assessed and (not data_ok):
            status = Status.INPUTS_MISSING
        elif not inputs_assessed:
            status = Status.INSTALLED
        elif by_name["execution_environment"].ok:
            status = Status.READY
        else:
            status = Status.INSTALLED
    return Availability(status, tuple(checks), exe_hash, meta, smoke)


__all__ = [
    "PROBE_INPUT",
    "PROBE_MARKER",
    "SMOKE_RECORD",
    "SUCCESS_MARKER",
    "Availability",
    "Check",
    "Status",
    "check_availability",
    "install_metadata",
    "locate_executable",
    "probe_invocation",
    "read_smoke_record",
    "write_smoke_record",
]
