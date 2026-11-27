"""Locate and probe an external LAMMPS executable.

The probe runs ``<exe> -h``, which prints the version, build and installed packages and exits
without reading an input or running any dynamics. ``READY`` means the probe succeeded; it says
nothing about whether a given potential or result is valid.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from matsim_agents.core.hashing import sha256_file

ENV_EXECUTABLE = "LAMMPS_EXECUTABLE"
ENV_POTENTIALS = "LAMMPS_POTENTIALS"
CANDIDATE_NAMES = ("lmp", "lmp_mpi", "lmp_serial")
PROBE_TIMEOUT_SECONDS = 60
_VERSION_RE = re.compile(r"Large-scale Atomic/Molecular Massively Parallel Simulator - (.+)")


class Status(StrEnum):
    NOT_INSTALLED = "NOT_INSTALLED"
    INSTALLED = "INSTALLED"
    READY = "READY"
    EXECUTION_FAILED = "EXECUTION_FAILED"


@dataclass(frozen=True)
class Availability:
    status: Status
    executable: str | None = None
    executable_sha256: str | None = None
    version: str | None = None
    packages: tuple[str, ...] = ()
    potentials_dir: str | None = None
    detail: str = ""
    build: dict[str, str] = field(default_factory=dict)

    def has_package(self, name: str) -> bool:
        return name.upper() in self.packages

    def as_dict(self) -> dict:
        return {
            "tool": "lammps",
            "status": self.status.value,
            "executable": self.executable,
            "executable_sha256": self.executable_sha256,
            "version": self.version,
            "packages": list(self.packages),
            "potentials_dir": self.potentials_dir,
            "build": dict(self.build),
            "detail": self.detail,
            "meaning": "tool readiness only; not scientific validation",
        }


def find_executable(
    executable: str | os.PathLike | None = None, environ: Mapping[str, str] | None = None
) -> Path | None:
    """Explicit argument, then ``LAMMPS_EXECUTABLE``, then ``lmp``/``lmp_mpi``/``lmp_serial``."""
    env = os.environ if environ is None else environ
    explicit = executable or env.get(ENV_EXECUTABLE)
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
        found = shutil.which(str(explicit), path=env.get("PATH"))
        return Path(found).resolve() if found else None
    for name in CANDIDATE_NAMES:
        found = shutil.which(name, path=env.get("PATH"))
        if found:
            return Path(found).resolve()
    return None


def parse_help(text: str) -> tuple[str | None, tuple[str, ...], dict[str, str]]:
    """Version, installed packages and build lines from ``lmp -h`` output."""
    version = None
    if m := _VERSION_RE.search(text):
        version = m.group(1).strip()
    packages: tuple[str, ...] = ()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Installed packages:":
            body = []
            for nxt in lines[i + 1 :]:
                if not nxt.strip():
                    if body:
                        break
                    continue
                body.extend(nxt.split())
            packages = tuple(p.upper() for p in body)
            break
    build = {}
    for key in ("OS", "Compiler"):
        if m := re.search(rf"^{key}: (.+)$", text, flags=re.MULTILINE):
            build[key.lower()] = m.group(1).strip()
    if m := re.search(r"^MPI v[\d.]+: (.+)$", text, flags=re.MULTILINE):
        build["mpi"] = m.group(1).strip()
    return version, packages, build


def check_availability(
    executable: str | os.PathLike | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    invoke: bool = True,
) -> Availability:
    env = os.environ if environ is None else environ
    potentials = env.get(ENV_POTENTIALS) or None
    exe = find_executable(executable, env)
    if exe is None:
        return Availability(
            Status.NOT_INSTALLED,
            potentials_dir=potentials,
            detail=f"no LAMMPS executable (set {ENV_EXECUTABLE} or put lmp on PATH)",
        )
    digest = sha256_file(exe)
    if not invoke:
        return Availability(
            Status.INSTALLED, str(exe), digest, potentials_dir=potentials, detail="not probed"
        )
    with tempfile.TemporaryDirectory(prefix="matsim-lammps-probe-") as tmp:
        try:
            proc = subprocess.run(  # noqa: S603 - argument list, no shell, explicit cwd/timeout
                [str(exe), "-h"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
                check=False,
                env=dict(env),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Availability(
                Status.EXECUTION_FAILED,
                str(exe),
                digest,
                potentials_dir=potentials,
                detail=str(exc),
            )
    version, packages, build = parse_help(proc.stdout or "")
    if proc.returncode != 0 or version is None:
        tail = ((proc.stderr or "") + (proc.stdout or ""))[-500:]
        return Availability(
            Status.EXECUTION_FAILED,
            str(exe),
            digest,
            potentials_dir=potentials,
            detail=f"'-h' probe returned {proc.returncode} without a LAMMPS banner: {tail}",
        )
    return Availability(
        Status.READY,
        str(exe),
        digest,
        version=version,
        packages=packages,
        potentials_dir=potentials,
        build=build,
        detail="'-h' probe succeeded",
    )


__all__ = [
    "CANDIDATE_NAMES",
    "ENV_EXECUTABLE",
    "ENV_POTENTIALS",
    "Availability",
    "Status",
    "check_availability",
    "find_executable",
    "parse_help",
]
