#!/usr/bin/env python3
"""Static portability checks for machine deployment assets."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENTS = ROOT / "deployments"

MOVED_PATH = re.compile(
    r"scripts/(?:advanced|setup|launchers|smoke-tests|download)/(?:frontier|aurora|perlmutter)"
)
EMBEDDED_ACCOUNT = re.compile(r"^#(?:SBATCH|PBS) -A\s+(?!<)", re.MULTILINE)
ABSOLUTE_SCHEDULER_LOG = re.compile(r"^#SBATCH -[oe]\s+/", re.MULTILINE)
SITE_PROJECT_PATH = re.compile(r"/(?:lustre/orion|lus/flare|global/cfs)/")
NONEXISTENT_MAIN_MODULE = re.compile(r"python(?:3)?\s+-m\s+matsim_agents\.main\b")
LEGACY_UQ_DRIVER = re.compile(r"examples/active_learning_uq\.py")

FACILITIES = ("frontier", "aurora", "perlmutter")
SHARED_JOB_CONTRACTS = {
    "job-single-relaxation-{facility}.sh": "deployments/common/run-mlip-relaxation.sh",
    "job-active-learning-uq-{facility}.sh": "deployments/common/run-active-learning.sh",
}


def validate() -> list[str]:
    errors: list[str] = []
    for path in sorted(DEPLOYMENTS.rglob("*")):
        if not path.is_file() or path.suffix not in {".sh", ".py", ".md", ".yaml", ".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        for label, pattern in (
            ("stale pre-migration path", MOVED_PATH),
            ("embedded scheduler account", EMBEDDED_ACCOUNT),
            ("absolute scheduler log path", ABSOLUTE_SCHEDULER_LOG),
            ("embedded site/project path", SITE_PROJECT_PATH),
        ):
            if match := pattern.search(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{rel}:{line}: {label}: {match.group(0)!r}")

        if (
            "matsim-agents " in text
            and "--mlp-checkpoint" in text
            and "examples/active_learning_uq.py" not in text
            and "step2_perturbation_diagnostic.py" not in text
        ):
            errors.append(
                f"{rel}: obsolete matsim-agents option --mlp-checkpoint; "
                "use --hydragnn-branch-mlp-checkpoint"
            )

        if match := NONEXISTENT_MAIN_MODULE.search(text):
            errors.append(f"{rel}: references nonexistent Python module: {match.group(0)!r}")
        if match := LEGACY_UQ_DRIVER.search(text):
            errors.append(
                f"{rel}: uses retired standalone UQ driver; use matsim-agents al run: "
                f"{match.group(0)!r}"
            )
        if "examples/paper_cases/singlepass.py" in text and (
            "MATSIM_LEGACY_MANUSCRIPT_REPRODUCTION=1" not in text
        ):
            errors.append(f"{rel}: singlepass manuscript driver must be explicitly marked legacy")

        if path.suffix == ".sh":
            result = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True, check=False
            )
            if result.returncode:
                errors.append(f"{rel}: bash -n failed: {result.stderr.strip()}")

    for facility in FACILITIES:
        for pattern, shared_runner in SHARED_JOB_CONTRACTS.items():
            name = pattern.format(facility=facility)
            path = DEPLOYMENTS / facility / "jobs" / name
            if not path.is_file():
                errors.append(f"missing cross-facility workflow job: {path.relative_to(ROOT)}")
                continue
            text = path.read_text(encoding="utf-8")
            if shared_runner not in text:
                errors.append(
                    f"{path.relative_to(ROOT)}: must delegate to shared runner {shared_runner}"
                )
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Deployment validation failed:", file=sys.stderr)
        print("\n".join(f"  - {error}" for error in errors), file=sys.stderr)
        return 1
    print("Deployment validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
