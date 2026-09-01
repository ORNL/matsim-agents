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

        if path.suffix == ".sh":
            result = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True, check=False
            )
            if result.returncode:
                errors.append(f"{rel}: bash -n failed: {result.stderr.strip()}")
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
