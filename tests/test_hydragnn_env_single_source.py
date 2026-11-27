"""Job scripts take the HydraGNN code location from deployments/common/hydragnn-env.sh only."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV_SCRIPT = ROOT / "deployments/common/hydragnn-env.sh"
PIN = json.loads((ROOT / "benchmarks/e4_mlip/frozen-source.json").read_text())["commit"]

# Scripts allowed to name a HydraGNN location themselves, with the reason.
ALLOWED = {
    # Reviewed pilot: the approved manifest supplies HYDRAGNN_ROOT and the job re-hashes inputs.
    "deployments/aurora/jobs/job-e4b-mlip-pilot.pbs",
    # Installers clone the tree; they cannot depend on it existing.
    "deployments/frontier/setup/install.sh",
    "deployments/aurora/setup/install.sh",
    "deployments/perlmutter/setup/install.sh",
    "scripts/setup_env.sh",
}

# An assignment of a HydraGNN location variable, or a literal path into a HydraGNN checkout.
ASSIGNMENT = re.compile(
    r"^\s*(?:export\s+)?(?:HYDRAGNN_ROOT|HYDRAGNN_DIR|HYDRAGNN_EXAMPLE|HG_SRC|HYDRAGNN_SC26_DIR)="
)
LITERAL = re.compile(r"/HydraGNN(?:-e4b[^/\s\"']*)?/(?:examples|hydragnn)\b")
PROJ_TREE = re.compile(r"PROJ\}?/HydraGNN(?![_\w])")


def _scripts() -> list[Path]:
    found: list[Path] = []
    for pattern in ("deployments/**/*.sh", "deployments/**/*.pbs", "scripts/**/*.sh"):
        found.extend(ROOT.glob(pattern))
    return sorted(p for p in found if p != ENV_SCRIPT)


def _code_lines(path: Path):
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.lstrip().startswith("#"):
            yield number, line


def test_no_script_names_a_hydragnn_location_outside_the_env_script():
    offenders = []
    for path in _scripts():
        relative = path.relative_to(ROOT).as_posix()
        if relative in ALLOWED:
            continue
        for number, line in _code_lines(path):
            if ASSIGNMENT.search(line) or LITERAL.search(line) or PROJ_TREE.search(line):
                offenders.append(f"{relative}:{number}: {line.strip()}")
    assert not offenders, "source deployments/common/hydragnn-env.sh instead:\n" + "\n".join(
        offenders
    )


def test_every_allowlisted_script_exists():
    assert [p for p in ALLOWED if not (ROOT / p).is_file()] == []


@pytest.mark.parametrize("path", _scripts(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_script_parses(path):
    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _source(root: Path, **env: str) -> subprocess.CompletedProcess[str]:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("HYDRAGNN_")}
    clean.pop("PROJ", None)
    return subprocess.run(
        [
            "bash",
            "-uc",
            f'source "{ENV_SCRIPT}"; rc=$?; '
            'echo "$HYDRAGNN_ROOT|$HYDRAGNN_BEST6_LOGDIR|$HYDRAGNN_DIR"; exit $rc',
        ],
        env={**clean, "HYDRAGNN_ROOT": str(root), **env},
        capture_output=True,
        text=True,
    )


def _checkout(path: Path) -> None:
    path.mkdir()
    _git("init", cwd=path)
    (path / "tracked.txt").write_text("x")
    _git("add", "tracked.txt", cwd=path)
    _git("commit", "-m", "x", cwd=path)


def test_env_script_refuses_a_checkout_at_another_commit(tmp_path):
    root = tmp_path / "HydraGNN"
    _checkout(root)
    result = _source(root)
    assert result.returncode == 1
    assert PIN in result.stderr


def test_env_script_refuses_a_checkout_with_tracked_changes(tmp_path):
    # Pin the script to this checkout's own commit so only the dirty-tree check can fail.
    root = tmp_path / "repo"
    (root / "deployments/common").mkdir(parents=True)
    (root / "benchmarks/e4_mlip").mkdir(parents=True)
    script = root / "deployments/common/hydragnn-env.sh"
    script.write_text(ENV_SCRIPT.read_text())
    hg = tmp_path / "HydraGNN"
    _checkout(hg)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=hg, capture_output=True, text=True, check=True
    ).stdout.strip()
    (root / "benchmarks/e4_mlip/frozen-source.json").write_text(json.dumps({"commit": head}))
    clean = {k: v for k, v in os.environ.items() if not k.startswith("HYDRAGNN_")}

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-uc", f'source "{script}"'],
            env={**clean, "HYDRAGNN_ROOT": str(hg)},
            capture_output=True,
            text=True,
        )

    assert run().returncode == 0
    (hg / "tracked.txt").write_text("changed")
    dirty = run()
    assert dirty.returncode == 1
    assert "uncommitted changes" in dirty.stderr


def test_weights_default_to_the_pinned_tree_when_they_live_there(tmp_path):
    root = tmp_path / "HydraGNN"
    _checkout(root)
    best6 = root / "examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64"
    best6.mkdir(parents=True)
    result = _source(root)  # wrong commit, but the variables are still exported
    assert result.stdout.split("|")[1] == str(best6)
