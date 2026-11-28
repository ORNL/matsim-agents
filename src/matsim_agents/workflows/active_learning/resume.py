"""Resume of an interrupted active-learning run from its output directory."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from matsim_agents.workflows.active_learning.config import (
    ALConfig,
)

log = logging.getLogger(__name__)


_ROOT_SEED_FILE = "random-seed.json"


def _prior_run_artifacts(root: Path) -> list[str]:
    """Evidence a previous run left in *root*: iterations, datasets, and seed structures."""
    names = [
        "dataset.extxyz",
        "dataset.db",
        "validation.extxyz",
        "seeds",
        "dft_label_cache",
        "dft_attempts",
    ]
    found = [name for name in names if (root / name).exists()]
    return found + sorted(path.name for path in root.glob("iteration_*"))


def _bind_root_seed(cfg: ALConfig, root: Path) -> None:
    """Record the run's root seed; existing evidence must already carry the same one.

    Checked before seed structures are resolved or written. A resume must match the
    recorded seed. Artifacts without a recorded seed (from before seed recording) and,
    for a fresh run, any prior artifacts are rejected, so retained datasets are never
    reused under a seed that did not produce them.
    """
    path = root / _ROOT_SEED_FILE
    artifacts = _prior_run_artifacts(root)
    if path.exists():
        recorded = int(json.loads(path.read_text())["random_seed"])
        if not cfg.loop.resume:
            raise ValueError(
                f"{root} already holds a run (random_seed {recorded}); set loop.resume to "
                "continue it or choose a new out_dir"
            )
        if recorded != cfg.random_seed:
            raise ValueError(
                f"{root} was started with random_seed {recorded}; resuming it with "
                f"{cfg.random_seed} would mix two seed streams in one run"
            )
        return
    if artifacts:
        raise ValueError(
            f"{root} holds artifacts {artifacts} without a recorded random_seed; they "
            "cannot be attributed to a seed; choose a new out_dir"
        )
    path.write_text(json.dumps({"random_seed": cfg.random_seed}) + "\n")


def _scan_resume(root: Path) -> tuple[int, Path | None]:
    """Return (start_iteration, current_logdir_or_None) based on existing state.

    The current_logdir is taken from the most recent completed iteration's
    ``state.json::new_logdir`` if present.
    """
    if not root.exists():
        return 0, None
    completed: list[tuple[int, Path | None]] = []
    for d in sorted(root.glob("iteration_*")):
        if (d / "dft").exists() or any(d.rglob("*-dft-neb")):
            raise ValueError(f"{d} contains DFT work outside the attempt archive; refusing resume")
        sf = d / "state.json"
        if not sf.exists():
            # Partial — nuke it so we don't double-count.
            log.warning("Removing incomplete iteration dir %s", d)
            shutil.rmtree(d, ignore_errors=True)
            continue
        try:
            data = json.loads(sf.read_text())
            if data.get("status") == "complete":
                new_logdir = data.get("new_logdir")
                completed.append((int(data["iteration"]), Path(new_logdir) if new_logdir else None))
            else:
                log.warning("Removing failed/partial iteration dir %s", d)
                shutil.rmtree(d, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not parse %s (%s); removing", sf, exc)
            shutil.rmtree(d, ignore_errors=True)
    if not completed:
        return 0, None
    last_i = max(iteration for iteration, _ in completed)
    promoted = [entry for entry in completed if entry[1] is not None]
    last_logdir = max(promoted, key=lambda entry: entry[0])[1] if promoted else None
    return last_i + 1, last_logdir
