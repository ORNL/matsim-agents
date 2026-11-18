#!/usr/bin/env python3
"""Deterministic controlled task executed by an MPI/srun/local step.

The payload burns CPU for a bounded duration, writes an artifact of a fixed
size, and reports a SHA-256 the driver recomputes independently. It is
launched through the same facility step mechanism as VASP or QE, so it
exercises scheduler step creation, node binding, and output collection.

Fault injection is deterministic: ``--fail-attempts N`` fails attempts
``1..N`` of a task, so retry accounting can be tested without randomness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

FAILURE_MODES = ("exit", "nonfinite", "no-result")


def expected_digest(task_id: str, seed: int, artifact_bytes: int) -> str:
    """Return the digest a correct execution of ``task_id`` must report."""

    return hashlib.sha256(_artifact(task_id, seed, artifact_bytes)).hexdigest()


def _artifact(task_id: str, seed: int, size: int) -> bytes:
    block = hashlib.sha256(f"{task_id}:{seed}".encode()).digest()
    return (block * (size // len(block) + 1))[:size]


def _burn(seconds: float) -> int:
    deadline = time.perf_counter() + seconds
    iterations = 0
    value = 1
    while time.perf_counter() < deadline:
        for _ in range(2000):
            value = (value * 1103515245 + 12345) & 0x7FFFFFFF
        iterations += 1
    return iterations


def _rank() -> int:
    for name in ("PMI_RANK", "PMIX_RANK", "SLURM_PROCID", "PALS_RANKID", "OMPI_COMM_WORLD_RANK"):
        if os.environ.get(name, "").isdigit():
            return int(os.environ[name])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--artifact-bytes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--fail-attempts", type=int, default=0)
    parser.add_argument("--failure-mode", choices=FAILURE_MODES, default="exit")
    args = parser.parse_args(argv)

    if args.seconds < 0 or args.artifact_bytes < 0:
        parser.error("--seconds and --artifact-bytes must be nonnegative")
    started = time.time()
    iterations = _burn(args.seconds)
    if _rank() != 0:
        return 0
    args.work_dir.mkdir(parents=True, exist_ok=True)
    failing = args.attempt <= args.fail_attempts
    if failing and args.failure_mode == "exit":
        print(f"injected failure on attempt {args.attempt}", file=sys.stderr)
        return 3
    if failing and args.failure_mode == "no-result":
        return 0
    payload = _artifact(args.task_id, args.seed, args.artifact_bytes)
    (args.work_dir / "artifact.bin").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    result = {
        "task_id": args.task_id,
        "attempt": args.attempt,
        "sha256": digest,
        "metric": float("nan") if failing else float(iterations),
        "seconds": time.time() - started,
        "host": socket.gethostname(),
        "ranks": int(os.environ.get("PMI_SIZE") or os.environ.get("SLURM_NTASKS") or 1),
    }
    temporary = args.work_dir / "result.json.tmp"
    temporary.write_text(json.dumps(result, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(args.work_dir / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
