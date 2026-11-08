#!/usr/bin/env python3
"""Dry-run-first qsub helper for the frozen E4 first-batch manifest."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_COUNTS = {"HydraGNN": 8, "GenesisHackathonOct26": 4, "GridAI": 8}
EXPECTED_CAPS = {"HydraGNN": 192, "GenesisHackathonOct26": 96, "GridAI": 192}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"submission validation failed: {message}")


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    raise AssertionError("unreachable")


def validate_case(
    case: dict[str, Any], manifest: dict[str, Any], check_output_collision: bool = True
) -> None:
    calc_id = case.get("calculation_id")
    if not isinstance(calc_id, str) or not re.fullmatch(r"E4-(BCC|FCC|LFP)-[0-9]{4}", calc_id):
        fail(f"invalid calculation ID {calc_id!r}")
    family_by_id = {"BCC": "bcc", "FCC": "fcc", "LFP": "lifepo4"}
    expected_family = family_by_id[calc_id.split("-")[1]]
    allocation_by_family = {
        "bcc": "HydraGNN",
        "fcc": "GenesisHackathonOct26",
        "lifepo4": "GridAI",
    }
    if case.get("material_family") != expected_family:
        fail(f"{calc_id} has wrong material family")
    if case.get("allocation") != allocation_by_family[expected_family]:
        fail(f"{calc_id} has wrong allocation")
    if case.get("nodes") != 1 or case.get("walltime_seconds") != 86400:
        fail(f"{calc_id} exceeds or changes the one-node/24-hour per-job cap")
    if case.get("repeat") != "repeat-01":
        fail(f"{calc_id} has unexpected output repeat")
    if case.get("pbs_script") != str(
        (
            SCRIPT_DIR
            / {
                "bcc": "job-bcc.pbs",
                "fcc": "job-fcc.pbs",
                "lifepo4": "job-lifepo4.pbs",
            }[expected_family]
        ).resolve()
    ):
        fail(f"{calc_id} points to an unexpected PBS script")
    pbs_script = Path(case["pbs_script"])
    if not pbs_script.is_file():
        fail(f"missing PBS script: {pbs_script}")
    script_text = pbs_script.read_text()
    expected_header = (
        f"#PBS -A {case['allocation']}",
        "#PBS -q capacity",
        "#PBS -l select=1",
        "#PBS -l walltime=24:00:00",
    )
    if any(line not in script_text.splitlines() for line in expected_header):
        fail(f"{calc_id} PBS resource/account directives do not match the manifest")
    case_dir = Path(case["case_dir"])
    if not case_dir.is_dir():
        fail(f"missing case directory: {case_dir}")
    for name in ("START", "POSCAR", "KPOINTS", "POTCAR"):
        path = case_dir / name
        expected_hash = case.get("input_hashes", {}).get(name)
        if not path.is_file() or sha256(path) != expected_hash:
            fail(f"{calc_id} input hash mismatch: {path}")
    inputs_sha = case_dir / "inputs.sha256"
    expected_lines = "".join(
        f"{case['input_hashes'][name]}  {name}\n"
        for name in ("START", "POSCAR", "KPOINTS", "POTCAR")
    )
    if not inputs_sha.is_file() or inputs_sha.read_text() != expected_lines:
        fail(f"{calc_id} inputs.sha256 does not match the frozen manifest")
    bundle = "".join(
        f"{name}:{case['input_hashes'][name]}\n"
        for name in ("START", "POSCAR", "KPOINTS", "POTCAR")
    )
    if hashlib.sha256(bundle.encode()).hexdigest() != case.get("prepared_input_bundle_sha256"):
        fail(f"{calc_id} prepared bundle hash mismatch")
    source_poscar = REPO / case["source_poscar"]
    if not source_poscar.is_file() or sha256(source_poscar) != case.get("source_poscar_sha256"):
        fail(f"{calc_id} frozen structure source hash mismatch")
    if case.get("source_poscar_sha256") != case["input_hashes"]["POSCAR"]:
        fail(f"{calc_id} prepared POSCAR differs from its frozen source")
    paw_components = case.get("paw_components")
    if not isinstance(paw_components, list) or not paw_components:
        fail(f"{calc_id} is missing PAW component provenance")
    assembled = bytearray()
    for component in paw_components:
        source = Path(component["source"])
        if not source.is_file() or sha256(source) != component["sha256"]:
            fail(f"{calc_id} PAW component hash mismatch: {source}")
        assembled.extend(source.read_bytes())
    if hashlib.sha256(assembled).hexdigest() != case.get("potcar_sha256"):
        fail(f"{calc_id} PAW component sequence differs from the frozen POTCAR hash")
    if check_output_collision:
        output_dir = case_dir / "alcf" / case["repeat"]
        if output_dir.exists():
            fail(f"{calc_id} output directory already exists; refusing overwrite: {output_dir}")
        for suffix in ("out", "err"):
            log_path = case_dir / f"pbs-{calc_id}.{suffix}"
            if log_path.exists():
                fail(f"{calc_id} PBS log already exists; refusing overwrite: {log_path}")
    if case.get("ediffg_eV_per_A") not in {"-0.01", "-0.02"}:
        fail(f"{calc_id} has invalid force convergence criterion")
    if case.get("stage_nsw_max") not in {200, 240}:
        fail(f"{calc_id} has invalid cumulative ionic-step limit")
    expected_ranks = 12 if expected_family == "lifepo4" else 8
    expected_kpar = 1 if expected_family == "lifepo4" else 8
    expected_ncore = 12 if expected_family == "lifepo4" else None
    if case.get("mpi_ranks") != expected_ranks or case.get("omp_threads_per_rank") != 8:
        fail(f"{calc_id} has an unexpected MPI/OpenMP launch layout")
    if case.get("kpar") != expected_kpar or case.get("ncore") != expected_ncore:
        fail(f"{calc_id} has an unexpected material-specific parallelization")
    for token in (
        f"E4_MPI_RANKS={expected_ranks}",
        "E4_OMP_THREADS=8",
    ):
        if token not in script_text:
            fail(f"{calc_id} PBS wrapper does not declare {token}")


def validate_manifest(
    path: Path, manifest: dict[str, Any], selected_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    if manifest.get("batch_id") != "E4-first-production-batch-01":
        fail("unexpected batch ID")
    if manifest.get("queue") != "capacity":
        fail("unexpected queue")
    executable = Path(manifest.get("executable", ""))
    if not executable.is_file() or sha256(executable) != manifest.get("executable_sha256"):
        fail(f"VASP executable missing or hash changed: {executable}")
    if (
        manifest.get("executable_sha256")
        != "3354871ac0b7a812ebe3444dbd3d5bb4176995a5c37c3859978b58b5c7b55eb6"
    ):
        fail("unexpected VASP executable identity")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 20:
        fail("manifest must contain exactly 20 cases")
    ids = [case.get("calculation_id") for case in cases]
    if len(set(ids)) != 20:
        fail("duplicate calculation IDs in manifest")
    allocation_counts: dict[str, int] = {}
    allocation_caps: dict[str, int] = {}
    for case in cases:
        validate_case(
            case,
            manifest,
            check_output_collision=selected_ids is None or case["calculation_id"] in selected_ids,
        )
        allocation = case["allocation"]
        allocation_counts[allocation] = allocation_counts.get(allocation, 0) + 1
        allocation_caps[allocation] = allocation_caps.get(allocation, 0) + (
            case["nodes"] * case["walltime_seconds"] // 3600
        )
    if allocation_counts != EXPECTED_COUNTS:
        fail(f"allocation-to-material case counts mismatch: {allocation_counts}")
    if allocation_caps != EXPECTED_CAPS or sum(allocation_caps.values()) != 480:
        fail(f"resource caps mismatch: {allocation_caps}")
    if manifest.get("resource_caps_node_hours") != EXPECTED_CAPS:
        fail("manifest allocation caps differ from the approved envelope")
    if manifest.get("total_resource_cap_node_hours") != 480:
        fail("manifest total cap differs from the approved 480 node-hours")
    return cases


def read_ledger(path: Path) -> set[str]:
    if not path.exists():
        return set()
    submitted = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        calc_id = entry.get("calculation_id")
        if calc_id and entry.get("status") != "qsub-failed":
            submitted.add(calc_id)
    return submitted


def active_job_names() -> set[str]:
    user = getpass.getuser()
    result = subprocess.run(
        ["qstat", "-f", "-u", user], text=True, capture_output=True, check=False
    )
    if result.returncode:
        fail(f"qstat failed; cannot rule out active duplicates: {result.stderr.strip()}")
    names = set()
    for match in re.finditer(r"Job_Name\s*=\s*([^\s]+)", result.stdout):
        names.add(match.group(1))
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=SCRIPT_DIR / "batch-manifest.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate only (the default)")
    mode.add_argument("--submit", action="store_true", help="submit after explicit authorization")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="CALCULATION_ID",
        help="submit only a named subset from the frozen 20-case manifest",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    if args.only:
        selected_ids = set(args.only)
        if len(selected_ids) != len(args.only):
            fail("duplicate calculation ID in --only")
        raw_cases = manifest.get("cases")
        known_ids = (
            {case.get("calculation_id") for case in raw_cases}
            if isinstance(raw_cases, list)
            else set()
        )
        unknown_ids = selected_ids.difference(known_ids)
        if unknown_ids:
            fail(
                f"--only contains IDs outside the frozen manifest: {', '.join(sorted(unknown_ids))}"
            )
    else:
        selected_ids = None
    all_cases = validate_manifest(manifest_path, manifest, selected_ids)
    if selected_ids is not None:
        cases = [case for case in all_cases if case["calculation_id"] in selected_ids]
    else:
        cases = all_cases
    ledger = manifest_path.parent / "submissions.jsonl"
    previously_attempted = read_ledger(ledger)
    repeated = previously_attempted.intersection(case["calculation_id"] for case in cases)
    if repeated:
        fail(f"batch already has submission records for: {', '.join(sorted(repeated))}")

    commands = []
    for case in cases:
        commands.append(
            [
                "qsub",
                "-N",
                case["calculation_id"],
                "-v",
                f"E4_CASE_DIR={case['case_dir']},E4_CALCULATION_ID={case['calculation_id']}",
                "-o",
                str(Path(case["case_dir"]) / f"pbs-{case['calculation_id']}.out"),
                "-e",
                str(Path(case["case_dir"]) / f"pbs-{case['calculation_id']}.err"),
                case["pbs_script"],
            ]
        )
    if not args.submit:
        print(f"DRY RUN: validated {len(cases)} cases; no qstat or qsub commands executed.")
        for case, command in zip(cases, commands, strict=True):
            print(f"{case['calculation_id']} [{case['allocation']}] " + " ".join(command))
        print("Maximum requested resource envelope: 480 node-hours.")
        return 0
    if manifest.get("execution_authorized") is not True:
        fail("manifest is not marked execution_authorized; obtain explicit user approval first")
    active_names = active_job_names()
    active_duplicates = sorted(active_names.intersection(case["calculation_id"] for case in cases))
    if active_duplicates:
        fail(f"matching active PBS jobs already exist: {', '.join(active_duplicates)}")

    with ledger.open("a", encoding="utf-8") as stream:
        for case, command in zip(cases, commands, strict=True):
            try:
                result = subprocess.run(command, text=True, capture_output=True, check=False)
            except OSError as error:
                entry = {
                    "calculation_id": case["calculation_id"],
                    "status": "qsub-error",
                    "error": str(error),
                }
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                fail(f"qsub could not run for {case['calculation_id']}: {error}; stopped")
            output = result.stdout.strip()
            if result.returncode:
                entry = {
                    "calculation_id": case["calculation_id"],
                    "status": "qsub-failed",
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                }
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                fail(f"qsub failed for {case['calculation_id']}: {result.stderr.strip()}; stopped")
            match = re.match(r"^([0-9]+(?:\.[A-Za-z0-9_.-]+)?)$", output)
            if not match:
                entry = {
                    "calculation_id": case["calculation_id"],
                    "status": "invalid-qsub-response",
                    "stdout": output,
                    "stderr": result.stderr.strip(),
                }
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                fail(f"unrecognized qsub job ID for {case['calculation_id']}: {output!r}; stopped")
            entry = {
                "calculation_id": case["calculation_id"],
                "pbs_job_id": match.group(1),
                "allocation": case["allocation"],
                "status": "submitted",
                "command": command,
            }
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            print(f"{case['calculation_id']} submitted as {match.group(1)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"submission error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
