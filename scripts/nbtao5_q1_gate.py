from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from pymatgen.io.vasp import Poscar

PROTOCOL_ID = "nbtao5-fixed-kmesh-v1"
Q1_CASE_ID = "NbTaO5-ordering-01-baseline"
Q1_JOB = "deployments/aurora/jobs/job-nbtao5-qualification-aurora.pbs"
# 28-atom relaxations take ~80 s/ionic step on one node; 30 min was too short (job 9039526).
Q1_WALLTIME = "02:00:00"
Q1_QUEUE = "capacity"


class GateError(Exception):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read JSON {path}: {error}") from error


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise GateError(f"required file is missing or empty: {path}")


def _incar_tags(path: Path) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        statement = line.split("!", 1)[0].split("#", 1)[0].strip()
        if not statement or "=" not in statement:
            continue
        key, value = statement.split("=", 1)
        tags.setdefault(key.strip().upper(), []).append(value.strip())
    return tags


def validate_case(case_dir: Path, repo: Path, paw_root: Path) -> dict:
    case_dir = case_dir.resolve(strict=True)
    repo = repo.resolve(strict=True)
    paw_root = paw_root.resolve(strict=True)
    if case_dir == repo or repo in case_dir.parents:
        raise GateError("Q1 case must be outside the Git checkout")

    manifest_path = repo / "research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json"
    runtime_path = repo / "research/nb-ta-o/pilot/vasp_runtime_manifest.json"
    manifest = read_json(manifest_path)
    runtime = read_json(runtime_path)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise GateError("active manifest does not use nbtao5-fixed-kmesh-v1")
    paw_library = runtime.get("paw_library", {})
    if not paw_library.get("release_identifier"):
        raise GateError("authoritative PAW release identifier is unverified")
    if not paw_library.get("homogeneous_release_verified"):
        raise GateError("homogeneous PAW release membership is unverified")
    if not paw_library.get("license_authorization_verified"):
        raise GateError("PAW license authorization is unverified")

    case = read_json(case_dir / "case.json")
    current = next(
        (
            item
            for item in manifest.get("cases", [])
            if item.get("case_id") == case.get("case_id")
        ),
        None,
    )
    if current is None or case != current:
        raise GateError("case metadata does not match the active manifest")
    if case.get("protocol_id") != PROTOCOL_ID:
        raise GateError("Q1 case uses a stale protocol identifier")
    if case.get("category") == "crystalline_numerical_qualification":
        q1_stages = [
            stage
            for stage in manifest.get("qualification_stages", [])
            if case["case_id"] in stage.get("workflow_ids", [])
        ]
        if len(q1_stages) != 1:
            raise GateError("qualification case is not assigned to exactly one stage")
        if case["case_id"] == Q1_CASE_ID and (
            q1_stages[0].get("stage_id") != "Q1-runtime-and-workflow-smoke"
        ):
            raise GateError("active manifest does not assign this case to Q1")

    for name, expected in case.get("input_hashes", {}).items():
        path = case_dir / name
        require_file(path)
        if sha256(path) != expected:
            raise GateError(f"scientific input hash mismatch: {name}")

    poscar = Poscar.from_file(case_dir / "POSCAR")
    if poscar.site_symbols != case.get("species_order"):
        raise GateError("POSCAR species order differs from case metadata")
    counts = dict(zip(poscar.site_symbols, poscar.natoms, strict=True))
    if counts != case.get("species_counts"):
        raise GateError("POSCAR species counts differ from case metadata")

    settings = case["settings"]
    if settings.get("encut_eV") != 600:
        raise GateError("Q1 ENCUT differs from the accepted 600 eV protocol")
    for incar_name in ("INCAR", "INCAR.static"):
        tags = _incar_tags(case_dir / incar_name)
        if "KSPACING" in tags or "KGAMMA" in tags:
            raise GateError(f"{incar_name} must not define KSPACING or KGAMMA")
        values = tags.get("ENCUT", [])
        if len(values) != 1 or float(values[0].split()[0]) != 600.0:
            raise GateError(f"{incar_name} must contain exactly ENCUT = 600")

    kpoints_lines = [
        line.strip()
        for line in (case_dir / "KPOINTS").read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if len(kpoints_lines) < 5 or kpoints_lines[1] != "0" or kpoints_lines[2].lower() != "gamma":
        raise GateError("KPOINTS must use an explicit Gamma-centered mesh")
    try:
        mesh = [int(value) for value in kpoints_lines[3].split()]
        shift = [float(value) for value in kpoints_lines[4].split()]
    except ValueError as error:
        raise GateError("KPOINTS mesh or shift is malformed") from error
    if mesh != case["kpoints"].get("mesh_dimensions"):
        raise GateError("KPOINTS mesh differs from case metadata")
    if case.get("case_id") == Q1_CASE_ID and mesh != [4, 8, 9]:
        raise GateError("Q1 KPOINTS mesh differs from the approved 4x8x9 mesh")
    if shift != [0.0, 0.0, 0.0]:
        raise GateError("Q1 Gamma-centered KPOINTS must have zero shift")
    kpoint_metadata = read_json(case_dir / "KPOINTS.json")
    if kpoint_metadata != case.get("kpoints"):
        raise GateError("KPOINTS metadata differs from case metadata")
    if sha256(case_dir / "KPOINTS") != kpoint_metadata.get("kpoints_sha256"):
        raise GateError("KPOINTS file hash differs from KPOINTS metadata")

    component_manifest = runtime.get("proposed_components", {})
    component_by_species = {"Nb": "Nb_pv", "Ta": "Ta_pv", "O": "O"}
    expected_order = [component_by_species[symbol] for symbol in poscar.site_symbols]
    if expected_order != case.get("potcar_required_components"):
        raise GateError("required PAW components do not follow POSCAR species order")
    order_lines = [
        line.split()
        for line in (case_dir / "POTCAR.order").read_text(encoding="ascii").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if any(len(line) != 2 for line in order_lines):
        raise GateError("POTCAR.order must contain component name and SHA-256 per line")
    if [line[0] for line in order_lines] != expected_order:
        raise GateError("POTCAR component order differs from POSCAR species order")
    for component, recorded_hash in order_lines:
        expected = component_manifest.get(component, {}).get("sha256")
        if expected is None or recorded_hash != expected:
            raise GateError(f"POTCAR.order has an unapproved identity/hash for {component}")

    potcar = case_dir / "POTCAR"
    checksum_file = case_dir / "POTCAR.sha256"
    require_file(potcar)
    require_file(checksum_file)
    checksum_parts = checksum_file.read_text(encoding="ascii").split()
    actual_potcar_hash = sha256(potcar)
    if checksum_parts != [actual_potcar_hash, "POTCAR"]:
        raise GateError("POTCAR.sha256 is missing, malformed, or does not match POTCAR")

    combined = hashlib.sha256()
    for component, expected_hash in order_lines:
        source = paw_root / component / "POTCAR"
        require_file(source)
        if sha256(source) != expected_hash:
            raise GateError(f"approved source PAW hash mismatch for {component}")
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                combined.update(chunk)
    if combined.hexdigest() != actual_potcar_hash:
        raise GateError("assembled POTCAR is not the ordered concatenation of approved components")

    vasp_path = repo / runtime["executable"]["path"]
    require_file(vasp_path)
    if not os.access(vasp_path, os.X_OK):
        raise GateError("VASP executable is not executable")
    if sha256(vasp_path) != runtime["executable"]["sha256"]:
        raise GateError("VASP executable fingerprint differs from the runtime manifest")
    return {
        "case_id": case["case_id"],
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest_sha256": sha256(manifest_path),
        "runtime_manifest_sha256": sha256(runtime_path),
        "vasp_executable_sha256": runtime["executable"]["sha256"],
        "case_dir": str(case_dir),
    }


def validate_preflight(record_path: Path, repo: Path, case_dir: Path) -> dict:
    repo = repo.resolve(strict=True)
    case_dir = case_dir.resolve(strict=True)
    record_path = record_path.resolve(strict=True)
    runtime_path = repo / "research/nb-ta-o/pilot/vasp_runtime_manifest.json"
    manifest_path = repo / "research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json"
    runtime = read_json(runtime_path)
    record = read_json(record_path)
    logs_dir = case_dir.parent.parent / "logs"
    if record_path.parent != logs_dir:
        raise GateError("preflight record must be under this attempt's logs directory")
    runner = repo / "deployments/aurora/launchers/run-nbtao5-case-aurora.sh"
    vasp = repo / runtime["executable"]["path"]

    required = {
        "status": "PASS",
        "exit_code": 0,
        "repo_real": str(repo),
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest_sha256": sha256(manifest_path),
        "runtime_manifest_sha256": sha256(runtime_path),
        "vasp_executable_sha256": runtime["executable"]["sha256"],
        "runner_path": str(runner.resolve(strict=True)),
        "runner_sha256": sha256(runner),
        "vasp_path": str(vasp.resolve(strict=True)),
        "dependencies_resolved": True,
        "gpu_visible": True,
    }
    for key, value in required.items():
        if record.get(key) != value:
            raise GateError(f"preflight record is missing or incompatible: {key}")
    # The preflight checks the runtime, so any case of this attempt may reuse it.
    if Path(record.get("case_dir", "")).parent != case_dir.parent:
        raise GateError("preflight record belongs to a different attempt")
    if record.get("failure") is not None or not record.get("gpu_devices"):
        raise GateError("preflight record lacks successful failure/GPU evidence")
    if not record.get("pbs_job_id") or record["pbs_job_id"] == "unknown":
        raise GateError("preflight record lacks a PBS job ID")

    expected_modules = runtime["platform"]["module_stack"]
    loaded_modules = record.get("modules_loaded", "")
    if record.get("expected_module_stack") != expected_modules:
        raise GateError("preflight record has an incompatible expected module stack")
    if not all(module in loaded_modules for module in expected_modules):
        raise GateError("preflight module stack differs from the required Aurora stack")
    mpi = record.get("mpi", {})
    if "/pals/" not in mpi.get("mpiexec_path", "").lower() or not mpi.get("version"):
        raise GateError("preflight did not verify the Aurora PALS mpiexec")
    required_options = ("--hosts", "--ppn", "--cpu-bind", "--gpu-bind")
    if not all(mpi.get("required_options", {}).get(option) for option in required_options):
        raise GateError("preflight MPI launcher lacks required options")

    stdout_path = Path(record.get("stdout_path", ""))
    if not stdout_path.is_absolute() or stdout_path.parent != logs_dir:
        raise GateError("preflight stdout must be under this attempt's logs directory")
    require_file(stdout_path)
    if "NBTAO5_PREFLIGHT_RESULT=PASS" not in stdout_path.read_text(errors="replace"):
        raise GateError("preflight stdout does not contain its PASS marker")
    qstat = shutil.which("qstat")
    if not qstat:
        raise GateError("qstat is unavailable to verify preflight scheduler accounting")
    result = subprocess.run(
        [qstat, "-xf", record["pbs_job_id"]], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise GateError("PBS accounting unavailable for the recorded preflight job")
    accounting = result.stdout
    job_state = re.search(r"^\s*job_state\s*=\s*(\w+)", accounting, re.MULTILINE)
    exit_status = re.search(r"^\s*Exit_status\s*=\s*(-?\d+)", accounting, re.MULTILINE)
    account = re.search(r"^\s*Account_Name\s*=\s*(\S+)", accounting, re.MULTILINE)
    if not job_state or job_state.group(1) != "F" or not exit_status or exit_status.group(1) != "0":
        raise GateError("PBS accounting does not show a completed preflight with exit status 0")
    collapsed_accounting = re.sub(r"\s+", "", accounting)
    if str(stdout_path) not in collapsed_accounting:
        raise GateError("PBS accounting output path differs from the recorded preflight stdout")
    if not account or account.group(1) != "HydraGNN":
        raise GateError("PBS accounting does not show the approved HydraGNN account")
    return record


def _require_yes(name: str) -> str:
    value = os.environ.get(name)
    if value != "YES":
        raise GateError(f"approval required: export {name}=YES only after approval is granted")
    return value


def submit_q1(dry_run: bool) -> int:
    for name in (
        "NBTAO5_QUALIFICATION_AUTHORIZED",
        "NBTAO5_RESOURCE_APPROVED",
        "NBTAO5_PAW_RELEASE_APPROVED",
    ):
        _require_yes(name)
    for name in (
        "NBTAO5_REPO_ROOT",
        "NBTAO5_CASE_DIR",
        "NBTAO5_LOCK_ROOT",
        "NBTAO5_PAW_ROOT",
        "NBTAO5_PREFLIGHT_RECORD",
        "NBTAO5_Q1_STDOUT",
    ):
        if not os.environ.get(name):
            raise GateError(f"required path is not set: {name}")

    repo_arg = Path(os.environ["NBTAO5_REPO_ROOT"])
    if not repo_arg.is_absolute():
        raise GateError("NBTAO5_REPO_ROOT must be an absolute path")
    repo = repo_arg.resolve(strict=True)
    if not (repo / "pyproject.toml").is_file():
        raise GateError("NBTAO5_REPO_ROOT is not an accessible checkout")
    case_dir = Path(os.environ["NBTAO5_CASE_DIR"]).resolve(strict=True)
    attempt_root = case_dir.parent.parent
    expected_case = attempt_root / "cases" / Q1_CASE_ID
    if case_dir != expected_case:
        raise GateError(
            "NBTAO5_CASE_DIR must be the Q1 case under the new attempt's cases directory"
        )
    if read_json(case_dir / "case.json").get("case_id") != Q1_CASE_ID:
        raise GateError("submission is restricted to the designated Q1 case")
    lock_root = Path(os.environ["NBTAO5_LOCK_ROOT"]).resolve(strict=True)
    if lock_root != attempt_root / "locks":
        raise GateError("NBTAO5_LOCK_ROOT must be this attempt's external locks directory")
    if repo == case_dir or repo in case_dir.parents:
        raise GateError("Q1 case must be outside the checkout")

    stdout_path = Path(os.environ["NBTAO5_Q1_STDOUT"])
    preflight_path = Path(os.environ["NBTAO5_PREFLIGHT_RECORD"])
    if not stdout_path.is_absolute() or not preflight_path.is_absolute():
        raise GateError("PBS stdout and preflight record paths must be absolute")
    logs_dir = attempt_root / "logs"
    if stdout_path.parent != logs_dir or preflight_path.parent != logs_dir:
        raise GateError(
            "Q1 stdout and preflight record must be under this attempt's logs directory"
        )
    if not logs_dir.is_dir() or stdout_path.exists():
        raise GateError("PBS output directory must exist and Q1 stdout path must be unused")
    if not preflight_path.is_file():
        raise GateError("successful compute-node preflight record is missing")

    paw_root = Path(os.environ["NBTAO5_PAW_ROOT"]).resolve(strict=True)
    if not paw_root.is_dir():
        raise GateError("NBTAO5_PAW_ROOT is not a directory")
    validate_case(case_dir, repo, paw_root)
    validate_preflight(preflight_path, repo, case_dir)

    pbs_script = repo / Q1_JOB
    pbs_text = pbs_script.read_text(encoding="ascii")
    directives = (
        "#PBS -l select=1",
        f"#PBS -l walltime={Q1_WALLTIME}",
        f"#PBS -q {Q1_QUEUE}",
        "#PBS -j oe",
    )
    for directive in directives:
        if directive not in pbs_text:
            raise GateError(f"Q1 PBS wrapper lacks required resource directive: {directive}")
    variables = [
        "NBTAO5_REPO_ROOT",
        "NBTAO5_CASE_DIR",
        "NBTAO5_LOCK_ROOT",
        "NBTAO5_PAW_ROOT",
        "NBTAO5_PREFLIGHT_RECORD",
        "NBTAO5_QUALIFICATION_AUTHORIZED",
        "NBTAO5_RESOURCE_APPROVED",
        "NBTAO5_PAW_RELEASE_APPROVED",
    ]
    exported = ",".join(f"{name}={os.environ[name]}" for name in variables)
    command = [
        "qsub", "-A", "HydraGNN", "-l", "select=1", "-l", f"walltime={Q1_WALLTIME}",
        "-q", Q1_QUEUE, "-o", str(stdout_path), "-v", exported, str(pbs_script),
    ]
    if dry_run:
        print("DRY RUN: all Q1 prerequisites validated; qsub was not invoked")
        print(shlex.join(command))
        return 0
    qsub = shutil.which("qsub")
    if not qsub:
        raise GateError("qsub is unavailable")
    command[0] = qsub
    result = subprocess.run(command, check=False, text=True)
    return result.returncode


def submit_preflight(dry_run: bool) -> int:
    _require_yes("NBTAO5_RESOURCE_APPROVED")
    for name in (
        "NBTAO5_REPO_ROOT",
        "NBTAO5_CASE_DIR",
        "NBTAO5_PREFLIGHT_RECORD",
        "NBTAO5_PREFLIGHT_STDOUT",
    ):
        if not os.environ.get(name):
            raise GateError(f"required path is not set: {name}")
    repo_arg = Path(os.environ["NBTAO5_REPO_ROOT"])
    if not repo_arg.is_absolute():
        raise GateError("NBTAO5_REPO_ROOT must be an absolute path")
    repo = repo_arg.resolve(strict=True)
    if not (repo / "pyproject.toml").is_file():
        raise GateError("NBTAO5_REPO_ROOT is not an accessible checkout")
    case_dir = Path(os.environ["NBTAO5_CASE_DIR"]).resolve(strict=True)
    attempt_root = case_dir.parent.parent
    if case_dir != attempt_root / "cases" / Q1_CASE_ID:
        raise GateError("preflight must target the fresh Q1 case")
    if repo == case_dir or repo in case_dir.parents:
        raise GateError("preflight case must be outside the checkout")
    logs_dir = attempt_root / "logs"
    record_path = Path(os.environ["NBTAO5_PREFLIGHT_RECORD"])
    stdout_path = Path(os.environ["NBTAO5_PREFLIGHT_STDOUT"])
    if not record_path.is_absolute() or not stdout_path.is_absolute():
        raise GateError("preflight record and PBS stdout paths must be absolute")
    if record_path.parent != logs_dir or stdout_path.parent != logs_dir:
        raise GateError("preflight outputs must be under this attempt's logs directory")
    if not logs_dir.is_dir() or record_path.exists() or stdout_path.exists():
        raise GateError("preflight logs directory must exist and output paths must be unused")
    preflight_script = repo / "deployments/aurora/jobs/job-nbtao5-preflight-aurora.pbs"
    text = preflight_script.read_text(encoding="ascii")
    for directive in ("#PBS -l select=1", "#PBS -l walltime=00:05:00", "#PBS -A HydraGNN"):
        if directive not in text:
            raise GateError(f"preflight PBS wrapper lacks required directive: {directive}")
    variables = (
        "NBTAO5_REPO_ROOT",
        "NBTAO5_CASE_DIR",
        "NBTAO5_PREFLIGHT_RECORD",
        "NBTAO5_PREFLIGHT_STDOUT",
        "NBTAO5_RESOURCE_APPROVED",
    )
    exported = ",".join(f"{name}={os.environ[name]}" for name in variables)
    command = [
        "qsub", "-A", "HydraGNN", "-l", "select=1", "-l", "walltime=00:05:00",
        "-q", "debug", "-o", str(stdout_path), "-v", exported, str(preflight_script),
    ]
    if dry_run:
        print("DRY RUN: preflight resources authorized; qsub was not invoked")
        print(shlex.join(command))
        return 0
    qsub = shutil.which("qsub")
    if not qsub:
        raise GateError("qsub is unavailable")
    command[0] = qsub
    result = subprocess.run(command, check=False, text=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    case_parser = subparsers.add_parser("validate-case")
    case_parser.add_argument("--repo", type=Path, required=True)
    case_parser.add_argument("--case", type=Path, required=True)
    case_parser.add_argument("--paw-root", type=Path, required=True)
    preflight_parser = subparsers.add_parser("validate-preflight")
    preflight_parser.add_argument("--repo", type=Path, required=True)
    preflight_parser.add_argument("--case", type=Path, required=True)
    preflight_parser.add_argument("--record", type=Path, required=True)
    subparsers.add_parser("submit-q1").add_argument("--dry-run", action="store_true")
    subparsers.add_parser("submit-preflight").add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "validate-case":
            validate_case(args.case, args.repo, args.paw_root)
            print("NBTAO5_CASE_VALIDATION=PASS")
            return 0
        if args.action == "validate-preflight":
            validate_preflight(args.record, args.repo, args.case)
            print("NBTAO5_PREFLIGHT_VALIDATION=PASS")
            return 0
        if args.action == "submit-preflight":
            return submit_preflight(args.dry_run)
        return submit_q1(args.dry_run)
    except (GateError, OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
        print(f"NBTAO5_Q1_GATE=BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())