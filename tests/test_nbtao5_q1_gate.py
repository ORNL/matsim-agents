from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nbtao5_q1_gate as gate  # noqa: E402

Q1_CASE_ID = "NbTaO5-ordering-01-baseline"
SOURCE_CASE = ROOT / "research/nb-ta-o/pilot/vasp_inputs/qualification/NbTaO5-ordering-01/baseline"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def make_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "checkout"
    case_dir = tmp_path / "attempt/cases" / Q1_CASE_ID
    attempt = case_dir.parent.parent
    paw_root = repo / "external/synthetic-paw"
    logs = attempt / "logs"
    locks = attempt / "locks"
    fakebin = tmp_path / "bin"
    for path in (case_dir, logs, locks, fakebin, paw_root):
        path.mkdir(parents=True, exist_ok=True)
    (repo / "research/nb-ta-o/pilot/vasp_inputs").mkdir(parents=True)
    (repo / "research/nb-ta-o/pilot").mkdir(parents=True, exist_ok=True)
    (repo / "deployments/aurora/jobs").mkdir(parents=True)
    (repo / "deployments/aurora/launchers").mkdir(parents=True)
    (repo / "external/vasp6/src/vasp.6.6.1/bin").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("synthetic fixture\n", encoding="ascii")
    pbs = repo / "deployments/aurora/jobs/job-nbtao5-qualification-aurora.pbs"
    pbs.write_text(
        "#PBS -l select=1\n#PBS -l walltime=02:00:00\n#PBS -q capacity\n#PBS -j oe\n",
        encoding="ascii",
    )
    preflight_pbs = repo / "deployments/aurora/jobs/job-nbtao5-preflight-aurora.pbs"
    preflight_pbs.write_text(
        "#PBS -l select=1\n#PBS -l walltime=00:05:00\n#PBS -A HydraGNN\n",
        encoding="ascii",
    )
    runner = repo / "deployments/aurora/launchers/run-nbtao5-case-aurora.sh"
    runner.write_text("#!/bin/bash\nexit 0\n", encoding="ascii")
    runner.chmod(0o755)

    source_manifest = json.loads(
        (ROOT / "research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json").read_text()
    )
    manifest = copy.deepcopy(source_manifest)
    case = next(item for item in manifest["cases"] if item["case_id"] == Q1_CASE_ID)
    case = copy.deepcopy(case)
    for name in (
        "POSCAR",
        "INCAR",
        "INCAR.static",
        "KPOINTS",
        "KPOINTS.json",
        "POTCAR.order",
    ):
        (case_dir / name).write_bytes((SOURCE_CASE / name).read_bytes())
    case["case_id"] = Q1_CASE_ID

    source_runtime = json.loads(
        (ROOT / "research/nb-ta-o/pilot/vasp_runtime_manifest.json").read_text()
    )
    runtime = copy.deepcopy(source_runtime)
    runtime["paw_library"].update(
        {
            "release_identifier": "SYNTHETIC-TEST-ONLY",
            "homogeneous_release_verified": True,
            "license_authorization_verified": True,
        }
    )
    component_lines = []
    payloads = {
        "Nb_pv": b"SYNTHETIC TEST PAW Nb_pv\n",
        "Ta_pv": b"SYNTHETIC TEST PAW Ta_pv\n",
        "O": b"SYNTHETIC TEST PAW O\n",
    }
    for component, payload in payloads.items():
        source = paw_root / component / "POTCAR"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
        component_hash = _sha(source)
        runtime["proposed_components"][component]["path"] = str(source.relative_to(repo))
        runtime["proposed_components"][component]["sha256"] = component_hash
        component_lines.append(f"{component} {component_hash}")
    (case_dir / "POTCAR.order").write_text("\n".join(component_lines) + "\n", encoding="ascii")
    case["input_hashes"]["POTCAR.order"] = _sha(case_dir / "POTCAR.order")
    _write_json(case_dir / "case.json", case)

    vasp = repo / runtime["executable"]["path"]
    vasp.write_bytes(b"SYNTHETIC TEST VASP EXECUTABLE\n")
    vasp.chmod(0o755)
    runtime["executable"]["sha256"] = _sha(vasp)
    _write_json(repo / "research/nb-ta-o/pilot/vasp_runtime_manifest.json", runtime)
    manifest["cases"] = [case]
    _write_json(repo / "research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json", manifest)

    potcar_payload = b"".join(payloads.values())
    (case_dir / "POTCAR").write_bytes(potcar_payload)
    (case_dir / "POTCAR.sha256").write_text(
        f"{hashlib.sha256(potcar_payload).hexdigest()}  POTCAR\n", encoding="ascii"
    )

    preflight_stdout = logs / "preflight.pbs.out"
    preflight_stdout.write_text("NBTAO5_PREFLIGHT_RESULT=PASS\n", encoding="ascii")
    preflight_record = logs / "preflight-record.json"
    record = {
        "status": "PASS",
        "exit_code": 0,
        "repo_real": str(repo.resolve()),
        "case_dir": str(case_dir.resolve()),
        "protocol_id": "nbtao5-fixed-kmesh-v1",
        "protocol_manifest_sha256": _sha(
            repo / "research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json"
        ),
        "runtime_manifest_sha256": _sha(repo / "research/nb-ta-o/pilot/vasp_runtime_manifest.json"),
        "vasp_executable_sha256": runtime["executable"]["sha256"],
        "runner_sha256": _sha(runner),
        "runner_path": str(runner.resolve()),
        "vasp_path": str(vasp.resolve()),
        "dependencies_resolved": True,
        "gpu_visible": True,
        "gpu_devices": "[level_zero:gpu:0] synthetic fixture only",
        "failure": None,
        "expected_module_stack": runtime["platform"]["module_stack"],
        "modules_loaded": "oneapi/release/2026.1.0 intel_gpu_umd_aicoe/2026.06.19 mpich/prd/5.0.0",
        "mpi": {
            "mpiexec_path": "/opt/cray/pals/1.8/bin/mpiexec",
            "version": "mpiexec version 1.8.0",
            "required_options": {
                "--hosts": True,
                "--ppn": True,
                "--cpu-bind": True,
                "--gpu-bind": True,
            },
        },
        "pbs_job_id": "12345.aurora",
        "stdout_path": str(preflight_stdout),
    }
    _write_json(preflight_record, record)

    qstat = fakebin / "qstat"
    qstat.write_text(
        "#!/bin/bash\ncat <<'EOF'\n"
        "Job Id: 12345.aurora\n    job_state = F\n    Exit_status = 0\n"
        f"    Account_Name = HydraGNN\n    Output_Path = compute-node:{preflight_stdout}\nEOF\n",
        encoding="ascii",
    )
    qstat.chmod(0o755)
    qsub_marker = tmp_path / "qsub-invoked"
    qsub = fakebin / "qsub"
    qsub.write_text(f"#!/bin/bash\ntouch '{qsub_marker}'\necho 12346.aurora\n", encoding="ascii")
    qsub.chmod(0o755)

    env_values = {
        "NBTAO5_REPO_ROOT": str(repo),
        "NBTAO5_CASE_DIR": str(case_dir),
        "NBTAO5_LOCK_ROOT": str(locks),
        "NBTAO5_PAW_ROOT": str(paw_root),
        "NBTAO5_PREFLIGHT_RECORD": str(preflight_record),
        "NBTAO5_PREFLIGHT_STDOUT": str(preflight_stdout),
        "NBTAO5_Q1_STDOUT": str(logs / "q1-pbs.out"),
        "NBTAO5_QUALIFICATION_AUTHORIZED": "YES",
        "NBTAO5_RESOURCE_APPROVED": "YES",
        "NBTAO5_PAW_RELEASE_APPROVED": "YES",
        "PATH": f"{fakebin}{os.pathsep}{os.environ['PATH']}",
    }
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)
    return {
        "repo": repo,
        "case": case_dir,
        "paw_root": paw_root,
        "record": preflight_record,
        "stdout": preflight_stdout,
        "qstat": qstat,
        "qsub_marker": qsub_marker,
        "qsub": qsub,
    }


def test_authorized_valid_fixture_dry_run_does_not_call_qsub(tmp_path, monkeypatch, capsys):
    fixture = make_fixture(tmp_path, monkeypatch)
    assert gate.submit_q1(dry_run=True) == 0
    output = capsys.readouterr().out
    assert "all Q1 prerequisites validated" in output
    assert "-A HydraGNN" in output and "walltime=02:00:00" in output
    assert not fixture["qsub_marker"].exists()


def test_resource_authorized_preflight_dry_run_does_not_call_qsub(tmp_path, monkeypatch, capsys):
    fixture = make_fixture(tmp_path, monkeypatch)
    fixture["record"].unlink()
    fixture["stdout"].unlink()
    assert gate.submit_preflight(dry_run=True) == 0
    output = capsys.readouterr().out
    assert "preflight resources authorized" in output
    assert "walltime=00:05:00" in output
    assert not fixture["qsub_marker"].exists()


def test_preflight_submission_requires_resource_approval(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    fixture["record"].unlink()
    fixture["stdout"].unlink()
    monkeypatch.delenv("NBTAO5_RESOURCE_APPROVED")
    with pytest.raises(gate.GateError, match="approval required"):
        gate.submit_preflight(dry_run=True)
    assert not fixture["qsub_marker"].exists()


@pytest.mark.parametrize(
    "authorization",
    [
        "NBTAO5_QUALIFICATION_AUTHORIZED",
        "NBTAO5_RESOURCE_APPROVED",
        "NBTAO5_PAW_RELEASE_APPROVED",
    ],
)
def test_missing_approval_blocks_before_qsub(tmp_path, monkeypatch, authorization):
    fixture = make_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv(authorization)
    with pytest.raises(gate.GateError, match="approval required"):
        gate.submit_q1(dry_run=True)
    assert not fixture["qsub_marker"].exists()


def test_changed_scientific_input_hash_is_rejected(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    with (fixture["case"] / "INCAR").open("a") as stream:
        stream.write("ENCUT = 520\n")
    with pytest.raises(gate.GateError, match="scientific input hash mismatch"):
        gate.validate_case(fixture["case"], fixture["repo"], fixture["paw_root"])


@pytest.mark.parametrize(
    ("filename", "old", "new", "expected_error"),
    [
        ("INCAR", "ENCUT = 600", "ENCUT = 520", "ENCUT = 600"),
        ("KPOINTS", "4 8 9", "4 8 8", "KPOINTS mesh differs"),
    ],
)
def test_scientific_values_rejected_even_if_hashes_are_rewritten(
    tmp_path, monkeypatch, filename, old, new, expected_error
):
    fixture = make_fixture(tmp_path, monkeypatch)
    case_file = fixture["case"] / "case.json"
    case = json.loads(case_file.read_text())
    target = fixture["case"] / filename
    updated = target.read_text(encoding="ascii").replace(old, new)
    target.write_text(updated, encoding="ascii")
    digest = _sha(target)
    case["input_hashes"][filename] = digest
    _write_json(case_file, case)
    manifest_path = fixture["repo"] / "research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"] = [case]
    _write_json(manifest_path, manifest)
    with pytest.raises(gate.GateError, match=expected_error):
        gate.validate_case(fixture["case"], fixture["repo"], fixture["paw_root"])


@pytest.mark.parametrize("checksum_state", ["missing", "invalid"])
def test_missing_or_invalid_potcar_checksum_is_rejected(tmp_path, monkeypatch, checksum_state):
    fixture = make_fixture(tmp_path, monkeypatch)
    checksum = fixture["case"] / "POTCAR.sha256"
    if checksum_state == "missing":
        checksum.unlink()
    else:
        checksum.write_text("0" * 64 + "  POTCAR\n", encoding="ascii")
    with pytest.raises(gate.GateError, match="POTCAR.sha256"):
        gate.validate_case(fixture["case"], fixture["repo"], fixture["paw_root"])


def test_unverified_paw_release_or_license_blocks_case(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    runtime_path = fixture["repo"] / "research/nb-ta-o/pilot/vasp_runtime_manifest.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["paw_library"]["license_authorization_verified"] = False
    _write_json(runtime_path, runtime)
    with pytest.raises(gate.GateError, match="PAW license authorization is unverified"):
        gate.validate_case(fixture["case"], fixture["repo"], fixture["paw_root"])


def test_paw_component_order_mismatch_is_rejected(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    case_file = fixture["case"] / "case.json"
    case = json.loads(case_file.read_text())
    order_file = fixture["case"] / "POTCAR.order"
    lines = order_file.read_text(encoding="ascii").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    order_file.write_text("\n".join(lines) + "\n", encoding="ascii")
    case["input_hashes"]["POTCAR.order"] = _sha(order_file)
    _write_json(case_file, case)
    manifest_path = fixture["repo"] / "research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"] = [case]
    _write_json(manifest_path, manifest)
    with pytest.raises(gate.GateError, match="component order"):
        gate.validate_case(fixture["case"], fixture["repo"], fixture["paw_root"])


def test_tampered_paw_component_bytes_are_rejected(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    component = fixture["paw_root"] / "Nb_pv/POTCAR"
    component.write_bytes(component.read_bytes() + b"tampered\n")
    with pytest.raises(gate.GateError, match="source PAW hash mismatch for Nb_pv"):
        gate.validate_case(fixture["case"], fixture["repo"], fixture["paw_root"])


def test_wrong_protocol_identifier_is_rejected(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    case_file = fixture["case"] / "case.json"
    case = json.loads(case_file.read_text())
    case["protocol_id"] = "stale-protocol"
    _write_json(case_file, case)
    with pytest.raises(gate.GateError, match="metadata does not match"):
        gate.validate_case(fixture["case"], fixture["repo"], fixture["paw_root"])


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("status", "FAIL"),
        ("gpu_visible", False),
        ("dependencies_resolved", False),
        ("runner_sha256", "0" * 64),
        ("runtime_manifest_sha256", "0" * 64),
    ],
)
def test_failed_or_incompatible_preflight_is_rejected(tmp_path, monkeypatch, field, bad_value):
    fixture = make_fixture(tmp_path, monkeypatch)
    record = json.loads(fixture["record"].read_text())
    record[field] = bad_value
    _write_json(fixture["record"], record)
    with pytest.raises(gate.GateError):
        gate.submit_q1(dry_run=True)


def test_missing_preflight_record_is_rejected(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    fixture["record"].unlink()
    with pytest.raises(gate.GateError, match="preflight record is missing"):
        gate.submit_q1(dry_run=True)


def test_relative_checkout_path_is_rejected(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("NBTAO5_REPO_ROOT", "relative/checkout")
    with pytest.raises(gate.GateError, match="must be an absolute path"):
        gate.submit_q1(dry_run=True)
    assert not fixture["qsub_marker"].exists()


def test_preflight_must_have_successful_pbs_accounting(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    fixture["qstat"].write_text(
        "#!/bin/bash\necho 'job_state = F'\necho 'Exit_status = 1'\n"
        "echo 'Account_Name = HydraGNN'\n",
        encoding="ascii",
    )
    fixture["qstat"].chmod(0o755)
    with pytest.raises(gate.GateError, match="completed preflight"):
        gate.submit_q1(dry_run=True)


def test_preflight_accounting_must_bind_the_recorded_stdout(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    fixture["qstat"].write_text(
        "#!/bin/bash\ncat <<'EOF'\n"
        "Job Id: 12345.aurora\n    job_state = F\n    Exit_status = 0\n"
        "    Account_Name = HydraGNN\n    Output_Path = compute-node:/tmp/unrelated.out\nEOF\n",
        encoding="ascii",
    )
    fixture["qstat"].chmod(0o755)
    with pytest.raises(gate.GateError, match="output path differs"):
        gate.submit_q1(dry_run=True)


def test_preflight_record_covers_other_cases_in_the_same_attempt(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    sibling = fixture["case"].parent / "Nb-bcc-baseline"
    sibling.mkdir()
    gate.validate_preflight(fixture["record"], fixture["repo"], sibling)


def test_preflight_record_from_another_attempt_is_rejected(tmp_path, monkeypatch):
    fixture = make_fixture(tmp_path, monkeypatch)
    record = json.loads(fixture["record"].read_text())
    record["case_dir"] = str(tmp_path / "other-attempt/cases" / Q1_CASE_ID)
    _write_json(fixture["record"], record)
    with pytest.raises(gate.GateError, match="different attempt"):
        gate.validate_preflight(fixture["record"], fixture["repo"], fixture["case"])


def test_q1_submission_rejects_production_case_and_gates_remain_separate():
    launcher = (ROOT / "deployments/aurora/launchers/run-nbtao5-case-aurora.sh").read_text()
    q1_wrapper = (ROOT / "deployments/aurora/jobs/job-nbtao5-qualification-aurora.pbs").read_text()
    assert "NBTAO5_PRODUCTION_AUTHORIZED:-" in launcher
    assert "NBTAO5_QUALIFICATION_AUTHORIZED:-" in launcher
    assert "NBTAO5_QUALIFICATION_AUTHORIZED" in q1_wrapper
    assert "NBTAO5_PRODUCTION_AUTHORIZED" not in q1_wrapper
