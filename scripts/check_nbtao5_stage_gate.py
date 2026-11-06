from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: check_nbtao5_stage_gate.py CASE_DIR REPO GATE_RECORD")

    case_dir, repo = map(Path, sys.argv[1:3])
    gate_path = Path(sys.argv[3]) if sys.argv[3] else None
    case = json.loads((case_dir / "case.json").read_text())
    manifest_path = repo / "research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    runtime = json.loads((repo / "research/nb-ta-o/pilot/vasp_runtime_manifest.json").read_text())
    manifest_cases = {item["case_id"]: item for item in manifest["cases"]}
    current_case = manifest_cases.get(case.get("case_id"))
    if current_case is None:
        raise SystemExit("Case is not present in the current input manifest.")
    if case.get("protocol_id") != manifest["protocol_id"]:
        raise SystemExit("Case uses a stale or unsupported protocol.")
    for key in (
        "input_hashes",
        "settings",
        "kpoints",
        "species_order",
        "species_counts",
        "acceptance_criteria",
        "gamma_only",
        "category",
        "potcar_required_components",
    ):
        if case.get(key) != current_case.get(key):
            raise SystemExit(f"Case {key} do not match the current input manifest.")
    matches = [
        stage
        for stage in manifest["qualification_stages"]
        if case["case_id"] in stage["workflow_ids"]
    ]
    if len(matches) != 1:
        raise SystemExit("Case is not assigned to exactly one qualification stage.")

    stage = matches[0]
    if not stage["depends_on"]:
        return 0
    if gate_path is None or not gate_path.is_file():
        raise SystemExit("This stage requires a reviewed NBTAO5_STAGE_GATE_RECORD.")

    gate = json.loads(gate_path.read_text())
    expected_paw = {name: item["sha256"] for name, item in runtime["proposed_components"].items()}
    required_stages = set(stage["depends_on"])
    completed_stages = set(gate.get("completed_stage_ids", []))
    if gate.get("protocol_id") != manifest["protocol_id"]:
        raise SystemExit("Stage gate record uses a different qualification protocol.")
    if gate.get("status") != "stage_passed_and_next_stage_authorized":
        raise SystemExit("Stage gate record is not a reviewed pass/authorization.")
    if gate.get("authorized_stage_id") != stage["stage_id"]:
        raise SystemExit("Stage gate record does not authorize this exact stage.")
    if not required_stages.issubset(completed_stages):
        raise SystemExit("Stage gate record does not include every required prior stage.")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if gate.get("protocol_manifest_sha256") != manifest_hash:
        raise SystemExit("Stage gate record does not match the current input manifest.")
    if gate.get("vasp_executable_sha256") != runtime["executable"]["sha256"]:
        raise SystemExit("Stage gate record uses a different VASP executable.")
    if gate.get("paw_component_sha256") != expected_paw:
        raise SystemExit("Stage gate record uses a different PAW component set.")
    if not gate.get("reviewer") or not gate.get("reviewed_at_utc"):
        raise SystemExit("Stage gate record must identify its reviewer and review time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
