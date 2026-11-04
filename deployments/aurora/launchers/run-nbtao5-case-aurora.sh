#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[[ $# -eq 1 ]] || { echo "Usage: $0 qualification|production" >&2; exit 2; }
STAGE="$1"
CASE_DIR="$(realpath -e "${NBTAO5_CASE_DIR:?set NBTAO5_CASE_DIR to an external staged case}")"
REPO_REAL="$(realpath -e "${REPO}")"
case "${CASE_DIR}/" in
  "${REPO_REAL}/"*) echo "Refusing run directory inside Git worktree." >&2; exit 2 ;;
esac

case "${STAGE}" in
  qualification)
    [[ "${NBTAO5_QUALIFICATION_AUTHORIZED:-}" == "YES" ]] || {
      echo "Qualification execution requires NBTAO5_QUALIFICATION_AUTHORIZED=YES." >&2
      exit 2
    }
    [[ "${NBTAO5_RESOURCE_APPROVED:-}" == "YES" ]] || {
      echo "Qualification execution requires NBTAO5_RESOURCE_APPROVED=YES for the facility/account cap." >&2
      exit 2
    }
    if ! "${REPO}/.venv/bin/python" "${REPO}/scripts/check_nbtao5_stage_gate.py" \
      "${CASE_DIR}" "${REPO}" "${NBTAO5_STAGE_GATE_RECORD:-}"; then
      exit 2
    fi
    ;;
  production)
    [[ "${NBTAO5_PRODUCTION_AUTHORIZED:-}" == "YES" ]] || {
      echo "Production relaxations require explicit NBTAO5_PRODUCTION_AUTHORIZED=YES." >&2
      exit 2
    }
    [[ "${NBTAO5_RESOURCE_APPROVED:-}" == "YES" ]] || {
      echo "Production execution requires NBTAO5_RESOURCE_APPROVED=YES for the facility/account cap." >&2
      exit 2
    }
    RECORD="${NBTAO5_QUALIFICATION_RECORD:?set an approved qualification record}"
    "${REPO}/.venv/bin/python" - "${RECORD}" "${CASE_DIR}" "${REPO}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

record_path, case_dir, repo = map(Path, sys.argv[1:])
record = json.loads(record_path.read_text())
case = json.loads((case_dir / "case.json").read_text())
if case.get("category") != "production_candidate":
  raise SystemExit("Production execution is restricted to the five selected pilot candidates.")
runtime = json.loads(
    (repo / "research/nb-ta-o/pilot/vasp_runtime_manifest.json").read_text()
)
protocol_path = repo / "research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json"
protocol_manifest = json.loads(protocol_path.read_text())
current_case = next(
  (item for item in protocol_manifest["cases"] if item["case_id"] == case.get("case_id")),
  None,
)
if current_case is None:
  raise SystemExit("Candidate is absent from the current input manifest.")
for key in (
  "protocol_id",
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
    raise SystemExit(f"Candidate {key} differs from the current input manifest.")
protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
expected_paw = {
    name: component["sha256"]
    for name, component in runtime["proposed_components"].items()
}
  if record.get("protocol_id") != protocol_manifest["protocol_id"] or record.get("status") != "qualified":
    raise SystemExit("Qualification record is absent or not marked qualified.")
if record.get("protocol_manifest_sha256") != protocol_sha256:
    raise SystemExit("Qualification record does not match the current protocol manifest.")
if record.get("vasp_executable_sha256") != runtime["executable"]["sha256"]:
    raise SystemExit("Qualification record uses a different VASP executable.")
if record.get("paw_component_sha256") != expected_paw:
    raise SystemExit("Qualification record uses a different PAW component set.")
if record.get("qualified_settings") != case.get("settings"):
    raise SystemExit("Candidate inputs do not match the qualified settings.")
PY
    ;;
  *) echo "Unknown Nb-Ta-O stage: ${STAGE}" >&2; exit 2 ;;
esac

for name in POSCAR INCAR INCAR.static KPOINTS KPOINTS.json POTCAR POTCAR.sha256; do
  [[ -s "${CASE_DIR}/${name}" ]] || { echo "Missing ${CASE_DIR}/${name}" >&2; exit 2; }
done
[[ -s "${CASE_DIR}/POTCAR.order" && -s "${CASE_DIR}/case.json" ]] || {
  echo "Missing POTCAR.order or case.json." >&2
  exit 2
}
[[ "${NBTAO5_PAW_RELEASE_APPROVED:-}" == "YES" ]] || {
  echo "Execution requires explicit approval of the homogeneous PAW release." >&2
  exit 2
}
PAW_ROOT="$(realpath -e "${NBTAO5_PAW_ROOT:?set NBTAO5_PAW_ROOT to the approved library}")"
"${REPO}/.venv/bin/python" "${REPO}/scripts/nbtao5_q1_gate.py" validate-case \
  --repo "${REPO}" --case "${CASE_DIR}" --paw-root "${PAW_ROOT}"

LOCK_ROOT="$(realpath -e "${NBTAO5_LOCK_ROOT:?set a shared external lock root}")"
case "${LOCK_ROOT}/" in
  "${REPO_REAL}/"*) echo "Lock root must be outside the Git worktree." >&2; exit 2 ;;
esac
LOCK_DIR="${LOCK_ROOT}/.nbtao5-active.lock"
mkdir "${LOCK_DIR}" 2>/dev/null || {
  echo "Another Nb-Ta-O workflow is active or left a stale lock: ${LOCK_DIR}" >&2
  exit 2
}
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

VASP_BIN="${REPO}/external/vasp6/src/vasp.6.6.1/bin/vasp_std"
EXPECTED_VASP_SHA256="ebd284027717ff564abf7ceaf93c3aca3261b129d7b60b3f2e3e2aa82d04c552"
[[ -x "${VASP_BIN}" ]] || { echo "Missing executable ${VASP_BIN}" >&2; exit 2; }
ACTUAL_VASP_SHA256="$(sha256sum "${VASP_BIN}" | awk '{print $1}')"
[[ "${ACTUAL_VASP_SHA256}" == "${EXPECTED_VASP_SHA256}" ]] || {
  echo "VASP executable hash differs from the reviewed fingerprint." >&2
  exit 2
}

JOB_ID="${PBS_JOBID:-manual}"
WORK_DIR="${CASE_DIR}/runs/${JOB_ID}"
mkdir -p "${WORK_DIR}/relax" "${WORK_DIR}/static"

RUN_STATUS="${WORK_DIR}/workflow_status.json"
STOP_MARKER="${WORK_DIR}/walltime_stop_requested"
ACTIVE_RUN_DIR=""
STOP_REQUESTED=0
STOP_REASON=""
RUN_START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

write_status() {
  "${REPO}/.venv/bin/python" - "${RUN_STATUS}" "$1" "$2" "${ACTIVE_RUN_DIR}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text()) if path.exists() else {}
record.update(
    status=sys.argv[2],
    detail=sys.argv[3],
    active_run_directory=sys.argv[4] or None,
    updated_at_utc=datetime.now(timezone.utc).isoformat(),
)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(record, indent=2) + "\n")
temporary.replace(path)
PY
}

request_stop() {
  STOP_REQUESTED=1
  STOP_REASON="$1"
  touch "${STOP_MARKER}"
  for run_dir in "${WORK_DIR}/relax" "${WORK_DIR}/static"; do
    [[ -d "${run_dir}" ]] && printf 'LSTOP = .TRUE.\n' >"${run_dir}/STOPCAR"
  done
  write_status "interrupted" "${STOP_REASON}"
}

trap 'request_stop "scheduler_signal_TERM"' TERM
trap 'request_stop "scheduler_signal_USR1"' USR1

cat >"${WORK_DIR}/provenance.json" <<EOF
{
  "case_id": $("${REPO}/.venv/bin/python" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["case_id"]))' "${CASE_DIR}/case.json"),
  "protocol_id": "$(${REPO}/.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol_id"])' "${REPO}/research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json")",
  "stage": "${STAGE}",
  "pbs_job_id": "${JOB_ID}",
  "host": "$(hostname)",
  "runner_environment_loaded_modules": "${LOADEDMODULES:-unknown}",
  "configured_module_stack": $("${REPO}/.venv/bin/python" - "${REPO}/research/nb-ta-o/pilot/vasp_runtime_manifest.json" <<'PY'
import json
import sys
print(json.dumps(json.load(open(sys.argv[1]))["platform"]["module_stack"]))
PY
),
  "started_at_utc": "${RUN_START_UTC}",
  "input_manifest_sha256": "$(sha256sum "${REPO}/research/nb-ta-o/pilot/vasp_inputs/inputs_manifest.json" | awk '{print $1}')",
  "vasp_executable_sha256": "${EXPECTED_VASP_SHA256}",
  "paw_component_sha256": $("${REPO}/.venv/bin/python" - "${REPO}/research/nb-ta-o/pilot/vasp_runtime_manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
print(json.dumps({key: value["sha256"] for key, value in manifest["proposed_components"].items()}))
PY
),
  "assembled_potcar_sha256": "$(sha256sum "${CASE_DIR}/POTCAR" | awk '{print $1}')",
  "input_hashes": $("${REPO}/.venv/bin/python" - "${CASE_DIR}/case.json" <<'PY'
import json
import sys
print(json.dumps(json.load(open(sys.argv[1]))["input_hashes"]))
PY
),
  "scientific_settings": $("${REPO}/.venv/bin/python" - "${CASE_DIR}/case.json" <<'PY'
import json
import sys
print(json.dumps(json.load(open(sys.argv[1]))["settings"]))
PY
)
}
EOF

write_status "running" "workflow_started"

# Request VASP's normal end-of-step stop five minutes before PBS walltime.
(
  sleep "${NBTAO5_SOFT_STOP_SECONDS:-1500}"
  touch "${STOP_MARKER}"
  for run_dir in "${WORK_DIR}/relax" "${WORK_DIR}/static"; do
    [[ -d "${run_dir}" ]] && printf 'LSTOP = .TRUE.\n' >"${run_dir}/STOPCAR"
  done
) &
WATCHDOG_PID=$!
trap 'kill "${WATCHDOG_PID}" 2>/dev/null || true; wait "${WATCHDOG_PID}" 2>/dev/null || true' EXIT

# Davidson (ALGO=Normal) hangs on Aurora when bands are split across ranks; KPAR=ranks avoids it.
VASP_RANKS=12

make_restartable_incar() {
  awk '!/^[[:space:]]*([Ll][Ww][Aa][Vv][Ee]|[Kk][Pp][Aa][Rr])[[:space:]]*=/ { print }
       END { print "LWAVE = .TRUE."; print "KPAR = '"${VASP_RANKS}"'" }' "$1" >"$2"
}

run_vasp() {
  local run_dir="$1"
  ACTIVE_RUN_DIR="${run_dir}"
  if [[ -e "${STOP_MARKER}" ]]; then
    request_stop "soft_walltime_limit"
    return 75
  fi
  write_status "running" "${run_dir##*/}_started"
  (
    cd "${run_dir}"
    bash "${LAUNCHER}"
  ) >"${run_dir}/launcher.stdout.log" 2>"${run_dir}/launcher.stderr.log" &
  local launcher_pid=$!
  local launcher_status=0
  set +e
  while true; do
    wait "${launcher_pid}"
    launcher_status=$?
    kill -0 "${launcher_pid}" 2>/dev/null || break
  done
  set -e
  ACTIVE_RUN_DIR=""
  if [[ "${STOP_REQUESTED}" -eq 1 || -e "${STOP_MARKER}" ]]; then
    write_status "interrupted" "${STOP_REASON:-soft_walltime_limit}"
    return 75
  fi
  return "${launcher_status}"
}

summarize_vasprun() {
  "${REPO}/.venv/bin/python" - "$1" "$2" "$3" <<'PY'
import json
import sys
import numpy as np
from pymatgen.io.vasp.outputs import Vasprun

xml_path, summary_path, phase, case_json = sys.argv[1:]
case = json.load(open(case_json))
run = Vasprun(xml_path, parse_dos=False, parse_eigen=False, exception_on_bad_xml=False)
last = run.ionic_steps[-1] if run.ionic_steps else {}
forces = np.asarray(last["forces"]) if "forces" in last else None
stress = np.asarray(last["stress"]) if "stress" in last else None
max_force = float(np.linalg.norm(forces, axis=1).max()) if forces is not None else None
principal_stress = (
  np.linalg.eigvalsh((stress + stress.T) / 2.0) if stress is not None else None
)
max_abs_principal_stress = (
  float(np.abs(principal_stress).max()) if principal_stress is not None else None
)
force_limit = case["acceptance_criteria"]["max_force_eV_A"]
stress_limit = case["acceptance_criteria"]["max_abs_principal_stress_kbar"]
force_passed = max_force is not None and max_force <= force_limit + 1e-8
stress_passed = (
  None
  if stress_limit is None
  else max_abs_principal_stress is not None and max_abs_principal_stress <= stress_limit
)
summary = {
    "phase": phase,
    "vasprun_xml": xml_path,
    "converged_electronic": bool(run.converged_electronic),
    "converged_ionic": bool(run.converged_ionic) if phase == "relax" else None,
    "ionic_steps": len(run.ionic_steps),
    "final_energy_eV": run.final_energy,
    "final_energy_per_atom_eV": run.final_energy_per_atom,
    "max_force_eV_per_A": max_force,
    "force_criterion_eV_per_A": force_limit,
    "force_criterion_passed": force_passed,
    "stress_kbar": stress.tolist() if stress is not None else None,
    "max_abs_principal_stress_kbar": max_abs_principal_stress,
    "stress_criterion_kbar": stress_limit,
    "stress_criterion_passed": stress_passed,
}
with open(summary_path, "w") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
if not run.converged_electronic:
  raise SystemExit("VASP finished without electronic convergence.")
if phase == "relax":
  if not run.converged_ionic or not force_passed:
    raise SystemExit("Relaxation failed ionic convergence or the 0.01 eV/A force criterion.")
  if stress_passed is False:
    raise SystemExit("Relaxation failed the 2 kbar residual stress criterion.")
PY
}

cp "${CASE_DIR}/POSCAR" "${CASE_DIR}/KPOINTS" "${CASE_DIR}/KPOINTS.json" "${CASE_DIR}/POTCAR" "${WORK_DIR}/relax/"
make_restartable_incar "${CASE_DIR}/INCAR" "${WORK_DIR}/relax/INCAR"
cp "${CASE_DIR}/POTCAR" "${WORK_DIR}/static/"
make_restartable_incar "${CASE_DIR}/INCAR.static" "${WORK_DIR}/static/INCAR"
  "${REPO}/.venv/bin/python" - "${WORK_DIR}/provenance.json" "${WORK_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

provenance_path, work_dir = map(Path, sys.argv[1:])
record = json.loads(provenance_path.read_text())
record["effective_incar_sha256"] = {
  phase: hashlib.sha256((work_dir / phase / "INCAR").read_bytes()).hexdigest()
  for phase in ("relax", "static")
}
record["calculation_kpoints"] = {
  phase: json.loads((work_dir / phase / "KPOINTS.json").read_text())
  for phase in ("relax",)
}
record["runtime_only_input_change"] = (
  "LWAVE=.TRUE. to retain restart WAVECAR files; KPAR=12 to avoid a multi-rank Davidson "
  "hang on Aurora (parallelization only); staged scientific inputs are unchanged"
)
provenance_path.write_text(json.dumps(record, indent=2) + "\n")
PY

export VASP_ROOT="${REPO}/external/vasp6/src/vasp.6.6.1"
export VASP_BIN
export VASP_VARIANT=std
export NRANKS="${VASP_RANKS}"
export OMP_NUM_THREADS=8
LAUNCHER="${REPO}/deployments/aurora/launchers/run-vasp-gpu-aurora.sh"
if run_vasp "${WORK_DIR}/relax"; then
  :
else
  status=$?
  if [[ "${STOP_REQUESTED}" -eq 1 || -e "${STOP_MARKER}" ]]; then
    write_status "interrupted" "${STOP_REASON:-soft_walltime_limit}; outputs_preserved"
    exit 75
  fi
  write_status "execution_failed" "relaxation_launcher_exit_${status}"
  exit "${status}"
fi
[[ -s "${WORK_DIR}/relax/CONTCAR" && -s "${WORK_DIR}/relax/vasprun.xml" ]] || {
  write_status "incomplete" "relaxation_outputs_missing"
  echo "Relaxation outputs missing; stopping before static single point." >&2
  exit 1
}
if ! summarize_vasprun "${WORK_DIR}/relax/vasprun.xml" "${WORK_DIR}/relax/summary.json" relax "${CASE_DIR}/case.json"; then
  if [[ "${STOP_REQUESTED}" -eq 1 || -e "${STOP_MARKER}" ]]; then
    write_status "interrupted" "walltime_or_scheduler_stop; relaxation_not_classified_as_physical_failure"
    exit 75
  fi
  write_status "physical_convergence_failed" "relaxation_not_converged"
  exit 1
fi
[[ ! -e "${STOP_MARKER}" ]] || {
  request_stop "soft_walltime_limit_after_relaxation"
  exit 75
}
cp "${WORK_DIR}/relax/CONTCAR" "${WORK_DIR}/static/POSCAR"
if [[ "$("${REPO}/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gamma_only"])' "${CASE_DIR}/case.json")" == "True" ]]; then
  "${REPO}/.venv/bin/python" "${REPO}/scripts/nbtao5_kpoints.py" \
    "${WORK_DIR}/static/POSCAR" "${WORK_DIR}/static/KPOINTS" \
    "${WORK_DIR}/static/KPOINTS.json" --gamma-only >/dev/null
else
  spacing="$("${REPO}/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["settings"]["kpoint_spacing_A^-1"])' "${CASE_DIR}/case.json")"
  "${REPO}/.venv/bin/python" "${REPO}/scripts/nbtao5_kpoints.py" \
    "${WORK_DIR}/static/POSCAR" "${WORK_DIR}/static/KPOINTS" \
    "${WORK_DIR}/static/KPOINTS.json" --spacing "${spacing}" >/dev/null
fi
"${REPO}/.venv/bin/python" - "${WORK_DIR}/provenance.json" "${WORK_DIR}/static/KPOINTS.json" <<'PY'
import json
import sys
from pathlib import Path
provenance = Path(sys.argv[1])
record = json.loads(provenance.read_text())
record.setdefault("calculation_kpoints", {})["static"] = json.loads(Path(sys.argv[2]).read_text())
provenance.write_text(json.dumps(record, indent=2) + "\n")
PY
if run_vasp "${WORK_DIR}/static"; then
  :
else
  status=$?
  if [[ "${STOP_REQUESTED}" -eq 1 || -e "${STOP_MARKER}" ]]; then
    write_status "interrupted" "${STOP_REASON:-soft_walltime_limit}; outputs_preserved"
    exit 75
  fi
  write_status "execution_failed" "static_launcher_exit_${status}"
  exit "${status}"
fi
[[ -s "${WORK_DIR}/static/vasprun.xml" ]] || {
  write_status "incomplete" "static_output_missing"
  echo "Static outputs missing." >&2
  exit 1
}
if ! summarize_vasprun "${WORK_DIR}/static/vasprun.xml" "${WORK_DIR}/static/summary.json" static "${CASE_DIR}/case.json"; then
  if [[ "${STOP_REQUESTED}" -eq 1 || -e "${STOP_MARKER}" ]]; then
    write_status "interrupted" "walltime_or_scheduler_stop; static_not_classified_as_physical_failure"
    exit 75
  fi
  write_status "physical_convergence_failed" "static_not_converged"
  exit 1
fi
write_status "complete" "relaxation_and_static_converged; review_required"
"${REPO}/.venv/bin/python" - "${WORK_DIR}/provenance.json" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text())
record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
work_dir = path.parent
record["output_sha256"] = {
  str(output.relative_to(work_dir)): hashlib.sha256(output.read_bytes()).hexdigest()
  for phase in ("relax", "static")
  for output in (work_dir / phase).rglob("*")
  if output.is_file() and output.name != "POTCAR"
}
path.write_text(json.dumps(record, indent=2) + "\n")
PY
printf 'Nb-Ta-O %s workflow complete: %s\n' "${STAGE}" "${WORK_DIR}"