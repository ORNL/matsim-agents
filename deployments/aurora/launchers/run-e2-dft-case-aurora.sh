#!/bin/bash
# Run one staged E2 DFT case: segmented, resumable ISIF=3 relaxation -> static -> summary.json.
# Usage: run-e2-dft-case-aurora.sh <case_dir>   (MATSIM_ASSIGNED_NODES selects the node)
set -uo pipefail
CASE="$(realpath -e "$1")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
MAX_SEGMENTS="${E2_MAX_SEGMENTS:-6}"
LAUNCHER="${REPO}/deployments/aurora/launchers/run-vasp-gpu-aurora.sh"
RANKS="$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["settings"]["ranks_per_node"])' "${CASE}/case.json")"

mkdir "${CASE}/.lock" 2>/dev/null || { echo "[e2] ${CASE} is locked by another run" >&2; exit 3; }
trap 'rmdir "${CASE}/.lock" 2>/dev/null' EXIT

status() {
  printf '{"state": "%s", "detail": "%s", "node": "%s", "time": "%s"}\n' \
    "$1" "$2" "${MATSIM_ASSIGNED_NODES:-unknown}" "$(date -u +%FT%TZ)" >"${CASE}/status.json"
  echo "[e2] $(basename "${CASE}") $1: $2"
}
relaxed() { grep -q 'reached required accuracy' "${CASE}/relax/OUTCAR" 2>/dev/null; }
run_vasp() {
  (cd "$1" && NRANKS="${RANKS}" OMP_NUM_THREADS=8 VASP_VARIANT=std bash "${LAUNCHER}" >>vasp.log 2>&1)
}

export VASP_ROOT="${REPO}/external/vasp6/src/vasp.6.6.1"
mkdir -p "${CASE}/relax"
if [[ ! -e "${CASE}/relax/POSCAR" ]]; then
  cp "${CASE}/POSCAR" "${CASE}/KPOINTS" "${CASE}/POTCAR" "${CASE}/relax/"
  cp "${CASE}/INCAR.relax" "${CASE}/relax/INCAR"
fi

segment=$(find "${CASE}/relax" -maxdepth 1 -name 'segment-*' -type d | wc -l)
while ! relaxed; do
  if [[ -s "${CASE}/relax/OUTCAR" ]]; then
    seg_dir="${CASE}/relax/segment-$(printf '%02d' "${segment}")"
    mkdir -p "${seg_dir}"
    for f in OUTCAR OSZICAR vasprun.xml XDATCAR CONTCAR POSCAR vasp.log; do
      [[ -e "${CASE}/relax/${f}" ]] && cp -p "${CASE}/relax/${f}" "${seg_dir}/"
    done
    if [[ -s "${CASE}/relax/CONTCAR" ]] && grep -q 'Iteration' "${CASE}/relax/OUTCAR"; then
      cp "${CASE}/relax/CONTCAR" "${CASE}/relax/POSCAR"
    fi
    # A WAVECAR from a killed run may be truncated; keep it only after a normal VASP exit.
    grep -q 'General timing' "${CASE}/relax/OUTCAR" || rm -f "${CASE}/relax/WAVECAR"
    rm -f "${CASE}/relax/OUTCAR" "${CASE}/relax/vasprun.xml"
    segment=$((segment + 1))
  fi
  ((segment < MAX_SEGMENTS)) || { status relax_unconverged "max_segments=${MAX_SEGMENTS}"; exit 1; }
  status running "relax segment ${segment}"
  run_vasp "${CASE}/relax" || { status execution_failed "relax segment ${segment} exit $?"; exit 1; }
  grep -q 'Iteration' "${CASE}/relax/OUTCAR" 2>/dev/null || { status execution_failed "no SCF iteration in relax segment ${segment}"; exit 1; }
done
rm -f "${CASE}/relax/WAVECAR"

mkdir -p "${CASE}/static"
if ! grep -q 'General timing' "${CASE}/static/OUTCAR" 2>/dev/null; then
  cp "${CASE}/relax/CONTCAR" "${CASE}/static/POSCAR"
  cp "${CASE}/KPOINTS" "${CASE}/POTCAR" "${CASE}/static/"
  cp "${CASE}/INCAR.static" "${CASE}/static/INCAR"
  status running static
  run_vasp "${CASE}/static" || { status execution_failed "static exit $?"; exit 1; }
fi

"${PY}" - "${CASE}" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from pymatgen.io.vasp import Outcar, Vasprun
case = Path(sys.argv[1])
out = {"case": case.name}
for phase in ("relax", "static"):
    vr = Vasprun(case / phase / "vasprun.xml", parse_dos=False, parse_eigen=False, parse_potcar_file=False)
    st = vr.final_structure
    fin = vr.ionic_steps[-1]
    oc = Outcar(case / phase / "OUTCAR")
    mag = oc.total_mag if hasattr(oc, "total_mag") else None
    out[phase] = {
        "converged_electronic": bool(vr.converged_electronic),
        "converged_ionic": bool(vr.converged_ionic) if phase == "relax" else None,
        "relax_accuracy_marker": "reached required accuracy" in (case / phase / "OUTCAR").read_text(errors="replace") if phase == "relax" else None,
        "energy_eV": float(vr.final_energy),
        "energy_per_atom_eV": float(vr.final_energy) / len(st),
        "max_force_eV_per_A": float(np.linalg.norm(np.array(fin["forces"]), axis=1).max()),
        "stress_kbar": np.array(fin["stress"]).tolist(),
        "total_magnetization_muB": None if mag is None else float(mag),
        "volume_A3": float(st.volume), "volume_per_atom_A3": float(st.volume) / len(st),
        "lattice_abc_A": [float(x) for x in st.lattice.abc],
        "lattice_angles_deg": [float(x) for x in st.lattice.angles],
        "ionic_steps_last_segment": len(vr.ionic_steps),
    }
out["accepted"] = bool(out["relax"]["relax_accuracy_marker"] and out["relax"]["converged_electronic"]
                       and out["static"]["converged_electronic"])
(case / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"case": case.name, "accepted": out["accepted"], "E_per_atom": out["static"]["energy_per_atom_eV"]}))
PY
if [[ $? -ne 0 ]]; then status summary_failed "summary parse error"; exit 1; fi
mapfile -t EXPECT < <("${PY}" - "${CASE}/case.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
s = c["settings"]["relax"]
for tag in ("ENCUT", "EDIFF", "PREC", "ISPIN", "IVDW", "ISMEAR", "SIGMA"):
    if tag in s:
        print("--expect"); print(f"{tag}={s[tag]}")
print("--expect-paw"); print(",".join(c["backend"]["paw"].values()))
PY
)
verified=1
"${PY}" "${REPO}/scripts/verify_vasp_run.py" "${CASE}/relax" --phase relax --max-force 0.02 "${EXPECT[@]}" \
  -o "${CASE}/verify-relax.json" || verified=0
"${PY}" "${REPO}/scripts/verify_vasp_run.py" "${CASE}/static" --phase static "${EXPECT[@]}" \
  -o "${CASE}/verify-static.json" || verified=0
if ((verified)); then
  status accepted "raw OUTCAR/OSZICAR verification passed for relax and static"
else
  status rejected "raw verification failed; see verify-relax.json and verify-static.json"
  exit 1
fi
