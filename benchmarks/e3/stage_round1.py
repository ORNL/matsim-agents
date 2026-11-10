"""Stage the frozen E3 Round 1 BCC acquisition calculations for the existing E4 runner.

Structures come from the frozen E3 acquisition pool and must match the structure
hashes recorded in the Round 1 decisions. START, KPOINTS and PAW components are the
frozen E4 BCC production inputs. Case directories are prepared with the unchanged
``prepare_batch.prepare_case``; the batch is validated by ``submit_batch.py``.
Nothing is submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read

from benchmarks.e3.run import REPO, _structure_hash
from benchmarks.e4.aurora_dft.prepare_stage1 import _write_poscar

JOBS = REPO / "deployments/aurora/jobs"
sys.path.insert(0, str(JOBS / "e4-first-batch"))
import prepare_batch  # noqa: E402

BATCH_ID = "E3-round1-bcc-01"
STAGE1_BCC = REPO / "research/paper/e4/aurora-dft-scaling/stage1-inputs-v2/bcc_refractory"
PAW_SOURCE = STAGE1_BCC / "E4-BCC-1801" / "POTCAR.spec"  # covers all six BCC elements
INPUTS = REPO / "research/paper/e3/round1-inputs"
DEFAULT_ROOT = REPO.parent / "e3-round1-bcc-01"
DEFAULT_MANIFEST = JOBS / "e3-round1-bcc" / "batch-manifest.json"


def e3_id(candidate_id: str) -> str:
    return "E3-R1-BCC-" + candidate_id.rsplit("-", 1)[1]


def _sites(atoms) -> list[tuple]:
    scaled = np.mod(atoms.get_scaled_positions(wrap=False), 1.0).round(8) % 1.0
    return sorted(zip(atoms.get_chemical_symbols(), map(tuple, scaled), strict=True))


def stage_source(candidate_id: str, atoms, decision: dict) -> Path:
    if _structure_hash(atoms) != decision["structure_sha256"]:
        raise ValueError(f"{candidate_id}: pool structure differs from the frozen decision")
    source = INPUTS / e3_id(candidate_id)
    source.mkdir(parents=True, exist_ok=False)
    _write_poscar(atoms.copy(), source / "POSCAR", e3_id(candidate_id))
    if _sites(read(source / "POSCAR", format="vasp")) != _sites(atoms) or not np.allclose(
        read(source / "POSCAR", format="vasp").cell.array, atoms.cell.array
    ):
        raise ValueError(f"{candidate_id}: written POSCAR is not the frozen structure")
    starts = {(STAGE1_BCC / c / "START").read_bytes() for c in ("E4-BCC-0001", "E4-BCC-1801")}
    kpoints = {(STAGE1_BCC / c / "KPOINTS").read_bytes() for c in ("E4-BCC-0001", "E4-BCC-1801")}
    if len(starts) != 1 or len(kpoints) != 1:
        raise ValueError("frozen E4 BCC START/KPOINTS are not uniform")
    (source / "START").write_bytes(starts.pop())
    (source / "KPOINTS").write_bytes(kpoints.pop())
    paw, _ = prepare_batch.parse_paw_spec(PAW_SOURCE)
    species = read(source / "POSCAR", format="vasp").get_chemical_symbols()
    order = list(dict.fromkeys(species))
    assembled = b"".join(paw[e][1].read_bytes() for e in order)
    for element in order:
        label, path, expected = paw[element]
        if prepare_batch.sha256(path) != expected:
            raise ValueError(f"PAW hash mismatch for {label}")
    (source / "POTCAR.order").write_text("".join(f"{paw[e][0]} {paw[e][2]}\n" for e in order))
    (source / "POTCAR.spec").write_text(
        "".join(f"{e} {paw[e][0]} {paw[e][1]} {paw[e][2]}\n" for e in order)
        + f"assembled_potcar_sha256 {hashlib.sha256(assembled).hexdigest()}\n"
    )
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="frozen acquisition.extxyz")
    parser.add_argument("--round1", type=Path, default=REPO / "research/paper/e3/round1-bcc.json")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    round1 = json.loads(args.round1.read_text())
    if round1["manifest"] != "e3-round1-frozen" or round1["material"] != "bcc_refractory":
        raise SystemExit("not the frozen BCC Round 1 record")
    decisions: dict[str, dict] = {}
    requesters: dict[str, list[str]] = {}
    for strategy, report in round1["strategies"].items():
        for decision in report["decisions"]:
            if decision["state"] != "frozen_awaiting_dft":
                raise SystemExit(f"{decision['candidate_id']} is not frozen awaiting DFT")
            decisions[decision["candidate_id"]] = decision
            requesters.setdefault(decision["candidate_id"], []).append(strategy)
    if len(decisions) != round1["unique_physical_calculations_requested"]:
        raise SystemExit("unique physical calculations differ from the frozen manifest")
    pool = {a.info["frame_id"]: a for a in read(args.pool, index=":")}
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty batch directory: {root}")
    if prepare_batch.sha256(prepare_batch.EXECUTABLE) != prepare_batch.VASP_SHA256:
        raise SystemExit("VASP executable hash mismatch")
    cases = []
    for candidate_id in sorted(decisions):
        decision = decisions[candidate_id]
        source = stage_source(candidate_id, pool[candidate_id], decision)
        record = {"atom_count": 128, "canonical_structure_id": decision["structure_sha256"]}
        case = prepare_batch.prepare_case(
            e3_id(candidate_id), "bcc", source, source / "POSCAR", record, root
        )
        case["e3"] = {
            "candidate_id": candidate_id,
            "requested_by": sorted(requesters[candidate_id]),
            "logical_attempts": len(requesters[candidate_id]),
            "round": 1,
            "frozen_round1_manifest_sha256": round1["sha256"],
            "decision_ids": {
                s: next(
                    d["decision_id"]
                    for d in round1["strategies"][s]["decisions"]
                    if d["candidate_id"] == candidate_id
                )
                for s in requesters[candidate_id]
            },
            "structure_sha256": decision["structure_sha256"],
            "committee_force_disagreement_eV_A": decision["committee_force_disagreement_eV_A"],
            "soap_novelty_to_references": decision["soap_novelty_to_references"],
        }
        cases.append(case)
    manifest = {
        "schema_version": 1,
        "batch_id": BATCH_ID,
        "execution_authorized": False,
        "executable": str(prepare_batch.EXECUTABLE),
        "executable_sha256": prepare_batch.VASP_SHA256,
        "queue": "capacity",
        "resource_caps_node_hours": {"HydraGNN": 24 * len(cases)},
        "total_resource_cap_node_hours": 24 * len(cases),
        "e3_accounting": {
            "logical_attempts": {
                s: sum(s in c["e3"]["requested_by"] for c in cases)
                for s in ("llm_guided_adaptation", "rule_based_adaptation")
            },
            "unique_physical_calculations": len(cases),
            "attempts_consumed_on": "qsub acceptance; restarts reuse the calculation ID",
        },
        "cases": cases,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(cases)} cases at {root}; manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
