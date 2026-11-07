#!/usr/bin/env python3
"""Generate E4 planning artifacts only; this command never contacts a scheduler."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.e4.planning import build_run_matrix, load_config  # noqa: E402
from benchmarks.portability.run import resolved_configuration, validate_inputs  # noqa: E402


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "config/campaign.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research/paper/e4/manifests")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    rows = build_run_matrix(config)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    structure = ROOT / config["candidate_contract"]["structure"]
    facilities = {}
    for facility in ("aurora", "frontier", "perlmutter"):
        resolved = resolved_configuration(facility)
        errors = validate_inputs(resolved)
        deployment = resolved["deployment"]
        facilities[facility] = {
            "overlay_path": f"benchmarks/portability/config/{facility}.yaml",
            "overlay_sha256": sha(ROOT / f"benchmarks/portability/config/{facility}.yaml"),
            "preflight_errors": errors,
            "launchers": {
                name: {"path": relative, "sha256": sha(ROOT / relative)}
                for name, relative in deployment["launchers"].items()
            },
        }
    qe_case_path = ROOT / "benchmarks/portability/config/relaxation/qe-si.yaml"
    qe_case = yaml.safe_load(qe_case_path.read_text(encoding="utf-8"))
    pseudo_dir = Path(qe_case["dft"]["pseudo_dir"]).expanduser()
    pseudo_file = pseudo_dir / "Si_r.upf"
    manifest = {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "status": "draft_not_authorized",
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha(config_path),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()
        or None,
        "candidate_structure": {"path": str(structure.relative_to(ROOT)), "sha256": sha(structure)},
        "candidate_qe_assets": {
            "config_path": str(qe_case_path.relative_to(ROOT)),
            "config_sha256": sha(qe_case_path),
            "pseudopotential_path": str(pseudo_file),
            "pseudopotential_exists": pseudo_file.is_file(),
            "pseudopotential_sha256": sha(pseudo_file),
        },
        "facility_preflight": facilities,
        "candidate_mlip": config["candidate_contract"]["mlip"],
        "candidate_dft": config["candidate_contract"]["dft"],
        "scientific_contract": config["candidate_contract"],
        "device_units": config["device_units"],
        "task_count": {
            "strong": config["scaling"]["strong_workload_task_count"],
            "weak_per_node": config["scaling"]["weak_workload_tasks_per_node"],
        },
        "resource_profile": config["aurora_resource_profile_candidate"],
        "planned_allocations": len(rows),
        "scheduler_submissions": 0,
        "note": (
            "Inputs and model are candidates, not frozen. Null hashes and task counts "
            "must be resolved at Gates 1-3."
        ),
        "open_items": config["candidate_contract"]["open_items"],
    }
    (output / "e4-workload-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (output / "e4-run-matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["authorization"] != "not_authorized":
            raise ValueError("planner must not authorize allocations")
    print(f"planned {len(rows)} allocations; submitted 0; artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
