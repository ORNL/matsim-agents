"""Run one MLIP relaxation pass over a checked-in paper-case seed set."""

from __future__ import annotations

import argparse
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from matsim_agents.orchestration.objective_graph import build_graph
from matsim_agents.orchestration.state import MatSimState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_ROOT = Path(__file__).resolve().parent / "seeds"
OUT_ROOT = Path(os.environ.get("OUT_DIR", PROJECT_ROOT / "out_singlepass"))
CASES = ("cu_bht", "hea_bcc", "hea_fcc", "lifepo4", "phosphorene", "zn_formate")


def stage_seeds(case: str, destination: Path) -> list[Path]:
    """Copy the checked-in seeds for one case into its run directory."""
    sources = sorted(SEED_ROOT.glob(f"{case}_*"))
    if not sources:
        raise FileNotFoundError(
            f"No seed structures found for {case} in {SEED_ROOT}. "
            "Cu-BHT requires a supplied cu_bht_monolayer.cif."
        )
    destination.mkdir(parents=True, exist_ok=True)
    staged = []
    for source in sources:
        target = destination / source.name
        shutil.copy2(source, target)
        staged.append(target.resolve())
    return staged


def make_objective(case: str, seed_paths: list[Path]) -> str:
    structures = "\n".join(f"- {path}" for path in seed_paths)
    return (
        f"Relax each supplied seed structure for the {case} paper case using the configured "
        "MLIP. Compare relaxed energies per atom, rank the structures, and report their "
        f"initial and optimized structure paths.\n{structures}"
    )


def run_case(case: str, run_dir: Path, args: argparse.Namespace) -> None:
    case_dir = run_dir / case
    seed_paths = stage_seeds(case, case_dir / "seeds")
    objective = make_objective(case, seed_paths)
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "mlip_backend": args.mlip_backend,
            "logdir": str(args.logdir) if args.logdir else None,
            "hydragnn_branch_mlp_checkpoint": args.hydragnn_branch_mlp_checkpoint,
            "checkpoint": args.checkpoint,
            "uma_model_name": args.uma_model_name,
            "uma_task": args.uma_task,
            "mlp_device": args.mlp_device,
            "output_dir": str(case_dir / "relaxations"),
        }
    }
    final = build_graph().invoke(MatSimState(objective=objective), config=config)
    analysis = str(final.get("analysis") or "")
    (case_dir / "objective.txt").write_text(objective + "\n", encoding="utf-8")
    (case_dir / "analysis.txt").write_text(analysis + "\n", encoding="utf-8")
    print(f"\n=== {case}: {len(final.get('results', []))} relaxation(s) ===")
    print(analysis)


def _run_directory() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_id = os.environ.get("RUN_ID", f"run-{timestamp}-{job_id}-{uuid.uuid4().hex[:8]}")
    path = OUT_ROOT / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", choices=CASES)
    selection.add_argument("--all", action="store_true")
    parser.add_argument(
        "--seeds-only", action="store_true", help="Stage seed files without running the graph."
    )
    parser.add_argument(
        "--mlip-backend",
        choices=("hydragnn", "uma"),
        default=os.getenv("MATSIM_MLIP_BACKEND", "hydragnn"),
    )
    parser.add_argument("--logdir", type=Path, default=os.getenv("MATSIM_HYDRAGNN_LOGDIR"))
    parser.add_argument(
        "--hydragnn-branch-mlp-checkpoint",
        default=os.getenv("HYDRAGNN_BRANCH_MLP_CHECKPOINT"),
    )
    parser.add_argument("--checkpoint", default=os.getenv("MATSIM_HYDRAGNN_CHECKPOINT"))
    parser.add_argument("--uma-model-name", default=os.getenv("MATSIM_UMA_MODEL_NAME", "uma-s-1p1"))
    parser.add_argument("--uma-task", default=os.getenv("MATSIM_UMA_TASK", "omat"))
    parser.add_argument("--mlp-device", default="cuda")
    args = parser.parse_args()

    cases = CASES if args.all else (args.case,)
    run_dir = _run_directory()
    print(f"Run directory: {run_dir}")
    for case in cases:
        try:
            if args.seeds_only:
                paths = stage_seeds(case, run_dir / case / "seeds")
                print(f"[{case}] staged {len(paths)} seed structure(s)")
            else:
                run_case(case, run_dir, args)
        except FileNotFoundError as exc:
            print(f"[{case}] skipped: {exc}")


if __name__ == "__main__":
    main()