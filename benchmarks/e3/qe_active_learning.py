"""Separate E3/QE active-learning orchestration; never modifies frozen results."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.e3 import production as e3
from matsim_agents.active_learning.candidates import Candidate
from matsim_agents.active_learning.config import ALConfig
from matsim_agents.active_learning.dft_protocol import path_identity, scientific_dft_payload
from matsim_agents.active_learning.loop import run_active_learning
from matsim_agents.execution.locking import exclusive_lock

log = logging.getLogger(__name__)
VARIANT = "e3-mlip-qe-active-learning"


def validate_config(cfg: ALConfig, material: str) -> None:
    from ase.formula import Formula

    if material not in e3.MATERIALS:
        raise ValueError(f"Unknown E3 material: {material}")
    if cfg.mlip.backend != "mace" or cfg.mlip.mace is None:
        raise ValueError("E3-QE requires MACE")
    if cfg.mlip.mace.family != "checkpoint" or cfg.mlip.mace.precision != "fp64":
        raise ValueError("E3-QE requires a local, fp64 MACE checkpoint")
    if (
        cfg.mlip.mace.device != "cuda"
        or cfg.mlip.mace.dispersion
        or cfg.mlip.mace.head is not None
        or cfg.mlip.mace.ensemble_models
    ):
        raise ValueError(
            "Use the unchanged single CUDA MACE head without added dispersion/ensemble"
        )
    if e3._sha256_file(Path(cfg.mlip.mace.model)) != e3.E3_MLIP_CHECKPOINT_SHA256:
        raise ValueError("The initial model must be the frozen MACE-OMAT-0 medium checkpoint")
    if cfg.dft.backend != "qe" or cfg.dft.qe is None:
        raise ValueError("E3-QE requires Quantum ESPRESSO")
    if cfg.dft.qe.pw_template is not None:
        raise ValueError("Pin QE settings explicitly, without an overriding pw_template")
    qe = cfg.dft.qe
    if not qe.pw_bin.is_file() or not qe.pw_wrapper.is_file():
        raise ValueError("QE executable and wrapper must exist before approving a campaign")
    for field in ("ecutwfc_ry", "ecutrho_ry", "kpts", "occupations", "pseudopotentials"):
        if getattr(qe, field) is None:
            raise ValueError(f"E3-QE requires reviewed, explicit dft.qe.{field}")
    elements = {
        symbol
        for action in e3._action_space(material)
        for symbol in Formula(action["formula"]).count()
    }
    if missing := elements - set(qe.pseudopotentials or {}):
        raise ValueError(f"QE pseudopotential mapping lacks {sorted(missing)} for {material}")
    for filename in (qe.pseudopotentials or {}).values():
        if not (qe.pseudo_dir / filename).is_file():
            raise ValueError(f"QE pseudopotential file does not exist: {filename}")
    if qe.extra_control.get("calculation", "scf") != "scf":
        raise ValueError("QE audits must be exact-geometry SCF single points")
    if cfg.acquisition.strategy != "random" or cfg.acquisition.diversity_filter:
        raise ValueError("Use uniform random QE audits: strategy=random, diversity_filter=false")
    if cfg.acquisition.min_uncertainty_eV_per_A != 0:
        raise ValueError("Random audit scores are not uncertainties; the score floor must be zero")
    if cfg.trainer.force_error_trigger_eV_per_A is None:
        raise ValueError("An explicit measured force-error training trigger is required")
    if not cfg.trainer.enabled or not cfg.trainer.compare_after_training:
        raise ValueError("Enable training and independent incumbent/candidate comparison")
    if cfg.trainer.validation_set is None or cfg.trainer.validation_fraction != 0:
        raise ValueError("Provide an independent, pre-existing QE held-out validation set")
    if cfg.trainer.promotion_min_evaluated_frames < 2:
        raise ValueError("At least two independent held-out frames are required")
    if cfg.loop.n_iterations != 3:
        raise ValueError("E3-QE uses three iterations")
    if cfg.loop.max_dft_calculations != 3 * cfg.acquisition.n_select:
        raise ValueError(
            "Set the per-material DFT cap to three times the per-iteration audit count"
        )
    if not cfg.loop.fail_fast or not cfg.loop.resume:
        raise ValueError("E3-QE requires fail_fast=true and resume=true")
    for existing in ("e3-mlip-production", "e3-mlip-collaborator-5000"):
        if Path(cfg.loop.out_dir).resolve().is_relative_to((e3.REPO / "runs" / existing).resolve()):
            raise ValueError("E3-QE must not write into existing MLIP-only campaign results")


def check_resume(root: Path) -> None:
    """Do not let generic AL cleanup erase interrupted E3/QE evidence."""
    if (
        root.exists()
        and not (root / "e3-qe-policy.json").exists()
        and any(path.name != ".e3-qe.lock" for path in root.iterdir())
    ):
        raise RuntimeError("Output directory contains evidence without an E3-QE policy")
    for iteration in root.glob("iteration_*"):
        if (iteration / "generation-stop.json").exists():
            raise RuntimeError(
                f"Campaign scientifically stopped at {iteration}; a new generation "
                "decision requires reviewed authorization, not automatic resume."
            )
        state = iteration / "state.json"
        if not state.is_file() or json.loads(state.read_text())["status"] != "complete":
            raise RuntimeError(
                f"Interrupted iteration retained at {iteration}; automatic partial replay is "
                "disabled. Reconcile DFT attempts and datasets before an approved recovery."
            )


class E3Sampler:
    def __init__(self, material: str) -> None:
        self.material = material

    def __call__(self, cfg: ALConfig, iteration: int, it_dir: Path) -> list[Candidate]:
        snapshot = it_dir / "e3-geometry-config.json"
        e3._atomic_json(snapshot, cfg.model_dump(mode="json"))
        command = [
            sys.executable,
            "-m",
            "benchmarks.e3.qe_active_learning",
            "geometry",
            "--config",
            str(snapshot.resolve()),
            "--material",
            self.material,
            "--iteration",
            str(iteration),
            "--iteration-dir",
            str(it_dir.resolve()),
            "--approve-compute",
        ]
        try:
            e3._run_supervised_round(command, seconds=e3.ROUND_SECONDS)
        except subprocess.TimeoutExpired as error:
            e3._atomic_json(
                it_dir / "geometry-deadline.json",
                {
                    "status": "failed",
                    "reason": "generation_geometry_deadline",
                    "seconds": e3.ROUND_SECONDS,
                    "partial_evidence_retained": True,
                    "qe_dispatched": False,
                },
            )
            raise TimeoutError(
                "E3 generation/geometry exceeded its parent-enforced deadline"
            ) from error
        receipt = json.loads((it_dir / "e3-geometry-result.json").read_text())
        if receipt["status"] == "stopped_generation_exhausted":
            return []
        return self._candidates(it_dir, Path(receipt["tasks_path"]), receipt["active_model"])

    def _run_geometry(self, cfg: ALConfig, iteration: int, it_dir: Path) -> None:
        assert cfg.mlip.mace is not None
        root = Path(cfg.loop.out_dir)
        generation_root = root / "e3"
        manifest = e3.build_campaign_manifest()
        manifest["campaign_id"] = VARIANT
        manifest["campaign_variant"] = VARIANT
        spec = e3._find_round(
            manifest, self.material, "llm_guided_adaptation", e3.BOUNDED_SEED, iteration + 1
        )
        history = e3._history_for_round(
            generation_root,
            self.material,
            "llm_guided_adaptation",
            e3.BOUNDED_SEED,
            iteration + 1,
        )
        # QE errors are appended to the same action-level feedback seen by the LLM.
        if iteration:
            previous = json.loads(
                (root / f"iteration_{iteration - 1:04d}" / "state.json").read_text()
            )
            pool = json.loads(
                (root / f"iteration_{iteration - 1:04d}" / "e3-audit-pool.json").read_text()
            )
            action_by_id = {row["candidate_id"]: row["action_id"] for row in pool}
            audits: dict[str, list[float]] = {}
            for candidate_id, error in previous["dft_force_errors_eV_per_A"].items():
                audits.setdefault(action_by_id[candidate_id], []).append(error)
            history.append(
                {
                    "evidence_id": f"qe-audit-{iteration - 1}",
                    "qe_audits_by_action": {
                        action: {
                            "n_audited": len(errors),
                            "maximum_force_mae_eV_per_A": max(errors),
                        }
                        for action, errors in audits.items()
                    },
                    "accuracy_triggered": previous["accuracy_triggered"],
                    "model_promoted": previous["model_promoted"],
                    "round": iteration,
                    "active_model": json.loads(
                        (
                            root / f"iteration_{iteration - 1:04d}" / "e3-geometry-result.json"
                        ).read_text()
                    )["active_model"],
                    "note": (
                        "QE error is measured at audited geometries only; "
                        "not a global accuracy guarantee."
                    ),
                }
            )
        started = time.monotonic()
        try:
            manifest_path, tasks_path, round_manifest = e3._materialize_round(
                manifest, spec, generation_root, history, e3._chat_model()
            )
        except e3.AcquisitionExhausted as error:
            e3._atomic_json(it_dir / "generation-stop.json", {"reason": str(error)})
            e3._atomic_json(
                it_dir / "e3-geometry-result.json",
                {
                    "status": "stopped_generation_exhausted",
                },
            )
            log.warning("E3 generation exhausted: %s", error)
            return
        checkpoint = Path(cfg.mlip.mace.model).resolve()
        model = {"path": str(checkpoint), "sha256": e3._sha256_file(checkpoint)}
        tasks = e3._read_tasks(tasks_path)
        if len(tasks) != 5000:
            raise ValueError("Refusing an incomplete 5,000-structure E3 geometry batch")
        for task in tasks:
            task.update(
                campaign_variant=VARIANT,
                active_model=model,
                optimizer=e3._hard_optimizer_policy(e3.HARD_OPTIMIZATION_SECONDS),
            )
        e3._write_jsonl(tasks_path, tasks)
        round_manifest.update(campaign_variant=VARIANT, active_model=model)
        round_manifest["execution_optimizer"] = e3._hard_optimizer_policy(
            e3.HARD_OPTIMIZATION_SECONDS
        )
        round_manifest["model"] = "MACE E3-QE incumbent"
        round_manifest["mace_model_path"] = model["path"]
        round_manifest["mace_model"] = model["path"]
        round_manifest["checkpoint_sha256"] = model["sha256"]
        e3._atomic_json(manifest_path, round_manifest)
        round_dir = tasks_path.parent
        e3._execute_round_tasks(
            tasks_path,
            round_dir,
            e3.WORKERS_PER_ARRAY_ELEMENT,
            e3.GPU_COUNT_PER_NODE,
            sys.executable,
            hard_optimization_seconds=e3.HARD_OPTIMIZATION_SECONDS,
            round_deadline=started + e3.ROUND_SECONDS,
        )
        outcomes = e3._round_action_outcomes(tasks, round_dir / "tasks")
        for outcome in outcomes:
            outcome.update(active_model=model, round=iteration + 1)
            outcome.pop("evidence_id")
            outcome["evidence_id"] = hashlib.sha256(
                e3._canonical_json(outcome).encode()
            ).hexdigest()
        e3._atomic_json(
            round_dir / "round-summary.json",
            {
                "status": "complete",
                "campaign_variant": VARIANT,
                "active_model": model,
                "action_outcomes": outcomes,
            },
        )
        e3._atomic_json(
            it_dir / "e3-geometry-result.json",
            {
                "status": "complete",
                "tasks_path": str(tasks_path.resolve()),
                "active_model": model,
            },
        )

    def _candidates(self, it_dir: Path, tasks_path: Path, model: dict[str, str]) -> list[Candidate]:
        from ase.io import read

        tasks = e3._read_tasks(tasks_path)
        round_dir = tasks_path.parent
        candidates = []
        for task in tasks:
            path = e3._task_result(round_dir / "tasks" / task["candidate_id"], task)
            if path is None:
                raise RuntimeError(f"Missing E3 result for {task['candidate_id']}")
            result = json.loads(path.read_text())
            if result["status"] == "error":
                raise RuntimeError(f"E3 geometry task failed: {result}")
            final = result.get("optimized_structure_path")
            atoms = read(final or task["structure_path"])
            atoms.calc = None
            atoms.info.update(
                sampling_method="e3_geometry_optimization",
                e3_action_id=task["action_id"],
                e3_geometry_stage="endpoint" if final else "initial_after_timeout",
                incumbent_checkpoint_sha256=model["sha256"],
            )
            candidates.append(
                Candidate(
                    candidate_id=task["candidate_id"],
                    atoms=atoms,
                    seed_path=task["structure_path"],
                    md_step=0,
                )
            )
        n_initial = sum(c.atoms.info["e3_geometry_stage"] != "endpoint" for c in candidates)
        if n_initial:
            log.warning(
                "%d timed-out tasks enter the audit pool at their initial geometry", n_initial
            )
        e3._atomic_json(
            it_dir / "e3-audit-pool.json",
            [
                {
                    "candidate_id": c.candidate_id,
                    "action_id": c.atoms.info["e3_action_id"],
                    "geometry_stage": c.atoms.info["e3_geometry_stage"],
                }
                for c in candidates
            ],
        )
        return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "run", "geometry"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--material", choices=sorted(e3.MATERIALS), required=True)
    parser.add_argument("--approve-compute", action="store_true")
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--iteration-dir", type=Path)
    args = parser.parse_args(argv)
    profile = os.environ.setdefault("MATSIM_E3_MLIP_STRUCTURES_PER_ROUND", "5000")
    if profile != "5000":
        raise ValueError("E3-QE uses the separate 5,000-structure profile")
    cfg = ALConfig.from_yaml(args.config)
    if args.command == "geometry":
        if (
            not args.approve_compute
            or not os.environ.get("SLURM_JOB_ID")
            or args.iteration not in range(3)
            or args.iteration_dir is None
        ):
            raise RuntimeError(
                "Internal geometry stage requires authorized allocation and iteration"
            )
        E3Sampler(args.material)._run_geometry(cfg, args.iteration, args.iteration_dir)
        return 0
    validate_config(cfg, args.material)
    root = Path(cfg.loop.out_dir)
    check_resume(root)
    policy = {
        "campaign_variant": VARIANT,
        "material": args.material,
        "max_structures_per_iteration": 5000,
        "al_config": cfg.model_dump(mode="json"),
        "dft_input_identities": scientific_dft_payload(
            cfg.dft, set(cfg.dft.qe.pseudopotentials or {})
        ),
        "dft_wrapper_identity": path_identity(cfg.dft.qe.pw_wrapper),
        "validation_identity": path_identity(cfg.trainer.validation_set),
        "elemental_reference_identity": path_identity(cfg.trainer.validation_reference_set),
        "objectives_identity": path_identity(e3.OBJECTIVES),
        "source_sha256": {
            path: e3._sha256_file(e3.REPO / path)
            for path in (
                "benchmarks/e3/qe_active_learning.py",
                "benchmarks/e3/production.py",
                "benchmarks/e3/structure_generation.py",
                "benchmarks/e3/feasibility.py",
                "benchmarks/e3/profiles.py",
                "benchmarks/e3/scientific-objectives.yaml",
                "src/matsim_agents/active_learning/loop.py",
                "src/matsim_agents/active_learning/config.py",
            )
        },
    }
    # Normalize tuples in the resolved DFT payload to their on-disk JSON form.
    policy = json.loads(json.dumps(policy))
    policy_path = root / "e3-qe-policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
        raise ValueError("Cannot change E3-QE scientific or resource policy on resume")
    if args.command == "check":
        print(json.dumps(policy, indent=2))
        return 0
    if not args.approve_compute or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run requires --approve-compute inside a reviewed Slurm allocation")
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".e3-qe.lock", timeout=0):
        check_resume(root)
        if policy_path.exists() and json.loads(policy_path.read_text()) != policy:
            raise ValueError("Cannot change E3-QE scientific or resource policy on resume")
        e3._atomic_json(policy_path, policy)
        run_active_learning(cfg, candidate_sampler=E3Sampler(args.material))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
