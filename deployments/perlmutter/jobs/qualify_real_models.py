"""Bounded real-model/QE qualification; no mock labels or scientific accuracy claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import read, write

from matsim_agents.backends.mlip.calculator import build_ensemble, make_mlip_calculator
from matsim_agents.active_learning.candidates import Candidate
from matsim_agents.active_learning.config import ALConfig
from matsim_agents.backends.mlip.config import HydraGNNConfig, MACEConfig, MLIPConfig
from matsim_agents.datasets.governance import structure_identity
from matsim_agents.backends.dft.base import DFTJobSpec, make_backend
from matsim_agents.active_learning.evaluate import evaluate_promotion_candidate
from matsim_agents.active_learning.loop import _dft_method_signature, run_active_learning
from matsim_agents.active_learning.trainer import retrain_hydragnn, retrain_mace
from matsim_agents.active_learning.uncertainty import score_ensemble
from matsim_agents.discovery.wrapper import explore_composition


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_metrics(comparison: dict) -> None:
    for role in ("candidate_metrics", "incumbent_metrics"):
        metrics = comparison.get(role)
        require(isinstance(metrics, dict) and bool(metrics), f"Missing {role}")
        for key in (
            "energy_mae_eV_per_atom",
            "formation_energy_mae_eV_per_atom",
            "force_mae_eV_per_A",
        ):
            value = metrics.get(key)
            require(value is not None and np.isfinite(value), f"Invalid {role}.{key}")


def identities(path: Path) -> set[str]:
    return {structure_identity(frame) for frame in read(path, index=":")}


def config(repo: Path, root: Path, backend: str = "qe") -> ALConfig:
    source = Path(__file__).resolve().parents[3]
    dft = {
        "backend": backend,
        "max_concurrent_jobs": 1,
    }
    if backend == "vasp":
        dft["vasp"] = {
            "vasp_bin": str(repo / "external/vasp6/src/vasp.6.6.1/bin/vasp_std"),
            "vasp_wrapper": str(
                source / "deployments/perlmutter/launchers/_vasp-step-perlmutter.sh"
            ),
            "incar_template": str(source / "examples/active_learning/INCAR.template"),
            "potcar_dir": str(repo / "external/vasp6/potcar/potpaw_PBE.64"),
            "nodes_per_job": 1,
            "ranks_per_node": 4,
            "threads_per_rank": 16,
            "timeout_sec": 1800,
            "extra_incar": {"LREAL": ".FALSE.", "KPAR": "1"},
        }
    elif backend == "qe":
        dft["qe"] = {
            "pw_bin": str(repo / "external/quantum-espresso/install-gpu/bin/pw.x"),
            "pw_wrapper": str(source / "deployments/perlmutter/launchers/_qe-step-perlmutter.sh"),
            "pseudo_dir": str(repo / "external/quantum-espresso/src/pseudo"),
            "pseudopotentials": {"Si": "Si_r.upf"},
            "ecutwfc_ry": 50,
            "ecutrho_ry": 400,
            "kpts": [2, 2, 2],
            "occupations": "fixed",
            "nodes_per_job": 1,
            "ranks_per_node": 4,
            "threads_per_rank": 16,
            "timeout_sec": 1800,
        }
    else:
        raise ValueError(f"Unknown DFT backend: {backend}")
    return ALConfig.model_validate(
        {
            "mlip": {"backend": "uma", "uma": {"device": "cuda"}},
            "md": {
                "seed_source": {"kind": "paths", "paths": [str(root / "Si.extxyz")]},
                "n_steps": 20,
                "sample_every": 5,
                "temperature_K": 300,
                "random_seed": 42,
            },
            "acquisition": {"strategy": "random", "n_select": 2, "diversity_filter": False},
            "dft": dft,
            "trainer": {
                "enabled": True,
                "train_script": str(source / "src/matsim_agents/training/finetune_uma.py"),
                "epochs_per_iter": 1,
                "nodes_for_train": 1,
                "ranks_per_node": 1,
                "validation_fraction": 0.2,
                "compare_after_training": True,
                "promote_model": True,
                "promotion_approved": True,
                "validation_reference_set": str(root / "elemental.json"),
            },
            "loop": {
                "out_dir": str(root / "al"),
                "n_iterations": 1,
                "max_dft_calculations": 6,
                "resume": True,
                "fail_fast": True,
            },
        }
    )


def bootstrap(repo: Path, root: Path, backend: str = "qe") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / "al").exists(), "Bootstrap cannot overwrite an existing AL run")
    require(not (root / "elemental.json").exists(), "Bootstrap reference already exists")
    seed_path = root / "Si.extxyz"
    write(seed_path, bulk("Si", "diamond", a=5.43))
    # Construct the DFT config without fabricating an elemental reference label.
    payload = {
        "backend": backend,
        "method_signature": "pending-real-dft",
        "references": {},
    }
    manifest = root / "elemental.json"
    manifest.write_text(json.dumps(payload))
    cfg = config(repo, root, backend)
    result = make_backend(cfg.dft).run_one(
        DFTJobSpec("elemental-Si", read(seed_path), str(root / f"reference-{backend}"))
    )
    require(
        result.converged
        and result.energy_eV is not None
        and np.isfinite(result.energy_eV)
        and result.forces_eV_per_A is not None
        and np.isfinite(result.forces_eV_per_A).all(),
        f"Elemental {backend} reference failed: {result.notes}",
    )
    payload["method_signature"] = _dft_method_signature(cfg, {"Si"})
    payload["references"] = {
        "Si": {
            "structure_path": seed_path.name,
            "structure_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
            "energy_eV": result.energy_eV,
        }
    }
    manifest.write_text(json.dumps(payload, indent=2))
    cfg = config(repo, root, backend)
    (root / "al-config.json").write_text(cfg.model_dump_json(indent=2))
    explored = explore_composition(
        "Si",
        output_dir=str(root / "budget-check"),
        mlip_backend="uma",
        n_random=0,
        max_relaxations=1,
        maxiter=5,
        mlp_device="cuda",
    )
    require(explored.candidate_counts["attempted"] == 1, "Real relaxation cap not exercised")
    require(bool(explored.relaxations), "The real UMA relaxation failed to return")
    require(explored.relaxation_budget_exhausted, "Need multiple Si prototype seeds for cap test")
    (root / "budget-check.json").write_text(explored.model_dump_json(indent=2))
    run_active_learning(cfg)
    state = json.loads((root / "al/iteration_0000/state.json").read_text())
    require(state["training_status"] == "deferred", "Two-label bootstrap did not defer")
    require(not state["model_promoted"], "Bootstrap unexpectedly promoted")
    require(state["n_training_frames_total"] == 1, "Expected one bootstrap training label")
    require(state["n_validation_frames_total"] == 1, "Expected one bootstrap held-out label")
    return {"bootstrap_state": state, "budget_counts": explored.candidate_counts}


def resume(root: Path) -> dict:
    cfg = ALConfig.model_validate_json((root / "al-config.json").read_text())
    training = root / "al/dataset.extxyz"
    validation = root / "al/validation.extxyz"
    old_training = identities(training)
    old_validation = identities(validation)
    original = json.loads((root / "al/iteration_0000/state.json").read_text())
    for iteration in (1, 2):
        cfg.md.random_seed = 42 + iteration
        cfg.loop.n_iterations = iteration + 1
        run_active_learning(cfg)
    require(old_training <= identities(training), "Restart moved old training membership")
    require(old_validation <= identities(validation), "Restart moved old validation membership")
    require(identities(training).isdisjoint(identities(validation)), "Held-out leakage")
    require(
        json.loads((root / "al/iteration_0000/state.json").read_text()) == original,
        "Restart modified completed iteration evidence",
    )
    states = [
        json.loads((root / f"al/iteration_{index:04d}/state.json").read_text())
        for index in range(3)
    ]
    require(
        all(
            state["status"] == "complete"
            and state["n_dft_converged"] == 2
            and state["n_dft_failed"] == 0
            for state in states
        ),
        "Each iteration must complete two real DFT labels",
    )
    require(len(identities(training) | identities(validation)) == 6, "Labels did not accumulate")
    require(
        any(state["training_status"] == "completed" for state in states[1:]),
        "No real training completed after accumulation",
    )
    comparisons = [state["model_comparison"] for state in states[1:] if state["model_comparison"]]
    require(bool(comparisons), "No held-out comparison produced")
    for comparison in comparisons:
        require_metrics(comparison)
    return {"states": states, "training_count": len(identities(training))}


def mace(repo: Path, root: Path) -> dict:
    cfg = ALConfig.model_validate_json((root / "al-config.json").read_text())
    cache = repo.parent / "models/artifacts/mace/mace"
    primary = cache / "macempa0mediummodel"
    member = cache / "20231203mace128L1_epoch199model"
    require(primary.is_file() and member.is_file(), "MACE ensemble assets unavailable")
    cfg.mlip = MLIPConfig(
        backend="mace",
        mace=MACEConfig(
            family="checkpoint",
            model=str(primary),
            device="cuda",
            precision="fp64",
            ensemble_models=[str(member)],
        ),
    )
    frame = read(root / "Si.extxyz")
    energies = []
    calculators = build_ensemble(cfg.mlip)
    scores = score_ensemble(
        [Candidate("Si-ensemble", frame, str(root / "Si.extxyz"), 0)], calculators
    )
    require(scores.shape == (1,) and np.isfinite(scores).all(), "Invalid ensemble scores")
    for calc in calculators:
        atoms = frame.copy()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        require(np.isfinite(energy) and np.isfinite(forces).all(), "Nonfinite MACE prediction")
        energies.append(energy)
    require(len(energies) == 2, "MACE ensemble member missing")
    cfg.trainer.train_script = (
        Path(__file__).resolve().parents[3] / "src/matsim_agents/training/finetune_mace.py"
    )
    cfg.trainer.validation_fraction = 0
    cfg.trainer.validation_set = root / "al/validation.extxyz"
    cfg.mlip.mace.ensemble_models = []
    checkpoint = retrain_mace(
        cfg.trainer, cfg.mlip.mace, root / "al/dataset.extxyz", 1, root / "mace-training"
    )
    require(checkpoint.is_file(), "MACE training did not produce an inference checkpoint")
    decision = evaluate_promotion_candidate(
        cfg, str(checkpoint), iteration=1, training_set=root / "al/dataset.extxyz"
    )
    # Rejection is valid evidence; failed/nonfinite inference is not.
    require_metrics(asdict(decision))
    candidate_cfg = cfg.mlip.model_copy(deep=True)
    candidate_cfg.mace.model = str(checkpoint)
    frame.calc = make_mlip_calculator(candidate_cfg)
    require(
        np.isfinite(frame.get_potential_energy()) and np.isfinite(frame.get_forces()).all(),
        "MACE checkpoint reload failed",
    )
    return {
        "ensemble_energies_eV": energies,
        "force_disagreement_eV_per_A": scores.tolist(),
        "promotion": asdict(decision),
    }


def hydragnn(repo: Path, root: Path) -> dict:
    logdir = repo / "HydraGNN/examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64"
    require((logdir / "config.json").is_file(), "HydraGNN logdir missing")
    predictions = {}
    for precision in ("fp64", "bf16"):
        cfg = MLIPConfig(
            backend="hydragnn",
            hydragnn=HydraGNNConfig(
                logdir=logdir,
                checkpoint="multidataset_hpo-BEST6-fp64_epoch_97.pk",
                inference_head="OMat24",
                precision=precision,
            ),
        )
        calc = make_mlip_calculator(cfg)
        atoms = bulk("Si", "diamond", a=5.43)
        atoms.calc = calc
        energies = []
        for displacement in (0.0, 0.01):
            atoms.positions[0, 0] += displacement
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            require(
                np.isfinite(energy) and np.isfinite(forces).all(),
                f"HydraGNN {precision} inference nonfinite",
            )
            energies.append(energy)
        predictions[precision] = energies
    return {"pinned_head": "OMat24", "checkpoint_logdir": str(logdir), "energies_eV": predictions}


def hydragnn_finetune(repo: Path, root: Path) -> dict:
    """Fine-tune with both built-in HydraGNN trainers, reload, infer, and relax."""
    import os

    from matsim_agents.backends.mlip.relaxation import RelaxStructureInput, relax_with_mlip

    cfg = ALConfig.model_validate_json((root / "al-config.json").read_text())
    dataset = root / "al/dataset.extxyz"
    require(dataset.is_file(), "Run the bootstrap and resume stages first")
    source = Path(__file__).resolve().parents[3]
    hydragnn_root = repo / "HydraGNN"
    example = hydragnn_root / "examples/multidataset_hpo_sc26"
    gfm = example / "multidataset_hpo-BEST6-fp64"
    branch_mlp = example / "mlp_branch_weights.pt"
    ft_repo = Path(
        os.environ.get("HYDRAGNN_FT_REPO", repo.parent / "HydraGNN_GFM_FineTuning4Materials")
    )
    require((gfm / "config.json").is_file() and branch_mlp.is_file(), "HydraGNN assets missing")
    seed = root / "Si.extxyz"
    results: dict = {}

    def relax(label: str, **kwargs) -> dict:
        outcome = relax_with_mlip(
            RelaxStructureInput(
                structure_path=str(seed),
                output_dir=str(root / f"relax-{label}"),
                mlip_backend="hydragnn",
                maxiter=20,
                random_displacement=True,
                **kwargs,
            )
        )
        require(
            np.isfinite(outcome.final_energy_eV) and np.isfinite(outcome.final_max_force_eV_per_A),
            f"HydraGNN {label} relaxation returned nonfinite values",
        )
        require(outcome.num_steps >= 1, f"HydraGNN {label} relaxation took no steps")
        return outcome.model_dump(mode="json")

    results["relax_pretrained_routed"] = relax(
        "pretrained-routed",
        logdir=str(gfm),
        checkpoint="multidataset_hpo-BEST6-fp64_epoch_97.pk",
        hydragnn_branch_mlp_checkpoint=str(branch_mlp),
    )
    for script, extra in (
        ("finetune_hydragnn_newhead", {"ft_repo": ft_repo}),
        ("finetune_hydragnn", {"hydragnn_branch_mlp_checkpoint": branch_mlp}),
    ):
        out = root / f"hydragnn-{script}"
        require(not out.exists(), f"{out} already exists")
        trainer = cfg.trainer.model_copy(
            update={
                "train_script": source / f"src/matsim_agents/training/{script}.py",
                "train_launcher": None,
                "epochs_per_iter": 1,
                "nodes_for_train": 1,
                "ranks_per_node": 1,
            },
            deep=True,
        )
        hydragnn = HydraGNNConfig(
            logdir=gfm,
            checkpoint="multidataset_hpo-BEST6-fp64_epoch_97.pk",
            hydragnn_root=hydragnn_root,
            **extra,
        )
        logdir = retrain_hydragnn(
            trainer,
            hydragnn,
            dataset,
            1,
            out,
            dft_config=cfg.dft,
            reference_root=root / f"hydragnn-{script}-references",
        )
        calculator = make_mlip_calculator(
            MLIPConfig(
                backend="hydragnn",
                hydragnn=hydragnn.model_copy(update={"logdir": logdir, "checkpoint": None}),
            )
        )
        atoms = read(seed)
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        require(np.isfinite(energy) and np.isfinite(forces).all(), f"{script} inference nonfinite")
        results[script] = {
            "logdir": str(logdir),
            "energy_eV": energy,
            "max_force_eV_per_A": float(np.linalg.norm(forces, axis=1).max()),
            "relaxation": relax(script, logdir=str(logdir)),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=["bootstrap", "resume", "mace", "hydragnn", "hydragnn-finetune"],
        required=True,
    )
    parser.add_argument("--dft-backend", choices=["qe", "vasp"], default="qe")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    if args.stage == "bootstrap":
        result = bootstrap(args.repo, args.root, args.dft_backend)
    elif args.stage == "resume":
        result = resume(args.root)
    elif args.stage == "mace":
        result = mace(args.repo, args.root)
    elif args.stage == "hydragnn":
        result = hydragnn(args.repo, args.root)
    else:
        result = hydragnn_finetune(args.repo, args.root)
    (args.root / f"{args.stage}-verified.json").write_text(json.dumps(result, indent=2))
    print(f"VERIFIED {args.stage}: {args.root}")


if __name__ == "__main__":
    main()
