"""ASE vs LAMMPS ML-IAP parity for one MLIP; run after sourcing activate-mliap.sh."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk

from matsim_agents.core.seeding import numpy_rng, stream_seed

MACE_SMALL = "20231210mace128L0_energy_epoch249model"


def build_calculator(kind: str, model: str, zbl: bool):
    if kind == "emt":
        from ase.calculators.emt import EMT

        return EMT(), "emt", None
    if kind == "mace":
        from matsim_agents.backends.mlip.calculator import build_mace_calculator
        from matsim_agents.backends.mlip.config import MACEConfig

        calc = build_mace_calculator(MACEConfig(model=model, device="cpu", precision="fp64"))
        ckpt = Path(model).expanduser()
        if not ckpt.is_file() and model == "small":
            ckpt = Path(os.environ.get("MACE_CACHE", "~/.cache/mace")).expanduser() / MACE_SMALL
        return calc, "mace", ckpt if ckpt.is_file() else None
    if kind == "hydragnn":
        from matsim_agents.backends.mlip.calculator import build_hydragnn_calculator
        from matsim_agents.backends.mlip.config import HydraGNNConfig

        PS = Path(__file__).resolve().parents[3]
        root = Path(os.environ.get("HYDRAGNN_DIR", str(PS / "HydraGNN-e4b-frozen-81d5b883")))
        ex = Path(os.environ.get("HYDRAGNN_WEIGHTS", str(PS / "HydraGNN")))
        ex = ex / "examples/multidataset_hpo_sc26"
        logdir = Path(model) if model != "small" else ex / "multidataset_hpo-BEST6-fp64"
        ckpt = logdir / "multidataset_hpo-BEST6-fp64_epoch_97.pk"
        cfg = HydraGNNConfig(
            logdir=logdir,
            checkpoint=str(ckpt),
            inference_head="OMat24",
            hydragnn_root=root,
            precision="fp64",
            device="cpu",
        )
        return build_hydragnn_calculator(cfg), "hydragnn", ckpt
    if kind == "uma":
        from matsim_agents.backends.mlip.calculator import build_uma_calculator
        from matsim_agents.backends.mlip.config import UMAConfig

        cfg = UMAConfig(model_name=model, device="cpu")
        ckpt = Path(os.environ["UMA_CHECKPOINT"]) if "UMA_CHECKPOINT" in os.environ else None
        return build_uma_calculator(cfg), "uma", ckpt
    raise SystemExit(f"unsupported --calc {kind}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", choices=["emt", "mace", "hydragnn", "uma"], required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--element", default="Cu")
    ap.add_argument("--zbl", action="store_true")
    ap.add_argument("--a", type=float, default=3.6)
    ap.add_argument("--rattle", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True, help="root seed of the run")
    ap.add_argument("--md-steps", type=int, default=0, help="NVE steps for the stability check")
    ap.add_argument("--md-temperature", type=float, default=300.0, help="K, initial velocities")
    ap.add_argument("--md-timestep-fs", type=float, default=1.0)
    ap.add_argument(
        "--pka-eV",
        type=float,
        default=0.0,
        help="add a recoil of this energy to atom 0 along <110> before the MD stage (WP1 smoke)",
    )
    args = ap.parse_args()

    from matsim_agents.backends.external.lammps.mliap.parity import (
        md_stability_check,
        parity_check,
        write_export_record,
    )
    from matsim_agents.science.cascade.zbl import build_short_range_calculator

    calc, label, ckpt = build_calculator(args.calc, args.model, args.zbl)
    short_range = None
    if args.zbl:
        calc, record = build_short_range_calculator(
            calc,
            native_short_range=False,
            additive_zbl=True,
            justification="parity check of the additive ZBL path",
        )
        short_range = {"policy": record.policy, "r_in": record.r_in, "r_out": record.r_out}

    atoms = bulk(args.element, "fcc", a=args.a, cubic=True).repeat((3, 3, 3))
    atoms.rattle(args.rattle, seed=stream_seed(args.seed, "rattle"))
    result = parity_check(atoms, calc, [args.element])
    payload = {"label": label, "zbl": args.zbl, "result": result.__dict__}
    md_passed = True
    if args.md_steps > 0:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

        md_atoms = atoms.copy()
        MaxwellBoltzmannDistribution(
            md_atoms, temperature_K=args.md_temperature, rng=numpy_rng(args.seed, "velocities")
        )
        Stationary(md_atoms)
        if args.pka_eV > 0:
            from matsim_agents.science.cascade.recoil import crystal_direction, initialize_recoil

            recoil = initialize_recoil(
                md_atoms,
                0,
                args.pka_eV,
                crystal_direction(md_atoms, (1, 1, 0)),
                expected_symbol=args.element,
                convention="add",
            )
            payload["pka"] = {
                "energy_eV": recoil.realized_energy_eV,
                "direction": recoil.direction,
                "convention": recoil.convention,
            }
        md = md_stability_check(
            md_atoms,
            calc,
            [args.element],
            steps=args.md_steps,
            timestep_fs=args.md_timestep_fs,
        )
        payload["md"] = md.__dict__
        md_passed = md.passed
        print(json.dumps(payload["md"], indent=2, default=float))
    rank, size = 0, 1
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        from mpi4py import MPI

        rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
        if size != int(os.environ["SLURM_NTASKS"]):
            raise SystemExit(f"MPI world size {size} != SLURM_NTASKS: launcher/MPI mismatch")
    payload["mpi_ranks"] = size
    if rank != 0:  # every rank ran the same collective checks; only rank 0 reports
        return 0 if result.passed and md_passed else 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if ckpt is not None:
        payload["export"] = write_export_record(
            args.out.with_suffix(".export.json"),
            checkpoint=ckpt,
            species=[args.element],
            short_range=short_range,
            parity=result,
        )
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(json.dumps(payload["result"], indent=2, default=float))
    return 0 if result.passed and md_passed else 1


if __name__ == "__main__":
    np.seterr(all="raise")
    sys.exit(main())
