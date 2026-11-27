"""Fine-tune an MLIP on a small labelled Cu set, then re-export it for LAMMPS ML-IAP.

The labels are EMT energies and forces, not DFT, so this checks the pipeline (fine-tune,
checkpoint, calculator reload, parity and NVE under LAMMPS), not model accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from matsim_agents.core.seeding import numpy_rng

PROJ = Path(__file__).resolve().parents[3]
MACE_SMALL = "20231210mace128L0_energy_epoch249model"


def make_inputs(root: Path, frames: int, seed: int) -> tuple[Path, Path]:
    from matsim_agents.data.governance import DatasetValidationSummary, write_dataset_manifest

    root.mkdir(parents=True, exist_ok=True)
    ref = bulk("Cu", "fcc", a=3.6)
    ref.calc = EMT()
    ref_energy = float(ref.get_potential_energy())
    ref_path = root / "elemental-Cu.extxyz"
    write(ref_path, ref)
    manifest = root / "elemental-references.json"
    manifest.write_text(
        json.dumps(
            {
                "backend": "qe",
                "method_signature": "emt-synthetic",
                "references": {
                    "Cu": {
                        "structure_path": ref_path.name,
                        "energy_eV": ref_energy,
                        "structure_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )
    rng = numpy_rng(seed, "finetune-inputs")
    out = []
    for _ in range(frames):
        atoms = bulk("Cu", "fcc", a=float(rng.uniform(3.5, 3.7)), cubic=True).repeat((2, 2, 2))
        atoms.rattle(0.05, seed=int(rng.integers(1 << 31)))
        atoms.calc = EMT()
        atoms.calc = SinglePointCalculator(
            atoms, energy=atoms.get_potential_energy(), forces=atoms.get_forces()
        )
        out.append(atoms)
    dataset = root / "dataset.extxyz"
    write(dataset, out)
    reference = json.loads(manifest.read_text())
    write_dataset_manifest(
        dataset,
        dft_backend=reference["backend"],
        method_signature=reference["method_signature"],
        energy_reference=f"{reference['backend']}:native_total_energy",
        validation=DatasetValidationSummary(),
    )
    return dataset, manifest


def finetune_hydragnn_case(dataset: Path, manifest: Path, out: Path, args):
    from matsim_agents.backends.mlip.config import HydraGNNConfig
    from matsim_agents.backends.mlip.training.finetune_hydragnn_newhead import (
        finetune_hydragnn_newhead,
    )

    src = Path(os.environ.get("HYDRAGNN_DIR", str(PROJ / "HydraGNN-e4b-frozen-81d5b883")))
    example = Path(os.environ.get("HYDRAGNN_WEIGHTS", str(PROJ / "HydraGNN")))
    example = example / "examples/multidataset_hpo_sc26"
    gfm = example / "multidataset_hpo-BEST6-fp64"
    ckpt = "multidataset_hpo-BEST6-fp64_epoch_97.pk"
    logdir = finetune_hydragnn_newhead(
        dataset,
        out,
        gfm_logdir=gfm,
        strategy="frozen",
        head="OMat24",
        gfm_checkpoint=ckpt,
        epochs=args.epochs,
        seed=args.seed,
        hydragnn_root=src,
        device="cpu",
        elemental_reference_manifest=manifest,
    )
    cfg = HydraGNNConfig(
        logdir=gfm,
        checkpoint=ckpt,
        inference_head="OMat24",
        hydragnn_root=src,
        precision="fp64",
        device="cpu",
    )
    return cfg, logdir


def finetune_mace_case(dataset: Path, manifest: Path, out: Path, args):
    from matsim_agents.backends.mlip.config import MACEConfig
    from matsim_agents.backends.mlip.training.finetune_mace import finetune_mace

    base = Path(os.environ.get("MACE_CACHE", "~/.cache/mace")).expanduser() / MACE_SMALL
    checkpoint = finetune_mace(
        dataset,
        out,
        elemental_reference_manifest=manifest,
        family="checkpoint",
        base_model=str(base),
        precision="fp64",
        device="cpu",
        epochs=args.epochs,
        lora=args.lora,
        seed=args.seed,
    )
    cfg = MACEConfig(family="checkpoint", model=str(base), precision="fp64", device="cpu")
    return cfg, checkpoint


def finetune_uma_case(dataset: Path, manifest: Path, out: Path, args):
    from matsim_agents.backends.mlip.config import UMAConfig
    from matsim_agents.backends.mlip.training.finetune_uma import finetune_uma

    model = os.environ.get("UMA_MODEL", "uma-s-1p1")
    checkpoint = finetune_uma(
        dataset,
        out,
        elemental_reference_manifest=manifest,
        base_model=model,
        device="cpu",
        epochs=args.epochs,
        lora=args.lora,
        seed=args.seed,
    )
    return UMAConfig(model_name=model, device="cpu"), checkpoint


CASES = {
    "hydragnn": finetune_hydragnn_case,
    "mace": finetune_mace_case,
    "uma": finetune_uma_case,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(CASES), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora", action="store_true", help="MACE and UMA only")
    ap.add_argument("--seed", type=int, required=True, help="root seed of the run")
    ap.add_argument("--md-steps", type=int, default=20)
    args = ap.parse_args()

    from matsim_agents.backends.external.lammps.mliap.export import export_checkpoint_for_lammps

    dataset, manifest = make_inputs(args.out / "inputs", args.frames, args.seed)
    cfg, checkpoint = CASES[args.backend](dataset, manifest, args.out / "ft", args)
    record = export_checkpoint_for_lammps(
        cfg,
        checkpoint,
        args.out / "export",
        species=["Cu"],
        seed=args.seed,
        md_steps=args.md_steps,
    )
    record["backend"] = args.backend
    record["finetune_checkpoint"] = str(checkpoint)
    record["labels"] = "EMT energies and forces on 2x2x2 Cu, not DFT"
    (args.out / "finetune-export.json").write_text(json.dumps(record, indent=2, default=float))
    print(json.dumps({k: record[k] for k in ("backend", "approved", "checkpoint_sha256")}))
    return 0 if record["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
