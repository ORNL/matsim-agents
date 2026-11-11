"""Zero-shot MLIP energies/forces for E3, run inside the backend's isolated venv.

MACE and UMA need incompatible e3nn versions, so each runs in its own environment
(``.venv-mace``, ``.venv-uma``). This module imports only ASE and the selected
backend. It never trains, relaxes or modifies structures; it writes per-frame
total energies (eV) and forces (eV/A) keyed by ``info['frame_id']``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read

MODELS = {
    # Frozen E3 zero-shot configurations.
    "mace-mp-0-small": {"backend": "mace", "model": "small", "dtype": "float64"},
    "uma-s-1p1-omat": {"backend": "uma", "model": "uma-s-1p1", "task": "omat"},
    # mace-foundations checkpoints (mace-torch 0.3.16 aliases).
    "mace-mp-0-medium": {"backend": "mace", "model": "medium", "dtype": "float64"},
    "mace-mp-0-large": {"backend": "mace", "model": "large", "dtype": "float64"},
    "mace-mp-0b3-medium": {"backend": "mace", "model": "medium-0b3", "dtype": "float64"},
    "mace-mpa-0-medium": {"backend": "mace", "model": "medium-mpa-0", "dtype": "float64"},
    "mace-omat-0-small": {"backend": "mace", "model": "small-omat-0", "dtype": "float64"},
    "mace-omat-0-medium": {"backend": "mace", "model": "medium-omat-0", "dtype": "float64"},
    "mace-matpes-pbe-0": {"backend": "mace", "model": "mace-matpes-pbe-0", "dtype": "float64"},
    # r2SCAN energies are not on the PBE scale of the DFT references.
    "mace-matpes-r2scan-0": {
        "backend": "mace", "model": "mace-matpes-r2scan-0", "dtype": "float64"
    },
    "mace-mh-1-omat-pbe": {
        "backend": "mace", "model": "mh-1", "head": "omat_pbe", "dtype": "float64"
    },
    "mace-mh-1-mp-pbe": {
        "backend": "mace", "model": "mh-1", "head": "mp_pbe_refit_add", "dtype": "float64"
    },
    "mace-mh-0-omat-pbe": {
        "backend": "mace", "model": "mh-0", "head": "omat_pbe", "dtype": "float64"
    },
}


def calculator(name: str, device: str):
    spec = MODELS[name]
    if spec["backend"] == "mace":
        from mace.calculators import mace_mp

        head = {"head": spec["head"]} if "head" in spec else {}
        return mace_mp(model=spec["model"], device=device, default_dtype=spec["dtype"], **head)
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    unit = pretrained_mlip.get_predict_unit(spec["model"], device=device)
    return FAIRChemCalculator(unit, task_name=spec["task"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    frames = read(args.input, index=":")
    calc = calculator(args.model, args.device)
    started = time.perf_counter()
    rows = []
    for atoms in frames:
        atoms = atoms.copy()
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            raise ValueError(f"non-finite prediction for {atoms.info['frame_id']}")
        rows.append(
            {
                "frame_id": atoms.info["frame_id"],
                "energy_eV": energy,
                "forces_eV_A": forces.tolist(),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "model": args.model,
                "config": MODELS[args.model],
                "device": args.device,
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "seconds": time.perf_counter() - started,
                "units": {"energy": "eV", "forces": "eV/A"},
                "frames": rows,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
