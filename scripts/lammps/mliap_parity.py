"""ASE vs LAMMPS ML-IAP parity for one MLIP; run after sourcing activate-mliap.sh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk


def build_calculator(kind: str, model: str, zbl: bool):
    if kind == "emt":
        from ase.calculators.emt import EMT

        return EMT(), "emt", None
    if kind == "mace":
        from matsim_agents.backends.mlip.calculator import build_mace_calculator
        from matsim_agents.backends.mlip.config import MACEConfig

        calc = build_mace_calculator(MACEConfig(model=model, device="cpu", precision="fp64"))
        ckpt = Path(model).expanduser()
        return calc, "mace", ckpt if ckpt.is_file() else None
    raise SystemExit(f"unsupported --calc {kind}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calc", choices=["emt", "mace"], required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--element", default="Cu")
    ap.add_argument("--zbl", action="store_true")
    ap.add_argument("--a", type=float, default=3.6)
    ap.add_argument("--rattle", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from matsim_agents.backends.external.lammps.mliap.parity import (
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
    atoms.rattle(args.rattle, seed=7)
    result = parity_check(atoms, calc, [args.element])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"label": label, "zbl": args.zbl, "result": result.__dict__}
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
    return 0 if result.passed else 1


if __name__ == "__main__":
    np.seterr(all="raise")
    sys.exit(main())
