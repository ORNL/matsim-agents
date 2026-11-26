"""Run a shard of a verification cascade study (threshold scan, size study or alloy) with MACE-MP-0.

Every shard builds the same task list from the arguments and runs the tasks whose id modulo
``--shards`` equals ``--shard``. Finished tasks are skipped, so a rerun resumes. The potential is
an unqualified foundation model; results are measurements of that model, not qualified cascades.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from matsim_agents.science.cascade.zbl import (
    build_short_range_calculator,
    check_pair_curve,
    pair_curve,
)
from matsim_agents.workflows.radiation.cascade_study import (
    plan_cascades,
    plan_threshold,
    run_task,
)

ZBL_JUSTIFICATION = (
    "MACE-MP-0 has no native short-range term and its training data do not reach the collision "
    "regime; an additive universal ZBL is used as a sensitivity choice, not a validated model"
)
# Arguments that only select which tasks run here or where; they may change between resumes.
RUNTIME_ARGS = frozenset({"shard", "shards", "device", "plan_only", "out"})


def build_tasks(args: argparse.Namespace) -> list[dict]:
    modes = ["none", "zbl"] if args.short_range == "both" else [args.short_range]
    tasks: list[dict] = []
    for sr in modes:
        if args.study == "threshold":
            tasks += plan_threshold(
                args.element,
                args.a,
                [float(e) for e in args.energies.split(",")],
                args.directions.split(","),
                args.realizations,
                sr,
                n_cells=args.n_cells,
                temperature_K=args.temperature,
                base_seed=args.seed,
                first_id=len(tasks),
            )
        else:
            counts = None
            if args.study == "alloy":
                fractions = json.loads(args.composition)

                def counts(n: int, fractions=fractions) -> dict[str, int]:
                    total = args.sites_per_cell * n**3
                    out = {k: int(round(v * total)) for k, v in fractions.items()}
                    first = next(iter(out))
                    out[first] += total - sum(out.values())
                    return out

            tasks += plan_cascades(
                args.study,
                args.element,
                args.a,
                [int(n) for n in args.sizes.split(",")],
                args.energy,
                args.realizations,
                sr,
                counts_for=counts,
                temperature_K=args.temperature,
                base_seed=args.seed,
                first_id=len(tasks),
            )
    return tasks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=["threshold", "size", "alloy"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--element", default="Fe")
    p.add_argument("--a", type=float, required=True, help="cubic lattice parameter, angstrom")
    p.add_argument("--short-range", choices=["none", "zbl", "both"], default="both")
    p.add_argument("--energies", default="")
    p.add_argument("--directions", default="100,110,111")
    p.add_argument("--n-cells", type=int, default=6)
    p.add_argument("--sizes", default="")
    p.add_argument("--energy", type=float, default=0.0)
    p.add_argument("--composition", default="{}")
    p.add_argument("--sites-per-cell", type=int, default=2)
    p.add_argument("--realizations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model", default="small")
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()

    tasks = build_tasks(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = out / "plan.json"
    settings = {k: v for k, v in vars(args).items() if k not in RUNTIME_ARGS}
    if plan.exists():
        old = json.loads(plan.read_text()).get("args", {})
        changed = sorted(
            k for k in settings.keys() | (old.keys() - RUNTIME_ARGS) if old.get(k) != settings.get(k)
        )
        if changed:
            raise SystemExit(
                f"{plan} was written with different settings {changed}; existing task records "
                "would be reused for new tasks. Use a new --out directory."
            )
    plan.write_text(json.dumps({"args": vars(args), "n_tasks": len(tasks)}))
    if args.plan_only:
        print(len(tasks), "tasks")
        return 0

    from mace.calculators import mace_mp

    base = mace_mp(model=args.model, device=args.device, default_dtype="float64")
    zbl, record = build_short_range_calculator(
        base, native_short_range=False, additive_zbl=True, justification=ZBL_JUSTIFICATION
    )
    r = np.geomspace(0.3, 5.0, 600)
    checks = {}
    used = {t["short_range"] for t in tasks}
    for name, calc in (("none", base), ("zbl", zbl)):
        if name not in used:
            continue
        e, f = pair_curve(calc, args.element, args.element, r)
        rep = check_pair_curve(r, e, f, repulsive_below=2.0)
        checks[name] = {"ok": rep.ok, "problems": list(rep.problems)}
    if args.shard == 0:
        (out / "short_range.json").write_text(
            json.dumps({"checks": checks, "zbl": record.__dict__}, default=str)
        )
    for name, c in checks.items():
        if not c["ok"]:
            raise SystemExit(f"pair curve for {name!r} failed: {c['problems']}")
    calcs = {"none": base, "zbl": zbl}
    mine = [t for t in tasks if t["id"] % args.shards == args.shard]
    for k, t in enumerate(mine):
        rec = run_task(t, calcs, out)
        print(
            f"[{k + 1}/{len(mine)}] id={t['id']} {t['study']} E={t['energy_eV']} "
            f"dir={t['direction']} sr={t['short_range']} FP={rec['n_frenkel_pairs']} "
            f"stable={rec['numerically_stable']} wall={rec['wall_s']:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
