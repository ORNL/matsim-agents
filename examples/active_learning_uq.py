"""Active-learning example: HydraGNN MLFF relaxation gated by a UQ proxy.

The HydraGNN multi-branch model returns per-branch weights for every
prediction (see :mod:`matsim_agents.tools.relaxation`). When the model is
confident, the top branch dominates and the per-branch weight distribution
has low entropy. When the model is *uncertain* — typically because the
structure is out of distribution — the weights are spread across many
branches and the top weight is small. We use this as a cheap, built-in
uncertainty-quantification (UQ) proxy.

Workflow per input structure:

    1. Run a HydraGNN-driven ASE relaxation via :func:`relax_structure`.
    2. Read the per-step branch weights from the CSV log emitted by
       the relaxation tool. Compute the average ``top_weight`` and
       the average normalized entropy of the branch-weight distribution.
    3. If ``top_weight < --top-weight-threshold`` OR
       ``entropy > --entropy-threshold``, mark the prediction as
       *unreliable* and trigger two reference DFT calculations on the
       optimized structure:
         * Quantum ESPRESSO ``pw.x`` via ``MATSIM_QE_LAUNCHER``
         * VASP ``vasp_std`` via ``MATSIM_VASP_LAUNCHER``
    4. Append the structure to a CSV ``training_candidates.csv`` so it
       can be folded back into the next HydraGNN training round.

Both DFT engines are launched through user-provided launcher commands
(strings that accept the run directory as the first argument). On Aurora
the matching wrappers are:

    MATSIM_QE_LAUNCHER=scripts/launchers/aurora/run-pw-gpu-aurora.sh
    MATSIM_VASP_LAUNCHER=<path-to-your-aurora-vasp-launcher>

Each launcher is expected to:
    * accept the run directory as ``$1``;
    * find its input file inside that directory (``pw.in`` for QE,
      ``INCAR``/``POSCAR``/``KPOINTS``/``POTCAR`` for VASP);
    * return a non-zero exit code on failure.

If a launcher is not configured the corresponding DFT call is skipped
and a warning is logged; the example still runs end-to-end so it can be
exercised in CI.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matsim_agents.tools.relaxation import RelaxStructureInput, _run as run_relaxation


# --------------------------------------------------------------------------- #
# UQ proxy                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class UQReport:
    structure: str
    optimized_structure: str
    final_energy_eV: float
    final_max_force_eV_per_A: float
    num_steps: int
    converged: bool
    mean_top_weight: float
    mean_entropy: float
    is_unreliable: bool
    reason: str


def _branch_weight_columns(header: list[str]) -> list[int]:
    return [i for i, name in enumerate(header) if name.startswith("w_branch_")]


def _normalized_entropy(weights: np.ndarray) -> float:
    """Return Shannon entropy normalized to ``[0, 1]`` (1.0 == uniform)."""
    w = np.clip(weights, 1e-12, None)
    w = w / w.sum()
    h = float(-(w * np.log(w)).sum())
    return h / math.log(len(w)) if len(w) > 1 else 0.0


def summarize_uq_from_csv(csv_path: str) -> tuple[float, float]:
    """Average top-branch weight and average normalized branch-weight entropy."""
    with open(csv_path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        branch_cols = _branch_weight_columns(header)
        try:
            top_col = header.index("top_weight")
        except ValueError:
            top_col = -1

        top_weights: list[float] = []
        entropies: list[float] = []
        for row in reader:
            if branch_cols:
                try:
                    weights = np.array(
                        [float(row[i]) for i in branch_cols], dtype=np.float64
                    )
                except (ValueError, IndexError):
                    continue
                if not np.all(np.isfinite(weights)):
                    continue
                entropies.append(_normalized_entropy(weights))
            if top_col >= 0:
                try:
                    top_weights.append(float(row[top_col]))
                except (ValueError, IndexError):
                    pass

    mean_top = float(np.mean(top_weights)) if top_weights else float("nan")
    mean_ent = float(np.mean(entropies)) if entropies else float("nan")
    return mean_top, mean_ent


# --------------------------------------------------------------------------- #
# DFT trigger                                                                 #
# --------------------------------------------------------------------------- #


def _run_launcher(label: str, launcher: str | None, run_dir: Path, log_path: Path) -> int:
    if not launcher:
        print(f"[{label}] launcher not configured -> skipping (set MATSIM_{label.upper()}_LAUNCHER).")
        return -1
    cmd = [*launcher.split(), str(run_dir)]
    print(f"[{label}] launching: {' '.join(cmd)}")
    with open(log_path, "w") as out:
        return subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)


def trigger_dft_fallbacks(
    optimized_structure: str,
    work_dir: Path,
    qe_launcher: str | None,
    vasp_launcher: str | None,
) -> dict[str, dict]:
    """Stage the optimized structure for QE and VASP and invoke each launcher."""
    qe_dir = work_dir / "qe"
    vasp_dir = work_dir / "vasp"
    qe_dir.mkdir(parents=True, exist_ok=True)
    vasp_dir.mkdir(parents=True, exist_ok=True)

    # Stage the relaxed structure into both run directories under canonical
    # filenames the launchers can pick up.
    shutil.copy(optimized_structure, qe_dir / "input_structure")
    shutil.copy(optimized_structure, vasp_dir / "POSCAR_active_learning")

    qe_log = work_dir / "qe.log"
    vasp_log = work_dir / "vasp.log"
    qe_rc = _run_launcher("qe", qe_launcher, qe_dir, qe_log)
    vasp_rc = _run_launcher("vasp", vasp_launcher, vasp_dir, vasp_log)

    return {
        "qe": {"return_code": qe_rc, "run_dir": str(qe_dir), "log": str(qe_log)},
        "vasp": {"return_code": vasp_rc, "run_dir": str(vasp_dir), "log": str(vasp_log)},
    }


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def process_structure(
    structure_path: str,
    args: argparse.Namespace,
    output_root: Path,
) -> UQReport:
    out_dir = output_root / Path(structure_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    relax_input = RelaxStructureInput(
        structure_path=structure_path,
        logdir=args.logdir,
        mlp_checkpoint=args.mlp_checkpoint,
        checkpoint=args.checkpoint,
        optimizer=args.optimizer,
        maxiter=args.maxiter,
        fmax=args.fmax,
        mlp_device=args.mlp_device,
        precision=args.precision,
        mlp_precision=args.mlp_precision,
        output_dir=str(out_dir),
    )
    print(f"\n>>> Relaxing {structure_path} with HydraGNN")
    res = run_relaxation(relax_input)

    mean_top, mean_ent = summarize_uq_from_csv(res.log_csv_path)
    reasons: list[str] = []
    if not math.isnan(mean_top) and mean_top < args.top_weight_threshold:
        reasons.append(
            f"mean_top_weight={mean_top:.3f} < {args.top_weight_threshold:.3f}"
        )
    if not math.isnan(mean_ent) and mean_ent > args.entropy_threshold:
        reasons.append(
            f"mean_entropy={mean_ent:.3f} > {args.entropy_threshold:.3f}"
        )
    is_unreliable = bool(reasons)
    reason = "; ".join(reasons) if reasons else "confidence within thresholds"

    print(
        f"    UQ: mean_top_weight={mean_top:.3f}  mean_entropy={mean_ent:.3f}"
        f"  -> {'UNRELIABLE' if is_unreliable else 'reliable'} ({reason})"
    )

    if is_unreliable:
        dft_dir = out_dir / "dft_fallback"
        dft_dir.mkdir(parents=True, exist_ok=True)
        trigger_dft_fallbacks(
            res.optimized_structure_path,
            dft_dir,
            qe_launcher=args.qe_launcher,
            vasp_launcher=args.vasp_launcher,
        )

    return UQReport(
        structure=structure_path,
        optimized_structure=res.optimized_structure_path,
        final_energy_eV=res.final_energy_eV,
        final_max_force_eV_per_A=res.final_max_force_eV_per_A,
        num_steps=res.num_steps,
        converged=res.converged,
        mean_top_weight=mean_top,
        mean_entropy=mean_ent,
        is_unreliable=is_unreliable,
        reason=reason,
    )


def write_summary(reports: list[UQReport], output_root: Path) -> Path:
    candidates = [r for r in reports if r.is_unreliable]
    summary_path = output_root / "uq_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "structure",
                "optimized_structure",
                "final_energy_eV",
                "final_max_force_eV_per_A",
                "num_steps",
                "converged",
                "mean_top_weight",
                "mean_entropy",
                "is_unreliable",
                "reason",
            ]
        )
        for r in reports:
            w.writerow(
                [
                    r.structure,
                    r.optimized_structure,
                    f"{r.final_energy_eV:.6f}",
                    f"{r.final_max_force_eV_per_A:.6f}",
                    r.num_steps,
                    r.converged,
                    f"{r.mean_top_weight:.6f}",
                    f"{r.mean_entropy:.6f}",
                    int(r.is_unreliable),
                    r.reason,
                ]
            )

    cand_path = output_root / "training_candidates.csv"
    with open(cand_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["structure", "mean_top_weight", "mean_entropy", "reason"])
        for r in candidates:
            w.writerow(
                [
                    r.optimized_structure,
                    f"{r.mean_top_weight:.6f}",
                    f"{r.mean_entropy:.6f}",
                    r.reason,
                ]
            )
    return summary_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "structures",
        nargs="+",
        help="Input structure files (.vasp, .cif, .xyz, ...).",
    )
    p.add_argument("--logdir", required=True, help="HydraGNN logdir with config.json + checkpoint.")
    p.add_argument("--mlp-checkpoint", required=True, help="BranchWeightMLP .pt file.")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-dir", default="./outputs/active_learning_uq")
    p.add_argument("--mlp-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--precision", default=None)
    p.add_argument("--mlp-precision", default=None)
    p.add_argument("--optimizer", default="FIRE", choices=["FIRE", "BFGS", "BFGSLineSearch"])
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--fmax", type=float, default=0.02)

    # UQ thresholds
    p.add_argument(
        "--top-weight-threshold",
        type=float,
        default=0.6,
        help="Trigger DFT when mean(top_branch_weight) over the relaxation drops below this.",
    )
    p.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.5,
        help="Trigger DFT when mean normalized branch-weight entropy exceeds this.",
    )

    # DFT launchers (env vars override CLI defaults)
    p.add_argument(
        "--qe-launcher",
        default=os.environ.get("MATSIM_QE_LAUNCHER"),
        help="Command to launch QE pw.x. Receives the run directory as $1.",
    )
    p.add_argument(
        "--vasp-launcher",
        default=os.environ.get("MATSIM_VASP_LAUNCHER"),
        help="Command to launch VASP. Receives the run directory as $1.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reports: list[UQReport] = []
    for s in args.structures:
        reports.append(process_structure(s, args, output_root))

    summary = write_summary(reports, output_root)

    n_unrel = sum(r.is_unreliable for r in reports)
    print(
        f"\nDone. {n_unrel}/{len(reports)} prediction(s) flagged as unreliable. "
        f"Summary: {summary}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
