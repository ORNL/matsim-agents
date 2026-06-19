"""Warm-start benchmark (Quantum ESPRESSO): HydraGNN pre-relax then compare DFT step counts.

For a given input crystal we run::

    1. HydraGNN structure optimisation     (cheap, ML-driven)
    2. pw.x relaxation from the *original* coordinates    (cold start)
    3. pw.x relaxation from the HydraGNN-relaxed coordinates  (warm start)

and report how many BFGS / SCF iterations each DFT run needed, plus the
agreement between the two final energies. The whole comparison is written to
a JSON file that the integration tests consume.

The HydraGNN step is delegated to the existing tool
``matsim_agents.tools.relaxation`` (which itself is optional — if HydraGNN is
not importable the warm-start phase is skipped and only the cold run is
performed, with a ``warm`` block left as ``None``).

For the VASP equivalent see :mod:`matsim_agents.tools.warmstart_benchmark_vasp`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from matsim_agents.tools.qe_relax import (
    PWResult,
    QESettings,
    recommend_settings,
    run_pw,
    write_pw_input,
)


@dataclass
class WarmstartComparison:
    """JSON-serialisable summary of a cold-vs-warm comparison."""

    structure_path: str
    work_dir: str
    cold: dict[str, Any]
    warm: dict[str, Any] | None
    hydragnn: dict[str, Any] | None
    delta_bfgs_steps: int | None
    delta_scf_iterations: int | None
    delta_energy_ev: float | None
    speedup_bfgs: float | None
    warm_helped: bool | None

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def run_warmstart_benchmark(
    structure_path: str,
    work_dir: str,
    pseudo_dir: str,
    qe_launcher: list[str] | str,
    *,
    qe_settings_overrides: dict[str, Any] | None = None,
    hydragnn_kwargs: dict[str, Any] | None = None,
    skip_hydragnn: bool = False,
    timeout_sec: int | None = None,
) -> WarmstartComparison:
    """Run the cold + warm DFT relaxations and return the comparison summary.

    ``qe_settings_overrides`` is forwarded to :func:`recommend_settings`.

    ``hydragnn_kwargs`` is forwarded to ``relax_structure`` (the HydraGNN
    tool). It must at least contain ``logdir`` and ``hydragnn_branch_mlp_checkpoint``; the
    output is written next to the input structure.
    """
    from ase.io import read, write

    structure_path = os.path.abspath(structure_path)
    work_dir = os.path.abspath(work_dir)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    cold_atoms = read(structure_path)
    overrides = dict(qe_settings_overrides or {})
    settings_cold = recommend_settings(cold_atoms, pseudo_dir, **overrides)

    cold_dir = os.path.join(work_dir, "cold")
    cold_input = os.path.join(cold_dir, "pw.in")
    write_pw_input(
        cold_atoms,
        settings_cold,
        cold_input,
        prefix="pwscf_cold",
        outdir=os.path.join(cold_dir, "tmp"),
        title=f"Cold-start relax of {os.path.basename(structure_path)}",
    )
    cold_result = run_pw(
        cold_input,
        work_dir=cold_dir,
        launcher_cmd=qe_launcher,
        stdout_name="pw.out",
        timeout_sec=timeout_sec,
    )

    warm_block: dict[str, Any] | None = None
    hydragnn_block: dict[str, Any] | None = None

    if not skip_hydragnn:
        try:
            warm_atoms_path, hydragnn_block = _hydragnn_relax(
                structure_path, work_dir, hydragnn_kwargs or {}
            )
        except Exception as exc:  # pragma: no cover - exercised on-cluster only
            hydragnn_block = {"error": f"{type(exc).__name__}: {exc}"}
            warm_atoms_path = None

        if warm_atoms_path is not None:
            warm_atoms = read(warm_atoms_path)
            warm_dir = os.path.join(work_dir, "warm")
            warm_input = os.path.join(warm_dir, "pw.in")
            # Reuse the *same* QE settings as cold to make the comparison fair.
            write_pw_input(
                warm_atoms,
                settings_cold,
                warm_input,
                prefix="pwscf_warm",
                outdir=os.path.join(warm_dir, "tmp"),
                title=f"Warm-start relax (HydraGNN-prerelaxed) of "
                f"{os.path.basename(structure_path)}",
            )
            warm_result = run_pw(
                warm_input,
                work_dir=warm_dir,
                launcher_cmd=qe_launcher,
                stdout_name="pw.out",
                timeout_sec=timeout_sec,
            )
            warm_block = warm_result.to_dict()

            # Round-trip the warm-input ASE structure to the work dir for
            # bookkeeping — handy when looking at runs later.
            write(os.path.join(warm_dir, "input_warm.cif"), warm_atoms)

    summary = _summarise(structure_path, work_dir, cold_result, warm_block, hydragnn_block)
    summary.to_json(os.path.join(work_dir, "comparison.json"))
    return summary


def _hydragnn_relax(
    structure_path: str,
    work_dir: str,
    kwargs: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Invoke the HydraGNN relax tool. Returns ``(optimized_path, info)``."""
    from matsim_agents.tools.relaxation import RelaxStructureInput, _run

    hydragnn_dir = os.path.join(work_dir, "hydragnn")
    Path(hydragnn_dir).mkdir(parents=True, exist_ok=True)

    payload = dict(kwargs)
    payload.setdefault("structure_path", structure_path)
    payload.setdefault("output_dir", hydragnn_dir)
    payload.setdefault("optimizer", "FIRE")
    payload.setdefault("fmax", 0.05)
    payload.setdefault("maxiter", 200)

    args = RelaxStructureInput(**payload)
    res = _run(args)
    return res.optimized_structure_path, {
        "num_steps": res.num_steps,
        "converged": res.converged,
        "final_energy_eV": res.final_energy_eV,
        "final_max_force_eV_per_A": res.final_max_force_eV_per_A,
        "optimized_structure_path": res.optimized_structure_path,
        "trajectory_path": res.trajectory_path,
        "log_csv_path": res.log_csv_path,
    }


def _summarise(
    structure_path: str,
    work_dir: str,
    cold: PWResult,
    warm_block: dict[str, Any] | None,
    hydragnn_block: dict[str, Any] | None,
) -> WarmstartComparison:
    cold_block = cold.to_dict()
    if warm_block is None or "bfgs_steps" not in warm_block:
        return WarmstartComparison(
            structure_path=structure_path,
            work_dir=work_dir,
            cold=cold_block,
            warm=warm_block,
            hydragnn=hydragnn_block,
            delta_bfgs_steps=None,
            delta_scf_iterations=None,
            delta_energy_ev=None,
            speedup_bfgs=None,
            warm_helped=None,
        )

    delta_bfgs = cold.bfgs_steps - int(warm_block["bfgs_steps"])
    delta_scf = cold.scf_iterations_total - int(warm_block["scf_iterations_total"])
    delta_e = cold.final_energy_ev - float(warm_block["final_energy_ev"])
    speedup = cold.bfgs_steps / max(1, int(warm_block["bfgs_steps"]))
    warm_helped = warm_block["bfgs_steps"] <= cold.bfgs_steps

    return WarmstartComparison(
        structure_path=structure_path,
        work_dir=work_dir,
        cold=cold_block,
        warm=warm_block,
        hydragnn=hydragnn_block,
        delta_bfgs_steps=delta_bfgs,
        delta_scf_iterations=delta_scf,
        delta_energy_ev=delta_e,
        speedup_bfgs=speedup,
        warm_helped=warm_helped,
    )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matsim-warmstart-benchmark-qe",
        description=(
            "Run a HydraGNN warm-started vs cold-started Quantum ESPRESSO "
            "geometry-optimisation comparison and emit a JSON summary."
        ),
    )
    p.add_argument("--structure", required=True, help="Path to input structure (.cif/.vasp/.xyz).")
    p.add_argument("--work-dir", required=True, help="Output directory for QE inputs/outputs.")
    p.add_argument(
        "--pseudo-dir",
        required=True,
        help="Directory containing .UPF pseudopotentials.",
    )
    p.add_argument(
        "--qe-launcher",
        required=True,
        help=(
            "Command (or absolute path) used to launch pw.x. The QE input file "
            "is appended as the final argument."
        ),
    )
    p.add_argument("--logdir", help="HydraGNN logdir (config.json + checkpoint).")
    p.add_argument("--mlp-checkpoint", help="HydraGNN BranchWeightMLP checkpoint .pt path.")
    p.add_argument("--checkpoint", default=None, help="Optional HydraGNN checkpoint filename.")
    p.add_argument("--mlp-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--hydragnn-fmax", type=float, default=0.05)
    p.add_argument("--hydragnn-maxiter", type=int, default=200)
    p.add_argument("--calculation", default="vc-relax", choices=["relax", "vc-relax"])
    p.add_argument(
        "--is-2d",
        action="store_true",
        help="Treat as 2-D (k-mesh = 1 in the longest cell direction).",
    )
    p.add_argument("--ecutwfc", type=float, default=None, help="Override ecutwfc (Ry).")
    p.add_argument("--ecutrho", type=float, default=None, help="Override ecutrho (Ry).")
    p.add_argument("--kpts", type=int, nargs=3, default=None, metavar=("KX", "KY", "KZ"))
    p.add_argument("--nstep", type=int, default=100)
    p.add_argument("--timeout-sec", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    overrides: dict[str, Any] = {
        "calculation": args.calculation,
        "nstep": args.nstep,
        "is_2d": args.is_2d,
    }
    if args.ecutwfc is not None:
        overrides["ecutwfc_ry"] = args.ecutwfc
    if args.ecutrho is not None:
        overrides["ecutrho_ry"] = args.ecutrho
    if args.kpts is not None:
        overrides["kpts"] = tuple(args.kpts)

    hydragnn_kwargs: dict[str, Any] = {}
    if args.logdir:
        hydragnn_kwargs["logdir"] = args.logdir
    if args.hydragnn_branch_mlp_checkpoint:
        hydragnn_kwargs["hydragnn_branch_mlp_checkpoint"] = args.hydragnn_branch_mlp_checkpoint
    if args.checkpoint:
        hydragnn_kwargs["checkpoint"] = args.checkpoint
    hydragnn_kwargs["mlp_device"] = args.mlp_device
    hydragnn_kwargs["fmax"] = args.hydragnn_fmax
    hydragnn_kwargs["maxiter"] = args.hydragnn_maxiter

    skip_hg = not (args.logdir and args.hydragnn_branch_mlp_checkpoint)

    qe_launcher: list[str] | str = (
        args.qe_launcher.split() if " " in args.qe_launcher else args.qe_launcher
    )

    summary = run_warmstart_benchmark(
        structure_path=args.structure,
        work_dir=args.work_dir,
        pseudo_dir=args.pseudo_dir,
        qe_launcher=qe_launcher,
        qe_settings_overrides=overrides,
        hydragnn_kwargs=hydragnn_kwargs,
        skip_hydragnn=skip_hg,
        timeout_sec=args.timeout_sec,
    )

    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Ensure the optional QESettings override 'is_2d' propagates cleanly. The
# dataclass stores it; recommend_settings honours it. Touch to silence linters
# that flag the import as unused.
_ = QESettings  # noqa: F401
