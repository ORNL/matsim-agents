"""Warm-start benchmark (VASP): HydraGNN pre-relax then compare DFT step counts.

Sibling of :mod:`matsim_agents.tools.warmstart_benchmark_qe`. For a given
input crystal we run::

    1. HydraGNN structure optimisation     (cheap, ML-driven)
    2. vasp_std relaxation from the *original* coordinates    (cold start)
    3. vasp_std relaxation from the HydraGNN-relaxed coordinates  (warm start)

and report how many ionic steps and total electronic-SCF iterations each
DFT run needed, plus the agreement between the two final energies. The full
comparison is written to a JSON file analogous to the QE flavour, so any
downstream analysis can consume both with the same schema.

The HydraGNN step is delegated to :mod:`matsim_agents.tools.relaxation`
(if it cannot be imported the warm-start phase is skipped and only the
cold DFT run is performed; the ``warm`` block is left ``None``).

Energy units throughout: **eV** (matches what VASP and the AL pipeline use).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from matsim_agents.tools.vasp_relax import (
    VASPRelaxResult,
    VASPSettings,
    parse_relax_outputs,
    prepare_relax_workdir,
    recommend_settings,
    run_vasp,
)


@dataclass
class WarmstartComparison:
    """JSON-serialisable summary of a cold-vs-warm VASP comparison."""

    structure_path: str
    work_dir: str
    cold: dict[str, Any]
    warm: dict[str, Any] | None
    hydragnn: dict[str, Any] | None
    delta_ionic_steps: int | None
    delta_scf_iterations: int | None
    delta_energy_ev: float | None
    speedup_ionic: float | None
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
    potcar_dir: str,
    vasp_launcher: list[str] | str,
    *,
    vasp_settings_overrides: dict[str, Any] | None = None,
    hydragnn_kwargs: dict[str, Any] | None = None,
    skip_hydragnn: bool = False,
    timeout_sec: int | None = None,
) -> WarmstartComparison:
    """Run the cold + warm VASP relaxations and return the comparison summary.

    ``vasp_settings_overrides`` is forwarded verbatim to
    :func:`matsim_agents.tools.vasp_relax.recommend_settings`.

    ``hydragnn_kwargs`` is forwarded to ``relax_structure`` (the HydraGNN
    tool). It must at least contain ``logdir`` and ``hydragnn_branch_mlp_checkpoint``; the
    output is written next to the input structure.
    """
    from ase.io import read, write

    structure_path = os.path.abspath(structure_path)
    work_dir = os.path.abspath(work_dir)
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    cold_atoms = read(structure_path)
    overrides = dict(vasp_settings_overrides or {})
    settings_cold = recommend_settings(cold_atoms, potcar_dir=potcar_dir, **overrides)

    cold_dir = os.path.join(work_dir, "cold")
    prepare_relax_workdir(cold_atoms, cold_dir, settings_cold, potcar_dir=potcar_dir)
    cold_result = run_vasp(
        cold_dir,
        launcher_cmd=vasp_launcher,
        stdout_name="vasp.out",
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
            # Reuse the *same* VASP settings as cold to make the comparison fair.
            prepare_relax_workdir(warm_atoms, warm_dir, settings_cold, potcar_dir=potcar_dir)
            warm_result = run_vasp(
                warm_dir,
                launcher_cmd=vasp_launcher,
                stdout_name="vasp.out",
                timeout_sec=timeout_sec,
            )
            warm_block = warm_result.to_dict()

            # Round-trip the warm-input ASE structure for bookkeeping.
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
    cold: VASPRelaxResult,
    warm_block: dict[str, Any] | None,
    hydragnn_block: dict[str, Any] | None,
) -> WarmstartComparison:
    cold_block = cold.to_dict()
    if warm_block is None or "n_ionic_steps" not in warm_block:
        return WarmstartComparison(
            structure_path=structure_path,
            work_dir=work_dir,
            cold=cold_block,
            warm=warm_block,
            hydragnn=hydragnn_block,
            delta_ionic_steps=None,
            delta_scf_iterations=None,
            delta_energy_ev=None,
            speedup_ionic=None,
            warm_helped=None,
        )

    warm_ionic = int(warm_block["n_ionic_steps"])
    warm_scf_total = sum(int(n) for n in warm_block.get("scf_iterations_per_step", []))
    cold_scf_total = sum(cold.scf_iterations_per_step)

    delta_ionic = cold.n_ionic_steps - warm_ionic
    delta_scf = cold_scf_total - warm_scf_total
    delta_e = (
        (cold.final_energy_eV - float(warm_block["final_energy_eV"]))
        if (cold.final_energy_eV is not None and warm_block.get("final_energy_eV") is not None)
        else None
    )
    speedup = cold.n_ionic_steps / max(1, warm_ionic)
    warm_helped = warm_ionic <= cold.n_ionic_steps

    return WarmstartComparison(
        structure_path=structure_path,
        work_dir=work_dir,
        cold=cold_block,
        warm=warm_block,
        hydragnn=hydragnn_block,
        delta_ionic_steps=delta_ionic,
        delta_scf_iterations=delta_scf,
        delta_energy_ev=delta_e,
        speedup_ionic=speedup,
        warm_helped=warm_helped,
    )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matsim-warmstart-benchmark-vasp",
        description=(
            "Run a HydraGNN warm-started vs cold-started VASP "
            "geometry-optimisation comparison and emit a JSON summary."
        ),
    )
    p.add_argument("--structure", required=True, help="Path to input structure (.cif/.vasp/.xyz).")
    p.add_argument("--work-dir", required=True, help="Output directory for VASP inputs/outputs.")
    p.add_argument(
        "--potcar-dir",
        required=True,
        help="Directory with per-element POTCARs (e.g. <potcar_dir>/Si/POTCAR).",
    )
    p.add_argument(
        "--vasp-launcher",
        required=True,
        help=(
            "Command (or absolute path) used to launch vasp_std. The wrapper "
            "is fully responsible for invoking VASP under the local MPI launcher. "
            "VASP reads its inputs from the cwd, so no input path is appended."
        ),
    )
    p.add_argument("--logdir", help="HydraGNN logdir (config.json + checkpoint).")
    p.add_argument("--mlp-checkpoint", help="HydraGNN BranchWeightMLP checkpoint .pt path.")
    p.add_argument("--checkpoint", default=None, help="Optional HydraGNN checkpoint filename.")
    p.add_argument("--mlp-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--hydragnn-fmax", type=float, default=0.05)
    p.add_argument("--hydragnn-maxiter", type=int, default=200)
    p.add_argument(
        "--calculation",
        default="vc-relax",
        choices=["relax", "vc-relax", "vc-relax-shape"],
    )
    p.add_argument("--encut", type=float, default=None, help="Override ENCUT (eV).")
    p.add_argument("--kspacing", type=float, default=None, help="Override KSPACING (1/Å).")
    p.add_argument("--nsw", type=int, default=100, help="Max ionic steps (NSW).")
    p.add_argument("--ediff", type=float, default=None, help="Override EDIFF (eV).")
    p.add_argument(
        "--ediffg",
        type=float,
        default=None,
        help="Override EDIFFG (eV/Å, sign convention: negative = force criterion).",
    )
    p.add_argument("--timeout-sec", type=int, default=None)
    p.add_argument(
        "--skip-hydragnn",
        action="store_true",
        help="Skip HydraGNN entirely; run only the cold DFT relax.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    overrides: dict[str, Any] = {
        "calculation": args.calculation,
        "nsw": args.nsw,
    }
    if args.encut is not None:
        overrides["encut_ev"] = args.encut
    if args.kspacing is not None:
        overrides["kspacing"] = args.kspacing
    if args.ediff is not None:
        overrides["ediff"] = args.ediff
    if args.ediffg is not None:
        overrides["ediffg_eV_per_A"] = args.ediffg

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

    skip_hg = args.skip_hydragnn or not (args.logdir and args.hydragnn_branch_mlp_checkpoint)

    vasp_launcher: list[str] | str = (
        args.vasp_launcher.split() if " " in args.vasp_launcher else args.vasp_launcher
    )

    summary = run_warmstart_benchmark(
        structure_path=args.structure,
        work_dir=args.work_dir,
        potcar_dir=args.potcar_dir,
        vasp_launcher=vasp_launcher,
        vasp_settings_overrides=overrides,
        hydragnn_kwargs=hydragnn_kwargs,
        skip_hydragnn=skip_hg,
        timeout_sec=args.timeout_sec,
    )

    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Touch to silence linters that flag the imports as unused (they form part of
# the documented public surface re-exported via this module).
_ = (VASPSettings, parse_relax_outputs)  # noqa: F841
