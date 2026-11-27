"""Plumbing smoke for uncertainty-gated escalation to a real ``neb.x`` run.

This builds a small BCC vacancy hop, hands ``_apply_neb_escalation`` a *stand-in* MLIP outcome whose
pathway uncertainty is unavailable (so the gate must escalate), and lets the real escalation code
prepare, launch and ingest a QE NEB. It checks the launch and ingest path only.

The stand-in outcome has no surrogate behind it, and the QE run is far below any converged setting
(few path steps, small cutoffs, 4 images). The recorded barrier is therefore a software-plumbing
number, not a physical result, and nothing here is a numerical or physical validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.science.defects.point_defects import generate_vacancy
from matsim_agents.science.neb.contracts import NebResult
from matsim_agents.science.neb.endpoints import generate_vacancy_hop, vacancy_neighbors
from matsim_agents.science.neb.images import generate_linear_images
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy
from matsim_agents.workflows.active_learning.config import NebEscalationConfig
from matsim_agents.workflows.active_learning.loop import IterationState, _apply_neb_escalation

LATTICE_A = 3.44  # Angstrom, BCC Li; chosen for a small system, not for accuracy


def build_outcome(n_images: int):
    parent = generate_alloy(
        AlloySpec(counts={"Li": 16}, repeats=(2, 2, 2), seed=0, lattice_parameter=LATTICE_A)
    )
    initial, defect = generate_vacancy(parent, "li-parent", 0)
    mover = vacancy_neighbors(initial, defect)[0]
    pair = generate_vacancy_hop(initial, defect, parent, mover)
    linear = generate_linear_images(pair.initial, pair.final, n_images)
    result = NebResult(
        state="converged_unqualified",
        method_id="mlip-neb-standin",
        n_images=n_images,
        forward_barrier_eV=0.0,
        reverse_barrier_eV=0.0,
        pathway_uncertainty_eV=None,  # unavailable, which must not read as low uncertainty
        provenance={"transition_id": pair.transition.transition_id, "backend": "standin"},
    )
    return SimpleNamespace(
        result=result,
        transition=pair.transition,
        images=linear.images,
        initial=pair.initial,
        final=pair.final,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--pseudo-file", default="li_pbe_v1.4.uspp.F.UPF")
    parser.add_argument("--neb-bin", required=True)
    parser.add_argument("--launcher", nargs="*", default=[], help="launcher prefix for neb.x")
    parser.add_argument("--nimage", type=int, default=1)
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--nstep-path", type=int, default=3)
    parser.add_argument("--ecutwfc", type=float, default=40.0)
    parser.add_argument("--ecutrho", type=float, default=320.0)
    parser.add_argument("--timeout-sec", type=float, default=420.0)
    args = parser.parse_args(argv)

    neb = QENebConfig(
        n_images=args.n_images,
        nstep_path=args.nstep_path,
        species=("Li",),
        pseudo_dir=args.pseudo_dir,
        pseudopotentials={"Li": args.pseudo_file},
        ecutwfc_ry=args.ecutwfc,
        ecutrho_ry=args.ecutrho,
        occupations="smearing",
        degauss_ry=0.02,
        kpts=(1, 1, 1),
        nspin=1,
        neb_executable=args.neb_bin,
        launcher=tuple(args.launcher),
        nimage=args.nimage,
    )
    policy = NebEscalationConfig(
        policy="uncertainty_gated",
        max_pathway_uncertainty_eV=0.2,
        max_dft_neb_runs=1,
        dft_neb=neb,
        allow_unrelaxed_endpoints=True,  # endpoints here are generated, not relaxed
    )
    iteration_dir = args.workdir / "iteration_0000"
    iteration_dir.mkdir(parents=True, exist_ok=False)
    record, dispatched = _apply_neb_escalation(
        policy,
        build_outcome(args.n_images),
        iteration_dir,
        iteration_dir / "neb",
        IterationState(iteration=0),
        completed_dft_neb=0,
        env=None,
        timeout_sec=args.timeout_sec,
    )
    summary = {
        "decision": record["decision"],
        "dft_dispatched": dispatched,
        "dft_start_path": record.get("dft_start_path"),
        "dft_state": (record.get("dft_result") or {}).get("state"),
        "dft_forward_barrier_eV": (record.get("dft_result") or {}).get("forward_barrier_eV"),
        "dft_run_directory": record.get("dft_run_directory"),
        "scope": "launch/ingest plumbing only; not a physical or numerical validation",
    }
    (args.workdir / "escalation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    # Plumbing succeeded only when neb.x ran and its output parsed into image energies (state
    # unconverged or better). "failed" and "incomplete" mean no usable output came back. The NEB
    # need not have converged within the few path steps allowed here.
    parsed = summary["dft_state"] in {"unconverged", "converged_unqualified", "qualified"}
    return 0 if dispatched and parsed else 1


if __name__ == "__main__":
    sys.exit(main())
