"""Vacancy-migration NEB with QE: lattice constant, endpoint relaxation, then ``neb.x``.

For any cubic crystal (sc, bcc, fcc, diamond). The hop distance is derived from the structure.
Steps: relax the unit-cell volume with ``pw.x``, build the vacancy supercell at that lattice
constant, relax both hop endpoints at fixed cell, then run a climbing-image NEB between them.

The result is at best ``converged_unqualified``: cutoff, k-point, supercell and image-count
convergence are not tested here, so a barrier from this script is a single-setting number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.backends.dft.qe_neb_stages import (
    ci_entry_problems,
    derive_stage_config,
    load_history,
    prepare_stage,
    run_stage,
)
from matsim_agents.backends.dft.qe_relax import QESettings, run_pw, write_pw_input
from matsim_agents.science.defects.point_defects import generate_vacancy, geometry_fingerprint
from matsim_agents.science.neb.contracts import NebEndpoint, NebTransition, build_correspondence
from matsim_agents.science.neb.endpoints import (
    MigrationPair,
    generate_vacancy_hop,
    vacancy_neighbors,
)
from matsim_agents.science.neb.serialization import read_pathway
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy
from matsim_agents.workflows.neb import ingest_neb_results, run_neb_workflow

ATOMS_PER_CELL = {"sc": 1, "bcc": 2, "fcc": 4, "diamond": 8}
ENDPOINT_METHOD = "qe-pw-relax"


def _relax(atoms: Atoms, settings: QESettings, directory: Path, launcher: list[str], timeout: int):
    directory.mkdir(parents=True, exist_ok=True)
    input_path = write_pw_input(
        atoms, settings, str(directory / "pw.in"), outdir=str(directory / "tmp")
    )
    result = run_pw(input_path, str(directory), launcher, timeout_sec=timeout)
    if not (result.converged and result.job_done):
        raise RuntimeError(f"pw.x did not converge in {directory}; see {result.stdout_path}")
    return ase_read(result.stdout_path, index=-1, format="espresso-out"), result


def _relaxed_endpoint(endpoint: NebEndpoint, atoms: Atoms) -> NebEndpoint:
    data = endpoint.model_dump(mode="python")
    data.update(
        geometry_fingerprint=geometry_fingerprint(atoms),
        status="relaxed",
        relaxation_converged=True,
        method=ENDPOINT_METHOD,
        qualification_evidence=(),
    )
    return NebEndpoint.model_validate(data)


def _relaxed_pair(pair: MigrationPair, initial: Atoms, final: Atoms) -> MigrationPair:
    transition = pair.transition
    new_initial = _relaxed_endpoint(transition.initial, initial)
    new_final = _relaxed_endpoint(transition.final, final)
    data = transition.model_dump(mode="python")
    data.update(
        initial=new_initial,
        final=new_final,
        correspondence=build_correspondence(new_initial, new_final),
    )
    return MigrationPair(initial, final, NebTransition.model_validate(data), pair.final_config)


def _with_positions(template: Atoms, relaxed: Atoms) -> Atoms:
    if relaxed.get_chemical_symbols() != template.get_chemical_symbols():
        raise RuntimeError("relaxed structure changed the atom order or species")
    if not np.allclose(relaxed.cell.array, template.cell.array, atol=1e-4):
        raise RuntimeError("fixed-cell relaxation changed the cell")
    out = template.copy()
    out.positions = relaxed.positions
    return out


CONTINUABLE = ("iteration_limit", "incomplete_output", "process_failure")


def advance_stages(
    neb_dir: Path,
    *,
    nstep_path: int,
    nstep_path_ci: int,
    max_continues: int,
    timeout_sec: float,
    skip_ci: bool = False,
):
    """Run any prepared stage, continue an unfinished one, then the climbing-image stage.

    Returns ``(outcome, note)``. Each ``continue`` stage keeps the CI settings and raises
    ``nstep_path`` by ``nstep_path`` above what was already done; at most ``max_continues`` run.
    """
    qe_dir = neb_dir / "qe"
    images = read_pathway(neb_dir / "pathway").images
    history = load_history(qe_dir)
    if not history.stages:
        raise RuntimeError(f"no NEB stage in {qe_dir}")
    if history.stages[-1].status == "prepared":
        run_stage(qe_dir, timeout_sec=timeout_sec)
    continues = 0
    while True:
        last = load_history(qe_dir).stages[-1]
        if last.outcome not in CONTINUABLE or continues >= max_continues:
            break
        base = QENebConfig.model_validate(last.config)
        target = max(base.nstep_path, last.iterations_completed or 0) + nstep_path
        prepare_stage(
            derive_stage_config(base, "continue", nstep_path=target), images, kind="continue"
        )
        run_stage(qe_dir, timeout_sec=timeout_sec)
        continues += 1
    outcome = ingest_neb_results(neb_dir)
    note = "climbing-image stage not run"
    problems = ["skipped by request"] if skip_ci else ci_entry_problems(last)
    if problems:
        note += ": " + "; ".join(problems)
    else:
        base = QENebConfig.model_validate(last.config)
        target = max(nstep_path_ci, (last.iterations_completed or 0) + nstep_path)
        prepare_stage(derive_stage_config(base, "ci", nstep_path=target), images, kind="ci")
        run_stage(qe_dir, timeout_sec=timeout_sec)
        outcome = ingest_neb_results(neb_dir)
        note = "climbing-image stage run"
    if continues:
        note += f"; {continues} continue stage(s) run"
    return outcome, note


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse the endpoints and path in an existing --workdir and only run NEB stages",
    )
    parser.add_argument(
        "--max-continues",
        type=int,
        default=0,
        help="continue stages to run after an unfinished stage (iteration limit or timeout)",
    )
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--pseudo-file", required=True)
    parser.add_argument("--pw-bin", required=True)
    parser.add_argument("--neb-bin", required=True)
    parser.add_argument("--pw-launcher", nargs="*", required=True, help="{work_dir} {input} ok")
    parser.add_argument("--neb-launcher", nargs="*", required=True)
    parser.add_argument("--element", default="Al")
    parser.add_argument("--structure", choices=sorted(ATOMS_PER_CELL), default="fcc")
    parser.add_argument("--lattice-guess", type=float, default=4.05)
    parser.add_argument("--repeats", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument("--kpts-unit", type=int, default=12)
    parser.add_argument("--kpts-super", type=int, default=3)
    parser.add_argument("--ecutwfc", type=float, default=30.0)
    parser.add_argument("--ecutrho", type=float, default=240.0)
    parser.add_argument("--degauss", type=float, default=0.02)
    parser.add_argument("--forc-conv", type=float, default=5.0e-4, help="Ry/Bohr")
    parser.add_argument("--n-images", type=int, default=7)
    parser.add_argument("--nstep-path", type=int, default=40)
    parser.add_argument("--nstep-path-ci", type=int, default=120, help="total, above --nstep-path")
    parser.add_argument("--path-thr", type=float, default=0.05, help="eV/Angstrom")
    parser.add_argument("--nimage", type=int, default=1)
    parser.add_argument("--npool", type=int, default=1)
    parser.add_argument("--pw-timeout-sec", type=int, default=1200)
    parser.add_argument("--neb-timeout-sec", type=float, default=1500.0)
    parser.add_argument(
        "--skip-ci", action="store_true", help="do not start the climbing-image stage"
    )
    args = parser.parse_args(argv)

    work = args.workdir
    if args.resume:
        return _resume(args)
    work.mkdir(parents=True, exist_ok=False)
    element = args.element
    pseudos = {element: args.pseudo_file}
    common = dict(
        pseudo_dir=args.pseudo_dir,
        ecutwfc_ry=args.ecutwfc,
        ecutrho_ry=args.ecutrho,
        occupations="smearing",
        smearing="marzari-vanderbilt",
        degauss_ry=args.degauss,
        pseudopotentials=pseudos,
        forc_conv_thr_ry_au=args.forc_conv,
    )
    pw_launcher = list(args.pw_launcher)

    def alloy(a: float, repeats) -> Atoms:
        spec = AlloySpec(
            crystal_structure=args.structure,
            counts={element: ATOMS_PER_CELL[args.structure] * int(np.prod(repeats))},
            repeats=tuple(repeats),
            lattice_parameter=a,
            seed=0,
        )
        return generate_alloy(spec)

    unit = alloy(args.lattice_guess, (1, 1, 1))
    relaxed_unit, _ = _relax(
        unit,
        QESettings(
            calculation="vc-relax",
            kpts=(args.kpts_unit,) * 3,
            extra_cell={"press_conv_thr": 0.2},
            **common,
        ),
        work / "unit-vc-relax",
        pw_launcher,
        args.pw_timeout_sec,
    )
    a0 = args.lattice_guess * (relaxed_unit.get_volume() / unit.get_volume()) ** (1.0 / 3.0)

    parent = alloy(a0, args.repeats)
    initial, config = generate_vacancy(parent, f"{element}-{args.structure}", 0)
    mover = vacancy_neighbors(initial, config)[0]
    pair = generate_vacancy_hop(initial, config, parent, mover)

    fixed_cell = QESettings(calculation="relax", kpts=(args.kpts_super,) * 3, **common)
    relaxed_i, res_i = _relax(
        pair.initial, fixed_cell, work / "endpoint-initial", pw_launcher, args.pw_timeout_sec
    )
    relaxed_f, res_f = _relax(
        pair.final, fixed_cell, work / "endpoint-final", pw_launcher, args.pw_timeout_sec
    )
    initial_atoms = _with_positions(pair.initial, relaxed_i)
    final_atoms = _with_positions(pair.final, relaxed_f)
    relaxed = _relaxed_pair(pair, initial_atoms, final_atoms)

    neb_config = QENebConfig(
        n_images=args.n_images,
        nstep_path=args.nstep_path,
        path_thr_ev_ang=args.path_thr,
        species=(element,),
        pseudo_dir=args.pseudo_dir,
        pseudopotentials=pseudos,
        ecutwfc_ry=args.ecutwfc,
        ecutrho_ry=args.ecutrho,
        occupations="smearing",
        smearing="marzari-vanderbilt",
        degauss_ry=args.degauss,
        kpts=(args.kpts_super,) * 3,
        nspin=1,
        neb_executable=args.neb_bin,
        launcher=tuple(args.neb_launcher),
        nimage=args.nimage,
        npool=args.npool,
    )

    outcome = run_neb_workflow(
        relaxed,
        neb_config,
        work / "neb",
        dispatch=True,
        timeout_sec=args.neb_timeout_sec,
        method_id=ENDPOINT_METHOD,
    )
    climbing_note = "climbing-image stage not run"
    if outcome.stage is not None:
        outcome, climbing_note = advance_stages(
            work / "neb",
            nstep_path=args.nstep_path,
            nstep_path_ci=args.nstep_path_ci,
            max_continues=args.max_continues,
            timeout_sec=args.neb_timeout_sec,
            skip_ci=args.skip_ci,
        )
    result = outcome.result
    summary = {
        "system": f"{element} {args.structure} {'x'.join(map(str, args.repeats))} vacancy hop",
        "lattice_constant_A": a0,
        "hop_distance_A": relaxed.transition.mechanism.parameters.get("hop_distance"),
        "settings": {
            "ecutwfc_Ry": args.ecutwfc,
            "ecutrho_Ry": args.ecutrho,
            "degauss_Ry": args.degauss,
            "kpts_supercell": args.kpts_super,
            "n_images": args.n_images,
        },
        "endpoint_energy_difference_eV": res_f.final_energy_ev - res_i.final_energy_ev,
        "endpoint_max_force_eV_per_A": [
            res_i.final_max_force_eV_per_A,
            res_f.final_max_force_eV_per_A,
        ],
        "plan_status": outcome.plan.status,
        "plan_reasons": list(outcome.plan.reasons),
        "state": None if result is None else result.state,
        "forward_barrier_eV": None if result is None else result.forward_barrier_eV,
        "reverse_barrier_eV": None if result is None else result.reverse_barrier_eV,
        "relative_energies_eV": None if result is None else list(result.relative_energies_eV),
        "optimization_steps": None if result is None else result.optimization_steps,
        "max_force_eV_per_A": None if result is None else result.max_force_eV_per_A,
        "climbing_image_used": None if result is None else result.climbing_image_used,
        "climbing_note": climbing_note,
        "qualification_blockers": list(outcome.qualification_blockers),
        "scope": "single-setting barrier; not converged in cutoff, k-points, supercell or images",
    }
    return _report(work, summary, result)


def _report(work: Path, summary: dict, result) -> int:
    (work / "physical-neb-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result is not None and result.state in {"converged_unqualified"} else 1


def _resume(args) -> int:
    """Run more NEB stages in an existing work directory; keep its endpoint facts."""
    work = args.workdir
    outcome, note = advance_stages(
        work / "neb",
        nstep_path=args.nstep_path,
        nstep_path_ci=args.nstep_path_ci,
        max_continues=args.max_continues,
        timeout_sec=args.neb_timeout_sec,
        skip_ci=args.skip_ci,
    )
    path = work / "physical-neb-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    result = outcome.result
    summary.update(
        plan_status=outcome.plan.status,
        plan_reasons=list(outcome.plan.reasons),
        state=None if result is None else result.state,
        forward_barrier_eV=None if result is None else result.forward_barrier_eV,
        reverse_barrier_eV=None if result is None else result.reverse_barrier_eV,
        relative_energies_eV=None if result is None else list(result.relative_energies_eV),
        optimization_steps=None if result is None else result.optimization_steps,
        max_force_eV_per_A=None if result is None else result.max_force_eV_per_A,
        climbing_image_used=None if result is None else result.climbing_image_used,
        climbing_note=note,
        qualification_blockers=list(outcome.qualification_blockers),
        resumed=True,
    )
    return _report(work, summary, result)


if __name__ == "__main__":
    sys.exit(main())
