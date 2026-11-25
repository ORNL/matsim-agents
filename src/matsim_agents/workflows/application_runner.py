"""Execute an application: generate its structures, then optionally relax them.

Structure generation and relaxation are the shared workflows (``science.structures`` and
``workflows.relaxation.run_relaxation``); an application only chooses what to generate and
how far to go (``mode``): validate, generate, mlip_relax or dft_relax.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

from matsim_agents.core.application import ApplicationConfig, load_application
from matsim_agents.execution.run_directory import ScientificRunDirectory
from matsim_agents.workflows import runner
from matsim_agents.workflows.applications import check_application

_RELAX_MODES = {"mlip_relax": {"mlip"}, "dft_relax": {"dft", "mlip-dft"}}
_SCIENTIFIC_STATUS = (
    "integrated and executed; not scientifically qualified until relaxed energies are "
    "checked against a reference hull or DFT"
)


def build_target_structure(app: ApplicationConfig) -> Atoms:
    """The application's fixed-composition supercell, with exact counts."""
    from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy

    target = app.target
    if target is None:
        raise ValueError(f"application {app.name!r} has no target supercell")
    spec = AlloySpec(
        crystal_structure=target.parent_lattice,
        repeats=tuple(target.supercell),
        lattice_parameter=float(app.parameters.get("lattice_parameter_angstrom", 3.16)),
        counts=dict(target.composition),
        seed=int(app.parameters.get("random_seed", 0)),
    )
    atoms = generate_alloy(spec)
    built = {el: atoms.get_chemical_symbols().count(el) for el in target.composition}
    if len(atoms) != target.expected_sites or built != dict(target.composition):
        raise ValueError(f"generated {built} ({len(atoms)} atoms) differs from the target")
    return atoms


def _structure_digest(atoms: Atoms) -> str:
    key = repr(
        (
            atoms.get_chemical_symbols(),
            [tuple(round(x, 5) for x in row) for row in atoms.cell.array],
            [tuple(round(x, 5) for x in row) for row in atoms.get_scaled_positions()],
        )
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _relax_template(app: ApplicationConfig) -> dict[str, Any]:
    from matsim_agents.backends.mlip.config import resolve_mlip_paths

    path = app.resolve(app.relaxation)
    raw = runner.load_yaml(path)
    if isinstance(raw.get("mlip"), dict):
        resolve_mlip_paths(raw["mlip"], path.resolve().parent)
    if raw.get("mode") not in _RELAX_MODES[app.mode]:
        raise ValueError(
            f"application mode {app.mode!r} needs relaxation mode in "
            f"{sorted(_RELAX_MODES[app.mode])}, not {raw.get('mode')!r}"
        )
    return raw


def _relax_config(template: dict[str, Any], structure: Path, out: Path):
    from matsim_agents.workflows.relaxation import ScientificRelaxationConfig

    return ScientificRelaxationConfig.model_validate(
        {**template, "structure_path": str(structure), "output_root": str(out)}
    )


def _candidates(app: ApplicationConfig, directory: Path) -> list[tuple[str, Atoms | Path]]:
    """Structures to process: one for a supercell target, deduplicated seeds per composition."""
    if app.target is not None:
        return [(app.target.formula, build_target_structure(app))]
    from matsim_agents.workflows.composition_exploration import generate_composition_candidates

    found: list[tuple[str, Atoms | Path]] = []
    seen: set[str] = set()
    for formula in app.compositions:
        _, seeds, _ = generate_composition_candidates(
            formula,
            str(directory / "candidates"),
            n_random=int(app.parameters.get("n_random", 0)),
            random_seed=int(app.parameters.get("random_seed", 0)),
        )
        for seed in seeds:
            digest = seed.structure_hash or seed.candidate_id
            if digest in seen:
                continue
            seen.add(digest)
            found.append((f"{formula}/{seed.candidate_id}", Path(seed.structure_path)))
    return found


def application_elements(app: ApplicationConfig) -> list[str]:
    """Every element the application's structures can contain."""
    from matsim_agents.core.composition import parse_composition

    found = set(app.species)
    if app.target is not None:
        found.update(app.target.elements)
    for formula in app.compositions:
        parsed = parse_composition(formula)
        if parsed is not None:
            found.update(parsed.elements)
    return sorted(found)


def coverage_blockers(app: ApplicationConfig, cfg) -> list[str]:
    """Elements the configured MLIP cannot model."""
    if cfg.mlip is None:
        return []
    missing = cfg.mlip.unsupported_elements(application_elements(app))
    if not missing:
        return []
    return [f"MLIP {cfg.mlip.model_reference} does not cover element(s) {', '.join(missing)}"]


def preflight(path: Path, options: runner.RunOptions) -> runner.Preflight:
    app = load_application(path)
    summary = {"application": app.name, "mode": app.mode, **check_application(app)}
    blockers: list[str] = []
    if app.mode.endswith("_relax"):
        template = _relax_template(app)
        probe = _relax_config(template, Path(path), Path("."))
        blockers = [b for b in runner.relaxation_blockers(probe) if "structure_path" not in b]
        blockers += coverage_blockers(app, probe)
    return runner.Preflight(summary, blockers)


def _relax_one(app, template, label: str, structure: Path, run: ScientificRunDirectory) -> dict:
    from matsim_agents.workflows.relaxation import run_relaxation

    cfg = _relax_config(template, structure, run.path / "relaxations" / label.replace("/", "_"))
    result = run_relaxation(cfg)
    stage = result.stages[-1] if result.stages else None
    return {
        "candidate": label,
        "status": result.status.value,
        "converged": bool(stage and stage.converged),
        "energy_eV": stage.energy_eV if stage else None,
        "max_force_eV_per_A": stage.max_force_eV_per_A if stage else None,
        "force_tolerance_eV_per_A": cfg.force_tolerance_eV_per_A,
        "backend": stage.backend if stage else None,
        "evidence_level": stage.evidence_level.value if stage else None,
        "run_directory": result.run_directory,
        "failure_reason": result.failure_reason,
    }


def execute(path: Path, run: ScientificRunDirectory, options: runner.RunOptions) -> runner.Outcome:
    app = load_application(path)
    summary: dict[str, Any] = {"application": app.name, "mode": app.mode}
    summary.update(check_application(app))
    if app.mode == "validate":
        return runner.Outcome("complete", str(run.path), summary)

    items = _candidates(app, run.path)
    structures_dir = run.path / "structures"
    structures_dir.mkdir(exist_ok=True)
    files: list[tuple[str, Path]] = []
    for label, item in items:
        if isinstance(item, Atoms):
            target = structures_dir / f"{label}.extxyz"
            write(target, item)
            item = target
        files.append((label, item))
    summary["structures"] = [{"candidate": label, "path": str(file)} for label, file in files]
    if app.mode == "generate":
        return runner.Outcome("complete", str(run.path), summary)

    template = _relax_template(app)
    results = [_relax_one(app, template, label, file, run) for label, file in files]
    summary["relaxations"] = results
    summary["scientific_status"] = _SCIENTIFIC_STATUS
    failed = [r for r in results if r["status"] not in {"complete", "partial"}]
    unconverged = [r for r in results if r["status"] == "complete" and not r["converged"]]
    summary["converged"] = sum(r["converged"] for r in results)
    summary["not_converged"] = len(unconverged)
    if failed:
        reasons = "; ".join(f"{r['candidate']}: {r['failure_reason']}" for r in failed)
        return runner.Outcome("failed", str(run.path), summary, reasons)
    state = "partial" if unconverged else "complete"
    return runner.Outcome(state, str(run.path), summary)
