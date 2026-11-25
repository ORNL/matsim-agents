"""Generate a user's own active-learning campaign: AL config and elemental reference plan.

The YAML files under ``examples/`` are illustrations. A campaign needs its own
configuration whose elemental reference plan covers every element it explores.
This module writes that configuration from the campaign's seeds or compositions.
It proposes one condensed-phase prototype per element from ASE's reference
states and flags every choice the user must review; it never approves the plan
or any DFT.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from ase import Atoms
from ase.build import bulk, molecule
from ase.data import atomic_numbers, reference_states
from ase.formula import Formula
from ase.io import read, write

from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.elemental_dft import ElementalReferencePlan

_BULK_SYMMETRIES = {"bcc", "fcc", "hcp", "diamond"}
# Ferromagnetic starting guesses for the 3d ferromagnets. The DFT result, not
# this guess, decides the final moment.
_FERROMAGNETS = {"Fe", "Co", "Ni"}
# Commensurate collinear antiferromagnetic approximations: a two-site cubic cell
# with antiparallel starting moments. The real ground state of Cr is an
# incommensurate spin-density wave.
_ANTIFERROMAGNETS = {"Cr": "bcc Cr ground state is an incommensurate spin-density wave"}
_AFM_QE_STARTING_MAGNETIZATION = 0.5
_AFM_VASP_MOMENT_MUB = 1.0
_F_BLOCK = set(range(57, 72)) | set(range(89, 104))
_NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe"}
TEMPLATE_DIRS = ("examples",)


@dataclass
class ReferenceProposal:
    """Proposed phases, their structures, and what the user must decide."""

    phases: list[dict] = field(default_factory=list)
    structures: dict[str, Atoms] = field(default_factory=dict)
    unresolved: dict[str, str] = field(default_factory=dict)
    review: list[str] = field(default_factory=list)


def elements_of(seed_paths: Iterable[Path] = (), compositions: Iterable[str] = ()) -> set[str]:
    """Elements present in seed structure files and reduced formulas."""
    elements: set[str] = set()
    for path in seed_paths:
        for atoms in read(path, index=":"):
            elements.update(atoms.get_chemical_symbols())
    for composition in compositions:
        elements.update(Formula(composition).count())
    return elements


def kpts_for_spacing(atoms: Atoms, spacing_inv_A: float) -> list[int]:
    """Smallest Monkhorst-Pack mesh with spacing <= ``spacing_inv_A`` (2*pi/A included)."""
    if not spacing_inv_A > 0:
        raise ValueError("k-point spacing must be positive")
    lengths = 2 * math.pi * (atoms.cell.reciprocal() ** 2).sum(axis=1) ** 0.5
    return [max(1, math.ceil(length / spacing_inv_A - 1e-9)) for length in lengths]


def kspacing_of_mesh(cell_lengths_A: Iterable[float], kpts: Iterable[int]) -> float:
    """Largest k-point spacing (2*pi/A) of a mesh on an orthogonal cell."""
    return max(2 * math.pi / length / n for length, n in zip(cell_lengths_A, kpts, strict=True))


def propose_reference_phases(
    elements: Iterable[str],
    *,
    backend: str,
    stress_tolerance_eV_per_A3: float,
    reference_kpts: tuple[int, int, int] | None = None,
    reference_kspacing_inv_A: float | None = None,
    relaxation_force_tolerance_eV_per_A: float | None = None,
) -> ReferenceProposal:
    """Propose one declared condensed phase per element, or flag it as unresolved."""
    if reference_kpts is not None and reference_kspacing_inv_A is not None:
        raise ValueError("Give reference k-points or a reference k-spacing, not both")
    proposal = ReferenceProposal()
    for element in sorted(elements):
        number = atomic_numbers.get(element)
        if number is None or element == "X":
            raise ValueError(f"Not a chemical element: {element!r}")
        state = reference_states[number] or {}
        symmetry = state.get("symmetry")
        if number in _F_BLOCK:
            proposal.unresolved[element] = (
                "f-electron element: declare its phase, magnetic state and any DFT+U protocol"
            )
            continue
        if symmetry in _BULK_SYMMETRIES:
            _propose_bulk(proposal, element, symmetry, backend, stress_tolerance_eV_per_A3)
            phase = proposal.phases[-1]
            if reference_kpts is not None:
                phase["kpts"] = list(reference_kpts)
            elif reference_kspacing_inv_A is not None:
                phase["kpts"] = kpts_for_spacing(
                    proposal.structures[phase["structure_path"]], reference_kspacing_inv_A
                )
        elif symmetry == "diatom":
            _propose_diatom(proposal, element, backend)
        elif symmetry == "atom" and element in _NOBLE_GASES:
            _propose_atom(proposal, element, backend)
        else:
            proposal.unresolved[element] = (
                f"ASE reference state {symmetry!r} has no generated prototype; "
                "supply the structure and declare the phase"
            )
            continue
        if relaxation_force_tolerance_eV_per_A is not None:
            proposal.phases[-1]["relaxation_force_tolerance_eV_per_A"] = (
                relaxation_force_tolerance_eV_per_A
            )
    if reference_kspacing_inv_A is not None and backend == "qe":
        proposal.review.append(
            f"Bulk phase k-meshes give a spacing <= {reference_kspacing_inv_A:.4f} 2pi/A on "
            "the unrelaxed prototype cells; not convergence-tested."
        )
    if (
        reference_kpts is None
        and reference_kspacing_inv_A is None
        and backend == "qe"
        and any(phase["kind"] == "bulk" for phase in proposal.phases)
    ):
        proposal.review.append(
            "Bulk phases use the shared dft.qe.kpts mesh; a primitive elemental cell "
            "usually needs a denser mesh than a supercell (set phase kpts or "
            "--reference-kpts)."
        )
    if backend == "vasp" and proposal.phases:
        proposal.review.append(
            "VASP k-points come from the shared INCAR/KPOINTS settings; check they suit "
            "both the primitive elemental cells and the molecule boxes."
        )
    return proposal


def _propose_bulk(
    proposal: ReferenceProposal,
    element: str,
    symmetry: str,
    backend: str,
    stress_tolerance: float,
) -> None:
    phase_id = f"{element.lower()}-{symmetry}"
    phase: dict = {
        "phase_id": phase_id,
        "element": element,
        "structure_path": f"references/{element}-{symmetry}.extxyz",
        "kind": "bulk",
        "geometry_status": "requires_relaxation",
        "magnetic_state": "nonmagnetic",
        "relaxation_stress_tolerance_eV_per_A3": stress_tolerance,
    }
    if element in _FERROMAGNETS:
        phase["magnetic_state"] = "ferromagnetic"
        if backend == "qe":
            phase["qe_magnetic_settings"] = {"nspin": 2, "starting_magnetization(1)": 0.5}
        else:
            phase["vasp_magnetic_settings"] = {"ISPIN": "2"}
        proposal.review.append(
            f"{phase_id}: ferromagnetic starting guess; confirm the converged moment."
        )
    atoms = bulk(element)
    if element in _ANTIFERROMAGNETS:
        atoms = bulk(element, cubic=True)
        if len(atoms) != 2:
            raise ValueError(f"{element}: expected a two-site cubic cell for AFM order")
        phase_id = f"{element.lower()}-{symmetry}-afm"
        phase["phase_id"] = phase_id
        phase["structure_path"] = f"references/{element}-{symmetry}-afm.extxyz"
        phase["magnetic_state"] = "antiferromagnetic"
        if backend == "qe":
            m = _AFM_QE_STARTING_MAGNETIZATION
            phase["qe_site_starting_magnetization"] = [m, -m]
        else:
            m = _AFM_VASP_MOMENT_MUB
            phase["vasp_magnetic_settings"] = {"ISPIN": "2", "MAGMOM": f"{m:g} {-m:g}"}
        proposal.review.append(
            f"{phase_id}: commensurate collinear AFM approximation of Cr "
            f"({_ANTIFERROMAGNETS[element]}); confirm the converged moments are "
            "antiparallel and nonzero."
        )
    proposal.phases.append(phase)
    proposal.structures[phase["structure_path"]] = atoms


def _propose_diatom(proposal: ReferenceProposal, element: str, backend: str) -> None:
    phase_id = f"{element.lower()}2-molecule"
    atoms = molecule(f"{element}2")
    atoms.center(vacuum=7.5)
    atoms.pbc = True
    phase: dict = {
        "phase_id": phase_id,
        "element": element,
        "structure_path": f"references/{element}2-vacuum.extxyz",
        "kind": "molecule",
        "geometry_status": "requires_relaxation",
        "magnetic_state": "singlet",
    }
    if element == "O":
        phase["magnetic_state"] = "triplet"
        if backend == "qe":
            phase["qe_magnetic_settings"] = {"nspin": 2, "tot_magnetization": 2}
        else:
            phase["vasp_magnetic_settings"] = {"ISPIN": "2", "NUPDOWN": "2"}
    if backend == "qe":
        phase["kpts"] = [1, 1, 1]
    proposal.review.append(
        f"{phase_id}: an isolated molecule in a 15 A box, not a condensed phase; "
        "check the vacuum and whether this reference suits your hull."
    )
    proposal.phases.append(phase)
    proposal.structures[phase["structure_path"]] = atoms


def _propose_atom(proposal: ReferenceProposal, element: str, backend: str) -> None:
    phase_id = f"{element.lower()}-atom"
    atoms = Atoms(element, cell=[15.0, 15.0, 15.0], pbc=True)
    atoms.center()
    phase: dict = {
        "phase_id": phase_id,
        "element": element,
        "structure_path": f"references/{element}-atom-vacuum.extxyz",
        "kind": "molecule",
        "geometry_status": "verified_converged",
        "magnetic_state": "closed_shell",
    }
    if backend == "qe":
        phase["kpts"] = [1, 1, 1]
    proposal.review.append(
        f"{phase_id}: an isolated closed-shell atom in a 15 A box, not a condensed phase; "
        "check the vacuum and whether this reference suits your hull."
    )
    proposal.phases.append(phase)
    proposal.structures[phase["structure_path"]] = atoms


def _placeholder(name: str, message: str) -> str:
    return f"${{{name}:?{message}}}"


def _dft_block(
    backend: str,
    ecutwfc_ry: float | None,
    ecutrho_ry: float | None,
    kpts: tuple[int, int, int] | None,
    settings: dict | None = None,
) -> dict:
    block = _default_dft_block(backend, ecutwfc_ry, ecutrho_ry, kpts)
    for key, value in (settings or {}).items():
        current = block[backend].get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            block[backend][key] = {**current, **value}
        else:
            block[backend][key] = value
    return block


def _default_dft_block(
    backend: str,
    ecutwfc_ry: float | None,
    ecutrho_ry: float | None,
    kpts: tuple[int, int, int] | None,
) -> dict:
    if backend == "vasp":
        return {
            "backend": "vasp",
            "vasp": {
                "vasp_bin": _placeholder("VASP_BIN", "path to vasp_std"),
                "vasp_wrapper": _placeholder("VASP_WRAPPER", "site VASP step launcher"),
                "incar_template": _placeholder("INCAR_TEMPLATE", "your campaign INCAR"),
                "potcar_dir": _placeholder("POTCAR_DIR", "POTCAR library"),
            },
        }
    return {
        "backend": "qe",
        "qe": {
            "pw_bin": _placeholder("PW_BIN", "path to pw.x"),
            "pw_wrapper": _placeholder("PW_WRAPPER", "site QE step launcher"),
            "pseudo_dir": _placeholder("PSEUDO_DIR", "pseudopotential library"),
            "ecutwfc_ry": ecutwfc_ry
            if ecutwfc_ry is not None
            else _placeholder("ECUTWFC_RY", "converged wavefunction cutoff (Ry)"),
            "ecutrho_ry": ecutrho_ry
            if ecutrho_ry is not None
            else _placeholder("ECUTRHO_RY", "converged charge-density cutoff (Ry)"),
            "kpts": list(kpts)
            if kpts is not None
            else [_placeholder(f"KPTS_{axis}", "k-points for labelled cells") for axis in "XYZ"],
            "koffset": [0, 0, 0],
            "occupations": "smearing",
            "smearing": "gaussian",
            "degauss_ry": 0.01,
        },
    }


def write_campaign(
    campaign_dir: Path,
    *,
    mlip_config: Path,
    dft_backend: str,
    seed_paths: list[Path] = (),
    compositions: list[str] = (),
    extra_elements: Iterable[str] = (),
    stress_tolerance_eV_per_A3: float = 0.01,
    reference_kpts: tuple[int, int, int] | None = None,
    reference_kspacing_inv_A: float | None = None,
    relaxation_force_tolerance_eV_per_A: float | None = None,
    ecutwfc_ry: float | None = None,
    ecutrho_ry: float | None = None,
    kpts: tuple[int, int, int] | None = None,
    dft_settings: dict | None = None,
    preset: str | None = None,
) -> ReferenceProposal:
    """Write ``al.yaml``, ``elemental-phases.yaml`` and reference structures; never overwrite."""
    if not seed_paths and not compositions:
        raise ValueError("A campaign needs seed structures or compositions")
    if seed_paths and compositions:
        raise ValueError("Give seed structures or compositions, not both")
    campaign_dir = Path(campaign_dir).resolve()
    seed_paths = [Path(path).resolve() for path in seed_paths]
    elements = elements_of(seed_paths, compositions) | set(extra_elements)
    proposal = propose_reference_phases(
        elements,
        backend=dft_backend,
        stress_tolerance_eV_per_A3=stress_tolerance_eV_per_A3,
        reference_kpts=reference_kpts,
        reference_kspacing_inv_A=reference_kspacing_inv_A,
        relaxation_force_tolerance_eV_per_A=relaxation_force_tolerance_eV_per_A,
    )
    if not proposal.phases:
        raise ValueError(f"No reference phase could be proposed: {proposal.unresolved}")
    plan_path = campaign_dir / "elemental-phases.yaml"
    al_path = campaign_dir / "al.yaml"
    targets = [plan_path, al_path, *(campaign_dir / rel for rel in proposal.structures)]
    if existing := [str(path) for path in targets if path.exists()]:
        raise FileExistsError(f"Refusing to overwrite {existing}")

    mlip_document = yaml.safe_load(Path(mlip_config).read_text())
    if not isinstance(mlip_document, dict) or not isinstance(mlip_document.get("mlip"), dict):
        raise ValueError(f"{mlip_config}: expected a top-level mlip: block")

    plan = {"protocol": "declared_phases_zero_pressure_v3", "phases": proposal.phases}
    seed_source = (
        {"kind": "paths", "paths": [str(path) for path in seed_paths]}
        if seed_paths
        else {"kind": "compositions", "compositions": list(compositions)}
    )
    config = {
        "random_seed": 0,
        "mlip": mlip_document["mlip"],
        "md": {"seed_source": seed_source},
        "acquisition": {"strategy": "random", "n_select": 32},
        "dft": _dft_block(dft_backend, ecutwfc_ry, ecutrho_ry, kpts, dft_settings),
        "trainer": {
            "enabled": True,
            "training_references": {
                "phase_plan": str(plan_path),
                "cache_dir": str(campaign_dir / "reference-cache"),
                "phases_approved": False,
                "dft_approved": False,
                # Each relaxed phase costs a relaxation (counted twice) and a single point.
                "max_dft_calculations": 3 * len(proposal.phases),
            },
        },
        "loop": {"out_dir": str(campaign_dir / "runs"), "max_dft_calculations": 0},
        "resources": {
            "preset": preset or _placeholder("RESOURCE_PRESET", "facility preset name"),
            "pools": {
                "dft": {"nodes": "all", "devices": "all", "share_with": ["training", "mlip"]},
                "training": {"nodes": 1, "devices": 1, "share_with": ["dft", "mlip"]},
                "mlip": {"nodes": 1, "devices": "all", "share_with": ["dft", "training"]},
            },
        },
    }

    for relative, atoms in proposal.structures.items():
        target = campaign_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write(target, atoms, format="extxyz")
    plan_path.write_text(_plan_header(elements, proposal) + yaml.safe_dump(plan, sort_keys=False))
    ElementalReferencePlan.from_yaml(plan_path)
    al_path.write_text(_config_header(plan_path) + yaml.safe_dump(config, sort_keys=False))
    return proposal


def _comment(lines: Iterable[str]) -> str:
    return "".join(f"# {line}".rstrip() + "\n" for line in lines)


def _plan_header(elements: set[str], proposal: ReferenceProposal) -> str:
    lines = [
        "Elemental reference plan for this campaign (generated by `matsim-agents al",
        "init-campaign`). It is a proposal, not an approved or verified reference set.",
        f"Campaign elements: {' '.join(sorted(elements))}",
        "Every element in the labelled structures needs a declared condensed phase here;",
        "formation energies, hulls and fine-tuning labels all use these references",
        "(docs/mlip-energy-conventions.md).",
        "Structures are ASE prototypes with ASE's reference lattice constants; every",
        "phase is relaxed at zero pressure before its single point.",
        "",
        "Review before setting trainer.training_references.phases_approved: true:",
    ]
    lines += [f"  - {note}" for note in proposal.review] or ["  - (no flagged items)"]
    if proposal.unresolved:
        lines += ["", "UNRESOLVED - add a declared phase for each before running:"]
        lines += [f"  - {element}: {reason}" for element, reason in proposal.unresolved.items()]
    return _comment(lines) + "\n"


def _config_header(plan_path: Path) -> str:
    return (
        _comment(
            [
                "Active-learning configuration for this campaign (generated by",
                "`matsim-agents al init-campaign`). Edit it for your system; the files under",
                "examples/ are illustrations only.",
                "",
                "Before running:",
                "  1. Set every required placeholder variable below (environment or a vars:",
                "     block); loading fails with its message while one is unset.",
                f"  2. Review {plan_path.name} and set training_references.phases_approved: true.",
                "  3. Approve the reference DFT (training_references.dft_approved: true) and",
                "     set loop.max_dft_calculations to your label budget.",
                "  4. Run `matsim-agents al validate-config al.yaml`, then `al run al.yaml`.",
            ]
        )
        + "\n"
    )


def reference_elements(path: Path) -> set[str]:
    """Elements declared by an elemental reference manifest (JSON) or phase plan (YAML)."""
    if Path(path).suffix.lower() == ".json":
        references = json.loads(Path(path).read_text()).get("references")
        return set(references) if isinstance(references, dict) else set()
    return {phase.element for phase in ElementalReferencePlan.from_yaml(Path(path)).phases}


def check_reference_coverage(trainer: TrainerConfig, elements: set[str]) -> None:
    """Fail before any DFT if the training references miss an element of the seeds."""
    if not trainer.enabled or not elements:
        return
    references = trainer.training_references
    source = (
        (references.manifest or references.phase_plan)
        if references is not None
        else trainer.validation_reference_set
    )
    if source is None:
        return
    missing = elements - reference_elements(source)
    if missing:
        raise ValueError(
            f"Elemental references {source} do not cover seed elements {sorted(missing)}. "
            "Each campaign needs its own condensed-phase references for every element it "
            "labels; generate them with `matsim-agents al init-campaign`."
        )


def is_repo_template(config_path: Path, repo_root: Path) -> bool:
    """Whether ``config_path`` is one of the illustrative configurations in the repository."""
    resolved = Path(config_path).resolve()
    return any(resolved.is_relative_to(repo_root / name) for name in TEMPLATE_DIRS)
