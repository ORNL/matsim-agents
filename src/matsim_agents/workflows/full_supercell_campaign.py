"""Full-size parent-supercell QE campaign: production config, validation gates and manifests.

Every structure handed to QE is checked against the configured parent supercell (exact atom
count, composition, chemical ordering and the defect's expected atom-count change). Smaller
cells are never substituted; a gate failure raises ``ValueError`` before any input is written.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from matsim_agents.backends.dft.qe_convergence import parse_observables
from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input
from matsim_agents.core.application import ATOMS_PER_CONVENTIONAL_CELL
from matsim_agents.core.fileio import atomic_write_text
from matsim_agents.core.hashing import sha256_file as _sha256
from matsim_agents.core.hashing import sha256_json
from matsim_agents.core.yamlio import load_yaml_mapping
from matsim_agents.science.defects.bcc import (
    generate_bcc_interstitial,
    generate_divacancy,
    generate_frenkel_pair,
)
from matsim_agents.science.defects.point_defects import DefectConfiguration, generate_vacancy
from matsim_agents.science.magnetism.competition import StateEvaluation, rank_states
from matsim_agents.science.magnetism.config import MagneticConfiguration
from matsim_agents.science.magnetism.generators import bcc_nn_cutoff
from matsim_agents.science.magnetism.policies import configurations_for_policy, select_policy
from matsim_agents.science.magnetism.results import classify_moments, parse_qe_magnetic_output
from matsim_agents.science.magnetism.survey import (
    SURVEY_MODES,
    MagneticSurvey,
    derive_survey_for_defect,
    survey_configurations,
)
from matsim_agents.science.neb.endpoints import (
    MigrationPair,
    generate_frenkel_recombination,
    generate_interstitial_migration,
    generate_vacancy_hop,
    vacancy_neighbors,
)
from matsim_agents.science.structures.alloy_ensemble import occupation_fingerprint
from matsim_agents.science.structures.alloys import (
    AlloySpec,
    build_supercell,
    generate_alloy,
)

DEFECT_ATOM_DELTA = {"vacancy": -1, "interstitial": 1, "frenkel_pair": 0, "divacancy": -2}
MANIFEST_FILE = "campaign_manifest.json"
Status = Literal["PREPARED", "INPUT_BLOCKED", "RESOURCE_BLOCKED"]


class ParentSupercell(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lattice: Literal["sc", "bcc", "fcc", "hcp", "diamond"] = "bcc"
    conventional_repeats: tuple[StrictInt, StrictInt, StrictInt]
    expected_parent_atoms: StrictInt = Field(gt=0)
    composition: dict[str, StrictInt]
    lattice_parameter: float = Field(gt=0)
    ordering: Literal["random"] = "random"
    seed: StrictInt = 1

    @model_validator(mode="after")
    def _consistent(self) -> ParentSupercell:
        sites = ATOMS_PER_CONVENTIONAL_CELL[self.lattice] * math.prod(self.conventional_repeats)
        total = sum(self.composition.values())
        if not (sites == self.expected_parent_atoms == total):
            raise ValueError(
                f"parent supercell inconsistent: {sites} {self.lattice} sites, "
                f"expected_parent_atoms="
                f"{self.expected_parent_atoms}, composition sums to {total}"
            )
        if any(n <= 0 for n in self.composition.values()):
            raise ValueError("composition counts must be positive")
        return self


class NumericalQualification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    require_full_parent_supercell: bool = True
    prohibit_smaller_cell_substitution: bool = True


class DefectSites(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vacancy_site: StrictInt = 0
    interstitial_site: StrictInt = 0
    interstitial_species: str = Field(min_length=1)
    interstitial_site_type: str = "octahedral"
    frenkel_site: StrictInt = 5
    frenkel_anchor: StrictInt = 40
    divacancy_site: StrictInt = 0
    divacancy_shell: StrictInt = 1


class QEControls(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pseudo_dir: str
    pseudo_manifest: str | None = None
    pseudopotentials: dict[str, str] = Field(default_factory=dict)
    ecutwfc_ry: float = 45.0
    ecutrho_ry: float = 360.0
    smearing: str = "mv"
    degauss_ry: float = 0.02
    kpts: tuple[StrictInt, StrictInt, StrictInt] = (1, 1, 1)
    conv_thr_ry: float = 1.0e-6
    mixing_beta: float = 0.2
    neb_images: StrictInt = 5

    @field_validator("pseudo_dir", "pseudo_manifest")
    @classmethod
    def _expand_env(cls, value: str | None) -> str | None:
        # An unset variable stays literal, so the missing-pseudopotential gate blocks the inputs.
        return os.path.expandvars(value) if value is not None else None


class SlurmBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    account: str = Field(min_length=1)
    qos: str = "debug"
    partition: str = "batch"
    nodes: StrictInt = Field(default=8, gt=0)
    walltime: str = "02:00:00"
    job_name: str = Field(min_length=1)
    max_concurrent_jobs: StrictInt = Field(default=1, gt=0)


class TransmutationExample(BaseModel):
    """One synthetic substitution of ``host`` atoms by product counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    products: dict[str, StrictInt] = Field(min_length=1)
    seed: StrictInt = 1
    with_vacancy_he: bool = Field(
        default=False, description="Also place a V1He1 complex in the transmuted cell."
    )


class TransmutationExamples(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(min_length=1)
    examples: tuple[TransmutationExample, ...] = Field(min_length=1)


class FullSupercellCampaign(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    campaign_id: str = Field(min_length=1)
    parent_supercell: ParentSupercell
    numerical_qualification: NumericalQualification = NumericalQualification()
    defects: DefectSites
    qe: QEControls
    slurm: SlurmBudget
    transmutation: TransmutationExamples | None = None
    magnetism: MagneticSurvey | None = Field(
        default=None,
        description=(
            "Elements tried nonmagnetic, ferromagnetic and antiferromagnetic in the parent "
            "and every defect cell. Without it the material-family policy decides."
        ),
    )

    @model_validator(mode="after")
    def _surveyed_elements_in_parent(self) -> FullSupercellCampaign:
        if self.magnetism is not None:
            absent = sorted(set(self.magnetism.elements) - set(self.parent_supercell.composition))
            if absent:
                raise ValueError(f"magnetism elements {absent} are not in the parent composition")
        return self

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> FullSupercellCampaign:
        return cls.model_validate(load_yaml_mapping(path))


# --------------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------------


def _parent_spec(p: ParentSupercell) -> AlloySpec:
    return AlloySpec(
        crystal_structure=p.lattice,
        counts=dict(p.composition),
        repeats=p.conventional_repeats,
        lattice_parameter=p.lattice_parameter,
        ordering=p.ordering,
        seed=p.seed,
    )


def build_parent(cfg: FullSupercellCampaign) -> Atoms:
    atoms = generate_alloy(_parent_spec(cfg.parent_supercell))
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    validate_parent(atoms, cfg)
    return atoms


def _composition(atoms: Atoms) -> dict[str, int]:
    symbols, counts = np.unique(atoms.get_chemical_symbols(), return_counts=True)
    return {str(s): int(n) for s, n in zip(symbols, counts, strict=True)}


def validate_parent(atoms: Atoms, cfg: FullSupercellCampaign) -> str:
    """Return the parent's ordering fingerprint, or raise if it is not the configured parent."""
    p = cfg.parent_supercell
    q = cfg.numerical_qualification
    if (q.require_full_parent_supercell or q.prohibit_smaller_cell_substitution) and len(
        atoms
    ) != p.expected_parent_atoms:
        raise ValueError(
            f"parent has {len(atoms)} atoms, campaign requires {p.expected_parent_atoms}; "
            "smaller-cell substitution is prohibited"
        )
    if _composition(atoms) != dict(p.composition):
        raise ValueError(f"parent composition {_composition(atoms)} != {dict(p.composition)}")
    expected_cell = build_supercell(_parent_spec(p)).cell.array
    if not np.allclose(atoms.cell.array, expected_cell, atol=1e-6):
        raise ValueError(f"parent cell does not match the {p.lattice} supercell of the config")
    return occupation_fingerprint(atoms)


def validate_defect(
    atoms: Atoms, config: DefectConfiguration, cfg: FullSupercellCampaign, parent_fingerprint: str
) -> None:
    n_parent = cfg.parent_supercell.expected_parent_atoms
    if config.n_parent_atoms != n_parent:
        raise ValueError(f"defect parent has {config.n_parent_atoms} atoms, expected {n_parent}")
    if config.parent_ordering_id != parent_fingerprint:
        raise ValueError("defect parent_ordering_id does not match the campaign parent")
    expected = n_parent + DEFECT_ATOM_DELTA[config.defect_type]
    if len(atoms) != expected or config.n_atoms != expected:
        raise ValueError(
            f"{config.defect_type} has {len(atoms)} atoms, expected {expected} "
            f"(parent {n_parent} {DEFECT_ATOM_DELTA[config.defect_type]:+d})"
        )


def validate_magnetic(atoms: Atoms, mag: MagneticConfiguration) -> None:
    ids = tuple(int(i) for i in atoms.get_array("atom_id"))
    if mag.atom_ids != ids:
        raise ValueError(f"magnetic config {mag.config_id} atom_ids do not match the structure")
    if mag.species != tuple(atoms.get_chemical_symbols()):
        raise ValueError(f"magnetic config {mag.config_id} species do not match the structure")


def validate_neb_pair(pair: MigrationPair, expected_atoms: int) -> None:
    a, b = pair.initial, pair.final
    for name, img in (("initial", a), ("final", b)):
        if len(img) != expected_atoms:
            raise ValueError(f"NEB {name} has {len(img)} atoms, expected {expected_atoms}")
    if not np.array_equal(a.get_array("atom_id"), b.get_array("atom_id")):
        raise ValueError("NEB endpoints have different atom_id order")
    if a.get_chemical_symbols() != b.get_chemical_symbols():
        raise ValueError("NEB endpoints have different species order")
    if not np.allclose(a.cell.array, b.cell.array):
        raise ValueError("NEB endpoints have different cells")


# --------------------------------------------------------------------------------------------
# Structure set
# --------------------------------------------------------------------------------------------


def build_structures(cfg: FullSupercellCampaign) -> dict[str, Any]:
    """Parent, defects and NEB pairs, all validated against the full-size parent."""
    if cfg.parent_supercell.lattice != "bcc":
        raise ValueError(
            "the full-supercell defect set uses bcc interstitial sites; "
            f"parent lattice {cfg.parent_supercell.lattice!r} is not supported by this workflow"
        )
    a = cfg.parent_supercell.lattice_parameter
    d = cfg.defects
    n = cfg.parent_supercell.expected_parent_atoms
    parent = build_parent(cfg)
    fp = validate_parent(parent, cfg)
    pid = f"{cfg.campaign_id}-parent"

    vac, vac_cfg = generate_vacancy(parent, pid, d.vacancy_site)
    inter, inter_cfg = generate_bcc_interstitial(
        parent, pid, d.interstitial_site, d.interstitial_species, d.interstitial_site_type, a
    )
    frenkel, frenkel_cfg = generate_frenkel_pair(parent, pid, d.frenkel_site, d.frenkel_anchor, a)
    divac, divac_cfg = generate_divacancy(
        parent, pid, d.divacancy_site, shell=d.divacancy_shell, lattice_parameter=a
    )
    defects = {
        "vacancy": (vac, vac_cfg),
        "interstitial": (inter, inter_cfg),
        "frenkel_pair": (frenkel, frenkel_cfg),
        "divacancy": (divac, divac_cfg),
    }
    for atoms, config in defects.values():
        validate_defect(atoms, config, cfg, fp)

    mover = vacancy_neighbors(vac, vac_cfg)[0]
    inter_id = next(s.atom_id for s in inter_cfg.sites if s.role == "interstitial")
    # Adjacent octahedral site, a/sqrt(2) away through the tetrahedral site.
    target = inter.get_positions()[-1] + np.array([-a / 2, a / 2, 0.0])
    target = inter.cell.scaled_positions(target[None]) % 1.0 @ inter.cell.array
    nebs = {
        "neb_vacancy": generate_vacancy_hop(vac, vac_cfg, parent, mover),
        "neb_interstitial": generate_interstitial_migration(
            inter, inter_cfg, {inter_id: tuple(float(x) for x in target[0])}
        ),
        "neb_recombination": generate_frenkel_recombination(frenkel, frenkel_cfg, parent),
    }
    for key, pair in nebs.items():
        delta = {"neb_vacancy": -1, "neb_interstitial": 1, "neb_recombination": 0}[key]
        validate_neb_pair(pair, n + delta)
    return {"parent": parent, "parent_fingerprint": fp, "defects": defects, "nebs": nebs}


# --------------------------------------------------------------------------------------------
# Inputs and manifest
# --------------------------------------------------------------------------------------------


def missing_pseudopotentials(cfg: FullSupercellCampaign) -> list[str]:
    elements = sorted(cfg.parent_supercell.composition)
    pdir = Path(cfg.qe.pseudo_dir)
    files = os.listdir(pdir) if pdir.is_dir() else []
    out = []
    for el in elements:
        name = cfg.qe.pseudopotentials.get(el)
        if name is not None:
            if name not in files:
                out.append(el)
        elif not any(
            f.lower().startswith((el.lower() + ".", el.lower() + "_", el.lower() + "-"))
            and f.lower().endswith(".upf")
            for f in files
        ):
            out.append(el)
    return out


def _pseudo_map(cfg: FullSupercellCampaign) -> dict[str, str]:
    files = sorted(os.listdir(cfg.qe.pseudo_dir), key=len)
    out = {}
    for el in sorted(cfg.parent_supercell.composition):
        out[el] = cfg.qe.pseudopotentials.get(el) or next(
            f
            for f in files
            if f.lower().startswith((el.lower() + ".", el.lower() + "_", el.lower() + "-"))
            and f.lower().endswith(".upf")
        )
    return out


def prepare_campaign(
    cfg: FullSupercellCampaign, root: str | os.PathLike, *, git_revision: str
) -> list[dict[str, Any]]:
    """Validate all structures, write pw.x inputs where possible and a JSON manifest.

    Nothing is submitted. Calculations without pseudopotentials are ``INPUT_BLOCKED``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    s = build_structures(cfg)
    n_parent = cfg.parent_supercell.expected_parent_atoms
    missing = missing_pseudopotentials(cfg)
    pseudos = {} if missing else _pseudo_map(cfg)
    blocked = f"INPUT_BLOCKED: missing pseudopotentials {','.join(missing)}" if missing else None
    settings = {
        "calculation": "scf",
        "ecutwfc_ry": cfg.qe.ecutwfc_ry,
        "ecutrho_ry": cfg.qe.ecutrho_ry,
        "occupations": "smearing",
        "smearing": cfg.qe.smearing,
        "degauss_ry": cfg.qe.degauss_ry,
        "kpts": list(cfg.qe.kpts),
        "conv_thr_ry": cfg.qe.conv_thr_ry,
        "mixing_beta": cfg.qe.mixing_beta,
    }
    common = {
        "campaign_id": cfg.campaign_id,
        "parent_atoms": n_parent,
        "parent_ordering_id": s["parent_fingerprint"],
        "pseudo_dir": cfg.qe.pseudo_dir,
        "pseudopotentials": pseudos,
        "pseudopotential_sha256": {
            el: _sha256(Path(cfg.qe.pseudo_dir) / name) for el, name in pseudos.items()
        },
        "settings": settings,
        "git_revision": git_revision,
        "job_id": None,
        "numerical_qualification": "not_run",
    }
    entries: list[dict[str, Any]] = []

    def scf(
        calc_id: str,
        atoms: Atoms,
        structure_id: str,
        mag: MagneticConfiguration | None,
        structure_key: str | None = None,
    ):
        if mag is not None:
            validate_magnetic(atoms, mag)
        d = root / calc_id
        d.mkdir(exist_ok=True)
        ase_write(d / "structure.extxyz", atoms, format="extxyz")
        entry = {
            **common,
            "calc_id": calc_id,
            "kind": "scf",
            "structure_id": structure_id,
            "structure_sha256": _sha256(d / "structure.extxyz"),
            "qe_atoms": len(atoms),
            "magnetic_config": None
            if mag is None
            else {
                "config_id": mag.config_id,
                "mode": mag.mode,
                "spin": mag.spin_polarized,
                "configuration": mag.model_dump(mode="json"),
            },
            "output_dir": str(d),
            "input_sha256": None,
            "status": blocked or "PREPARED",
        }
        if structure_key is not None:
            # Rank the states sharing a structure_key (magnetism.competition.rank_states); the
            # lowest converged energy is that structure's reference, all energies are kept.
            entry["structure_key"] = structure_key
        if (
            not blocked
            and mag is not None
            and mag.spin_polarized
            and mag.qe_starting_magnetization is None
        ):
            # Moment -> QE polarization needs the real pseudopotentials' valence counts.
            entry["status"] = "INPUT_BLOCKED: starting_magnetization needs verified UPF valence"
        elif not missing:
            from matsim_agents.backends.dft.pseudo_validation import load_manifest

            qs = QESettings(
                pseudo_dir=cfg.qe.pseudo_dir,
                calculation="scf",
                ecutwfc_ry=cfg.qe.ecutwfc_ry,
                ecutrho_ry=cfg.qe.ecutrho_ry,
                occupations="smearing",
                smearing=cfg.qe.smearing,
                degauss_ry=cfg.qe.degauss_ry,
                kpts=cfg.qe.kpts,
                conv_thr_ry=cfg.qe.conv_thr_ry,
                mixing_beta=cfg.qe.mixing_beta,
                pseudopotentials=pseudos,
                pseudo_manifest=load_manifest(cfg.qe.pseudo_manifest)
                if cfg.qe.pseudo_manifest
                else None,
            )
            path = Path(
                write_pw_input(
                    atoms, qs, str(d / "pw.in"), prefix=calc_id, outdir="./tmp", magnetic_config=mag
                )
            )
            entry["input_sha256"] = _sha256(path)
        entries.append(entry)

    parent = s["parent"]
    pid = f"{cfg.campaign_id}-parent"
    survey = cfg.magnetism
    if survey is not None:
        parent_mags = survey_configurations(
            parent, pid, survey, nn_cutoff_A=bcc_nn_cutoff(cfg.parent_supercell.lattice_parameter)
        )
        for mag in parent_mags:
            scf(f"pristine_{mag.mode}", parent, pid, mag, structure_key="pristine")
        for kind, (atoms, config) in s["defects"].items():
            derived = derive_survey_for_defect(parent_mags, atoms, config.config_id, kind, survey)
            for parent_mag, mag in zip(parent_mags, derived, strict=True):
                scf(f"{kind}_{parent_mag.mode}", atoms, config.config_id, mag, structure_key=kind)
    else:
        policy = select_policy(parent)
        mags = configurations_for_policy(parent, pid, policy) if policy else []
        for mag in mags:
            scf(f"pristine_{mag.mode}", parent, pid, mag)
        if not any(m.mode == "nonmagnetic" for m in mags):
            scf("pristine_nonmagnetic", parent, pid, None)
        for kind, (atoms, config) in s["defects"].items():
            scf(kind, atoms, config.config_id, None)

    for key, pair in s["nebs"].items():
        d = root / key
        d.mkdir(exist_ok=True)
        ase_write(d / "initial.extxyz", pair.initial, format="extxyz")
        ase_write(d / "final.extxyz", pair.final, format="extxyz")
        entries.append(
            {
                **common,
                "calc_id": key,
                "kind": "neb",
                "structure_id": pair.final_config.config_id if pair.final_config else key,
                "qe_atoms": len(pair.initial),
                "n_images": cfg.qe.neb_images,
                "magnetic_config": None,
                "output_dir": str(d),
                "input_sha256": None,
                "status": blocked or "PREPARED",
                "note": "NEB requires relaxed endpoints from the defect SCF/relax first"
                + (
                    "; use the lowest-energy converged magnetic state of the defect"
                    if survey is not None
                    else ""
                ),
            }
        )
    atomic_write_text(root / MANIFEST_FILE, json.dumps(entries, indent=2, sort_keys=True))
    return entries


def collect_magnetic_survey(root: str | os.PathLike) -> dict[str, Any]:
    """Rank completed QE survey states per structure; incomplete groups stay provisional."""
    root = Path(root)
    entries = json.loads((root / MANIFEST_FILE).read_text())
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if entry.get("kind") == "scf" and entry.get("structure_key") is not None:
            groups.setdefault(entry["structure_key"], []).append(entry)
    if not groups:
        raise ValueError("manifest has no magnetic survey entries")
    results: dict[str, Any] = {}
    for key, group in groups.items():
        evaluations: list[StateEvaluation] = []
        observations: list[dict[str, Any]] = []
        output_hashes: dict[str, str] = {}
        failures: list[dict[str, str]] = []
        structure_identities: set[str] = set()
        methods: set[str] = set()
        modes = [entry["calc_id"].removeprefix(f"{key}_") for entry in group]
        if len(set(modes)) != len(modes) or set(modes) - set(SURVEY_MODES):
            raise ValueError(f"{key}: duplicate or unsupported magnetic survey states")
        for mode in SURVEY_MODES:
            if mode not in modes:
                failures.append(
                    {"calc_id": f"{key}_{mode}", "reason": "survey state missing from manifest"}
                )
        for entry in group:
            calc_id = entry["calc_id"]
            directory = root / calc_id
            config_data = (entry.get("magnetic_config") or {}).get("configuration")
            if config_data is None:
                raise ValueError(
                    f"{calc_id}: manifest lacks magnetic configuration; prepare the campaign again"
                )
            config = MagneticConfiguration.model_validate(config_data)
            structure_path = directory / "structure.extxyz"
            if (
                not structure_path.is_file()
                or not entry.get("structure_sha256")
                or _sha256(structure_path) != entry["structure_sha256"]
            ):
                raise ValueError(f"{calc_id}: prepared structure is missing or changed")
            atoms = ase_read(structure_path)
            config.validate_against(atoms)
            structure_identities.add(
                sha256_json(
                    {
                        "numbers": atoms.numbers.tolist(),
                        "positions": atoms.positions.tolist(),
                        "cell": atoms.cell.array.tolist(),
                        "atom_ids": atoms.get_array("atom_id").tolist(),
                    }
                )
            )
            pseudo_hashes = entry.get("pseudopotential_sha256")
            if not pseudo_hashes:
                failures.append({"calc_id": calc_id, "reason": "pseudopotential hashes missing"})
                continue
            pseudo_dir = Path(entry["pseudo_dir"])
            if any(
                not (pseudo_dir / name).is_file()
                or _sha256(pseudo_dir / name) != pseudo_hashes.get(element)
                for element, name in entry["pseudopotentials"].items()
            ):
                failures.append(
                    {"calc_id": calc_id, "reason": "pseudopotential missing or changed"}
                )
                continue
            method_hash = sha256_json(
                {
                    "settings": entry["settings"],
                    "pseudopotentials": pseudo_hashes,
                    "git_revision": entry["git_revision"],
                }
            )
            methods.add(method_hash)
            input_path = directory / "pw.in"
            if not input_path.is_file() or _sha256(input_path) != entry["input_sha256"]:
                failures.append({"calc_id": calc_id, "reason": "QE input missing or changed"})
                continue
            output = directory / "pw.out"
            if not output.is_file():
                failures.append({"calc_id": calc_id, "reason": "QE output missing"})
                continue
            obs = parse_observables(output, len(atoms))
            if not obs.usable:
                failures.append(
                    {
                        "calc_id": calc_id,
                        "reason": "QE output incomplete or unconverged",
                        "output_sha256": _sha256(output),
                    }
                )
                continue
            magnetic = parse_qe_magnetic_output(
                output.read_text(errors="replace"), atoms, spin_polarized=config.spin_polarized
            )
            output_hashes[config.config_id] = _sha256(output)
            evaluations.append(
                StateEvaluation(
                    config,
                    method_hash,
                    obs.energy_eV,
                    len(atoms),
                    magnetic,
                    composition=atoms.get_chemical_formula(),
                )
            )
            observations.append(
                {
                    "calc_id": calc_id,
                    "requested_mode": config.mode,
                    "energy_eV": obs.energy_eV,
                    "output_sha256": output_hashes[config.config_id],
                    "classification": (
                        classify_moments(magnetic) if config.spin_polarized else "nonmagnetic"
                    ),
                    "classification_source": (
                        "qe_site_moments" if config.spin_polarized else "non_spin_polarized_input"
                    ),
                }
            )
        if len(structure_identities) != 1:
            raise ValueError(f"{key}: magnetic states do not share the same structure")
        if len(methods) > 1:
            raise ValueError(f"{key}: magnetic states use different numerical methods")
        ranked = rank_states(evaluations)
        calc_by_config = {e["magnetic_config"]["config_id"]: e["calc_id"] for e in group}
        ranking = [
            {
                **asdict(state),
                "calc_id": calc_by_config[state.config_id],
                "output_sha256": output_hashes[state.config_id],
            }
            for state in ranked.ranking
        ]
        complete = not failures and len(evaluations) == len(group)
        ties = sum(state.within_tie_tolerance for state in ranked.ranking) > 1
        missing_moments = [
            calc_by_config[e.config.config_id]
            for e in evaluations
            if e.config.spin_polarized and (e.result is None or not e.result.local_moments_muB)
        ]
        results[key] = {
            "status": "complete" if complete else "provisional",
            "selected_calc_id": ranking[0]["calc_id"] if ranking else None,
            "ready_for_review": bool(complete and ranking and not ties and not missing_moments),
            "downstream_allowed": False,
            "ranking": ranking,
            "observations": observations,
            "incomplete": failures,
            "excluded": list(ranked.excluded),
            "missing_evidence": list(ranked.missing_evidence)
            + [f"{calc_id}: site moments missing" for calc_id in missing_moments]
            + ["QE output-to-input execution provenance is not verified"],
            "statement": ranked.statement,
        }
    report = {"campaign_id": entries[0]["campaign_id"], "structures": results}
    atomic_write_text(
        root / "magnetic_selection.json", json.dumps(report, indent=2, sort_keys=True)
    )
    return report
