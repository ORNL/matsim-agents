"""Full-size parent-supercell QE campaign: production config, validation gates and manifests.

Every structure handed to QE is checked against the configured parent supercell (exact atom
count, composition, chemical ordering and the defect's expected atom-count change). Smaller
cells are never substituted; a gate failure raises ``ValueError`` before any input is written.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from ase import Atoms
from ase.io import write as ase_write
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input
from matsim_agents.core.application import ATOMS_PER_CONVENTIONAL_CELL
from matsim_agents.core.hashing import sha256_file as _sha256
from matsim_agents.science.defects.bcc import (
    generate_bcc_interstitial,
    generate_divacancy,
    generate_frenkel_pair,
)
from matsim_agents.science.defects.point_defects import DefectConfiguration, generate_vacancy
from matsim_agents.science.magnetism.config import MagneticConfiguration
from matsim_agents.science.magnetism.policies import configurations_for_policy, select_policy
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
    pseudopotentials: dict[str, str] = Field(default_factory=dict)
    ecutwfc_ry: float = 45.0
    ecutrho_ry: float = 360.0
    smearing: str = "mv"
    degauss_ry: float = 0.02
    kpts: tuple[StrictInt, StrictInt, StrictInt] = (1, 1, 1)
    conv_thr_ry: float = 1.0e-6
    mixing_beta: float = 0.2
    neb_images: StrictInt = 5

    @field_validator("pseudo_dir")
    @classmethod
    def _expand_env(cls, value: str) -> str:
        # An unset variable stays literal, so the missing-pseudopotential gate blocks the inputs.
        return os.path.expandvars(value)


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

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> FullSupercellCampaign:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


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

    mover = vacancy_neighbors(vac, vac_cfg, a)[0]
    inter_id = next(s.atom_id for s in inter_cfg.sites if s.role == "interstitial")
    # Adjacent octahedral site, a/sqrt(2) away through the tetrahedral site.
    target = inter.get_positions()[-1] + np.array([-a / 2, a / 2, 0.0])
    target = inter.cell.scaled_positions(target[None]) % 1.0 @ inter.cell.array
    nebs = {
        "neb_vacancy": generate_vacancy_hop(vac, vac_cfg, parent, mover, a),
        "neb_interstitial": generate_interstitial_migration(
            inter, inter_cfg, {inter_id: tuple(float(x) for x in target[0])}
        ),
        "neb_recombination": generate_frenkel_recombination(frenkel, frenkel_cfg, parent, a),
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
        "settings": settings,
        "git_revision": git_revision,
        "job_id": None,
        "numerical_qualification": "not_run",
    }
    entries: list[dict[str, Any]] = []

    def scf(calc_id: str, atoms: Atoms, structure_id: str, mag: MagneticConfiguration | None):
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
            "qe_atoms": len(atoms),
            "magnetic_config": None
            if mag is None
            else {"config_id": mag.config_id, "mode": mag.mode, "spin": mag.spin_polarized},
            "output_dir": str(d),
            "input_sha256": None,
            "status": blocked or "PREPARED",
        }
        if (
            not blocked
            and mag is not None
            and mag.spin_polarized
            and mag.qe_starting_magnetization is None
        ):
            # Moment -> QE polarization needs the real pseudopotentials' valence counts.
            entry["status"] = "INPUT_BLOCKED: starting_magnetization needs verified UPF valence"
        elif not missing:
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
                "note": "NEB requires relaxed endpoints from the defect SCF/relax first",
            }
        )
    (root / MANIFEST_FILE).write_text(json.dumps(entries, indent=2, sort_keys=True))
    return entries
