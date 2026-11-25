"""Full-size (1,024-site) QE campaign for transmutation and H/He vacancy-complex configurations.

Every configuration starts from the full parent supercell. Its atom count follows
``N_atoms = N_parent - N_vac + N_He + N_H`` (transmutation substitutes and keeps the count). A cell
is never shrunk. Per configuration the manifest records the configuration identity, numerical
settings, input hash, requested and actual cost and the job id. Status values:

* ``INPUT_BLOCKED``: a pseudopotential is missing or fails its header check; no input is written.
* ``RESOURCE_BLOCKED``: the input is written and kept, but the queue limits or the node-hour
  ledger cannot cover the request.
* ``READY``: input verified and the request fits.
* ``SUBMITTING`` / ``SUBMITTED`` / ``NEEDS_MANUAL_CHECK``: dispatch states; nothing is polled.

Nothing here is a validated numerical result: no job has been run, the k-mesh and cutoffs are
whatever the campaign file states, and their convergence is ``not_run``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from ase.io import write as ase_write

from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input
from matsim_agents.campaign.budget import BudgetLedger
from matsim_agents.campaign.hierarchy_campaign import DispatchOutcome
from matsim_agents.campaign.qe_plan import ResourceLimits, ResourceRequest, resource_problems
from matsim_agents.core.hashing import sha256_file as _sha256
from matsim_agents.science.defects.gas import (
    GasComplexSpec,
    GasPlacement,
    generate_gas_complex,
)
from matsim_agents.science.irradiation.transmutation import (
    TransmutationScenario,
    apply_transmutation,
)
from matsim_agents.science.structures.alloy_ensemble import occupation_fingerprint
from matsim_agents.science.structures.species_support import assess_pseudopotentials
from matsim_agents.workflows.full_supercell_campaign import (
    FullSupercellCampaign,
    build_parent,
)

GAS_MANIFEST_FILE = "gas_campaign_manifest.json"
GAS_LEDGER_FILE = "gas_campaign_ledger.json"
LEDGER_CATEGORY = "defect"
NUMERICAL_NOTE = (
    "k-mesh and cutoffs are not convergence-tested for this alloy; no result is validated"
)


class GasCampaignError(ValueError):
    """The campaign cannot be prepared without breaking a full-size invariant."""


def expected_gas_atoms(n_parent: int, n_vacancies: int, n_he: int, n_h: int) -> int:
    return n_parent - n_vacancies + n_he + n_h


@dataclass(frozen=True)
class GasExample:
    """One configuration with its identity and an exact atom-count record."""

    calc_id: str
    atoms: Atoms
    config_id: str
    notation: str
    n_vacancies: int
    n_he: int
    n_h: int
    n_re: int = 0
    n_os: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)


def _nearest_neighbors(parent: Atoms, site: int, a: float, count: int) -> list[int]:
    nn = a * math.sqrt(3.0) / 2.0
    _, d = get_distances(parent.positions[site], parent.positions, cell=parent.cell, pbc=True)
    close = [int(i) for i in np.flatnonzero(np.abs(d[0] - nn) < 0.05 * nn) if int(i) != site]
    if len(close) < count:
        raise GasCampaignError(f"site {site} has only {len(close)} nearest neighbours")
    return close[:count]


def _gas(parent: Atoms, a: float, calc_id: str, parent_id: str, spec: GasComplexSpec, base: Atoms):
    atoms, config = generate_gas_complex(base, parent_id, spec)
    n_parent = len(parent)
    want = expected_gas_atoms(n_parent, config.n_vacancies, config.n_he, config.n_h)
    if len(atoms) != want or config.n_atoms != want:
        raise GasCampaignError(f"{calc_id}: {len(atoms)} atoms, expected {want}")
    return atoms, config


def build_gas_examples(cfg: FullSupercellCampaign) -> list[GasExample]:
    """Ten example configurations on the full parent: gas complexes and Re/Os transmutation."""
    parent = build_parent(cfg)
    a = cfg.parent_supercell.lattice_parameter
    parent_id = f"{cfg.campaign_id}-parent"
    n_parent = len(parent)
    nbr = _nearest_neighbors(parent, 0, a, 4)

    def gas(name: str, spec: GasComplexSpec) -> GasExample:
        atoms, config = _gas(parent, a, name, parent_id, spec, parent)
        return GasExample(
            calc_id=name,
            atoms=atoms,
            config_id=config.config_id,
            notation=config.notation,
            n_vacancies=config.n_vacancies,
            n_he=config.n_he,
            n_h=config.n_h,
            provenance={"parent_ordering_id": config.parent_ordering_id},
        )

    def spec(vac: Sequence[int] = (), *placements: GasPlacement) -> GasComplexSpec:
        return GasComplexSpec(vacancy_sites=tuple(vac), gas=placements, lattice_parameter=a)

    he_cav = GasPlacement(species="He", placement="vacancy", anchor_site=0)
    out = [
        gas("V1", spec((0,))),
        gas("He1", spec((), GasPlacement(species="He", placement="tetrahedral", anchor_site=0))),
        gas("H1", spec((), GasPlacement(species="H", placement="tetrahedral", anchor_site=0))),
        gas("V1He1", spec((0,), he_cav)),
        gas("V1H1", spec((0,), GasPlacement(species="H", placement="vacancy", anchor_site=0))),
        gas(
            "V2He2",
            spec(
                (0, nbr[0]),
                GasPlacement(species="He", placement="cluster_centroid"),
                GasPlacement(species="He", placement="cluster_centroid"),
            ),
        ),
        gas(
            "V1He2H1",
            spec(
                (0,),
                he_cav,
                he_cav,
                GasPlacement(species="H", placement="tetrahedral", anchor_site=nbr[1]),
            ),
        ),
    ]

    def transmuted(scenario_id: str, n_re: int, n_os: int, seed: int):
        scenario = TransmutationScenario(
            scenario_id=scenario_id,
            parent_alloy_id=parent_id,
            exposure_scenario="synthetic-example",
            n_re=n_re,
            n_os=n_os,
            seed=seed,
        )
        atoms, record = apply_transmutation(parent, parent_id, scenario, lattice_parameter=a)
        return atoms, record

    re5, rec = transmuted("re5", 5, 0, 1)
    out.append(
        GasExample(
            calc_id="Re5",
            atoms=re5,
            config_id=f"transmutation-{rec.geometry_fingerprint[:12]}",
            notation="Re5",
            n_vacancies=0,
            n_he=0,
            n_h=0,
            n_re=5,
            provenance={"scenario": rec.scenario.scenario_id, "evidence_level": "synthetic"},
        )
    )
    reos, rec = transmuted("re3os2", 3, 2, 2)
    out.append(
        GasExample(
            calc_id="Re3Os2",
            atoms=reos,
            config_id=f"transmutation-{rec.geometry_fingerprint[:12]}",
            notation="Re3Os2",
            n_vacancies=0,
            n_he=0,
            n_h=0,
            n_re=3,
            n_os=2,
            provenance={"scenario": rec.scenario.scenario_id, "evidence_level": "synthetic"},
        )
    )
    mixed, rec = transmuted("re2os1", 2, 1, 3)
    atoms, config = _gas(parent, a, "Re2Os1_V1He1", parent_id, spec((0,), he_cav), mixed)
    out.append(
        GasExample(
            calc_id="Re2Os1_V1He1",
            atoms=atoms,
            # The gas-complex id does not see the transmuted host, so the host is appended.
            config_id=f"{config.config_id}-{rec.geometry_fingerprint[:12]}",
            notation="Re2Os1+V1He1",
            n_vacancies=1,
            n_he=1,
            n_h=0,
            n_re=2,
            n_os=1,
            provenance={
                "scenario": rec.scenario.scenario_id,
                "transmuted_parent_ordering_id": occupation_fingerprint(mixed),
            },
        )
    )
    for ex in out:
        want = expected_gas_atoms(n_parent, ex.n_vacancies, ex.n_he, ex.n_h)
        if ex.n_atoms != want:
            raise GasCampaignError(f"{ex.calc_id}: {ex.n_atoms} atoms, expected {want}")
    return out


def _composition(atoms: Atoms) -> dict[str, int]:
    symbols, counts = np.unique(atoms.get_chemical_symbols(), return_counts=True)
    return {str(s): int(n) for s, n in zip(symbols, counts, strict=True)}


def _atomic_json(path: Path, payload: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".gas-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def prepare_gas_campaign(
    cfg: FullSupercellCampaign,
    root: str | os.PathLike,
    *,
    git_revision: str,
    request: ResourceRequest,
    limits: ResourceLimits,
    ledger: BudgetLedger | None = None,
    extra_pseudopotentials: Mapping[str, str] | None = None,
    examples: Sequence[GasExample] | None = None,
) -> list[dict[str, Any]]:
    """Validate, write inputs where pseudopotentials exist, gate resources and write a manifest.

    Nothing is submitted. A blocked configuration keeps its structure (and its input when it
    exists) at full size.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    examples = list(examples) if examples is not None else build_gas_examples(cfg)
    files = {**cfg.qe.pseudopotentials, **(extra_pseudopotentials or {})}
    n_parent = cfg.parent_supercell.expected_parent_atoms
    resource_blockers = resource_problems(request, limits)
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
        "nspin": 1,
    }
    entries: list[dict[str, Any]] = []
    for ex in examples:
        want = expected_gas_atoms(n_parent, ex.n_vacancies, ex.n_he, ex.n_h)
        if ex.n_atoms != want:
            raise GasCampaignError(f"{ex.calc_id}: {ex.n_atoms} atoms, expected {want}")
        d = root / ex.calc_id
        d.mkdir(exist_ok=True)
        ase_write(d / "structure.extxyz", ex.atoms, format="extxyz")
        composition = _composition(ex.atoms)
        pseudo = assess_pseudopotentials(composition, cfg.qe.pseudo_dir, files)
        entry: dict[str, Any] = {
            "calc_id": ex.calc_id,
            "config_id": ex.config_id,
            "notation": ex.notation,
            "campaign_id": cfg.campaign_id,
            "n_parent_atoms": n_parent,
            "n_vacancies": ex.n_vacancies,
            "n_he": ex.n_he,
            "n_h": ex.n_h,
            "n_re": ex.n_re,
            "n_os": ex.n_os,
            "qe_atoms": ex.n_atoms,
            "composition": composition,
            "atom_count_rule": "N_atoms = N_parent - N_vac + N_He + N_H",
            "provenance": dict(ex.provenance),
            "settings": settings,
            "magnetic_treatment": "nonmagnetic",
            "pseudopotentials": {r.species: r.file for r in pseudo.records},
            "pseudopotential_sha256": {r.species: r.sha256 for r in pseudo.records if r.sha256},
            "git_revision": git_revision,
            "output_dir": str(d),
            "input_sha256": None,
            "job_id": None,
            "cost": {
                "requested_node_hours": request.node_hours,
                "actual_node_hours": None,
                "nodes": request.nodes,
                "walltime_hours": request.walltime_hours,
            },
            "numerical_qualification": "not_run",
            "numerical_note": NUMERICAL_NOTE,
        }
        if not pseudo.ready:
            entry["status"] = "INPUT_BLOCKED"
            entry["blockers"] = list(pseudo.blockers)
            entries.append(entry)
            continue
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
            pseudopotentials={r.species: r.file for r in pseudo.records},
        )
        path = Path(
            write_pw_input(ex.atoms, qs, str(d / "pw.in"), prefix=ex.calc_id, outdir="./tmp")
        )
        entry["input_sha256"] = _sha256(path)
        blockers = list(resource_blockers)
        if ledger is not None and not ledger.can_dispatch(LEDGER_CATEGORY, request.node_hours):
            blockers.append(
                f"{request.node_hours:.1f} node-hours exceeds available "
                f"{ledger.available(LEDGER_CATEGORY):.1f} in {LEDGER_CATEGORY}"
            )
        if blockers:
            entry["status"] = "RESOURCE_BLOCKED"
            entry["blockers"] = blockers
        else:
            entry["status"] = "READY"
        entries.append(entry)
    _atomic_json(root / GAS_MANIFEST_FILE, entries)
    return entries


def load_gas_manifest(root: str | os.PathLike) -> list[dict[str, Any]]:
    return json.loads((Path(root) / GAS_MANIFEST_FILE).read_text())


GasSubmitter = Callable[[dict[str, Any]], str]


def _hhmmss(hours: float) -> str:
    total = int(round(hours * 3600))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def sbatch_gas_submitter(account: str, qos: str, job_script: str) -> GasSubmitter:
    """``sbatch --parsable`` with the calculation directory passed as ``CALC_DIR``."""

    def submit(entry: dict[str, Any]) -> str:
        cost = entry["cost"]
        proc = subprocess.run(
            [
                "sbatch",
                "--parsable",
                "-q",
                qos,
                "-A",
                account,
                f"--job-name={entry['calc_id'][:40]}",
                "-N",
                str(cost["nodes"]),
                "-t",
                _hhmmss(cost["walltime_hours"]),
                f"--export=ALL,CALC_DIR={entry['output_dir']}",
                job_script,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip().split(";")[0]

    return submit


def dispatch_gas_campaign(
    root: str | os.PathLike,
    ledger: BudgetLedger,
    submitter: GasSubmitter,
    *,
    dry_run: bool = True,
    approved_by: str | None = None,
    expected_approver: str | None = None,
    max_jobs: int = 1,
) -> DispatchOutcome:
    """Submit ``READY`` entries once each, writing state before the scheduler is contacted.

    Dry-run (the default), a missing or wrong approver or an exhausted ledger stops it. There is no
    polling: submitted entries stay ``SUBMITTED`` until a later, explicit refresh.
    """
    root = Path(root)
    out = DispatchOutcome()
    if dry_run:
        out.stopped_because = "dry_run is true; nothing was submitted"
        return out
    if not approved_by or not approved_by.strip() or approved_by != expected_approver:
        out.stopped_because = "approved_by does not match the expected approver"
        return out
    entries = load_gas_manifest(root)
    for e in entries:
        if e["status"] == "SUBMITTING":
            e["status"] = "NEEDS_MANUAL_CHECK"
        if e["status"] == "NEEDS_MANUAL_CHECK":
            out.needs_manual_check.append(e["calc_id"])
    n = 0
    for e in entries:
        if e["status"] != "READY":
            if e["status"] in ("INPUT_BLOCKED", "RESOURCE_BLOCKED"):
                out.blocked[e["calc_id"]] = e["status"]
            elif e["status"] == "SUBMITTED":
                out.skipped_duplicates.append(e["calc_id"])
            continue
        if n >= max_jobs:
            out.stopped_because = f"max_jobs={max_jobs} reached"
            break
        hours = e["cost"]["requested_node_hours"]
        if not ledger.can_dispatch(LEDGER_CATEGORY, hours):
            e["status"] = "RESOURCE_BLOCKED"
            e["blockers"] = ["node-hour ledger exhausted"]
            out.blocked[e["calc_id"]] = "RESOURCE_BLOCKED"
            continue
        ledger.reserve(e["calc_id"], LEDGER_CATEGORY, hours)
        e["status"] = "SUBMITTING"
        _atomic_json(root / GAS_MANIFEST_FILE, entries)
        try:
            job_id = submitter(e)
        except Exception:  # noqa: BLE001 - unconfirmed submission must not be retried blindly
            e["status"] = "NEEDS_MANUAL_CHECK"
            out.needs_manual_check.append(e["calc_id"])
            out.stopped_because = f"submitter failed for {e['calc_id']}"
            _atomic_json(root / GAS_MANIFEST_FILE, entries)
            break
        ledger.commit(e["calc_id"], job_id)
        e["status"], e["job_id"] = "SUBMITTED", job_id
        out.submitted.append(e["calc_id"])
        n += 1
        _atomic_json(root / GAS_MANIFEST_FILE, entries)
    _atomic_json(root / GAS_MANIFEST_FILE, entries)
    _atomic_json(root / GAS_LEDGER_FILE, json.loads(ledger.model_dump_json()))
    return out


__all__ = [
    "GAS_MANIFEST_FILE",
    "GasCampaignError",
    "GasExample",
    "build_gas_examples",
    "dispatch_gas_campaign",
    "expected_gas_atoms",
    "load_gas_manifest",
    "prepare_gas_campaign",
    "sbatch_gas_submitter",
]
