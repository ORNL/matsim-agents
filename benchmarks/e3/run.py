"""E3 six-material inventory and paired-strategy dry run.

``inventory`` audits the frozen per-material candidate sources and blockers.
``dry-run`` exercises the full paired protocol for selected materials without any
DFT or training: it freezes a candidate pool, partitions it with the existing
``make_split`` (validation/test withheld), runs four rounds for both strategies with
separate evidence ledgers, deduplicates identical physical requests through a shared
cache, and writes the three-level accounting.

The two strategies are "LLM-guided adaptation" and "Rule-based adaptation".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.io import read, write

from benchmarks.e4.aurora_dft.prepare_stage1 import _lattice_occupations
from matsim_agents.active_learning.candidates import Candidate
from matsim_agents.active_learning.config import AcquisitionConfig
from matsim_agents.active_learning.uncertainty import select_candidates
from matsim_agents.datasets.splits import SplitConfig, make_split
from matsim_agents.execution.contracts import EvidenceLevel
from matsim_agents.execution.ledger import (
    CostRecord,
    DecisionRecord,
    EvidenceLedger,
    EvidenceOutcome,
    EvidenceRecord,
    LedgerQuantity,
    MethodRecord,
)

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("protocol.yaml")
LLM = "llm_guided_adaptation"
RULE = "rule_based_adaptation"
STRATEGIES = (LLM, RULE)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_protocol(path: Path = PROTOCOL) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = yaml.safe_load(raw)
    if set(protocol["strategies"]) != set(STRATEGIES):
        raise ValueError("protocol must define exactly the two E3 adaptation strategies")
    budget = protocol["budget"]
    if budget["rounds"] * budget["max_attempts_per_round"] < budget["max_attempts_per_strategy"]:
        raise ValueError("round structure cannot reach the per-strategy attempt ceiling")
    return protocol, sha256_bytes(raw)


def e4_excluded_ids(protocol: dict[str, Any]) -> set[str]:
    path = REPO / protocol["exclusions"]["e4_first_batch_manifest"]
    if not path.is_file():
        return set()
    return {case["calculation_id"] for case in json.loads(path.read_text())["cases"]}


# ------------------------------------------------------------------ pools --
@dataclass(frozen=True)
class PoolEntry:
    candidate_id: str
    group: str
    atoms: Atoms
    structure_sha256: str


def _structure_hash(atoms: Atoms) -> str:
    payload = {
        "symbols": atoms.get_chemical_symbols(),
        "cell": np.round(atoms.cell.array, 8).tolist(),
        "positions": np.round(atoms.positions, 8).tolist(),
    }
    return sha256_bytes(json.dumps(payload, separators=(",", ":")).encode())


def _limited(rows: list[Any], limit: int | None) -> list[Any]:
    if not limit or len(rows) <= limit:
        return rows
    # Deterministic, evenly spaced subset over the frozen inventory order.
    picks = np.linspace(0, len(rows) - 1, limit).round().astype(int)
    return [rows[i] for i in sorted(set(picks.tolist()))]


def candidate_pool(
    spec: dict[str, Any], excluded: set[str], exclude_compositions: tuple[dict, ...] = ()
) -> list[PoolEntry]:
    """Frozen pool; excluded IDs and reference compositions are removed before limiting."""
    kind = spec["kind"]
    if kind == "manifest_poscars":
        manifest = REPO / spec["manifest"]
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        rows = [
            row
            for row in rows
            if row["calculation_id"] not in excluded
            and row["composition"] not in exclude_compositions
            and row["configuration_category"] not in spec.get("exclude_categories", [])
        ]
        entries = []
        for row in _limited(rows, spec.get("limit")):
            atoms = read(manifest.parent / row["poscar"], format="vasp")
            entries.append(
                PoolEntry(
                    row["calculation_id"], row[spec["group_field"]], atoms, _structure_hash(atoms)
                )
            )
        return entries
    if kind == "lattice_occupations":
        parent = read(REPO / spec["parent_poscar"], format="vasp")
        rows = [
            json.loads(line)
            for line in (REPO / spec["inventory"]).read_text().splitlines()
            if line.strip()
        ]
        rows = [
            row
            for row in rows
            if row["material_family"] == spec["family"]
            and row["calculation_id"] not in excluded
            and row["composition"] not in exclude_compositions
        ]
        entries = []
        for row in _limited(rows, spec.get("limit")):
            atoms = _lattice_occupations(parent, row["composition"], row["random_seed"])
            group = "-".join(row["elements"]) + f"/c{row['composition_index']}"
            entries.append(PoolEntry(row["calculation_id"], group, atoms, _structure_hash(atoms)))
        return entries
    if kind == "poscar_dir":
        return [
            PoolEntry(path.stem, re.sub(r"_\d+$", "", path.stem), atoms, _structure_hash(atoms))
            for path in sorted((REPO / spec["path"]).glob("*.POSCAR"))
            for atoms in [read(path, format="vasp")]
        ]
    if kind == "seeds_only":
        return []
    raise ValueError(f"unknown pool kind {kind!r}")


# -------------------------------------------------------------- inventory --
def inventory(protocol: dict[str, Any], protocol_sha: str) -> dict[str, Any]:
    excluded = e4_excluded_ids(protocol)
    materials = []
    for key, spec in protocol["materials"].items():
        pool = spec["pool"]
        seeds = [path for path in pool.get("seeds", []) if (REPO / path).is_file()]
        try:
            materialized = len(candidate_pool(pool, excluded))
            pool_error = None
        except (OSError, KeyError, ValueError) as exc:
            materialized, pool_error = 0, str(exc)
        ready = spec["method_signature"] is not None and materialized > 0 and not spec["blockers"]
        materials.append(
            {
                "material": key,
                "name": spec["name"],
                "priority": spec["priority"],
                "split_unit": spec["split_unit"],
                "pool_kind": pool["kind"],
                "pool_materialized_candidates": materialized,
                "pool_seed_structures": len(seeds),
                "pool_error": pool_error,
                "method_signature": spec["method_signature"],
                "e3_qualified_reference_labels": 0,
                "mlip_cells": {
                    model: "pending_support_audit" for model in ("HydraGNN", "UMA", "MACE")
                },
                "activation": "ready" if ready else "blocked",
                "blockers": spec["blockers"],
            }
        )
    first = [m["material"] for m in materials if m["priority"] == 1]
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "strategies": [protocol["strategies"][s]["name"] for s in STRATEGIES],
        "budget": protocol["budget"],
        "e4_first_batch_ids_excluded": len(excluded),
        "first_paired_comparison": {
            "materials": first,
            "eligible": all(
                m["activation"] == "ready" for m in materials if m["material"] in first
            ),
        },
        "materials": materials,
    }


# --------------------------------------------------------------- strategies --
def _features(entry: PoolEntry) -> dict[str, Any]:
    symbols = entry.atoms.get_chemical_symbols()
    return {
        "id": entry.candidate_id,
        "group": entry.group,
        "composition": {s: symbols.count(s) for s in sorted(set(symbols))},
        "atoms": len(entry.atoms),
    }


def rule_based_select(pool: list[PoolEntry], n: int, seed: int) -> tuple[list[str], str]:
    """Frozen uncertainty/diversity acquisition through the existing selector.

    Live mode supplies the cell's checkpoint and ``strategy='mc_dropout'``. The dry run
    has no model loaded, so it uses the same selector with ``strategy='random'`` (seeded,
    frozen) as an explicitly labelled uncertainty placeholder.
    """
    candidates = [Candidate(e.candidate_id, e.atoms, e.candidate_id, 0) for e in pool]
    cfg = AcquisitionConfig(strategy="random", n_select=n, diversity_filter=True)
    chosen, _ = select_candidates(candidates, cfg, primary_calculator=None, seed=seed)
    return [c.candidate_id for c in chosen], "dry-run uncertainty placeholder + diversity"


def llm_prompt(material: str, pool: list[PoolEntry], evidence: list[EvidenceRecord], n: int) -> str:
    return (
        f"You are selecting up to {n} DFT calculations for E3 ({material}).\n"
        "Choose candidates whose labels best resolve the scientific question and the "
        "surrogate's failure. Ground every choice in the evidence below and cite its IDs.\n"
        "Respond with JSON only: "
        '{"selected": [candidate ids], "cited_evidence_ids": [ids], "rationale": "..."}\n'
        f"Evidence: {json.dumps([_evidence_summary(e) for e in evidence])}\n"
        f"Candidates: {json.dumps([_features(e) for e in pool])}\n"
    )


def _evidence_summary(record: EvidenceRecord) -> dict[str, Any]:
    return {
        "id": record.evidence_id,
        "kind": record.kind,
        "outcome": record.outcome.value,
        "values": [f"{v.name}={v.value}{v.unit}" for v in record.values],
    }


class DryRunChat:
    """Offline stand-in for the LLM: returns a well-formed, evidence-citing reply.

    It exists to exercise prompt construction, reply validation and ledger writing.
    Its selections carry no scientific reasoning and are labelled as such.
    """

    model_id = "dry-run-stub"

    def invoke(self, prompt: str) -> Any:
        evidence = json.loads(prompt.split("Evidence: ", 1)[1].split("\n", 1)[0])
        pool = json.loads(prompt.split("Candidates: ", 1)[1].split("\n", 1)[0])
        n = int(re.search(r"up to (\d+)", prompt).group(1))
        # Prefer unseen composition groups, then candidate order: deterministic only.
        seen, picks = set(), []
        for item in pool:
            if item["group"] not in seen and len(picks) < n:
                picks.append(item["id"])
                seen.add(item["group"])
        reply = {
            "selected": picks,
            "cited_evidence_ids": [evidence[0]["id"]],
            "rationale": "dry-run stub; no scientific reasoning performed",
        }
        return type("Reply", (), {"content": json.dumps(reply)})()


def llm_guided_select(
    chat: Any, material: str, pool: list[PoolEntry], ledger: EvidenceLedger, n: int
) -> tuple[list[str], list[str], str]:
    evidence = ledger.evidence()
    reply = chat.invoke(llm_prompt(material, pool, evidence, n)).content
    match = re.search(r"\{.*\}", reply, re.S)
    if match is None:
        raise ValueError("LLM reply contains no JSON object")
    parsed = json.loads(match.group(0))
    ids = {e.candidate_id for e in pool}
    selected = list(dict.fromkeys(parsed.get("selected", [])))
    cited = parsed.get("cited_evidence_ids", [])
    if not selected or len(selected) > n or not set(selected) <= ids:
        raise ValueError(f"LLM selection is empty, too large or outside the pool: {selected}")
    if not cited or not all(c in ledger for c in cited):
        raise ValueError("LLM selection must cite evidence present in its own ledger")
    return selected, cited, parsed.get("rationale", "")


# ---------------------------------------------------------------- dry run --
def _split(material: str, pool: list[PoolEntry], protocol: dict[str, Any], out: Path):
    frames = []
    for entry in pool:
        atoms = entry.atoms.copy()
        atoms.info.update(candidate_id=entry.candidate_id, e3_group=entry.group)
        frames.append(atoms)
    dataset = out / "candidate-pool.extxyz"
    write(dataset, frames, format="extxyz")
    sidecar = dataset.with_suffix(dataset.suffix + ".manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "dft_backend": "unlabelled-candidate-pool",
                "method_signature": None,
                "energy_reference": "none: candidate structures carry no DFT labels",
            },
            indent=2,
        )
    )
    fractions = protocol["splits"]["fractions"]
    cfg = SplitConfig(
        strategy="generic_group",
        group_by="e3_group",
        train_fraction=fractions["train"],
        validation_fraction=fractions["validation"],
        test_fraction=fractions["test"],
        seed=protocol["splits"]["seed"],
    )
    manifest = make_split(
        dataset,
        cfg,
        out / "split",
        objective=f"E3 {material}: acquisition pool vs withheld validation/test groups",
    )
    order = [e.candidate_id for e in pool]
    roles = {
        role: {order[frame.index] for frame in manifest.frames if frame.role == role}
        for role in ("train", "validation", "test")
    }
    return manifest, roles


def _seed_evidence(
    material: str, protocol_sha: str, split_sha: str, pool_size: int
) -> EvidenceRecord:
    return EvidenceRecord(
        kind="external",
        subject=material,
        campaign_id="e3",
        method=MethodRecord(backend="e3-protocol", method_signature=protocol_sha),
        evidence_level=EvidenceLevel.HYPOTHESIS,
        outcome=EvidenceOutcome.MEASURED,
        values=[LedgerQuantity(name="acquisition_pool_candidates", value=pool_size, unit="count")],
        input_sha256={"protocol": protocol_sha, "split_manifest": split_sha},
        physical_assumptions=["identical initial evidence for both adaptation strategies"],
    )


def dry_run_material(
    material: str, protocol: dict[str, Any], protocol_sha: str, out: Path, chat: Any
) -> dict[str, Any]:
    spec = protocol["materials"][material]
    pool = candidate_pool(spec["pool"], e4_excluded_ids(protocol))
    if not pool:
        return {"material": material, "status": "skipped", "reason": "no materialized pool"}
    out.mkdir(parents=True, exist_ok=False)
    manifest, roles = _split(material, pool, protocol, out)
    if manifest.failed_checks:
        raise RuntimeError(f"{material}: split leakage checks failed {manifest.failed_checks}")
    acquisition = [e for e in pool if e.candidate_id in roles["train"]]
    withheld = roles["validation"] | roles["test"]
    seed = _seed_evidence(material, protocol_sha, manifest.content_sha256, len(acquisition))
    budget = protocol["budget"]
    signature = spec["method_signature"] or "UNFROZEN"
    by_id = {e.candidate_id: e for e in acquisition}

    physical: dict[str, dict[str, Any]] = {}  # shared cache: one physical calculation per key
    report: dict[str, Any] = {}
    for strategy in STRATEGIES:
        ledger = EvidenceLedger(out / strategy / "ledger.jsonl")
        ledger.append(seed)
        requested: list[str] = []
        rounds = []
        for round_index in range(1, budget["rounds"] + 1):
            remaining = budget["max_attempts_per_strategy"] - len(requested)
            n = min(budget["max_attempts_per_round"], remaining)
            open_pool = [e for e in acquisition if e.candidate_id not in requested]
            if n <= 0 or not open_pool:
                break
            if strategy == RULE:
                round_seed = int(
                    sha256_bytes(f"{protocol['protocol_id']}|{material}|{round_index}".encode())[
                        :8
                    ],
                    16,
                )
                picks, rationale = rule_based_select(open_pool, n, round_seed)
                cited, actor = [seed.evidence_id], RULE
                version = protocol["strategies"][RULE]["policy_version"]
            else:
                picks, cited, rationale = llm_guided_select(chat, material, open_pool, ledger, n)
                actor, version = f"{LLM}:{getattr(chat, 'model_id', 'llm')}", None
            if set(picks) & withheld:
                raise RuntimeError(f"{strategy} selected withheld validation/test structures")
            for candidate_id in picks:
                key = sha256_bytes(f"{by_id[candidate_id].structure_sha256}|{signature}".encode())
                shared = key in physical
                physical.setdefault(
                    key, {"candidate_id": candidate_id, "first_requester": strategy}
                )
                ledger.append(
                    DecisionRecord(
                        campaign_id="e3",
                        iteration=round_index,
                        action="request_dft_label",
                        subject=candidate_id,
                        evidence_ids=cited,
                        rationale=rationale,
                        actor=actor,
                        policy_version=version,
                        expected_cost=CostRecord(dft_calculations=1),
                    )
                )
                requested.append(candidate_id)
                rounds.append(
                    {
                        "round": round_index,
                        "candidate_id": candidate_id,
                        "physical_key": key,
                        "shared_physical": shared,
                    }
                )
        report[strategy] = {
            "name": protocol["strategies"][strategy]["name"],
            "logical_attempts": len(requested),
            "rounds": rounds,
            "ledger": str((out / strategy / "ledger.jsonl").relative_to(out)),
            "ledger_records": len(ledger),
            "fine_tuning": "not triggered: no labels acquired in dry run",
        }
    accounting = {
        "material": material,
        "protocol_sha256": protocol_sha,
        "method_signature": signature,
        "split_manifest_sha256": manifest.content_sha256,
        "pool": {
            "acquisition": len(acquisition),
            "validation": len(roles["validation"]),
            "test": len(roles["test"]),
        },
        "leakage": {
            "split_checks": {c.name: c.status for c in manifest.checks},
            "withheld_selected": 0,
        },
        "logical_attempts": {s: report[s]["logical_attempts"] for s in STRATEGIES},
        "unique_physical_calculations": len(physical),
        "total_expenditure": {
            "dft_node_hours": None,
            "llm_tokens": None,
            "note": "dry run: nothing executed; measured costs are recorded only in live mode",
        },
        "identical_initial_evidence_id": seed.evidence_id,
        "strategies": report,
    }
    (out / "accounting.json").write_text(json.dumps(accounting, indent=2) + "\n")
    return accounting


# ----------------------------------------------------------------- replay --
MLIP_ENVS = {"mace-mp-0-small": ".venv-mace", "uma-s-1p1-omat": ".venv-uma"}


def _composition(atoms: Atoms) -> dict[str, int]:
    symbols = atoms.get_chemical_symbols()
    return {s: symbols.count(s) for s in sorted(set(symbols))}


def _reference_roles(groups: list[str]) -> dict[str, list[str]]:
    ordered = sorted(set(groups))
    return {"validation": ordered[0::2], "test": ordered[1::2]}


def prepare_replay(material: str, protocol: dict[str, Any], out: Path) -> dict[str, Any]:
    """Freeze pool, split, references and overlap checks; writes MLIP input files."""
    from benchmarks.e3 import references

    spec = protocol["materials"][material]
    out.mkdir(parents=True, exist_ok=False)
    refs, audit = [], []
    if material == "bcc_refractory":
        refs, audit = references.bcc_reference_frames(spec["reference_incar_check"])
    ref_compositions = tuple(dict(c) for c in {tuple(_composition(a).items()) for a in refs})
    pool = candidate_pool(spec["pool"], e4_excluded_ids(protocol), ref_compositions)
    manifest, roles = _split(material, pool, protocol, out)
    if manifest.failed_checks:
        raise RuntimeError(f"{material}: split leakage checks failed {manifest.failed_checks}")
    acquisition = [e for e in pool if e.candidate_id in roles["train"]]
    frames = []
    for entry in acquisition:
        atoms = entry.atoms.copy()
        atoms.info.update(frame_id=entry.candidate_id, e3_group=entry.group)
        frames.append(atoms)
    write(out / "acquisition.extxyz", frames, format="extxyz")
    ref_roles = _reference_roles([a.info["e3_group"] for a in refs])
    ref_hashes = {_structure_hash(a) for a in refs}
    pool_hashes = {e.structure_sha256 for e in pool}
    pool_compositions = [_composition(e.atoms) for e in pool]
    overlap = {
        "reference_structures_in_pool": len(ref_hashes & pool_hashes),
        "reference_compositions_in_pool": sum(c in ref_compositions for c in pool_compositions),
        "e4_first_batch_ids_in_pool": len(
            {e.candidate_id for e in pool} & e4_excluded_ids(protocol)
        ),
    }
    if any(overlap.values()):
        raise RuntimeError(f"{material}: reference/pool overlap {overlap}")
    sha = references.write_references(refs, out / "references.extxyz") if refs else None
    record = {
        "material": material,
        "method_signature": spec["method_signature"],
        "pool": {role: len(ids) for role, ids in roles.items()},
        "split_manifest_sha256": manifest.content_sha256,
        "split_checks": {c.name: c.status for c in manifest.checks},
        "references": {
            "admitted_frames": len(refs),
            "independent_groups": len(set(a.info["e3_group"] for a in refs)),
            "roles": ref_roles,
            "sha256": sha,
            "audit": audit,
            "excluded_sources": [
                s for s in references.EXCLUDED_SOURCES if s["material"] == material
            ],
        },
        "overlap": overlap,
    }
    (out / "prepare.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _predictions(directory: Path, name: str, stem: str) -> dict[str, dict[str, Any]]:
    path = directory / f"{stem}.{name}.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {row["frame_id"]: row for row in data["frames"]}


def zero_shot_metrics(refs: list[Atoms], predicted: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Force errors and group-relative energy errors (absolute conventions differ)."""
    if not refs or not predicted:
        return {"frames": 0}
    force_err, dft_e, mlip_e, groups = [], [], [], []
    for atoms in refs:
        row = predicted[atoms.info["frame_id"]]
        force_err.append(np.asarray(row["forces_eV_A"]) - atoms.get_forces())
        dft_e.append(atoms.info["energy_eV"] / len(atoms))
        mlip_e.append(row["energy_eV"] / len(atoms))
        groups.append(atoms.info["e3_group"])
    errors = np.concatenate([e.ravel() for e in force_err])
    dft_e, mlip_e, groups = np.array(dft_e), np.array(mlip_e), np.array(groups)
    relative = []
    for group in set(groups):
        mask = groups == group
        relative.extend((mlip_e[mask] - mlip_e[mask].mean()) - (dft_e[mask] - dft_e[mask].mean()))
    return {
        "frames": len(refs),
        "force_component_mae_eV_A": float(np.mean(np.abs(errors))),
        "force_component_rmse_eV_A": float(np.sqrt(np.mean(errors**2))),
        "relative_energy_mae_meV_atom": float(np.mean(np.abs(relative)) * 1000),
        "absolute_energy_offset_eV_atom": float(np.mean(mlip_e - dft_e)),
    }


class StoredForces:
    """ASE calculator returning precomputed committee forces by ``info['frame_id']``.

    MACE and UMA run in separate environments; this lets the existing
    ``score_ensemble`` consume their stored forces unchanged.
    """

    def __init__(self, rows: dict[str, dict[str, Any]]):
        self._rows = rows

    def get_forces(self, atoms: Atoms) -> np.ndarray:
        return np.asarray(self._rows[atoms.info["frame_id"]]["forces_eV_A"])

    def get_property(self, name, atoms=None, allow_calculation=True):
        return self.get_forces(atoms) if name == "forces" else None


def _committee_features(
    entry: PoolEntry, committee: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    features = _features(entry)
    forces = []
    for name, rows in committee.items():
        row = rows[entry.candidate_id]
        f = np.asarray(row["forces_eV_A"])
        forces.append(f)
        features[f"{name}_energy_eV_atom"] = round(row["energy_eV"] / len(entry.atoms), 5)
        features[f"{name}_max_force_eV_A"] = round(float(np.linalg.norm(f, axis=1).max()), 4)
    std = np.stack(forces).std(axis=0)
    features["committee_force_disagreement_eV_A"] = round(float(np.sqrt(np.mean(std**2))), 5)
    return features


def llm_replay_prompt(
    material: str,
    spec: dict[str, Any],
    features: list[dict[str, Any]],
    evidence: list[EvidenceRecord],
    n: int,
    round_index: int,
    rounds: int,
) -> str:
    return (
        f"E3 adaptive acquisition, material: {spec['name']}. Round {round_index} of {rounds}; "
        f"select up to {n} candidate structures for new DFT labels under the frozen protocol "
        f"{spec['method_signature']}.\n"
        "Goal: choose the calculations that most improve the reliability of the pretrained "
        "interatomic potentials for this material's scientific domain, given the evidence.\n"
        "For each selected candidate state its specific scientific purpose and cite the "
        "evidence IDs that support it. Use only candidate IDs and evidence IDs listed below.\n"
        'Respond with JSON only: {"selections": [{"candidate_id": "...", '
        '"scientific_purpose": "...", "cited_evidence_ids": ["..."]}], '
        '"round_rationale": "..."}\n'
        f"Evidence: {json.dumps([_evidence_summary(e) for e in evidence])}\n"
        f"Candidates: {json.dumps(features)}\n"
    )


def parse_llm_selections(
    reply: str, pool_ids: set[str], ledger: EvidenceLedger, n: int
) -> tuple[list[dict[str, Any]], str]:
    """Admissibility: in-pool, within budget, purposeful, citing this ledger's evidence."""
    match = re.search(r"\{.*\}", reply, re.S)
    if match is None:
        raise ValueError("LLM reply contains no JSON object")
    parsed = json.loads(match.group(0))
    selections = parsed.get("selections") or []
    ids = [s.get("candidate_id") for s in selections]
    if not selections or len(selections) > n or len(set(ids)) != len(ids):
        raise ValueError(f"LLM selection count invalid: {ids}")
    for item in selections:
        if item.get("candidate_id") not in pool_ids:
            raise ValueError(f"candidate outside the acquisition pool: {item.get('candidate_id')}")
        if not str(item.get("scientific_purpose", "")).strip():
            raise ValueError(f"no scientific purpose for {item['candidate_id']}")
        cited = item.get("cited_evidence_ids") or []
        if not cited or not all(isinstance(ledger.get(c), EvidenceRecord) for c in cited):
            raise ValueError(f"{item['candidate_id']} cites evidence absent from its ledger")
    return selections, str(parsed.get("round_rationale", ""))


def replay_material(
    material: str,
    protocol: dict[str, Any],
    protocol_sha: str,
    out: Path,
    chat: Any,
    model_label: str,
) -> dict[str, Any]:
    """Round-1 decisions of both strategies on real MLIP evidence (no DFT executed)."""
    spec = protocol["materials"][material]
    prepared = json.loads((out / "prepare.json").read_text())
    acquisition_atoms = read(out / "acquisition.extxyz", index=":")
    pool = [
        PoolEntry(a.info["frame_id"], a.info["e3_group"], a, _structure_hash(a))
        for a in acquisition_atoms
    ]
    refs = (
        read(out / "references.extxyz", index=":") if (out / "references.extxyz").is_file() else []
    )
    for atoms in refs:
        atoms.info["energy_eV"] = atoms.get_potential_energy()
    names = protocol["strategies"][RULE]["parameters"]["committee"]
    committee = {name: _predictions(out, name, "acquisition") for name in names}
    if not all(len(rows) == len(pool) for rows in committee.values()):
        raise RuntimeError(f"{material}: committee predictions missing for the acquisition pool")

    zero_shot, statuses = {}, {}
    min_val = protocol["splits"]["min_validation_groups"]
    min_test = protocol["splits"]["min_test_groups"]
    roles = prepared["references"]["roles"]
    for name in names:
        zero_shot[name] = zero_shot_metrics(refs, _predictions(out, name, "references"))
        if len(roles["validation"]) < min_val or len(roles["test"]) < min_test:
            statuses[name] = "insufficient_validation_evidence"
        else:  # pragma: no cover - no material currently has enough independent groups
            criteria = spec["accuracy_criteria"]
            ok = zero_shot[name]["force_component_mae_eV_A"] <= criteria["force_component_mae_eV_A"]
            statuses[name] = "pass" if ok else "fail"
    zero_shot["hydragnn-multidataset-hpo-best6-epoch97"] = {
        "status": "not_evaluated",
        "reason": protocol["zero_shot_models"]["hydragnn-multidataset-hpo-best6-epoch97"]["reason"],
    }

    seed = _seed_evidence(material, protocol_sha, prepared["split_manifest_sha256"], len(pool))
    shared_evidence = [seed]
    for name in names:
        values = [
            LedgerQuantity(name="acquisition_candidates_scored", value=len(pool), unit="count")
        ]
        values += [
            LedgerQuantity(name=k, value=round(v, 6), unit=k.rsplit("_", 1)[-1])
            for k, v in zero_shot[name].items()
            if isinstance(v, float)
        ]
        shared_evidence.append(
            EvidenceRecord(
                kind="characterization",
                subject=f"{material}:{name}",
                campaign_id="e3",
                method=MethodRecord(backend=name.split("-")[0], model=name),
                evidence_level=EvidenceLevel.MLIP_PREDICTION,
                outcome=EvidenceOutcome.MEASURED,
                values=values,
                input_sha256={
                    "acquisition_predictions": hashlib.sha256(
                        (out / f"acquisition.{name}.json").read_bytes()
                    ).hexdigest()
                },
                physical_assumptions=[protocol["zero_shot_models"][name]["convention"]],
            )
        )
    if refs:
        shared_evidence.append(
            EvidenceRecord(
                kind="characterization",
                subject=f"{material}:dft-reference-group",
                campaign_id="e3",
                method=MethodRecord(
                    backend="vasp",
                    backend_version="6.6.1",
                    method_signature=spec["method_signature"],
                ),
                evidence_level=EvidenceLevel.CONVERGED_DFT,
                outcome=EvidenceOutcome.NOT_CONVERGED,
                failure_reason="ionic relaxation incomplete; per-frame SCF-converged labels only",
                values=[LedgerQuantity(name="reference_frames", value=len(refs), unit="count")],
                input_sha256={"references": prepared["references"]["sha256"]},
            )
        )

    budget = protocol["budget"]
    n = budget["max_attempts_per_round"]
    features = [_committee_features(e, committee) for e in pool]
    by_id = {e.candidate_id: e for e in pool}
    physical: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {}
    for strategy in STRATEGIES:
        ledger = EvidenceLedger(out / strategy / "ledger.jsonl")
        for record in shared_evidence:
            ledger.append(record)
        cited_default = [r.evidence_id for r in shared_evidence]
        cost = CostRecord()
        if strategy == RULE:
            params = protocol["strategies"][RULE]["parameters"]
            candidates = [Candidate(e.candidate_id, e.atoms, e.candidate_id, 0) for e in pool]
            for c in candidates:
                c.atoms.info["frame_id"] = c.candidate_id
            cfg = AcquisitionConfig(
                strategy=params["strategy"],
                n_select=n,
                diversity_filter=params["diversity_filter"],
                min_uncertainty_eV_per_A=params["min_uncertainty_eV_per_A"],
            )
            calculators = [StoredForces(committee[name]) for name in names]
            chosen, scores = select_candidates(candidates, cfg, None, calculators)
            score = dict(zip([c.candidate_id for c in candidates], scores, strict=True))
            selections = [
                {
                    "candidate_id": c.candidate_id,
                    "scientific_purpose": (
                        f"committee force disagreement {score[c.candidate_id]:.4f} eV/A "
                        f"(rank {int((scores > score[c.candidate_id]).sum()) + 1} "
                        f"of {len(scores)}); "
                        "diversity-filtered"
                    ),
                    "cited_evidence_ids": cited_default,
                }
                for c in chosen
            ]
            rationale, actor = "frozen ensemble uncertainty + diversity rule", RULE
            version = protocol["strategies"][RULE]["policy_version"]
        else:
            prompt = llm_replay_prompt(
                material, spec, features, ledger.evidence(), n, 1, budget["rounds"]
            )
            started = time.perf_counter()
            try:
                response = chat.invoke(prompt)
            except Exception as exc:  # recorded, never replaced by another model
                report[strategy] = {
                    "name": protocol["strategies"][strategy]["name"],
                    "status": "blocked_llm_unavailable",
                    "error": f"{type(exc).__name__}: {exc}"[:600],
                    "logical_attempts": 0,
                    "decisions": [],
                    "ledger_records": len(ledger),
                }
                continue
            usage = getattr(response, "usage_metadata", None) or {}
            cost = CostRecord(
                llm_prompt_tokens=int(usage.get("input_tokens", 0)),
                llm_completion_tokens=int(usage.get("output_tokens", 0)),
                llm_seconds=time.perf_counter() - started,
            )
            (out / strategy / "llm-round-1.json").write_text(
                json.dumps({"prompt": prompt, "reply": response.content}, indent=2)
            )
            selections, rationale = parse_llm_selections(response.content, set(by_id), ledger, n)
            actor, version = f"{LLM}:{model_label}", None
        withheld = set(prepared["references"]["roles"]["validation"]) | set(
            prepared["references"]["roles"]["test"]
        )
        rounds = []
        for item in selections:
            cid = item["candidate_id"]
            if by_id[cid].group in withheld:
                raise RuntimeError(f"{strategy} selected a withheld reference group")
            key = sha256_bytes(f"{by_id[cid].structure_sha256}|{spec['method_signature']}".encode())
            shared = key in physical
            physical.setdefault(key, {"candidate_id": cid, "first_requester": strategy})
            ledger.append(
                DecisionRecord(
                    campaign_id="e3",
                    iteration=1,
                    action="request_dft_label",
                    subject=cid,
                    evidence_ids=item["cited_evidence_ids"],
                    rationale=f"{item['scientific_purpose']} | round: {rationale}",
                    actor=actor,
                    policy_version=version,
                    expected_cost=CostRecord(dft_calculations=1),
                    observed_cost=cost if strategy == LLM else None,
                )
            )
            rounds.append(
                {
                    "round": 1,
                    **item,
                    "shared_physical": shared,
                    "features": next(f for f in features if f["id"] == cid),
                }
            )
        report[strategy] = {
            "name": protocol["strategies"][strategy]["name"],
            "logical_attempts": len(rounds),
            "decisions": rounds,
            "llm_cost": cost.model_dump() if strategy == LLM else None,
            "ledger_records": len(ledger),
        }
    result = {
        "material": material,
        "protocol_sha256": protocol_sha,
        "method_signature": spec["method_signature"],
        "pool": prepared["pool"],
        "split_checks": prepared["split_checks"],
        "overlap": prepared["overlap"],
        "references": {k: v for k, v in prepared["references"].items() if k != "audit"},
        "reference_audit": prepared["references"]["audit"],
        "zero_shot": zero_shot,
        "cell_status": statuses,
        "logical_attempts": {s: report[s]["logical_attempts"] for s in STRATEGIES},
        "unique_physical_calculations": len(physical),
        "dft_executed": 0,
        "rounds_completed": 1,
        "rounds_pending": "2-4 require the DFT outcomes of round 1 (not authorized)",
        "identical_initial_evidence_ids": [r.evidence_id for r in shared_evidence],
        "strategies": report,
    }
    (out / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ---------------------------------------------------------------- round 1 --
def soap_descriptors(
    atoms_list: list[Atoms], species: list[str], scratch: Path
) -> list[np.ndarray]:
    """Existing campaign.registry SOAP descriptor for each structure (file-based API)."""
    from matsim_agents.campaign.registry import soap_descriptor

    scratch.mkdir(parents=True, exist_ok=True)
    out = []
    for index, atoms in enumerate(atoms_list):
        path = scratch / f"{index:04d}.extxyz"
        write(path, atoms, format="extxyz")
        out.append(soap_descriptor(path, species=species))
    return out


def soap_rule_select(
    ids: list[str],
    uncertainty: np.ndarray,
    descriptors: list[np.ndarray],
    reference_descriptors: list[np.ndarray],
    n: int,
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Frozen Rule-based adaptation: uncertainty-eligible SOAP max-min selection."""
    from matsim_agents.campaign.registry import soap_novelty

    order = sorted(range(len(ids)), key=lambda i: (-uncertainty[i], ids[i]))
    eligible = order[: max(n, int(np.ceil(params["eligible_top_fraction"] * len(ids))))]
    nearest = [
        soap_novelty(descriptors[i], [d for j, d in enumerate(descriptors) if j != i])
        for i in range(len(ids))
    ]
    duplicate_tol = params["duplicate_fraction_of_median_nn"] * float(np.median(nearest))
    rank = {i: r + 1 for r, i in enumerate(order)}
    u_max = float(uncertainty[eligible].max())
    chosen: list[int] = []
    table: list[dict[str, Any]] = []
    for step in range(n):
        anchors = reference_descriptors + [descriptors[j] for j in chosen]
        rows = []
        for i in eligible:
            if i in chosen:
                continue
            novelty = soap_novelty(descriptors[i], anchors)
            rows.append({"i": i, "novelty": novelty, "duplicate": novelty <= duplicate_tol})
        valid = [r for r in rows if not r["duplicate"]]
        if not valid:
            break
        d_max = max(r["novelty"] for r in valid) or 1.0
        for r in valid:
            r["value"] = float(
                uncertainty[r["i"]] / u_max + params["diversity_weight"] * (r["novelty"] / d_max)
            )
        best = min(valid, key=lambda r: (-r["value"], ids[r["i"]]))
        chosen.append(best["i"])
        table.extend(
            {
                "step": step + 1,
                "candidate_id": ids[r["i"]],
                "uncertainty_eV_A": round(float(uncertainty[r["i"]]), 6),
                "uncertainty_rank": rank[r["i"]],
                "soap_novelty": round(r["novelty"], 6),
                "near_duplicate": r["duplicate"],
                "value": round(r.get("value", float("nan")), 6) if not r["duplicate"] else None,
                "selected": r["i"] == best["i"],
            }
            for r in rows
        )
    selections = []
    for step, i in enumerate(chosen, start=1):
        row = next(r for r in table if r["step"] == step and r["selected"])
        selections.append(
            {
                "candidate_id": ids[i],
                **{
                    k: row[k]
                    for k in ("uncertainty_eV_A", "uncertainty_rank", "soap_novelty", "value")
                },
            }
        )
    meta = {
        "eligible": len(eligible),
        "duplicate_tolerance": duplicate_tol,
        "median_nearest_neighbour_novelty": float(np.median(nearest)),
    }
    return selections, [meta, *table]


_DECIMAL = re.compile(r"(?<![\w.])-?\d+\.\d+")


def unsupported_numbers(text: str, prompt: str) -> list[str]:
    """Decimal values in ``text`` that do not appear (to stated precision) in the prompt."""
    allowed = [float(x) for x in _DECIMAL.findall(prompt)]
    bad = []
    for token in _DECIMAL.findall(text):
        value, places = float(token), len(token.split(".")[1])
        tol = 0.5 * 10**-places + 1e-12
        if not any(abs(value - a) <= tol or abs(abs(value) - abs(a)) <= tol for a in allowed):
            bad.append(token)
    return bad


LLM_FIELDS = (
    "scientific_question",
    "informativeness",
    "expected_effect",
    "uncertainty_diversity_considerations",
)


def admissibility(
    item: dict[str, Any],
    pool_ids: set[str],
    withheld: set[str],
    ledger: EvidenceLedger,
    required: tuple[str, ...],
    prompt: str | None,
) -> list[str]:
    """Shared deterministic admissibility for both strategies' proposals."""
    problems = []
    cid = item.get("candidate_id")
    if cid in withheld:
        problems.append("held-out structure")
    elif cid not in pool_ids:
        problems.append("unknown candidate")
    cited = item.get("cited_evidence_ids") or []
    if not cited:
        problems.append("missing evidence citations")
    elif not all(isinstance(ledger.get(c), EvidenceRecord) for c in cited):
        problems.append("cites evidence absent from this strategy's ledger")
    for field in required:
        if not str(item.get(field, "")).strip():
            problems.append(f"missing {field}")
    if prompt is not None:
        text = " ".join(str(item.get(f, "")) for f in required)
        bad = unsupported_numbers(text, prompt)
        if bad:
            problems.append(f"unsupported numerical claims {bad}")
        others = set(re.findall(r"E4-[A-Z]+-\d{4}", text)) - pool_ids
        if others:
            problems.append(f"refers to structures outside the pool {sorted(others)}")
    return problems


def round1_prompt(
    spec: dict[str, Any],
    features: list[dict[str, Any]],
    evidence: list[EvidenceRecord],
    budget: dict[str, Any],
    n: int,
) -> str:
    return (
        "E3 adaptive characterization under a matched DFT budget (LLM-guided adaptation).\n"
        f"Material: {spec['name']}. Frozen DFT protocol: {spec['method_signature']} "
        f"({json.dumps(spec['method'])}).\n"
        f"Scientific objective: {spec['objective']}\n"
        f"Current hypothesis: {spec['hypothesis']}\n"
        f"Provisional validation criteria (not yet testable): "
        f"{json.dumps(spec['accuracy_criteria'])}\n"
        f"Budget: round 1 of {budget['rounds']}; select exactly {n} candidates now; "
        f"{budget['max_attempts_per_strategy']} DFT attempts remain for this strategy. "
        "Selections are frozen before any DFT result is available.\n"
        "Candidate descriptors: committee = MACE-MP-0 small and UMA-s-1p1 (omat); "
        "committee_force_disagreement_eV_A is the RMS per-component force standard "
        "deviation across them; soap_novelty_to_references is the SOAP cosine distance/2 "
        "to the qualified DFT reference structures (0 = identical).\n"
        "Rules: choose only listed candidate IDs; cite only listed evidence IDs; do not "
        "assert any numerical value or DFT result that is not given here.\n"
        'Respond with JSON only: {"selections": [{"candidate_id": "...", '
        '"scientific_question": "...", "cited_evidence_ids": ["..."], '
        '"informativeness": "why this DFT calculation is informative", '
        '"expected_effect": "expected effect on the hypothesis", '
        '"uncertainty_diversity_considerations": "..."}], "round_rationale": "..."}\n'
        f"Evidence: {json.dumps([_evidence_summary(e) for e in evidence])}\n"
        f"Candidates: {json.dumps(features)}\n"
    )


def round1_material(
    material: str,
    protocol: dict[str, Any],
    protocol_sha: str,
    src: Path,
    out: Path,
    chat: Any,
) -> dict[str, Any]:
    """Frozen round-1 decisions of both strategies; no DFT is requested."""
    spec = protocol["materials"][material]
    budget = protocol["budget"]
    n = budget["max_attempts_per_round"]
    prepared = json.loads((src / "prepare.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    pool_atoms = read(src / "acquisition.extxyz", index=":")
    ids = [a.info["frame_id"] for a in pool_atoms]
    refs = (
        read(src / "references.extxyz", index=":") if (src / "references.extxyz").is_file() else []
    )
    for atoms in refs:
        atoms.info["energy_eV"] = atoms.get_potential_energy()
    params = protocol["strategies"][RULE]["parameters"]
    names = params["committee"]
    committee = {name: _predictions(src, name, "acquisition") for name in names}
    if not all(len(rows) == len(ids) for rows in committee.values()):
        raise RuntimeError(f"{material}: committee predictions missing")

    from matsim_agents.active_learning.uncertainty import score_ensemble

    started = time.perf_counter()
    candidates = [Candidate(cid, a, cid, 0) for cid, a in zip(ids, pool_atoms, strict=True)]
    uncertainty = score_ensemble(candidates, [StoredForces(committee[m]) for m in names])
    uncertainty_seconds = time.perf_counter() - started
    species = sorted({s for a in [*pool_atoms, *refs] for s in a.get_chemical_symbols()})
    started = time.perf_counter()
    descriptors = soap_descriptors(pool_atoms, species, out / "soap" / "pool")
    ref_descriptors = soap_descriptors(refs, species, out / "soap" / "references")
    soap_seconds = time.perf_counter() - started
    from matsim_agents.campaign.registry import soap_novelty

    entries = {
        cid: PoolEntry(cid, a.info["e3_group"], a, _structure_hash(a))
        for cid, a in zip(ids, pool_atoms, strict=True)
    }
    features = []
    for k, cid in enumerate(ids):
        row = _committee_features(entries[cid], committee)
        row["soap_novelty_to_references"] = round(soap_novelty(descriptors[k], ref_descriptors), 5)
        features.append(row)
    by_feature = {f["id"]: f for f in features}

    zero_shot = {
        name: zero_shot_metrics(refs, _predictions(src, name, "references")) for name in names
    }
    seed = _seed_evidence(material, protocol_sha, prepared["split_manifest_sha256"], len(ids))
    shared = [seed]
    for name in names:
        values = [
            LedgerQuantity(name=k, value=round(v, 6), unit=k.rsplit("_", 1)[-1])
            for k, v in zero_shot[name].items()
            if isinstance(v, float)
        ]
        values.append(
            LedgerQuantity(name="acquisition_candidates_scored", value=len(ids), unit="count")
        )
        shared.append(
            EvidenceRecord(
                kind="characterization",
                subject=f"{material}:{name}",
                campaign_id="e3",
                method=MethodRecord(backend=name.split("-")[0], model=name),
                evidence_level=EvidenceLevel.MLIP_PREDICTION,
                outcome=EvidenceOutcome.MEASURED,
                values=values,
                input_sha256={
                    "acquisition_predictions": hashlib.sha256(
                        (src / f"acquisition.{name}.json").read_bytes()
                    ).hexdigest()
                },
                physical_assumptions=[
                    protocol["zero_shot_models"][name]["convention"],
                    "single correlated reference group; not a pass/fail test",
                ],
            )
        )
    if refs:
        shared.append(
            EvidenceRecord(
                kind="characterization",
                subject=f"{material}:dft-reference-group",
                campaign_id="e3",
                method=MethodRecord(
                    backend="vasp",
                    backend_version="6.6.1",
                    method_signature=spec["method_signature"],
                ),
                evidence_level=EvidenceLevel.CONVERGED_DFT,
                outcome=EvidenceOutcome.NOT_CONVERGED,
                failure_reason="ionic relaxation incomplete; per-frame SCF-converged labels only",
                values=[
                    LedgerQuantity(name="reference_frames", value=len(refs), unit="count"),
                    LedgerQuantity(
                        name="independent_groups",
                        value=prepared["references"]["independent_groups"],
                        unit="count",
                    ),
                ],
                input_sha256={"references": prepared["references"]["sha256"]},
                physical_assumptions=[f"composition {refs[0].get_chemical_formula()}"],
            )
        )
    shared.append(
        EvidenceRecord(
            kind="characterization",
            subject=f"{material}:acquisition-pool-uncertainty",
            campaign_id="e3",
            method=MethodRecord(backend="e3-committee", model="+".join(names)),
            evidence_level=EvidenceLevel.MLIP_PREDICTION,
            outcome=EvidenceOutcome.MEASURED,
            values=[
                LedgerQuantity(
                    name="disagreement_min", value=round(float(uncertainty.min()), 5), unit="eV/A"
                ),
                LedgerQuantity(
                    name="disagreement_median",
                    value=round(float(np.median(uncertainty)), 5),
                    unit="eV/A",
                ),
                LedgerQuantity(
                    name="disagreement_max", value=round(float(uncertainty.max()), 5), unit="eV/A"
                ),
            ],
            physical_assumptions=["committee disagreement is an uncalibrated acquisition signal"],
        )
    )
    withheld = set(prepared["references"]["roles"]["validation"]) | set(
        prepared["references"]["roles"]["test"]
    )
    pool_ids = set(ids)
    physical: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {}
    rule_diagnostics = None
    for strategy in (RULE, LLM):
        ledger = EvidenceLedger(out / strategy / "ledger.jsonl")
        for record in shared:
            ledger.append(record)
        cited = [r.evidence_id for r in shared]
        cost, prompt, rejected, diagnostics = CostRecord(), None, [], None
        started = time.perf_counter()
        if strategy == RULE:
            picks, diagnostics = soap_rule_select(
                ids, uncertainty, descriptors, ref_descriptors, n, params
            )
            proposals = [
                {
                    "candidate_id": p["candidate_id"],
                    "scientific_question": "largest committee disagreement not yet "
                    "represented structurally by the qualified reference evidence",
                    "cited_evidence_ids": cited,
                    "informativeness": f"uncertainty rank {p['uncertainty_rank']} of {len(ids)}",
                    "expected_effect": "tests force transfer where the committee disagrees most",
                    "uncertainty_diversity_considerations": (
                        f"disagreement {p['uncertainty_eV_A']} eV/A; SOAP novelty "
                        f"{p['soap_novelty']}; frozen value {p['value']}"
                    ),
                    "rule_scores": p,
                }
                for p in picks
            ]
            actor, version, rationale = RULE, protocol["strategies"][RULE]["policy_version"], ""
            decision_seconds = time.perf_counter() - started + uncertainty_seconds + soap_seconds
        else:
            prompt = round1_prompt(spec, features, ledger.evidence(), budget, n)
            response = chat.invoke(prompt)
            decision_seconds = time.perf_counter() - started
            usage = getattr(response, "usage_metadata", None) or {}
            meta = getattr(response, "response_metadata", None) or {}
            cost = CostRecord(
                llm_prompt_tokens=int(usage.get("input_tokens", 0)),
                llm_completion_tokens=int(usage.get("output_tokens", 0)),
                llm_seconds=decision_seconds,
            )
            (out / strategy / "llm-round-1.json").write_text(
                json.dumps(
                    {
                        "prompt": prompt,
                        "reply": response.content,
                        "usage": usage,
                        "response_metadata": meta,
                    },
                    indent=2,
                    default=str,
                )
            )
            match = re.search(r"\{.*\}", response.content, re.S)
            parsed = json.loads(match.group(0)) if match else {}
            proposals = (parsed.get("selections") or [])[: n + 2]
            rationale = str(parsed.get("round_rationale", ""))
            actor = f"{LLM}:{protocol['strategies'][LLM]['llm']['model']}"
            version = protocol["strategies"][LLM]["llm"]["prompt_template"]
        accepted = []
        for item in proposals:
            problems = admissibility(
                item, pool_ids, withheld, ledger, LLM_FIELDS, prompt if strategy == LLM else None
            )
            if not problems and len(accepted) >= n:
                problems = ["exceeds the round selection limit"]
            if not problems and item["candidate_id"] in {a["candidate_id"] for a in accepted}:
                problems = ["duplicate selection"]
            if problems:
                rejected.append({"proposal": item, "problems": problems})
                continue
            accepted.append(item)
        (out / strategy / "rejected-proposals.json").write_text(
            json.dumps(rejected, indent=2) + "\n"
        )
        decisions = []
        for item in accepted:
            cid = item["candidate_id"]
            key = sha256_bytes(
                f"{entries[cid].structure_sha256}|{spec['method_signature']}".encode()
            )
            shared_physical = key in physical
            physical.setdefault(key, {"candidate_id": cid, "first_requester": strategy})
            record = ledger.append(
                DecisionRecord(
                    campaign_id="e3",
                    iteration=1,
                    action="freeze_dft_request",
                    subject=cid,
                    evidence_ids=item["cited_evidence_ids"],
                    rationale=json.dumps({f: item.get(f) for f in LLM_FIELDS}),
                    actor=actor,
                    policy_version=version,
                    expected_cost=CostRecord(dft_calculations=1),
                    observed_cost=cost if strategy == LLM else None,
                )
            )
            decisions.append(
                {
                    "candidate_id": cid,
                    "state": "frozen_awaiting_dft",
                    "decision_id": record.decision_id,
                    "composition": by_feature[cid]["composition"],
                    "group": entries[cid].group,
                    "structure_sha256": entries[cid].structure_sha256,
                    "committee_force_disagreement_eV_A": by_feature[cid][
                        "committee_force_disagreement_eV_A"
                    ],
                    "uncertainty_rank": int((uncertainty > uncertainty[ids.index(cid)]).sum()) + 1,
                    "soap_novelty_to_references": by_feature[cid]["soap_novelty_to_references"],
                    "shared_physical_calculation": shared_physical,
                    **{f: item.get(f) for f in ("cited_evidence_ids", *LLM_FIELDS)},
                    **({"rule_scores": item["rule_scores"]} if "rule_scores" in item else {}),
                }
            )
        if diagnostics is not None:
            rule_diagnostics = diagnostics
            (out / strategy / "rule-scores.json").write_text(json.dumps(diagnostics, indent=2))
        report[strategy] = {
            "name": protocol["strategies"][strategy]["name"],
            "proposed": len(proposals),
            "validated": len(accepted),
            "rejected": len(rejected),
            "frozen_awaiting_dft": len(decisions),
            "dispatched": 0,
            "logical_attempts_consumed": 0,
            "remaining_logical_attempts": budget["max_attempts_per_strategy"],
            "round_rationale": rationale,
            "decision_seconds": round(decision_seconds, 3),
            "llm_tokens": {
                "prompt": cost.llm_prompt_tokens,
                "completion": cost.llm_completion_tokens,
            }
            if strategy == LLM
            else None,
            "decisions": decisions,
        }
    overlap = sorted(
        set(d["candidate_id"] for d in report[RULE]["decisions"])
        & set(d["candidate_id"] for d in report[LLM]["decisions"])
    )
    manifest = {
        "manifest": "e3-round1-frozen",
        "material": material,
        "protocol_sha256": protocol_sha,
        "method_signature": spec["method_signature"],
        "split_manifest_sha256": prepared["split_manifest_sha256"],
        "llm": protocol["strategies"][LLM]["llm"],
        "rule_policy_version": protocol["strategies"][RULE]["policy_version"],
        "selections": {s: [d["candidate_id"] for d in report[s]["decisions"]] for s in STRATEGIES},
        "overlap": overlap,
        "unique_physical_calculations_requested": len(physical),
        "dft_dispatched": 0,
    }
    manifest["sha256"] = sha256_bytes(json.dumps(manifest, sort_keys=True).encode())
    (out / "round1-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = {
        **manifest,
        "zero_shot": zero_shot,
        "costs": {
            "mlip_inference_seconds": {
                name: json.loads((src / f"acquisition.{name}.json").read_text())["seconds"]
                for name in names
            },
            "uncertainty_seconds": round(uncertainty_seconds, 3),
            "soap_seconds": round(soap_seconds, 3),
            "dft_node_hours": 0.0,
        },
        "rule_diagnostics": rule_diagnostics[0] if rule_diagnostics else None,
        "strategies": report,
    }
    (out / "round1.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--output", type=Path, required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--output", type=Path, required=True)
    dry.add_argument("--materials", nargs="+", default=["lifepo4", "bcc_refractory"])
    r1 = sub.add_parser("round1")
    r1.add_argument("--source", type=Path, required=True, help="prepared replay directory")
    r1.add_argument("--output", type=Path, required=True)
    r1.add_argument("--material", default="bcc_refractory")
    for name in ("prepare", "replay"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--materials", nargs="+", default=["lifepo4", "bcc_refractory"])
    ing = sub.add_parser("ingest", help="ingest finished DFT into copied strategy ledgers")
    ing.add_argument("--material", default="bcc_refractory")
    ing.add_argument("--round", type=int, required=True)
    ing.add_argument("--manifest", type=Path, required=True)
    ing.add_argument("--ledgers", type=Path, required=True, help="copy of the round's ledger directory")
    ing.add_argument("--replay", type=Path, required=True, help="prepared replay directory")
    ing.add_argument("--state", type=Path, required=True)
    ing.add_argument("--reassess", action="store_true", help="deterministic reassessment once ingested")
    args = parser.parse_args(argv)
    if args.command == "ingest":
        from benchmarks.e3 import controller, reassessment

        state = controller.ingest_round(
            material=args.material,
            round_number=args.round,
            manifest_path=args.manifest,
            ledger_dir=args.ledgers,
            replay_dir=args.replay,
            state_path=args.state,
        )
        if args.reassess:
            ctx = controller.load_context(args.material, args.replay, controller.DEFAULT_REFERENCE_ROLES)
            reassessment.reassess_round(state, ctx, args.ledgers.resolve())
            state.save(args.state)
        print(json.dumps(state.round(args.round).model_dump(mode="json"), indent=2))
        return 0
    protocol, protocol_sha = load_protocol()
    if args.command == "prepare":
        results = [prepare_replay(m, protocol, args.output / m) for m in args.materials]
        (args.output / "prepare-summary.json").write_text(json.dumps(results, indent=2) + "\n")
        return 0
    if args.command == "round1":
        from matsim_agents.backends.llm.provider import get_chat_model

        llm = protocol["strategies"][LLM]["llm"]
        chat = get_chat_model(
            provider=llm["provider"],
            model=llm["model"],
            temperature=llm["temperature"],
            max_tokens=llm["max_tokens"],
            mag_provider=llm["mag_provider"],
        )
        round1_material(
            args.material,
            protocol,
            protocol_sha,
            args.source / args.material,
            args.output / args.material,
            chat,
        )
        return 0
    if args.command == "replay":
        from matsim_agents.backends.llm.provider import get_chat_model

        llm = protocol["strategies"][LLM]["llm"]
        chat = get_chat_model(
            provider=llm["provider"],
            model=llm["model"],
            temperature=llm["temperature"],
            max_tokens=llm["max_tokens"],
            mag_provider=llm["mag_provider"],
        )
        results = [
            replay_material(m, protocol, protocol_sha, args.output / m, chat, llm["model"])
            for m in args.materials
        ]
        (args.output / "replay-summary.json").write_text(json.dumps(results, indent=2) + "\n")
        return 0
    if args.command == "inventory":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(inventory(protocol, protocol_sha), indent=2) + "\n")
        return 0
    results = [
        dry_run_material(m, protocol, protocol_sha, args.output / m, DryRunChat())
        for m in args.materials
    ]
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
