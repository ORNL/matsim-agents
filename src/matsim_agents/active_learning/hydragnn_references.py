"""Resolve and freeze HydraGNN training references independently of promotion."""

from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write

from matsim_agents.active_learning.config import DFTConfig, TrainerConfig
from matsim_agents.active_learning.dataset_governance import DatasetManifest, sha256_file
from matsim_agents.active_learning.dft_protocol import dft_method_signature
from matsim_agents.active_learning.elemental_dft import (
    ElementalReferencePlan,
    prepare_elemental_references,
)
from matsim_agents.discovery.energy_references import (
    load_elemental_reference_manifest,
    validate_dataset_reference_method,
)


def _verify_frozen_inputs(identity_path: Path, identity: dict, manifest: Path) -> None:
    recorded = json.loads(identity_path.read_text())
    if recorded["inputs"] != identity:
        raise ValueError("HydraGNN training reference inputs changed; use a new AL dataset")
    if recorded["snapshot_manifest_sha256"] != sha256_file(manifest):
        raise ValueError("Frozen HydraGNN training reference manifest hash mismatch")


def _save_frozen_inputs(identity_path: Path, identity: dict, manifest: Path) -> None:
    identity_path.write_text(
        json.dumps(
            {"inputs": identity, "snapshot_manifest_sha256": sha256_file(manifest)},
            indent=2,
        )
        + "\n"
    )


def resolve_training_references(
    trainer: TrainerConfig,
    dataset: str | Path,
    *,
    dft_config: DFTConfig | None,
    reference_root: Path,
) -> Path:
    dataset = Path(dataset)
    dataset_metadata = DatasetManifest.model_validate_json(
        dataset.with_suffix(dataset.suffix + ".manifest.json").read_text()
    )
    if dataset_metadata.energy_reference != f"{dataset_metadata.dft_backend}:native_total_energy":
        raise ValueError("HydraGNN reference preparation requires native DFT total-energy labels")
    if sha256_file(dataset) != dataset_metadata.sha256:
        raise ValueError("DFT dataset manifest hash does not match the training data")
    frames = list(read(dataset, index=":"))
    elements = {symbol for atoms in frames for symbol in atoms.get_chemical_symbols()}
    if not elements:
        raise ValueError("HydraGNN training dataset is empty")
    config = trainer.hydragnn_training_references
    source = config.manifest if config is not None else trainer.validation_reference_set
    if config is not None and config.phase_plan is not None:
        if dft_config is None or config.cache_dir is None:
            raise ValueError("Automatic HydraGNN references require DFTConfig and cache_dir")
        plan = ElementalReferencePlan.from_yaml(config.phase_plan)
        identity = {
            "plan_sha256": sha256_file(config.phase_plan),
            "geometry_sha256": {
                phase.phase_id: sha256_file(phase.structure_path) for phase in plan.phases
            },
            "compound_method_signature": dft_method_signature(dft_config, elements),
        }
        source = reference_root / "elemental-references.json"
        identity_path = reference_root / "training-reference-inputs.json"
        if reference_root.exists():
            _verify_frozen_inputs(identity_path, identity, source)
        else:
            validate_dataset_reference_method(
                dataset,
                reference_backend=dft_config.backend,
                reference_method_signature=identity["compound_method_signature"],
                require_sidecar=True,
            )
            source = prepare_elemental_references(
                plan,
                dft_config,
                required_elements=elements,
                cache_dir=config.cache_dir,
                output_dir=reference_root,
                phases_approved=config.phases_approved,
                dft_approved=config.dft_approved,
                max_dft_calculations=config.max_dft_calculations,
            )
            _save_frozen_inputs(identity_path, identity, source)
    if source is None:
        raise ValueError(
            "HydraGNN training requires hydragnn_training_references or validation_reference_set "
            "even when model comparison/promotion is disabled"
        )
    manifest, _, _ = load_elemental_reference_manifest(source, required_elements=elements)
    validate_dataset_reference_method(
        dataset,
        reference_backend=manifest["backend"],
        reference_method_signature=manifest["method_signature"],
        require_sidecar=True,
    )
    if config is None or config.phase_plan is None:
        identity = {"manifest_sha256": sha256_file(source)}
        identity_path = reference_root / "training-reference-inputs.json"
        frozen = reference_root / "elemental-references.json"
        if reference_root.exists():
            _verify_frozen_inputs(identity_path, identity, frozen)
            load_elemental_reference_manifest(frozen, required_elements=elements)
        else:
            _, source_bytes, references = load_elemental_reference_manifest(
                source, required_elements=elements
            )
            reference_root.mkdir(parents=True)
            copied = {}
            for element, atoms, energy, _ in references:
                geometry = reference_root / f"{element}.extxyz"
                write(geometry, atoms)
                copied[element] = {
                    "structure_path": geometry.name,
                    "structure_sha256": sha256_file(geometry),
                    "energy_eV": energy,
                }
            frozen.write_text(
                json.dumps({**json.loads(source_bytes), "references": copied}, indent=2) + "\n"
            )
            _save_frozen_inputs(identity_path, identity, frozen)
        source = frozen
    return source
