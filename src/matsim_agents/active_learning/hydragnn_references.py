"""Resolve and freeze HydraGNN training references independently of promotion."""

from __future__ import annotations

import json
import shutil
import tempfile
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


def _phase_plan_identity(
    plan_path: Path,
    plan: ElementalReferencePlan,
    dft_config: DFTConfig,
    elements: set[str],
) -> dict:
    return {
        "plan_sha256": sha256_file(plan_path),
        "geometry_sha256": {
            phase.phase_id: sha256_file(phase.structure_path) for phase in plan.phases
        },
        "base_method_signature": dft_method_signature(dft_config, set()),
        "element_method_signatures": {
            elem: dft_method_signature(dft_config, {elem}) for elem in sorted(elements)
        },
        "compound_method_signature": dft_method_signature(dft_config, elements),
    }


def _verify_phase_plan_identity(
    recorded_inputs: dict,
    plan_path: Path,
    plan: ElementalReferencePlan,
    dft_config: DFTConfig,
    existing_elements: set[str],
) -> None:
    if recorded_inputs.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("HydraGNN training reference inputs changed; use a new AL dataset")
    recorded_geometries = recorded_inputs.get("geometry_sha256", {})
    current_geometries = {
        phase.phase_id: sha256_file(phase.structure_path) for phase in plan.phases
    }
    if any(
        recorded_geometries.get(pid) != current_geometries.get(pid) for pid in recorded_geometries
    ):
        raise ValueError("HydraGNN training reference inputs changed; use a new AL dataset")
    if "base_method_signature" in recorded_inputs and recorded_inputs[
        "base_method_signature"
    ] != dft_method_signature(dft_config, set()):
        raise ValueError("HydraGNN training reference inputs changed; use a new AL dataset")
    if "element_method_signatures" in recorded_inputs:
        for elem in existing_elements:
            if recorded_inputs["element_method_signatures"].get(elem) != dft_method_signature(
                dft_config, {elem}
            ):
                raise ValueError("HydraGNN training reference inputs changed; use a new AL dataset")
    elif recorded_inputs.get("compound_method_signature") != dft_method_signature(
        dft_config, existing_elements
    ):
        raise ValueError("HydraGNN training reference inputs changed; use a new AL dataset")


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
    training_elements = {symbol for atoms in frames for symbol in atoms.get_chemical_symbols()}
    if not training_elements:
        raise ValueError("HydraGNN training dataset is empty")
    config = trainer.hydragnn_training_references
    if config is not None and config.phase_plan is not None:
        recorded_scope = dataset_metadata.method_signature_elements
        if recorded_scope is None:
            raise ValueError(
                "HydraGNN phase-plan references require the dataset manifest to record "
                "method_signature_elements"
            )
        signature_elements = set(recorded_scope)
        if len(signature_elements) != len(recorded_scope) or not signature_elements:
            raise ValueError("Dataset manifest has an invalid method-signature element scope")
        uncovered_elements = training_elements - signature_elements
        if uncovered_elements:
            raise ValueError(
                "Dataset manifest method-signature element scope does not cover training "
                f"elements: {sorted(uncovered_elements)}"
            )
        elements = signature_elements
    else:
        elements = training_elements
    source = config.manifest if config is not None else trainer.validation_reference_set
    if config is not None and config.phase_plan is not None:
        if dft_config is None or config.cache_dir is None:
            raise ValueError("Automatic HydraGNN references require DFTConfig and cache_dir")
        plan = ElementalReferencePlan.from_yaml(config.phase_plan)
        source = reference_root / "elemental-references.json"
        identity_path = reference_root / "training-reference-inputs.json"
        if reference_root.exists():
            recorded = json.loads(identity_path.read_text())
            if recorded.get("snapshot_manifest_sha256") != sha256_file(source):
                raise ValueError("Frozen HydraGNN training reference manifest hash mismatch")
            existing_manifest = json.loads(source.read_text())
            existing_elements = set(existing_manifest.get("references", {}).keys())
            _verify_phase_plan_identity(
                recorded.get("inputs", {}),
                config.phase_plan,
                plan,
                dft_config,
                existing_elements,
            )
            new_elements = elements - existing_elements
            if new_elements:
                all_elements = existing_elements | new_elements
                target_compound_signature = dft_method_signature(dft_config, all_elements)
                validate_dataset_reference_method(
                    dataset,
                    reference_backend=dft_config.backend,
                    reference_method_signature=target_compound_signature,
                    require_sidecar=True,
                )
                with tempfile.TemporaryDirectory(prefix="ext-references-") as temp_dir:
                    temp_output = Path(temp_dir) / "snapshot"
                    prepare_elemental_references(
                        plan,
                        dft_config,
                        required_elements=new_elements,
                        cache_dir=config.cache_dir,
                        output_dir=temp_output,
                        phases_approved=config.phases_approved,
                        dft_approved=config.dft_approved,
                        max_dft_calculations=config.max_dft_calculations,
                    )
                    new_manifest = json.loads(
                        (temp_output / "elemental-references.json").read_text()
                    )
                    for elem in new_elements:
                        geom_file = f"{elem}.extxyz"
                        shutil.copy2(temp_output / geom_file, reference_root / geom_file)
                        existing_manifest["references"][elem] = new_manifest["references"][elem]
                    existing_selection = existing_manifest.setdefault("selection", {})
                    new_selection = new_manifest.get("selection", {})
                    if "initial_approval" not in existing_selection:
                        existing_selection["initial_approval"] = {
                            "phases_approved": existing_selection.get("phases_approved", False),
                            "dft_approved": existing_selection.get("dft_approved", False),
                            "dft_attempts": existing_selection.get("dft_attempts", 0),
                        }
                    existing_selection.setdefault("extensions", []).append(
                        {
                            "elements": sorted(new_elements),
                            "phases_approved": new_selection.get("phases_approved", False),
                            "dft_approved": new_selection.get("dft_approved", False),
                            "dft_attempts": new_selection.get("dft_attempts", 0),
                        }
                    )
                    existing_selection.setdefault("phases", []).extend(
                        new_selection.get("phases", [])
                    )
                    existing_selection["phases_approved"] = bool(
                        existing_selection.get("phases_approved")
                        and new_selection.get("phases_approved")
                    )
                    existing_selection["dft_approved"] = bool(
                        existing_selection.get("dft_approved") or new_selection.get("dft_approved")
                    )
                    existing_selection["dft_attempts"] = existing_selection.get(
                        "dft_attempts", 0
                    ) + new_selection.get("dft_attempts", 0)
                    existing_manifest["method_signature"] = target_compound_signature
                    source.write_text(json.dumps(existing_manifest, indent=2) + "\n")
                    new_identity = _phase_plan_identity(
                        config.phase_plan, plan, dft_config, all_elements
                    )
                    _save_frozen_inputs(identity_path, new_identity, source)
        else:
            identity = _phase_plan_identity(config.phase_plan, plan, dft_config, elements)
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
    else:
        if source is None:
            raise ValueError(
                "HydraGNN training requires hydragnn_training_references or "
                "validation_reference_set even when model comparison/promotion is disabled"
            )
        identity = {"manifest_sha256": sha256_file(source)}
        identity_path = reference_root / "training-reference-inputs.json"
        frozen = reference_root / "elemental-references.json"
        if reference_root.exists():
            _verify_frozen_inputs(identity_path, identity, frozen)
            existing_frozen_manifest = json.loads(frozen.read_text())
            existing_elements = set(existing_frozen_manifest.get("references", {}).keys())
            new_elements = elements - existing_elements
            if new_elements:
                manifest, _, references = load_elemental_reference_manifest(
                    source, required_elements=elements
                )
                validate_dataset_reference_method(
                    dataset,
                    reference_backend=manifest["backend"],
                    reference_method_signature=manifest["method_signature"],
                    require_sidecar=True,
                )
                copied = dict(existing_frozen_manifest.get("references", {}))
                for element, atoms, energy, _ in references:
                    if element in new_elements:
                        geometry = reference_root / f"{element}.extxyz"
                        write(geometry, atoms)
                        copied[element] = {
                            "structure_path": geometry.name,
                            "structure_sha256": sha256_file(geometry),
                            "energy_eV": energy,
                        }
                source_manifest = json.loads(source.read_text())
                frozen.write_text(
                    json.dumps({**source_manifest, "references": copied}, indent=2) + "\n"
                )
                _save_frozen_inputs(identity_path, identity, frozen)
        else:
            manifest, source_bytes, references = load_elemental_reference_manifest(
                source, required_elements=elements
            )
            validate_dataset_reference_method(
                dataset,
                reference_backend=manifest["backend"],
                reference_method_signature=manifest["method_signature"],
                require_sidecar=True,
            )
            reference_root.mkdir(parents=True)
            copied = {}
            for element, atoms, energy, _ in references:
                if element in elements:
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
    manifest, _, _ = load_elemental_reference_manifest(source, required_elements=elements)
    validate_dataset_reference_method(
        dataset,
        reference_backend=manifest["backend"],
        reference_method_signature=manifest["method_signature"],
        require_sidecar=True,
    )
    return source
