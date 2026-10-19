"""Resolve and freeze HydraGNN training references independently of promotion."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from ase.io import read, write

from matsim_agents.active_learning.config import TrainerConfig
from matsim_agents.active_learning.dft_protocol import dft_method_signature
from matsim_agents.active_learning.elemental_dft import (
    ElementalReferencePlan,
    prepare_elemental_references,
)
from matsim_agents.backends.dft.config import DFTConfig
from matsim_agents.datasets.energy_references import (
    load_elemental_reference_manifest,
    validate_dataset_reference_method,
)
from matsim_agents.datasets.governance import DatasetManifest, sha256_file
from matsim_agents.execution.locking import exclusive_lock
from matsim_agents.execution.placement import ProcessLaunch


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


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_snapshot(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _remove_previous_snapshot(path: Path, reference_root: Path) -> None:
    target = path.resolve(strict=False) if path.is_symlink() else None
    _remove_snapshot(path)
    if (
        target is not None
        and target.parent == reference_root.parent.resolve()
        and target.name.startswith(f".{reference_root.name}.generation-")
    ):
        _remove_snapshot(target)


def _remove_transient_links(reference_root: Path) -> None:
    for link in reference_root.parent.glob(f".{reference_root.name}.next-*"):
        if link.is_symlink():
            link.unlink()


@contextmanager
def _staged_reference_snapshot(reference_root: Path, *, copy_existing: bool):
    reference_root.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(prefix=f".{reference_root.name}.staging-", dir=reference_root.parent)
    )
    snapshot = workspace / "snapshot"
    if copy_existing:
        shutil.copytree(reference_root, snapshot)
    try:
        yield snapshot
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _recover_reference_publication(reference_root: Path) -> None:
    marker = reference_root.parent / f".{reference_root.name}.publication.json"
    if not marker.is_file():
        return
    publication = json.loads(marker.read_text())
    if any(
        not isinstance(publication.get(key), str) or Path(publication[key]).name != publication[key]
        for key in ("generation", "backup")
    ):
        raise RuntimeError("Invalid HydraGNN reference publication marker")
    generation = reference_root.parent / publication["generation"]
    backup = reference_root.parent / publication["backup"]
    if not generation.is_dir():
        if _path_exists(backup) and not _path_exists(reference_root):
            backup.replace(reference_root)
        marker.unlink()
        raise RuntimeError("Incomplete HydraGNN reference publication could not be recovered")
    if reference_root.is_symlink() and reference_root.resolve() == generation.resolve():
        _remove_previous_snapshot(backup, reference_root)
        _remove_transient_links(reference_root)
        marker.unlink()
        return
    if _path_exists(reference_root) and not _path_exists(backup):
        reference_root.replace(backup)
    next_link = reference_root.parent / f".{reference_root.name}.next-{uuid.uuid4().hex}"
    next_link.symlink_to(generation.name, target_is_directory=True)
    next_link.replace(reference_root)
    _remove_previous_snapshot(backup, reference_root)
    _remove_transient_links(reference_root)
    marker.unlink()


def _publish_reference_snapshot(reference_root: Path, staged_snapshot: Path) -> None:
    generation = reference_root.parent / f".{reference_root.name}.generation-{uuid.uuid4().hex}"
    backup = reference_root.parent / f".{reference_root.name}.previous-{uuid.uuid4().hex}"
    staged_snapshot.replace(generation)
    marker = reference_root.parent / f".{reference_root.name}.publication.json"
    marker_tmp = marker.with_suffix(".tmp")
    marker_tmp.write_text(json.dumps({"generation": generation.name, "backup": backup.name}) + "\n")
    marker_tmp.replace(marker)
    _recover_reference_publication(reference_root)


def resolve_training_references(
    trainer: TrainerConfig,
    dataset: str | Path,
    *,
    dft_config: DFTConfig | None,
    dft_launch: ProcessLaunch | None = None,
    reference_root: Path,
) -> Path:
    reference_root = Path(reference_root)
    reference_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = reference_root.parent / f".{reference_root.name}.lock"
    with exclusive_lock(lock_path):
        _recover_reference_publication(reference_root)
        return _resolve_training_references_unlocked(
            trainer,
            dataset,
            dft_config=dft_config,
            dft_launch=dft_launch,
            reference_root=reference_root,
        )


def _training_element_scope(dataset: Path) -> tuple[set[str], bool]:
    """Elements the training references must cover, and whether the manifest records them.

    The scope is the dataset manifest's recorded ``method_signature_elements``
    when present (it must cover every element in the frames), otherwise the
    elements found in the frames.
    """
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
    recorded_scope = dataset_metadata.method_signature_elements
    if recorded_scope is None:
        return training_elements, False
    signature_elements = set(recorded_scope)
    if len(signature_elements) != len(recorded_scope) or not signature_elements:
        raise ValueError("Dataset manifest has an invalid method-signature element scope")
    uncovered_elements = training_elements - signature_elements
    if uncovered_elements:
        raise ValueError(
            "Dataset manifest method-signature element scope does not cover training "
            f"elements: {sorted(uncovered_elements)}"
        )
    return signature_elements, True


def _validate_against_manifest(dataset: Path, manifest: dict) -> None:
    validate_dataset_reference_method(
        dataset,
        reference_backend=manifest["backend"],
        reference_method_signature=manifest["method_signature"],
        reference_base_method_signature=manifest.get("base_method_signature"),
        reference_element_method_signatures=manifest.get("element_method_signatures"),
        require_sidecar=True,
    )


def _merge_extension_selection(
    existing_selection: dict, new_selection: dict, new_elements: set[str]
) -> None:
    """Record an extension's approvals and DFT attempts in the frozen manifest's audit."""
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
    existing_selection.setdefault("phases", []).extend(new_selection.get("phases", []))
    existing_selection["phases_approved"] = bool(
        existing_selection.get("phases_approved") and new_selection.get("phases_approved")
    )
    existing_selection["dft_approved"] = bool(
        existing_selection.get("dft_approved") or new_selection.get("dft_approved")
    )
    existing_selection["dft_attempts"] = existing_selection.get(
        "dft_attempts", 0
    ) + new_selection.get("dft_attempts", 0)


def _prepare_plan_references(
    config, plan, placed: tuple[DFTConfig, ProcessLaunch], elements, output_dir: Path
):
    dft_config, launch = placed
    return prepare_elemental_references(
        plan,
        dft_config,
        launch=launch,
        required_elements=elements,
        cache_dir=config.cache_dir,
        output_dir=output_dir,
        phases_approved=config.phases_approved,
        dft_approved=config.dft_approved,
        max_dft_calculations=config.max_dft_calculations,
    )


def _extend_planned_references(
    config,
    plan,
    placed: tuple[DFTConfig, ProcessLaunch],
    dataset: Path,
    elements: set[str],
    reference_root: Path,
) -> None:
    """Add DFT references for elements missing from an existing frozen phase-plan snapshot."""
    dft_config = placed[0]
    source = reference_root / "elemental-references.json"
    recorded = json.loads((reference_root / "training-reference-inputs.json").read_text())
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
    if not new_elements:
        return
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
        _prepare_plan_references(config, plan, placed, new_elements, temp_output)
        new_manifest = json.loads((temp_output / "elemental-references.json").read_text())
        if existing_manifest.get("base_method_signature") != new_manifest.get(
            "base_method_signature"
        ):
            raise ValueError("HydraGNN training reference inputs changed; use a new AL dataset")
        with _staged_reference_snapshot(reference_root, copy_existing=True) as staged:
            for elem in new_elements:
                geom_file = f"{elem}.extxyz"
                shutil.copy2(temp_output / geom_file, staged / geom_file)
                existing_manifest["references"][elem] = new_manifest["references"][elem]
                existing_manifest.setdefault("element_method_signatures", {})[elem] = new_manifest[
                    "element_method_signatures"
                ][elem]
            _merge_extension_selection(
                existing_manifest.setdefault("selection", {}),
                new_manifest.get("selection", {}),
                new_elements,
            )
            existing_manifest["method_signature"] = target_compound_signature
            staged_source = staged / "elemental-references.json"
            staged_source.write_text(json.dumps(existing_manifest, indent=2) + "\n")
            new_identity = _phase_plan_identity(config.phase_plan, plan, dft_config, all_elements)
            _save_frozen_inputs(
                staged / "training-reference-inputs.json",
                new_identity,
                staged_source,
            )
            _publish_reference_snapshot(reference_root, staged)


def _create_planned_references(
    config,
    plan,
    placed: tuple[DFTConfig, ProcessLaunch],
    dataset: Path,
    elements: set[str],
    reference_root: Path,
) -> None:
    """Compute and freeze DFT references for every element from the approved phase plan."""
    dft_config = placed[0]
    identity = _phase_plan_identity(config.phase_plan, plan, dft_config, elements)
    validate_dataset_reference_method(
        dataset,
        reference_backend=dft_config.backend,
        reference_method_signature=identity["compound_method_signature"],
        require_sidecar=True,
    )
    with _staged_reference_snapshot(reference_root, copy_existing=False) as staged:
        staged_source = _prepare_plan_references(config, plan, placed, elements, staged)
        _save_frozen_inputs(staged / "training-reference-inputs.json", identity, staged_source)
        _publish_reference_snapshot(reference_root, staged)


def _copy_reference_geometries(references, elements: set[str], staged: Path) -> dict:
    """Write each selected reference geometry into the snapshot and describe it."""
    copied = {}
    for element, atoms, energy, _ in references:
        if element in elements:
            geometry = staged / f"{element}.extxyz"
            write(geometry, atoms)
            copied[element] = {
                "structure_path": geometry.name,
                "structure_sha256": sha256_file(geometry),
                "energy_eV": energy,
            }
    return copied


def _freeze_declared_references(
    source: Path, dataset: Path, elements: set[str], reference_root: Path
) -> None:
    """Freeze (or extend) a snapshot of the required elements from a declared manifest."""
    identity = {"manifest_sha256": sha256_file(source)}
    identity_path = reference_root / "training-reference-inputs.json"
    frozen = reference_root / "elemental-references.json"
    if reference_root.exists():
        _verify_frozen_inputs(identity_path, identity, frozen)
        existing_frozen_manifest = json.loads(frozen.read_text())
        existing_elements = set(existing_frozen_manifest.get("references", {}).keys())
        new_elements = elements - existing_elements
        if not new_elements:
            return
        manifest, _, references = load_elemental_reference_manifest(
            source, required_elements=elements
        )
        _validate_against_manifest(dataset, manifest)
        with _staged_reference_snapshot(reference_root, copy_existing=True) as staged:
            copied = dict(existing_frozen_manifest.get("references", {}))
            copied.update(_copy_reference_geometries(references, new_elements, staged))
            source_manifest = json.loads(source.read_text())
            staged_frozen = staged / "elemental-references.json"
            staged_frozen.write_text(
                json.dumps({**source_manifest, "references": copied}, indent=2) + "\n"
            )
            _save_frozen_inputs(staged / "training-reference-inputs.json", identity, staged_frozen)
            _publish_reference_snapshot(reference_root, staged)
        return

    manifest, source_bytes, references = load_elemental_reference_manifest(
        source, required_elements=elements
    )
    _validate_against_manifest(dataset, manifest)
    with _staged_reference_snapshot(reference_root, copy_existing=False) as staged:
        staged.mkdir()
        copied = _copy_reference_geometries(references, elements, staged)
        staged_frozen = staged / "elemental-references.json"
        staged_frozen.write_text(
            json.dumps({**json.loads(source_bytes), "references": copied}, indent=2) + "\n"
        )
        _save_frozen_inputs(staged / "training-reference-inputs.json", identity, staged_frozen)
        _publish_reference_snapshot(reference_root, staged)


def _resolve_training_references_unlocked(
    trainer: TrainerConfig,
    dataset: str | Path,
    *,
    dft_config: DFTConfig | None,
    dft_launch: ProcessLaunch | None = None,
    reference_root: Path,
) -> Path:
    dataset = Path(dataset)
    elements, scope_recorded = _training_element_scope(dataset)
    config = trainer.hydragnn_training_references
    if config is not None and config.phase_plan is not None:
        if not scope_recorded:
            raise ValueError(
                "HydraGNN phase-plan references require the dataset manifest to record "
                "method_signature_elements"
            )
        if dft_config is None or dft_launch is None or config.cache_dir is None:
            raise ValueError(
                "Automatic HydraGNN references require a DFT config, a dft pool launch, "
                "and cache_dir"
            )
        plan = ElementalReferencePlan.from_yaml(config.phase_plan)
        placed = (dft_config, dft_launch)
        if reference_root.exists():
            _extend_planned_references(config, plan, placed, dataset, elements, reference_root)
        else:
            _create_planned_references(config, plan, placed, dataset, elements, reference_root)
    else:
        source = config.manifest if config is not None else trainer.validation_reference_set
        if source is None:
            raise ValueError(
                "HydraGNN training requires hydragnn_training_references or "
                "validation_reference_set even when model comparison/promotion is disabled"
            )
        _freeze_declared_references(source, dataset, elements, reference_root)
    source = reference_root / "elemental-references.json"
    manifest, _, _ = load_elemental_reference_manifest(source, required_elements=elements)
    _validate_against_manifest(dataset, manifest)
    return source
