"""MP2: property characterization recipes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write
from pydantic import ValidationError

from matsim_agents.campaign.characterization import (
    CharacterizationContract,
    outstanding_steps,
    run_characterization,
)
from matsim_agents.campaign.property_recipes import (
    METHODS,
    PROPERTY_RECIPES,
    BackendContext,
    BackendReadiness,
    list_recipes,
    method_readiness,
)
from matsim_agents.core.contracts import CharacterizationStatus, EvidenceLevel
from matsim_agents.execution.properties import (
    PropertyConditions,
    PropertyEvidence,
    PropertyStatus,
)

pytest.importorskip("pymatgen")
T = 1000.0


def _structure(path: Path) -> Path:
    atoms = Atoms("TaW")
    atoms.positions = [[0, 0, 0], [2.0, 0, 0]]
    atoms.cell = [8, 8, 8]
    atoms.pbc = True
    write(path, atoms)
    return path


def _evidence(prop, value, unit, **kw):
    base = dict(
        structure_id="s1",
        composition="TaW",
        property=prop,
        status=PropertyStatus.CALCULATED,
        value=value,
        unit=unit,
        evidence_level=EvidenceLevel.CONVERGED_DFT,
        converged=True,
        uncertainty=0.1 * abs(value) if isinstance(value, float) else None,
        evidence_ids=["ev-1"],
        method="shengbte",
        backend="shengbte",
        conditions=PropertyConditions(temperature_K=T),
    )
    base.update(kw)
    return PropertyEvidence(**base)


def _write(path: Path, items) -> Path:
    path.write_text(json.dumps([i.model_dump(mode="json") for i in items]))
    return path


def _contract(tmp_path, recipe, evidence_items=(), **kw):
    files = [_write(tmp_path / "evidence.json", evidence_items)] if evidence_items else []
    return CharacterizationContract(
        contract_id=f"{recipe}-test",
        objective="property test",
        formula="TaW",
        structures=[_structure(tmp_path / "s1.xyz")],
        recipe=recipe,
        property_evidence=files,
        conditions=PropertyConditions(temperature_K=T),
        output_root=tmp_path / "runs",
        **kw,
    )


def test_every_property_recipe_is_declared_and_discoverable():
    listed = {r["recipe"]: r for r in list_recipes()}
    assert set(listed) == set(PROPERTY_RECIPES)
    for spec in listed.values():
        assert spec["required_observables"] and spec["methods"] and spec["limitations"]
        assert all(m in METHODS for m in spec["methods"])
        assert spec["forbidden_fallbacks"]


def test_existing_recipes_are_unchanged(tmp_path):
    contract = CharacterizationContract(
        contract_id="relax-test",
        objective="x",
        formula="TaW",
        structures=[_structure(tmp_path / "s1.xyz")],
        recipe="relaxation",
        output_root=tmp_path / "runs",
    )
    assert [o.name for o in contract.observables] == ["final_energy", "max_force"] or (
        contract.observables
    )
    assert outstanding_steps(contract)


def test_missing_property_is_inconclusive_and_names_the_calculation(tmp_path):
    contract = _contract(tmp_path, "thermal_transport")
    report = run_characterization(contract)
    assert report.status == CharacterizationStatus.INCONCLUSIVE
    assert any("thermal_conductivity_total" in m for m in report.missing)
    assert report.observables == []
    assert "shengbte" in report.recommended_next_test
    assert "dft_approved" in report.recommended_next_test


def test_no_silent_fallback_to_a_related_property(tmp_path):
    related = [
        _evidence("electrical_resistivity", 1e-6, "ohm m"),
        _evidence(
            "yield_strength",
            900.0,
            "MPa",
            method="measurement",
            status=PropertyStatus.MEASURED,
            evidence_level=EvidenceLevel.EXPERIMENTAL,
        ),
    ]
    contract = _contract(tmp_path, "thermal_transport", related)
    report = run_characterization(contract)
    assert report.status == CharacterizationStatus.INCONCLUSIVE
    assert report.observables == []


def test_sigma_over_tau_is_not_conductivity(tmp_path):
    items = [_evidence("electrical_conductivity_over_relaxation_time", 1e20, "S/(m s)")]
    report = run_characterization(_contract(tmp_path, "electrical_transport", items))
    assert report.status == CharacterizationStatus.INCONCLUSIVE
    assert any("sigma/tau" in m for m in report.missing)


def test_compatible_evidence_is_reused(tmp_path):
    items = [_evidence("thermal_conductivity_total", 25.0, "W/(m K)")]
    report = run_characterization(_contract(tmp_path, "thermal_transport", items))
    assert report.status == CharacterizationStatus.SUFFICIENTLY_CHARACTERIZED
    assert report.observables[0].value == 25.0
    assert report.observables[0].unit == "W/(m K)"


def test_temperature_mismatch_is_rejected(tmp_path):
    items = [
        _evidence(
            "thermal_conductivity_total",
            25.0,
            "W/(m K)",
            conditions=PropertyConditions(temperature_K=300.0),
        )
    ]
    report = run_characterization(_contract(tmp_path, "thermal_transport", items))
    assert report.status == CharacterizationStatus.INCONCLUSIVE
    assert any("temperature" in m for m in report.missing)


def test_irradiated_evidence_is_not_pristine_evidence(tmp_path):
    items = [
        _evidence(
            "thermal_conductivity_total",
            25.0,
            "W/(m K)",
            conditions=PropertyConditions(temperature_K=T, irradiation="1 dpa"),
        )
    ]
    report = run_characterization(_contract(tmp_path, "thermal_transport", items))
    assert report.status == CharacterizationStatus.INCONCLUSIVE


def test_screening_level_evidence_is_below_required_fidelity(tmp_path):
    items = [
        _evidence(
            "thermal_conductivity_total",
            25.0,
            "W/(m K)",
            evidence_level=EvidenceLevel.MLIP_PREDICTION,
            converged=None,
            status=PropertyStatus.SURROGATE_PREDICTED,
            method="surrogate",
        )
    ]
    contract = _contract(tmp_path, "thermal_transport", items)
    contract.observables[0].minimum_evidence_level = EvidenceLevel.CONVERGED_DFT
    report = run_characterization(contract)
    assert report.status == CharacterizationStatus.INCONCLUSIVE
    assert any("below" in m for m in report.missing)


def test_transport_recipes_need_a_temperature(tmp_path):
    with pytest.raises(ValidationError, match="temperature_K"):
        CharacterizationContract(
            contract_id="t",
            objective="x",
            formula="TaW",
            structures=[_structure(tmp_path / "s1.xyz")],
            recipe="electrical_transport",
            output_root=tmp_path / "runs",
        )


def test_backend_readiness_statuses(tmp_path):
    ctx = BackendContext()
    state, why = method_readiness("stress_strain_mlip", "bulk_modulus", ctx)
    assert state == BackendReadiness.METHOD_UNSUPPORTED and "stress" in why
    state, _ = method_readiness("shengbte", "electrical_conductivity", ctx)
    assert state == BackendReadiness.METHOD_UNSUPPORTED
    state, _ = method_readiness("kubo_greenwood", "electrical_conductivity", ctx)
    assert state == BackendReadiness.NOT_CONFIGURED
    exe = tmp_path / "ShengBTE"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    configured = BackendContext(executables={"shengbte": str(exe)})
    state, why = method_readiness("shengbte", "thermal_conductivity_lattice", configured)
    assert state == BackendReadiness.RESOURCE_BLOCKED and "benchmark" in why
    costed = BackendContext(
        executables={"shengbte": str(exe)},
        calculation_cost_node_hours=50.0,
        remaining_node_hours=10.0,
    )
    state, _ = method_readiness("shengbte", "thermal_conductivity_lattice", costed)
    assert state == BackendReadiness.RESOURCE_BLOCKED
    ok = BackendContext(
        executables={"shengbte": str(exe)},
        calculation_cost_node_hours=5.0,
        remaining_node_hours=10.0,
    )
    state, _ = method_readiness("shengbte", "thermal_conductivity_lattice", ok)
    assert state == BackendReadiness.AVAILABLE


def test_property_recipes_run_no_calculation(tmp_path):
    contract = _contract(tmp_path, "mechanical_elasticity")
    assert outstanding_steps(contract) == []
    report = run_characterization(contract)
    assert report.status == CharacterizationStatus.INCONCLUSIVE
    assert any("elastic_stiffness_tensor" in m for m in report.missing)
