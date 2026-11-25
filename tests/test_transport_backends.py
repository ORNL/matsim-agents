"""MP4: transport interfaces, Wiedemann-Franz, kappa combination, and data ingestion."""

from __future__ import annotations

import json

import pytest

from matsim_agents.backends.external.transport import (
    INGEST_FORMAT,
    SOMMERFELD_LORENZ,
    TransportStatus,
    check_backend,
    combine_total_thermal_conductivity,
    ingest_transport_record,
    wiedemann_franz,
)
from matsim_agents.core.contracts import EvidenceLevel
from matsim_agents.science.properties.calculators import (
    PropertyConditions,
    PropertyEvidence,
    PropertyStatus,
)

COND = PropertyConditions(temperature_K=300.0)


def _ev(prop, value, unit, **kw):
    base = dict(
        structure_id="s1",
        composition="W2",
        property=prop,
        status=PropertyStatus.CALCULATED,
        value=value,
        unit=unit,
        evidence_level=EvidenceLevel.CONVERGED_DFT,
        converged=True,
        uncertainty=0.1 * value,
        evidence_ids=["ev"],
        conditions=COND,
        method="x",
    )
    base.update(kw)
    return PropertyEvidence(**base)


def _exe(tmp_path, name):
    path = tmp_path / name
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


def test_wiedemann_franz_is_an_labelled_approximation():
    sigma = _ev("electrical_conductivity", 1.0e7, "S/m")
    l_value, source = SOMMERFELD_LORENZ
    kappa = wiedemann_franz(sigma, lorenz_number=l_value, lorenz_source=source, temperature_K=300.0)
    assert kappa.status == PropertyStatus.APPROXIMATED
    assert kappa.value == pytest.approx(l_value * 1e7 * 300.0)
    assert kappa.value == pytest.approx(73.3, rel=1e-2)
    assert "Wiedemann" in kappa.limitations[0] and source in kappa.approximation
    assert kappa.uncertainty == pytest.approx(0.1 * kappa.value)
    assert kappa.derived_from == ["electrical_conductivity"]
    assert sigma.property_id() in kappa.evidence_ids


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"lorenz_source": " "}, "needs a source"),
        ({"lorenz_number": -1.0}, "positive"),
        ({"temperature_K": 400.0}, "not 400"),
        ({"temperature_K": 0.0}, "temperature"),
    ],
)
def test_wiedemann_franz_refuses_unsourced_or_mismatched_inputs(kw, match):
    sigma = _ev("electrical_conductivity", 1.0e7, "S/m")
    args = dict(lorenz_number=2.44e-8, lorenz_source="s", temperature_K=300.0)
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        wiedemann_franz(sigma, **args)


def test_wiedemann_franz_refuses_sigma_over_tau():
    tau = _ev("electrical_conductivity_over_relaxation_time", 1e20, "S/(m s)")
    with pytest.raises(ValueError, match="not sigma/tau"):
        wiedemann_franz(tau, lorenz_number=2.44e-8, lorenz_source="s", temperature_K=300.0)


def test_total_conductivity_is_the_sum_with_quadrature_uncertainty():
    ke = _ev("thermal_conductivity_electronic", 20.0, "W/(m K)", uncertainty=3.0)
    kp = _ev("thermal_conductivity_lattice", 5.0, "W/(m K)", uncertainty=4.0)
    total = combine_total_thermal_conductivity(ke, kp)
    assert total.value == 25.0 and total.uncertainty == pytest.approx(5.0)
    assert total.status == PropertyStatus.CALCULATED
    assert total.derived_from == ["thermal_conductivity_electronic", "thermal_conductivity_lattice"]


def test_total_inherits_the_approximation_of_wiedemann_franz():
    sigma = _ev("electrical_conductivity", 1.0e6, "S/m")
    ke = wiedemann_franz(sigma, lorenz_number=2.44e-8, lorenz_source="s", temperature_K=300.0)
    kp = _ev("thermal_conductivity_lattice", 5.0, "W/(m K)")
    total = combine_total_thermal_conductivity(ke, kp)
    assert total.status == PropertyStatus.APPROXIMATED
    assert "L sigma T" in total.approximation


def test_total_refuses_different_structures_or_conditions():
    ke = _ev("thermal_conductivity_electronic", 20.0, "W/(m K)")
    other = _ev("thermal_conductivity_lattice", 5.0, "W/(m K)", structure_id="s2")
    with pytest.raises(ValueError, match="different structures"):
        combine_total_thermal_conductivity(ke, other)
    hot = _ev(
        "thermal_conductivity_lattice",
        5.0,
        "W/(m K)",
        conditions=PropertyConditions(temperature_K=900.0),
    )
    with pytest.raises(ValueError, match="different conditions"):
        combine_total_thermal_conductivity(ke, hot)


def test_backend_statuses(tmp_path):
    r = check_backend("shengbte", "electrical_conductivity", ordered_crystal=True)
    assert r.status == TransportStatus.METHOD_UNSUPPORTED
    r = check_backend("shengbte", "thermal_conductivity_lattice", ordered_crystal=False)
    assert r.status == TransportStatus.METHOD_UNSUPPORTED and "disordered" in r.reasons[0]
    r = check_backend("almabte", "thermal_conductivity_lattice")
    assert r.status == TransportStatus.NOT_CONFIGURED
    exe = _exe(tmp_path, "ShengBTE")
    r = check_backend(
        "shengbte",
        "thermal_conductivity_lattice",
        executable=exe,
        ordered_crystal=True,
        n_atoms=1024,
    )
    assert r.status == TransportStatus.RESOURCE_BLOCKED and "reference" in r.reasons[0]


def test_cost_scales_from_a_measured_reference_and_blocks_on_budget(tmp_path):
    exe = _exe(tmp_path, "epw.x")
    common = dict(
        executable=exe,
        ordered_crystal=True,
        reference_node_hours=2.0,
        reference_atoms=2,
    )
    blocked = check_backend(
        "epw", "electrical_conductivity", n_atoms=16, remaining_node_hours=100.0, **common
    )
    assert blocked.status == TransportStatus.RESOURCE_BLOCKED
    assert blocked.estimated_node_hours == pytest.approx(2.0 * 8**3)
    ok = check_backend(
        "epw", "electrical_conductivity", n_atoms=4, remaining_node_hours=100.0, **common
    )
    assert ok.ready and ok.estimated_node_hours == pytest.approx(16.0)


def _record(**kw):
    base = {
        "format": INGEST_FORMAT,
        "structure_id": "s1",
        "composition": "W2",
        "method": "shengbte",
        "backend": "shengbte",
        "evidence_level": "converged_dft",
        "converged": True,
        "conditions": {"temperature_K": 300.0},
        "properties": {
            "thermal_conductivity_lattice": {"value": 0.5, "unit": "W/(cm K)", "uncertainty": 0.05},
            "electrical_conductivity": {"value": 10.0, "unit": "MS/m", "uncertainty": 1.0},
        },
    }
    base.update(kw)
    return base


def test_ingestion_converts_units_and_records_file_provenance(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_record()))
    items = {e.property: e for e in ingest_transport_record(path)}
    assert items["thermal_conductivity_lattice"].value == pytest.approx(50.0)
    assert items["electrical_conductivity"].value == pytest.approx(1e7)
    assert items["electrical_conductivity"].status == PropertyStatus.CALCULATED
    assert items["electrical_conductivity"].evidence_ids[0].startswith("file-")
    assert path.name in items["electrical_conductivity"].output_sha256


def test_ingestion_rejects_bad_format_missing_uncertainty_and_foreign_properties(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_record(format="other")))
    with pytest.raises(ValueError, match="format"):
        ingest_transport_record(path)
    bad = _record()
    del bad["properties"]["electrical_conductivity"]["uncertainty"]
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        ingest_transport_record(path)
    path.write_text(
        json.dumps(
            _record(properties={"bulk_modulus": {"value": 1.0, "unit": "GPa", "uncertainty": 0.1}})
        )
    )
    with pytest.raises(ValueError, match="not transport"):
        ingest_transport_record(path)


def test_unconverged_ingested_record_has_no_scoreable_value(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_record(converged=False)))
    items = ingest_transport_record(path)
    assert {e.status for e in items} == {PropertyStatus.UNCONVERGED}
    assert all(e.uncertainty is None for e in items)
