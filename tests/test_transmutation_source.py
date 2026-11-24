from __future__ import annotations

import pytest

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.transmutation import apply_transmutation
from matsim_agents.discovery.transmutation_source import (
    InventoryProvenance,
    InventorySourceError,
    inventory_from_rows,
    load_inventory,
)

PROV = InventoryProvenance(source_id="syn-1", kind="synthetic")
ROWS = [
    {"time_s": 0.0, "re": 0.0, "os": 0.0},
    {"time_s": 100.0, "re": 0.004, "os": 0.0005},
    {"time_s": 200.0, "re": 0.006, "os": 0.002},
]


def test_interpolation_and_count_mapping():
    inv = inventory_from_rows(ROWS, PROV)
    s = inv.sample(50.0, 1024)
    assert s.fractions == pytest.approx({"re": 0.002, "os": 0.00025})
    assert s.counts == {"re": 2, "os": 0}  # 2.048 and 0.256
    assert s.rounding_error_atoms["re"] == pytest.approx(2 - 2.048)
    assert s.evidence_level == "synthetic"
    assert inv.sample(100.0, 1024).counts == {"re": 4, "os": 1}
    assert inv.sample(200.0, 1000).counts == {"re": 6, "os": 2}


def test_re_may_decrease_decay_chain_is_not_assumed_monotonic():
    rows = [
        {"time_s": 0.0, "re": 0.0, "os": 0.0},
        {"time_s": 10.0, "re": 0.005, "os": 0.0},
        {"time_s": 20.0, "re": 0.003, "os": 0.002},
    ]
    s = inventory_from_rows(rows, PROV).sample(15.0, 1000)
    assert s.fractions["re"] == pytest.approx(0.004)


def test_appm_conversion_and_gas_columns():
    rows = [
        {"time_s": 0.0, "re": 0.0, "os": 0.0, "he": 0.0},
        {"time_s": 10.0, "re": 5000.0, "os": 1000.0, "he": 20000.0},
    ]
    s = inventory_from_rows(rows, PROV, units="appm").sample(10.0, 1024)
    assert s.fractions == pytest.approx({"re": 0.005, "os": 0.001, "he": 0.02})
    assert s.counts["he"] == 20  # gas is not subject to the Re/Os campaign cap
    assert "h" not in s.counts  # not supplied, not invented


def test_scenario_provenance_and_integration_with_generator():
    inv = inventory_from_rows(ROWS, PROV, parent_alloy_id="alloy")
    scen, sampled = inv.scenario_at(
        100.0, scenario_id="t100", parent_alloy_id="alloy", n_lattice_sites=1024, seed=1
    )
    assert (scen.n_re, scen.n_os) == (4, 1)
    assert scen.inventory_source == "syn-1" and scen.evidence_level == "synthetic"
    assert scen.exposure_scenario == "syn-1@t=100s"
    parent = generate_alloy(
        AlloySpec(
            counts={"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64},
            repeats=(8, 8, 8),
            seed=0,
        )
    )
    atoms, rec = apply_transmutation(parent, "alloy", scen)
    assert rec.n_re == sampled.counts["re"] and len(atoms) == 1024


def test_cap_is_not_relaxed_silently():
    inv = inventory_from_rows(ROWS, PROV)
    with pytest.raises(ValueError, match="justification"):
        inv.scenario_at(
            200.0,
            scenario_id="x",
            parent_alloy_id="a",
            n_lattice_sites=1024,
            seed=1,
            max_solute_fraction=0.05,
        )
    scen, _ = inv.scenario_at(
        200.0, scenario_id="x", parent_alloy_id="a", n_lattice_sites=1024, seed=1
    )
    assert scen.n_solutes == 8  # within cap
    with pytest.raises(ValueError, match="exceed|limit"):
        big = inventory_from_rows(
            [{"time_s": 0, "re": 0, "os": 0}, {"time_s": 1, "re": 0.02, "os": 0}], PROV
        )
        s, _ = big.scenario_at(
            1.0, scenario_id="y", parent_alloy_id="a", n_lattice_sites=1024, seed=1
        )
        generate = generate_alloy(AlloySpec(counts={"W": 1024}, repeats=(8, 8, 8), seed=0))
        apply_transmutation(generate, "a", s)


def test_rejections():
    inv = inventory_from_rows(ROWS, PROV, parent_alloy_id="alloy")
    with pytest.raises(InventorySourceError, match="outside"):
        inv.sample(201.0, 1024)
    with pytest.raises(InventorySourceError, match="outside"):
        inv.sample(float("nan"), 1024)
    with pytest.raises(InventorySourceError, match="parent"):
        inv.sample(10.0, 1024, parent_alloy_id="other")
    with pytest.raises(InventorySourceError, match="W atoms"):
        inv.sample(200.0, 1024, w_atoms=5)
    with pytest.raises(InventorySourceError, match="increasing"):
        inventory_from_rows(list(reversed(ROWS)), PROV)
    with pytest.raises(InventorySourceError):
        inventory_from_rows(ROWS[:1], PROV)
    with pytest.raises(InventorySourceError, match="exceeds one atom"):
        inventory_from_rows(
            [{"time_s": 0, "re": 0.7, "os": 0.5}, {"time_s": 1, "re": 0, "os": 0}], PROV
        )
    with pytest.raises(InventorySourceError, match="only some rows"):
        inventory_from_rows(
            [{"time_s": 0, "re": 0, "os": 0, "he": 0.1}, {"time_s": 1, "re": 0, "os": 0}], PROV
        )
    with pytest.raises(InventorySourceError):
        inventory_from_rows(
            [{"time_s": 0, "re": -1, "os": 0}, {"time_s": 1, "re": 0, "os": 0}], PROV
        )


def test_provenance_requirements():
    with pytest.raises(ValueError, match="external_calculation"):
        InventoryProvenance(source_id="x", kind="external_calculation", code="FISPACT")
    with pytest.raises(ValueError, match="spectrum"):
        InventoryProvenance(source_id="x", kind="user_table")
    ok = InventoryProvenance(
        source_id="x",
        kind="external_calculation",
        code="inv-code",
        nuclear_data_library="lib-v1",
        spectrum_reference="spec-ref",
        requires_temperature_history=True,
    )
    with pytest.raises(InventorySourceError, match="temperature"):
        inventory_from_rows(ROWS, ok)
    rows = [dict(r, temperature_k=600.0) for r in ROWS]
    s = inventory_from_rows(rows, ok).sample(10.0, 1024)
    assert s.evidence_level == "predicted" and s.temperature_k == 600.0


def test_csv_loading_and_errors(tmp_path):
    good = tmp_path / "inv.csv"
    good.write_text("time_s,re,os,sigma_re,sigma_os\n0,0,0,0,0\n100,0.004,0.0005,0.001,0.0001\n")
    inv = load_inventory(good, PROV)
    s = inv.sample(50.0, 1024)
    assert s.sigma_fractions == pytest.approx({"re": 0.0005, "os": 0.00005})
    bad = tmp_path / "bad.csv"
    bad.write_text("time_s,re\n0,0\n1,0\n")
    with pytest.raises(InventorySourceError, match="missing columns"):
        load_inventory(bad, PROV)
    extra = tmp_path / "extra.csv"
    extra.write_text("time_s,re,os,rate\n0,0,0,1\n1,0,0,1\n")
    with pytest.raises(InventorySourceError, match="unrecognized"):
        load_inventory(extra, PROV)
    with pytest.raises(InventorySourceError, match="not found"):
        load_inventory(tmp_path / "nope.csv", PROV)
