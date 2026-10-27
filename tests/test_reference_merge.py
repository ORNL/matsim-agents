"""How a campaign accumulates DFT reference sets across formula runs."""

from __future__ import annotations

import pytest

from matsim_agents.campaign.orchestrator import _merge_reference_energy_sets
from matsim_agents.discovery.stability import (
    ElementalReferenceEntry,
    ReferenceCompletenessPolicy,
    ReferenceEnergySet,
    ReferencePhaseEntry,
)

SIG = "qe:pbe:ecut60"


def _element(element, energy, phase_id=None):
    return ElementalReferenceEntry(
        element=element,
        phase_id=phase_id or f"{element}-ref",
        reference_formula=element,
        energy_eV_per_atom=energy,
        method_signature=SIG,
        backend="qe",
    )


def _phase(phase_id, formula, energy):
    return ReferencePhaseEntry(
        phase_id=phase_id,
        formula=formula,
        formation_energy_eV_per_atom=energy,
        method_signature=SIG,
        backend="qe",
    )


def _set(elements, *, phases=(), candidates=(), policy=None, **fields):
    return ReferenceEnergySet(
        identifier=fields.get("identifier", "campaign"),
        method_signature=fields.get("method_signature", SIG),
        backend=fields.get("backend", "qe"),
        elemental_energies_eV_per_atom={e: v for e, v in elements.items()},
        elemental_entries={e: _element(e, v) for e, v in elements.items()},
        elemental_reference_candidates=list(candidates),
        phase_entries=list(phases),
        completeness_policy=policy or ReferenceCompletenessPolicy(),
    )


def test_first_formula_run_is_taken_as_is():
    incoming = _set({"Si": -5.4})
    assert _merge_reference_energy_sets(None, incoming) is incoming


def test_merge_takes_the_union_and_keeps_the_lowest_competing_energy():
    existing = _set(
        {"Si": -5.4, "C": -9.2},
        phases=[_phase("SiC-3C", "SiC", -0.30)],
        candidates=[_element("Si", -5.4, "Si-diamond")],
        policy=ReferenceCompletenessPolicy(required_formulas=["SiC"]),
    )
    incoming = _set(
        {"Si": -5.4, "Ge": -4.5},
        phases=[
            _phase("SiC-3C", "SiC", -0.30),
            _phase("SiC-2H", "SiC", -0.35),
            _phase("SiGe-zb", "SiGe", -0.01),
        ],
        candidates=[_element("Si", -5.4, "Si-diamond"), _element("Ge", -4.5, "Ge-diamond")],
        policy=ReferenceCompletenessPolicy(
            required_formulas=["SiGe"],
            require_binary_subsystems=False,
            require_ternary_competitor=True,
        ),
    )
    merged = _merge_reference_energy_sets(existing, incoming)

    assert merged.elemental_energies_eV_per_atom == {"Si": -5.4, "C": -9.2, "Ge": -4.5}
    assert set(merged.elemental_entries) == {"Si", "C", "Ge"}
    assert [e.phase_id for e in merged.phase_entries] == ["SiC-3C", "SiC-2H", "SiGe-zb"]
    assert [e.phase_id for e in merged.elemental_reference_candidates] == [
        "Si-diamond",
        "Ge-diamond",
    ]
    assert [entry.formation_energy_eV_per_atom for entry in merged.phase_entries] == [
        -0.30,
        -0.35,
        -0.01,
    ]
    assert merged.completeness_policy.required_formulas == ["SiC", "SiGe"]
    assert merged.completeness_policy.require_binary_subsystems
    assert merged.completeness_policy.require_ternary_competitor
    assert [entry.phase_id for entry in existing.phase_entries] == ["SiC-3C"]


@pytest.mark.parametrize(
    "change",
    [{"identifier": "other"}, {"method_signature": "vasp:pbe"}, {"backend": "vasp"}],
)
def test_sets_from_different_methods_do_not_merge(change):
    incoming = _set({}, **change)
    with pytest.raises(ValueError, match="incompatible DFT reference sets"):
        _merge_reference_energy_sets(_set({}), incoming)


def test_a_disagreeing_elemental_energy_is_rejected():
    with pytest.raises(ValueError, match="elemental reference energy for Si"):
        _merge_reference_energy_sets(_set({"Si": -5.4}), _set({"Si": -5.5}))


def test_a_disagreeing_elemental_entry_is_rejected():
    existing = _set({"Si": -5.4})
    incoming = existing.model_copy(deep=True)
    incoming.elemental_entries["Si"] = _element("Si", -5.4, "Si-beta-tin")
    with pytest.raises(ValueError, match="elemental reference entry for Si"):
        _merge_reference_energy_sets(existing, incoming)


def test_a_disagreeing_unary_candidate_is_rejected():
    existing = _set({}, candidates=[_element("Si", -5.4, "Si-diamond")])
    incoming = _set({}, candidates=[_element("Si", -5.3, "Si-diamond")])
    with pytest.raises(ValueError, match="unary reference candidate 'Si-diamond'"):
        _merge_reference_energy_sets(existing, incoming)


def test_a_disagreeing_competing_phase_is_rejected():
    existing = _set({}, phases=[_phase("SiC-3C", "SiC", -0.30)])
    incoming = _set({}, phases=[_phase("SiC-3C", "SiC", -0.31)])
    with pytest.raises(ValueError, match="competing-phase reference 'SiC-3C'"):
        _merge_reference_energy_sets(existing, incoming)
