from __future__ import annotations

from matsim_agents.discovery.formula import FormulaGenerationPolicy, enumerate_formulas


def _nb_ta_o_policy(**overrides) -> FormulaGenerationPolicy:
    defaults = dict(
        elements=["Nb", "Ta", "O"],
        minimum_species=2,
        maximum_species=3,
        minimum_coefficient=1,
        maximum_coefficient=6,
        maximum_atoms_in_reduced_formula=12,
        require_charge_balance=True,
        oxidation_states={"Nb": [3, 4, 5], "Ta": [3, 4, 5], "O": [-2]},
    )
    defaults.update(overrides)
    return FormulaGenerationPolicy(**defaults)


def test_enumerate_formulas_includes_known_charge_balanced_binaries():
    # Canonical formulas follow the repo's strict alphabetical Hill order
    # (e.g. Ta2O5 -> "O5Ta2"), matching matsim_agents.discovery.composition.
    formulas = {c.reduced_formula: c for c in enumerate_formulas(_nb_ta_o_policy())}
    for expected in ["NbO2", "Nb2O3", "Nb2O5", "O2Ta", "O3Ta2", "O5Ta2"]:
        assert expected in formulas, formulas.keys()
        assert formulas[expected].active
        assert formulas[expected].charge_balanced


def test_enumerate_formulas_includes_known_charge_balanced_ternaries():
    # Nb2O7Ta2 needs a raw O coefficient of 7, beyond the default
    # maximum_coefficient=6, so it is intentionally outside this grid (see
    # test_formula_merge.py for it being picked up as an LLM-only addition).
    formulas = {c.reduced_formula: c for c in enumerate_formulas(_nb_ta_o_policy())}
    for expected in ["NbO3Ta", "NbO4Ta", "NbO5Ta", "Nb2O5Ta", "NbO5Ta2"]:
        assert expected in formulas, formulas.keys()
        assert formulas[expected].active
        assert formulas[expected].generation_source == "deterministic"


def test_enumerate_formulas_rejects_but_retains_charge_imbalanced_formula():
    # NbO would require Nb in oxidation state +2, which is outside the
    # configured [3, 4, 5]; it must be reported, not silently dropped.
    formulas = {c.reduced_formula: c for c in enumerate_formulas(_nb_ta_o_policy())}
    assert "NbO" in formulas
    assert formulas["NbO"].active is False
    assert formulas["NbO"].charge_balanced is False
    assert formulas["NbO"].rejection_reason is not None


def test_enumerate_formulas_respects_atom_and_coefficient_limits():
    policy = _nb_ta_o_policy(maximum_coefficient=2, maximum_atoms_in_reduced_formula=4)
    for candidate in enumerate_formulas(policy):
        assert sum(candidate.elements.values()) <= 4


def test_enumerate_formulas_honors_species_count_toggles():
    binary_only = _nb_ta_o_policy(include_mixed_oxides=False)
    assert all(len(c.elements) == 2 for c in enumerate_formulas(binary_only))

    ternary_only = _nb_ta_o_policy(include_binary_endmembers=False)
    assert all(len(c.elements) >= 3 for c in enumerate_formulas(ternary_only))


def test_enumerate_formulas_without_charge_balance_marks_everything_active():
    policy = _nb_ta_o_policy(require_charge_balance=False, oxidation_states={})
    formulas = enumerate_formulas(policy)
    assert formulas
    assert all(candidate.active for candidate in formulas)
    assert all(candidate.charge_balanced for candidate in formulas)
