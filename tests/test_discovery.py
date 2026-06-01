"""Tests for the discovery module that do not require HydraGNN."""

from __future__ import annotations

from matsim_agents.discovery.composition import (
    extract_compositions,
    parse_composition,
)


def test_parse_simple_binary():
    c = parse_composition("LiCoO2")
    assert c is not None
    # Hill order without C/H is alphabetical -> Co, Li, O.
    assert c.formula == "CoLiO2"
    assert c.elements == {"Li": 1, "Co": 1, "O": 2}


def test_parse_reduces_common_factor():
    c = parse_composition("Li4Mn2O6")
    assert c is not None
    assert c.elements == {"Li": 2, "Mn": 1, "O": 3}
    assert c.formula == "Li2MnO3"


def test_extract_from_prose():
    text = (
        "Consider Li2MnO3 as a high-capacity cathode, and compare it with "
        "Cs2AgBiBr6 perovskite. Plain words like Carbon should not match."
    )
    found = {c.formula for c in extract_compositions(text)}
    assert "Li2MnO3" in found
    # Cs2AgBiBr6 is reformatted into Hill order (alphabetical, no C/H):
    assert "AgBiBr6Cs2" in found


def test_rejects_garbage():
    assert parse_composition("Hello world") is None
    assert parse_composition("Xx2Yy3") is None


def test_rejects_acronyms_and_roman_numerals():
    # Crystal-structure abbreviations tokenize into valid element sequences
    # (BCC -> B+C+C -> C2B, FCC -> F+C+C -> C2F, HCP -> H+C+P -> CHP) and
    # Roman numerals do the same (IV -> I+V, VI -> V+I, ...). All such all-
    # uppercase alphabetic tokens must be rejected.
    text = (
        "MoNbTaW forms a single-phase BCC solid solution. The group-IV and "
        "group-V transition metals can also crystallize in FCC or HCP packing."
    )
    found = {c.formula for c in extract_compositions(text)}
    assert found == {"MoNbTaW"}, f"unexpected matches: {found}"
