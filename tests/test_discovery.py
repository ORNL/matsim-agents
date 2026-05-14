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
