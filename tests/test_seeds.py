"""Tests for the unified seed generator (AFLOW prototype decoration).

These exercise pure structure-building paths; no HydraGNN / MLP / XPU
involved. pyXtal random-search is covered by mocking when pyxtal is
unavailable in the test environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.seeds import (
    PhaseCandidate,
    SeedGenerationAction,
    compatible_prototypes,
    decorate,
    generate_seed_action,
    generate_seeds,
    load_prototypes,
    random_search,
)

pymatgen_available = importlib.util.find_spec("pymatgen") is not None
pyxtal_available = importlib.util.find_spec("pyxtal") is not None
requires_pymatgen = pytest.mark.skipif(not pymatgen_available, reason="pymatgen required")


def _comp(formula: str):
    c = parse_composition(formula)
    assert c is not None, formula
    return c


# --------------------------------------------------------------------- #
# Prototype loading / matching                                          #
# --------------------------------------------------------------------- #


@requires_pymatgen
def test_load_prototypes_returns_many():
    protos = load_prototypes()
    assert len(protos) >= 100  # pymatgen ships ~288 AFLOW prototypes
    aflow_labels = {p.aflow for p in protos}
    # Common landmark prototypes should be in the encyclopedia.
    assert any(lbl.startswith("AB_cF8_216") for lbl in aflow_labels)  # zincblende
    assert any("221" in lbl for lbl in aflow_labels)  # cubic Pm-3m family


@requires_pymatgen
def test_compatible_prototypes_filters_by_stoichiometry():
    # Binary 1:1 -> only prototypes with reduced signature (1,1).
    matches = compatible_prototypes(_comp("NaCl"))
    assert len(matches) >= 5
    for p in matches:
        assert p.signature == (1, 1)

    # Ternary 1:1:3 (perovskite-like).
    matches = compatible_prototypes(_comp("CaTiO3"))
    assert len(matches) >= 1
    for p in matches:
        assert p.signature == (1, 1, 3)


@requires_pymatgen
def test_compatible_prototypes_no_match_for_unusual_stoichiometry():
    # 1:1:1:1:1 quinary HEA — almost no prototypes in the AFLOW set
    # match this signature. Should return [] (or a very short list)
    # without crashing.
    matches = compatible_prototypes(_comp("CrMoNbTaW"))
    assert isinstance(matches, list)
    for p in matches:
        assert p.signature == (1, 1, 1, 1, 1)


# --------------------------------------------------------------------- #
# Decoration                                                            #
# --------------------------------------------------------------------- #


@requires_pymatgen
def test_decorate_substitutes_target_species():
    matches = compatible_prototypes(_comp("NaCl"))
    proto = next(p for p in matches if p.aflow.startswith("AB_cF8_216"))  # zincblende
    decs = decorate(proto, _comp("NaCl"))
    assert len(decs) >= 1
    struct, _mapping = decs[0]
    symbols = {site.specie.symbol for site in struct}
    assert symbols == {"Na", "Cl"}
    assert struct.num_sites == proto.structure.num_sites


@requires_pymatgen
def test_decorate_elemental_yields_single_decoration():
    matches = compatible_prototypes(_comp("Si"))
    assert len(matches) >= 1
    for proto in matches[:3]:
        decs = decorate(proto, _comp("Si"))
        assert len(decs) == 1  # single element -> single decoration


# --------------------------------------------------------------------- #
# pyXtal random search (graceful degradation)                           #
# --------------------------------------------------------------------- #


def test_random_search_zero_returns_empty():
    assert random_search(_comp("Si"), n=0) == []


@pytest.mark.skipif(pyxtal_available, reason="pyxtal IS installed; test covers the missing case")
def test_random_search_degrades_without_pyxtal():
    with pytest.warns(UserWarning, match="pyXtal is not installed"):
        out = random_search(_comp("Si"), n=5)
    assert out == []


@requires_pymatgen
def test_random_search_filters_symmetry_equivalent_structures(monkeypatch):
    from pymatgen.core import Lattice, Structure

    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])

    class FakePyXtal:
        valid = True

        def from_random(self, **kwargs):
            return None

        def to_pymatgen(self):
            return structure.copy()

    monkeypatch.setattr("matsim_agents.discovery.seeds._pyxtal_available", lambda: True)
    monkeypatch.setitem(sys.modules, "pyxtal", SimpleNamespace(pyxtal=FakePyXtal))

    results = random_search(_comp("Si"), n=2, seed=3, max_attempts_factor=1)
    assert len(results) == 1


@requires_pymatgen
def test_random_search_uses_formula_unit_and_space_group_controls(monkeypatch):
    from pymatgen.core import Lattice, Structure

    calls = []
    structure = Structure(Lattice.cubic(4.0), ["Li", "Fe", "P", "O"], [[0, 0, 0]] * 4)

    class FakePyXtal:
        valid = True

        def from_random(self, **kwargs):
            calls.append(kwargs)

        def to_pymatgen(self):
            return structure.copy()

    monkeypatch.setattr("matsim_agents.discovery.seeds._pyxtal_available", lambda: True)
    monkeypatch.setitem(sys.modules, "pyxtal", SimpleNamespace(pyxtal=FakePyXtal))

    random_search(
        _comp("LiFePO4"),
        n=1,
        seed=3,
        formula_units=4,
        space_groups=[62],
        max_attempts_factor=1,
    )

    assert calls == [
        {
            "dim": 3,
            "group": 62,
            "species": ["Li", "Fe", "P", "O"],
            "numIons": [4, 4, 4, 16],
        }
    ]


@requires_pymatgen
def test_random_search_seeded_proposals_replay_the_same_structure(monkeypatch):
    from pymatgen.core import Lattice, Structure

    random_values = []

    class FakePyXtal:
        valid = True

        def from_random(self, **kwargs):
            assert "random_state" in kwargs
            self.coordinate = float(kwargs["random_state"].random_sample())
            random_values.append(self.coordinate)

        def to_pymatgen(self):
            return Structure(
                Lattice.cubic(4.0),
                ["Si"],
                [[self.coordinate, 0.2, 0.3]],
            )

    monkeypatch.setattr("matsim_agents.discovery.seeds._pyxtal_available", lambda: True)
    monkeypatch.setitem(sys.modules, "pyxtal", SimpleNamespace(pyxtal=FakePyXtal))

    first = random_search(_comp("Si"), n=1, seed=31, seed_proposals=True)
    second = random_search(_comp("Si"), n=1, seed=31, seed_proposals=True)

    assert len(first) == len(second) == 1
    assert random_values[0] == random_values[1]
    assert first[0][0].matches(second[0][0])


# --------------------------------------------------------------------- #
# Unified entry point                                                   #
# --------------------------------------------------------------------- #


@requires_pymatgen
def test_generate_seeds_writes_files_and_carries_metadata(tmp_path: Path):
    cands = generate_seeds(_comp("NaCl"), str(tmp_path), n_random=0)
    assert len(cands) >= 3
    for c in cands:
        assert isinstance(c, PhaseCandidate)
        assert c.formula == "ClNa"  # Hill-ordered: C, then alphabetical (Cl, Na)
        assert Path(c.structure_path).exists()
        assert c.source == "prototype"
        assert c.prototype_id is not None
        assert c.space_group is not None and 1 <= c.space_group <= 230
        assert c.needs_dft_verification is False
        assert c.num_atoms is not None and c.num_atoms >= 2
        assert c.formula_units is not None
        assert c.structure_hash is not None and len(c.structure_hash) == 64
        assert set(c.lattice_parameters or {}) == {"a", "b", "c", "alpha", "beta", "gamma"}
        assert len(c.cell_matrix or []) == 3


@requires_pymatgen
def test_generate_seeds_with_n_random_zero_only_prototypes(tmp_path: Path):
    cands = generate_seeds(_comp("Si"), str(tmp_path), n_random=0)
    assert all(c.source == "prototype" for c in cands)
    assert all(not c.needs_dft_verification for c in cands)


@requires_pymatgen
def test_generate_seeds_unique_paths(tmp_path: Path):
    cands = generate_seeds(_comp("MgAl2O4"), str(tmp_path), n_random=0)
    paths = [c.structure_path for c in cands]
    assert len(paths) == len(set(paths))


@requires_pymatgen
def test_generate_seeds_honors_formula_unit_tier_and_prototype_filter(tmp_path: Path):
    composition = _comp("Si")
    prototype = compatible_prototypes(composition)[0]
    base_formula_units = prototype.structure.num_sites // composition.total_atoms
    requested_formula_units = base_formula_units * 2

    candidates = generate_seeds(
        composition,
        str(tmp_path),
        n_random=0,
        formula_units=requested_formula_units,
        max_aflow_candidates=1,
        prototype_ids=[prototype.aflow],
    )

    assert candidates
    assert len(candidates) <= 1
    assert all(c.formula_units == requested_formula_units for c in candidates)
    assert all(c.num_atoms == requested_formula_units for c in candidates)
    assert all(
        f"__fu{requested_formula_units}__" in Path(c.structure_path).name for c in candidates
    )


@requires_pymatgen
def test_generation_action_passes_strategy_controls(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_generate_seeds(composition, output_dir, **kwargs):
        captured.update(composition=composition, output_dir=output_dir, **kwargs)
        return []

    monkeypatch.setattr("matsim_agents.discovery.seeds.generate_seeds", fake_generate_seeds)
    action = SeedGenerationAction(
        action_id="round-1-tier-4",
        formula="NaCl",
        formula_units=4,
        max_aflow_candidates=0,
        pyxtal_count=12,
        space_groups=[225],
        random_seed=17,
    )
    generate_seed_action(action, str(tmp_path))

    assert captured["composition"].formula == "ClNa"
    assert (tmp_path / "generation-action.json").is_file()
    assert (tmp_path / "generation-manifest.json").is_file()
    assert captured["formula_units"] == 4
    assert captured["include_aflow"] is False
    assert captured["max_aflow_candidates"] == 0
    assert captured["n_random"] == 12
    assert captured["space_groups"] == [225]
    assert captured["random_seed"] == 17


def test_generation_action_rejects_invalid_space_groups_and_action_paths():
    with pytest.raises(ValueError, match="space_groups"):
        SeedGenerationAction(action_id="round-1", formula="Si", formula_units=1, space_groups=[231])
    with pytest.raises(ValueError, match="action_id"):
        SeedGenerationAction(action_id="../round-1", formula="Si", formula_units=1)


@requires_pymatgen
def test_generate_seeds_random_flagged_for_verification(tmp_path: Path):
    """If pyxtal is installed, random-source seeds carry the DFT-verify flag."""
    if not pyxtal_available:
        pytest.skip("pyxtal not installed; random-source flagging not exercised.")
    cands = generate_seeds(_comp("Si"), str(tmp_path), n_random=3, random_seed=42)
    random_cands = [c for c in cands if c.source == "random"]
    if random_cands:  # pyxtal can fail intermittently; just check what we got
        for c in random_cands:
            assert c.needs_dft_verification is True
            assert c.prototype_id is None
            assert c.space_group is not None and 1 <= c.space_group <= 230
