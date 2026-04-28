"""Tests for phase enumeration: supercell, 2-D, orderings, lattice scales.

These exercise pure structure-building paths; no HydraGNN / MLP involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.phase_explorer import (
    PhaseCandidate,
    enumerate_phases,
)


def _comp(formula: str):
    c = parse_composition(formula)
    assert c is not None, formula
    return c


def test_elemental_supercell_autotile(tmp_path: Path):
    cands = enumerate_phases(_comp("Si"), str(tmp_path), min_atoms=64)
    # 5 elemental prototypes, all auto-tiled.
    phases = sorted(c.phase for c in cands)
    assert phases == ["bcc", "diamond", "fcc", "hcp", "sc"]
    for c in cands:
        assert c.dimensionality == "3D"
        assert c.num_atoms is not None and c.num_atoms >= 64
        assert c.num_layers is None
        # Single element -> no non-trivial orderings.
        assert c.ordering_index == 0
        assert c.lattice_scale == 1.0
        assert Path(c.structure_path).exists()


def test_explicit_supercell_overrides_min_atoms(tmp_path: Path):
    cands = enumerate_phases(
        _comp("Si"), str(tmp_path), supercell=(2, 2, 2), min_atoms=10_000
    )
    for c in cands:
        assert c.supercell == (2, 2, 2)


def test_2d_disabled_by_default(tmp_path: Path):
    cands = enumerate_phases(_comp("MoS2"), str(tmp_path), min_atoms=12)
    assert all(c.dimensionality == "3D" for c in cands)
    assert not any(c.phase.startswith("mx2") for c in cands)


def test_2d_monolayer_and_multilayer(tmp_path: Path):
    cands = enumerate_phases(
        _comp("MoS2"),
        str(tmp_path),
        min_atoms=12,
        include_2d=True,
        num_layers=3,
        vacuum=20.0,
    )
    twod = [c for c in cands if c.dimensionality == "2D"]
    assert {c.phase for c in twod} == {"mx2_2h", "mx2_1t"}
    for c in twod:
        # nz must be 1 for 2-D (never tile through vacuum).
        assert c.supercell is not None and c.supercell[2] == 1
        assert c.num_layers == 3
        # 3 layers x 3 atoms/monolayer = 9 atoms, possibly tiled in-plane.
        assert c.num_atoms is not None and c.num_atoms >= 9 and c.num_atoms % 9 == 0
        assert "_L3" in Path(c.structure_path).name


def test_graphene_only_for_single_element(tmp_path: Path):
    cands = enumerate_phases(
        _comp("C"), str(tmp_path), min_atoms=8, include_2d=True
    )
    twod = [c for c in cands if c.dimensionality == "2D"]
    assert {c.phase for c in twod} == {"graphene"}


def test_double_perovskite_detected(tmp_path: Path):
    cands = enumerate_phases(_comp("Cs2AgBiBr6"), str(tmp_path), min_atoms=40)
    phases = {c.phase for c in cands}
    assert "double_perovskite" in phases
    dp = next(c for c in cands if c.phase == "double_perovskite")
    # Proper 2x2x2 Fm-3m cell -> 40 atoms in the primitive seed.
    assert dp.num_atoms == 40


def test_spinel_detected_for_AB2X4(tmp_path: Path):
    cands = enumerate_phases(_comp("MgAl2O4"), str(tmp_path), min_atoms=14)
    phases = {c.phase for c in cands}
    assert "spinel" in phases
    sp = next(c for c in cands if c.phase == "spinel")
    assert sp.num_atoms == 14


def test_n_orderings_collapses_for_single_element(tmp_path: Path):
    cands = enumerate_phases(_comp("Si"), str(tmp_path), min_atoms=8, n_orderings=5)
    assert all(c.ordering_index == 0 for c in cands)


def test_n_orderings_produces_distinct_files(tmp_path: Path):
    cands = enumerate_phases(
        _comp("GaAs"), str(tmp_path), min_atoms=32, n_orderings=4
    )
    rs = [c for c in cands if c.phase == "rocksalt"]
    assert len(rs) >= 2
    indices = {c.ordering_index for c in rs}
    assert 0 in indices and len(indices) >= 2
    paths = {c.structure_path for c in rs}
    assert len(paths) == len(rs)  # no filename collisions


def test_lattice_scales_replicate_each_ordering(tmp_path: Path):
    scales = (0.97, 1.0, 1.03)
    cands = enumerate_phases(
        _comp("Si"), str(tmp_path), min_atoms=8,
        n_orderings=1, lattice_scales=scales,
    )
    by_phase: dict[str, list[PhaseCandidate]] = {}
    for c in cands:
        by_phase.setdefault(c.phase, []).append(c)
    for phase, group in by_phase.items():
        assert {c.lattice_scale for c in group} == set(scales), phase


def test_lattice_scaling_actually_scales_cell(tmp_path: Path):
    from ase.io import read

    cands = enumerate_phases(
        _comp("Si"), str(tmp_path), min_atoms=8,
        n_orderings=1, lattice_scales=(1.0, 1.05),
    )
    fcc = sorted(
        (c for c in cands if c.phase == "fcc"),
        key=lambda c: c.lattice_scale,
    )
    v0 = read(fcc[0].structure_path).get_volume()
    v1 = read(fcc[1].structure_path).get_volume()
    # Volume scales as the cube of the linear scale factor.
    assert (v1 / v0) == pytest.approx(1.05 ** 3, rel=1e-3)


def test_orderings_times_scales_count(tmp_path: Path):
    cands = enumerate_phases(
        _comp("GaAs"), str(tmp_path), min_atoms=32,
        n_orderings=3, lattice_scales=(0.98, 1.0, 1.02),
    )
    rs = [c for c in cands if c.phase == "rocksalt"]
    # At most 3 orderings x 3 scales; orderings can collapse if dedup hits.
    assert 1 <= len({c.ordering_index for c in rs}) <= 3
    assert {c.lattice_scale for c in rs} == {0.98, 1.0, 1.02}
    # Per ordering, every requested scale is present.
    by_o: dict[int, set[float]] = {}
    for c in rs:
        by_o.setdefault(c.ordering_index, set()).add(c.lattice_scale)
    for scales in by_o.values():
        assert scales == {0.98, 1.0, 1.02}
