from __future__ import annotations

import pytest
from ase.geometry import get_distances

from matsim_agents.discovery.alloys import AlloySpec, generate_alloy
from matsim_agents.discovery.point_defects import generate_vacancy
from matsim_agents.discovery.transmutation import (
    TransmutationError,
    TransmutationRecord,
    TransmutationScenario,
    apply_transmutation,
)

PARENT_COUNTS = {"W": 640, "Ta": 128, "Cr": 64, "V": 96, "Hf": 32, "Mo": 64}


@pytest.fixture(scope="module")
def parent():
    return generate_alloy(AlloySpec(counts=PARENT_COUNTS, repeats=(8, 8, 8), seed=3))


def scenario(**kw):
    base = dict(
        scenario_id="s",
        parent_alloy_id="W640Ta128Cr64V96Hf32Mo64",
        exposure_scenario="synthetic-illustration",
        n_re=8,
        n_os=2,
        seed=5,
    )
    base.update(kw)
    return TransmutationScenario(**base)


def test_illustrative_composition_exact(parent):
    atoms, rec = apply_transmutation(parent, "p", scenario())
    assert len(atoms) == 1024
    assert rec.final_composition == {
        "Cr": 64,
        "Hf": 32,
        "Mo": 64,
        "Os": 2,
        "Re": 8,
        "Ta": 128,
        "V": 96,
        "W": 630,
    }
    assert rec.evidence_level == "synthetic"
    assert rec.solute_fraction == pytest.approx(10 / 1024)
    assert len(parent) == 1024 and parent.get_chemical_symbols().count("W") == 640


def test_site_identity_and_provenance(parent):
    atoms, rec = apply_transmutation(parent, "p", scenario())
    for s in rec.sites:
        assert s.original_species == "W" and s.final_species in ("Re", "Os")
        assert s.atom_id == s.lattice_site
        assert atoms.get_array("atom_id")[s.atom_id] == s.atom_id
        assert atoms[s.atom_id].symbol == s.final_species
        assert parent[s.atom_id].symbol == "W"
        assert sum(s.first_shell.values()) == 8 and sum(s.second_shell.values()) == 6
    assert rec.parent_id == "p" and rec.scenario.seed == 5 and rec.parent_ordering_id


def test_deterministic_and_seed_sensitive(parent):
    a1, r1 = apply_transmutation(parent, "p", scenario())
    a2, r2 = apply_transmutation(parent, "p", scenario())
    _, r3 = apply_transmutation(parent, "p", scenario(seed=6))
    assert r1 == r2 and r1.geometry_fingerprint == r2.geometry_fingerprint
    assert a1.get_chemical_symbols() == a2.get_chemical_symbols()
    assert r1.config_id != r3.config_id


def test_serialization_round_trip(parent):
    _, rec = apply_transmutation(parent, "p", scenario())
    assert TransmutationRecord.model_validate_json(rec.model_dump_json()) == rec


def test_concentration_limit_default_and_override(parent):
    with pytest.raises(TransmutationError, match="exceed"):
        apply_transmutation(parent, "p", scenario(n_re=9, n_os=2))
    apply_transmutation(parent, "p", scenario(n_re=8, n_os=2))
    with pytest.raises(ValueError, match="justification"):
        scenario(n_re=9, n_os=2, max_solute_fraction=0.02)
    _, rec = apply_transmutation(
        parent,
        "p",
        scenario(
            n_re=12,
            n_os=2,
            max_solute_fraction=0.02,
            limit_justification="synthetic test of an explicit override",
            inventory_source="synthetic-inventory",
        ),
    )
    assert rec.n_re + rec.n_os == 14


def test_only_w_replaced_and_no_zero_scenario(parent):
    atoms, rec = apply_transmutation(parent, "p", scenario(n_re=5, n_os=5))
    before = parent.get_chemical_symbols()
    changed = {
        i
        for i, (x, y) in enumerate(zip(before, atoms.get_chemical_symbols(), strict=True))
        if x != y
    }
    assert changed == {s.atom_id for s in rec.sites}
    assert all(before[i] == "W" for i in changed)
    with pytest.raises(ValueError):
        scenario(n_re=0, n_os=0)


def test_isolated_re_and_os(parent):
    for dist, kw, species in (
        ("isolated_re", dict(n_re=4, n_os=0), "Re"),
        ("isolated_os", dict(n_re=0, n_os=4), "Os"),
    ):
        atoms, rec = apply_transmutation(parent, "p", scenario(distribution=dist, **kw))
        assert {s.final_species for s in rec.sites} == {species}
        idx = [s.atom_id for s in rec.sites]
        _, d = get_distances(atoms.positions[idx], cell=atoms.cell, pbc=True)
        a = AlloySpec.model_validate_json(parent.info["alloy_spec"]).lattice_parameter
        assert d[d > 0].min() > 1.2 * a


def test_pairs_are_first_neighbors(parent):
    atoms, rec = apply_transmutation(
        parent, "p", scenario(distribution="re_os_pair", n_re=2, n_os=2)
    )
    re = [s for s in rec.sites if s.final_species == "Re"]
    os_ = [s for s in rec.sites if s.final_species == "Os"]
    assert len(re) == len(os_) == 2
    a = AlloySpec.model_validate_json(parent.info["alloy_spec"]).lattice_parameter
    for r in re:
        _, d = get_distances(
            atoms.positions[[r.atom_id]],
            atoms.positions[[o.atom_id for o in os_]],
            cell=atoms.cell,
            pbc=True,
        )
        assert d.min() < 0.9 * a  # first shell is sqrt(3)/2 a = 0.866 a


def test_enriched_is_more_compact_than_random(parent):
    def spread(dist):
        atoms, rec = apply_transmutation(parent, "p", scenario(distribution=dist, n_re=5, n_os=5))
        idx = [s.atom_id for s in rec.sites]
        _, d = get_distances(atoms.positions[idx], cell=atoms.cell, pbc=True)
        return d.max()

    assert spread("enriched") < spread("random")


def test_near_defect_and_parent_already_has_vacancy(parent):
    vac, vconf = generate_vacancy(parent, "p", 100)
    atoms, rec = apply_transmutation(
        vac, "p", scenario(distribution="near_defect", n_re=2, n_os=0), defect=vconf
    )
    assert len(atoms) == 1023 and rec.n_lattice_sites == 1024
    assert all(s.atom_id != 100 for s in rec.sites)
    _, d = get_distances(
        [vconf.sites[0].position],
        atoms.positions[[s.atom_id - (s.atom_id > 100) for s in rec.sites]],
        cell=atoms.cell,
        pbc=True,
    )
    assert d.max() < 3.0
    with pytest.raises(TransmutationError, match="near_defect"):
        apply_transmutation(parent, "p", scenario(distribution="near_defect"))


def test_rejections(parent):
    atoms, _ = apply_transmutation(parent, "p", scenario())
    with pytest.raises(TransmutationError, match="already contains"):
        apply_transmutation(atoms, "p", scenario())
    with pytest.raises(ValueError):
        scenario(distribution="isolated_re", n_os=1)
    with pytest.raises(ValueError):
        scenario(distribution="re_os_pair", n_re=2, n_os=1)
