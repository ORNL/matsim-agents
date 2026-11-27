import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from matsim_agents.science.cascade.boundaries import (
    assess_cascade_boundaries,
    minimum_cell_length_A,
)
from matsim_agents.science.cascade.events import (
    CascadeEvent,
    EventError,
    UnsupportedChannelError,
    event_from_pka,
    structure_hash,
    validate_event,
)
from matsim_agents.science.cascade.qualification import (
    REQUIRED_CHECKS,
    Check,
    Domain,
    Level,
    QualificationRecord,
    Query,
    require_level,
    resolve_level,
)
from matsim_agents.science.irradiation.contracts import PKAEvent, RecoilSpecies


def _w():
    atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat((4, 4, 4))
    atoms.symbols[10] = "Ta"
    return atoms


def _pka(site=0, element="W", energy_mev=1e-3):
    return PKAEvent(
        index=0,
        species=RecoilSpecies(element=element),
        energy_mev=energy_mev,
        site_index=site,
        probability=0.5,
    )


def _event(atoms, **kw):
    args = dict(
        parent_material_id="W-bcc",
        channel="elastic",
        seed=1,
        nuclear_data_provenance="unit-test",
        scenario_id="s",
        rate_basis="probability",
    )
    args.update(kw)
    return event_from_pka(_pka(), atoms, **args)


def test_event_energy_unit_and_hash_binding():
    atoms = _w()
    ev = _event(atoms)
    assert ev.energy_eV == pytest.approx(1000.0)
    atoms.positions[3, 0] += 0.5
    with pytest.raises(EventError, match="does not match"):
        validate_event(ev, atoms)


def test_species_and_site_consistency():
    atoms = _w()
    with pytest.raises(EventError, match="holds Ta"):
        event_from_pka(
            _pka(site=10),
            atoms,
            parent_material_id="m",
            channel="elastic",
            seed=1,
            nuclear_data_provenance="x",
            scenario_id="s",
            rate_basis="p",
        )
    with pytest.raises(EventError, match="no site"):
        event_from_pka(
            PKAEvent(index=0, species=RecoilSpecies(element="W"), energy_mev=1e-3, probability=1.0),
            atoms,
            parent_material_id="m",
            channel="elastic",
            seed=1,
            nuclear_data_provenance="x",
            scenario_id="s",
            rate_basis="p",
        )


def test_unsupported_channel_and_light_ion_are_refused_with_reasons():
    atoms = _w()
    with pytest.raises(UnsupportedChannelError, match="not supported"):
        _event(atoms, channel="(n,p)")
    he = CascadeEvent(
        event_id="e",
        parent_material_id="m",
        parent_structure_hash=structure_hash(atoms),
        kind="emitted_light_ion",
        recoil_species=RecoilSpecies(element="He"),
        parent_species=None,
        site_index=0,
        energy_eV=100.0,
        scenario_id="s",
        channel="(n,a)",
        seed=1,
        nuclear_data_provenance="x",
        rate_or_probability=1.0,
        rate_or_probability_basis="p",
    )
    with pytest.raises(UnsupportedChannelError, match="emitted light ion"):
        validate_event(he, atoms)


def test_transmutation_daughter_must_sit_on_parent_site():
    atoms = _w()
    base = dict(
        event_id="e",
        parent_material_id="m",
        parent_structure_hash=structure_hash(atoms),
        kind="transmutation_daughter",
        recoil_species=RecoilSpecies(element="Re"),
        parent_species=RecoilSpecies(element="W"),
        energy_eV=100.0,
        scenario_id="s",
        channel="(n,gamma)",
        seed=1,
        nuclear_data_provenance="x",
        rate_or_probability=1.0,
        rate_or_probability_basis="p",
    )
    validate_event(CascadeEvent(site_index=0, parent_site_index=0, **base), atoms)
    with pytest.raises(EventError, match="parent nucleus's site"):
        validate_event(CascadeEvent(site_index=1, parent_site_index=0, **base), atoms)
    with pytest.raises(EventError, match="needs its parent"):
        validate_event(CascadeEvent(site_index=0, **base), atoms)


def _record(passing, domain=None, checkpoint="ck1"):
    dom = domain or Domain(
        elements=frozenset({"W"}),
        temperature_K=(0.0, 600.0),
        recoil_energy_eV=(10.0, 5000.0),
        n_atoms=(100, 50000),
        precision="float64",
        short_range_model="zbl-additive",
    )
    names = [n for v in REQUIRED_CHECKS.values() for n in v]
    return QualificationRecord(
        checkpoint, dom, tuple(Check(n, n in passing, "c", 1.0, "d") for n in names)
    )


def _query(**kw):
    args = dict(
        elements=frozenset({"W"}),
        temperature_K=300.0,
        recoil_energy_eV=500.0,
        n_atoms=2000,
        precision="float64",
        short_range_model="zbl-additive",
    )
    args.update(kw)
    return Query(**args)


def test_levels_nest_and_stop_at_first_missing_group():
    allnames = {n for v in REQUIRED_CHECKS.values() for n in v}
    assert resolve_level(_record(set()), _query(), checkpoint="ck1").level == Level.UNASSESSED
    eq = set(REQUIRED_CHECKS[Level.EQUILIBRIUM_QUALIFIED])
    assert (
        resolve_level(_record(eq), _query(), checkpoint="ck1").level == Level.EQUILIBRIUM_QUALIFIED
    )
    only_cascade = set(REQUIRED_CHECKS[Level.CASCADE_QUALIFIED])
    assert (
        resolve_level(_record(only_cascade), _query(), checkpoint="ck1").level == Level.UNASSESSED
    )
    assert (
        resolve_level(_record(allnames), _query(), checkpoint="ck1").level
        == Level.CASCADE_QUALIFIED
    )


def test_isolated_defect_qualification_does_not_cover_cascades():
    names = set(REQUIRED_CHECKS[Level.EQUILIBRIUM_QUALIFIED]) | set(
        REQUIRED_CHECKS[Level.DEFECT_QUALIFIED]
    )
    res = resolve_level(_record(names), _query(), checkpoint="ck1")
    assert res.level == Level.DEFECT_QUALIFIED
    with pytest.raises(PermissionError, match="CASCADE_QUALIFIED"):
        require_level(res, Level.CASCADE_QUALIFIED)


def test_qualification_does_not_leave_its_domain():
    allnames = {n for v in REQUIRED_CHECKS.values() for n in v}
    rec = _record(allnames)
    for bad in (
        _query(elements=frozenset({"W", "Cr"})),
        _query(temperature_K=900.0),
        _query(recoil_energy_eV=1e5),
        _query(n_atoms=10),
        _query(precision="float32"),
        _query(short_range_model="none"),
    ):
        assert resolve_level(rec, bad, checkpoint="ck1").level == Level.UNASSESSED
    assert resolve_level(rec, _query(), checkpoint="other").level == Level.UNASSESSED


def test_failed_required_check_blocks_level_and_is_reported():
    allnames = {n for v in REQUIRED_CHECKS.values() for n in v} - {"elastic"}
    res = resolve_level(_record(allnames), _query(), checkpoint="ck1")
    assert res.level == Level.UNASSESSED and "elastic" in " ".join(res.reasons)


def test_conflicting_duplicate_check_does_not_grant_a_level():
    allnames = {n for v in REQUIRED_CHECKS.values() for n in v}
    rec = _record(allnames)
    rec = QualificationRecord(
        rec.checkpoint, rec.domain, (*rec.checks, Check("elastic", False, "c", 2.0, "d"))
    )
    res = resolve_level(rec, _query(), checkpoint="ck1")
    assert res.level == Level.UNASSESSED and "elastic" in " ".join(res.reasons)


def test_bad_domain_ranges_are_rejected():
    with pytest.raises(ValueError):
        Domain(frozenset({"W"}), (5, 1), (1, 2), (1, 2), "f", "s")


def test_boundary_report_flags_periodic_image_interaction_and_reservoir():
    start = _w()
    end = start.copy()
    end.positions[100] += [1.0, 0.0, 0.0]
    far = int(np.argmax(np.linalg.norm(start.positions - start.positions[0], axis=1)))
    end.positions[far] += [1.0, 0.0, 0.0]
    rep = assess_cascade_boundaries(start, end, pka_index=0)
    assert not rep.ok and "periodic images" in " ".join(rep.findings)
    clean = assess_cascade_boundaries(start, start.copy(), pka_index=0)
    assert clean.ok and clean.damaged_extent_A == 0.0
    mask = np.zeros(len(start), bool)
    mask[far] = True
    rep2 = assess_cascade_boundaries(start, end, pka_index=0, boundary_mask=mask)
    text = " ".join(rep2.findings)
    assert "thermostatted layer" in text and "reservoir" in text


def test_recirculation_and_minimum_cell_length():
    start = _w()
    rep = assess_cascade_boundaries(
        start, start.copy(), pka_index=0, far_side_ke_eV=5.0, far_side_ke_limit_eV=1.0
    )
    assert not rep.ok and "recirculates" in rep.findings[0]
    assert minimum_cell_length_A(30.0, safety=1.0) == pytest.approx(60.0)


def test_displacements_use_the_nearest_image_in_a_skewed_cell():
    cell = np.array([[10.0, 0, 0], [9.0, 1.5, 0], [0, 0, 10.0]])
    start = Atoms("W2", positions=[[0, 0, 0], [5, 0.7, 5]], cell=cell, pbc=True)
    end = start.copy()
    shift = np.array([-1.9, 2.2, 0.0])  # rounding fractional coordinates gives 9.1 A here
    end.positions[1] += shift
    images = [i * cell[0] + j * cell[1] for i in range(-3, 4) for j in range(-3, 4)]
    nearest = min(np.linalg.norm(shift + g) for g in images)
    rep = assess_cascade_boundaries(start, end, pka_index=0)
    assert rep.max_displacement_A == pytest.approx(nearest)
    with pytest.raises(ValueError):
        minimum_cell_length_A(-1.0)
