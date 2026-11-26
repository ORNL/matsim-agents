import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.mixing import SumCalculator

from matsim_agents.science.cascade.dynamics import StepControl, TabulatedStopping
from matsim_agents.science.cascade.events import event_from_pka
from matsim_agents.science.cascade.protocol import (
    Phase,
    ProtocolError,
    analyze_cascade,
    run_protocol,
    thermalize,
)
from matsim_agents.science.cascade.qualification import Level, Resolution
from matsim_agents.science.cascade.zbl import ZBLCalculator
from matsim_agents.science.irradiation.contracts import PKAEvent, RecoilSpecies


def _calc():
    return SumCalculator([LennardJones(sigma=2.5, epsilon=0.3, rc=6.0), ZBLCalculator(0.6, 1.2)])


def _cell():
    atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat((3, 3, 3))
    atoms.calc = _calc()
    return atoms


def _event(atoms, energy_mev=2e-5, site=0):
    pka = PKAEvent(
        index=0,
        species=RecoilSpecies(element="W"),
        energy_mev=energy_mev,
        site_index=site,
        probability=0.5,
    )
    return event_from_pka(
        pka,
        atoms,
        parent_material_id="W",
        channel="elastic",
        seed=7,
        nuclear_data_provenance="unit-test",
        scenario_id="s",
        rate_basis="probability",
    )


def _phases():
    ctl = StepControl(max_displacement_A=0.02, dt_min_fs=1e-3, dt_max_fs=1.0)
    return (
        Phase("ballistic", 20.0, ctl, True, False),
        Phase("thermal_spike", 20.0, ctl, True, True),
        Phase("cooling", 20.0, ctl, False, True),
    )


def _qual(level):
    return Resolution(level, (), {})


def _run(atoms=None, **kw):
    atoms = atoms or _cell()
    thermalize(atoms, 300.0, seed=1)
    mask = np.zeros(len(atoms), bool)
    mask[-8:] = True
    kw.setdefault("qualification", _qual(Level.CASCADE_QUALIFIED))
    return run_protocol(
        atoms,
        _event(atoms),
        direction=(1.0, 0.3, 0.1),
        phases=_phases(),
        boundary_mask=mask,
        boundary_target_K=300.0,
        **kw,
    )


def test_unqualified_potential_is_refused_unless_explicitly_waived():
    with pytest.raises(ProtocolError, match="CASCADE_QUALIFIED"):
        _run(qualification=None)
    with pytest.raises(ProtocolError, match="EQUILIBRIUM_QUALIFIED"):
        _run(qualification=_qual(Level.EQUILIBRIUM_QUALIFIED))
    res = _run(qualification=None, require_level=None)
    assert res.qualification is None


def test_transmutation_daughter_is_substituted_before_the_recoil():
    atoms = _cell()
    thermalize(atoms, 300.0, seed=1)
    pka = PKAEvent(
        index=0, species=RecoilSpecies(element="Re"), energy_mev=2e-5, site_index=0, probability=1
    )
    ev = event_from_pka(
        pka,
        atoms,
        parent_material_id="W",
        channel="(n,gamma)",
        seed=7,
        nuclear_data_provenance="unit-test",
        scenario_id="s",
        rate_basis="probability",
        kind="transmutation_daughter",
        parent_species=RecoilSpecies(element="W"),
        parent_site_index=0,
    )
    res = run_protocol(atoms, ev, direction=(1.0, 0.3, 0.1), phases=_phases(), require_level=None)
    assert res.initial_atoms.get_chemical_symbols()[0] == "Re"
    assert res.initial_atoms.get_masses()[0] == pytest.approx(186.207, abs=1e-2)
    assert res.recoil.realized_energy_eV == pytest.approx(20.0)


def test_output_contract_is_complete():
    res = _run(qualification=_qual(Level.CASCADE_QUALIFIED))
    assert res.phases == ("ballistic", "thermal_spike", "cooling")
    assert res.duration_fs == pytest.approx(60.0)
    assert res.recoil.realized_energy_eV == pytest.approx(res.event.energy_eV)
    assert res.max_force_eV_per_A > 0
    assert len(res.initial_atoms) == len(res.final_atoms)
    assert res.provenance["seed"] == 7 and res.provenance["protocol_hash"]
    assert all(len(r.trace.dt_fs) == len(r.trace.time_fs) for r in res.runs)
    assert res.energy_thermostat_eV != 0.0 or res.energy_electronic_eV == 0.0


def test_thermostat_is_off_in_ballistic_phase():
    res = _run()
    assert res.runs[0].boundary_used is False and res.runs[1].boundary_used is True


def test_stopping_energy_is_recorded_only_in_stopping_phases():
    stop = TabulatedStopping({"W": ([0.01, 1e5], [0.2, 0.2])}, source="unit-test table")
    res = _run(stopping=stop)
    assert res.runs[0].electronic_loss_eV > 0 and res.runs[2].electronic_loss_eV == 0.0
    assert res.provenance["stopping_source"] == "unit-test table"


def test_run_is_reproducible():
    a, b = _run(), _run()
    assert np.array_equal(a.final_atoms.positions, b.final_atoms.positions)


def test_checkpoint_restart_matches_uninterrupted_run(tmp_path):
    full = _run(checkpoint_dir=tmp_path / "full")
    atoms = _cell()
    thermalize(atoms, 300.0, seed=1)
    mask = np.zeros(len(atoms), bool)
    mask[-8:] = True
    ev = _event(atoms)
    part = run_protocol(
        atoms,
        ev,
        direction=(1.0, 0.3, 0.1),
        phases=_phases()[:1],
        boundary_mask=mask,
        boundary_target_K=300.0,
        qualification=_qual(Level.CASCADE_QUALIFIED),
        checkpoint_dir=tmp_path / "part",
    )
    assert part.phases == ("ballistic",)
    fresh = _cell()
    resumed = run_protocol(
        fresh,
        ev,
        direction=(1.0, 0.3, 0.1),
        phases=_phases(),
        boundary_mask=mask,
        boundary_target_K=300.0,
        qualification=_qual(Level.CASCADE_QUALIFIED),
        checkpoint_dir=tmp_path / "part",
        resume=True,
    )
    assert resumed.phases == ("thermal_spike", "cooling")
    assert resumed.provenance["resumed_from_phase"] == 1
    assert np.allclose(resumed.final_atoms.positions, full.final_atoms.positions, atol=1e-9)


def test_resume_without_checkpoint_and_mask_without_target_fail(tmp_path):
    atoms = _cell()
    thermalize(atoms, 300.0, seed=1)
    ev = _event(atoms)
    with pytest.raises(ProtocolError, match="checkpoint"):
        run_protocol(
            atoms, ev, direction=(1, 0, 0), phases=_phases(), require_level=None, resume=True
        )
    with pytest.raises(ProtocolError, match="target temperature"):
        run_protocol(
            atoms,
            ev,
            direction=(1, 0, 0),
            phases=_phases(),
            require_level=None,
            boundary_mask=np.ones(len(atoms), bool),
        )


def test_analysis_of_unperturbed_run_finds_no_defects_and_flags_qualification():
    ref = _cell()
    res = _run(qualification=_qual(Level.EQUILIBRIUM_QUALIFIED), require_level=None)
    out = analyze_cascade(res, ref, cluster_cutoff=3.5, cluster_justification="unit test")
    assert out.summary.n_frenkel_pairs == 0 and out.summary.n_vacancies == 0
    assert any("not qualified" in c for c in out.caveats)
    assert out.event_id == res.event.event_id


def test_analysis_counts_a_displaced_atom_as_a_frenkel_pair():
    ref = _cell()
    res = _run(qualification=_qual(Level.CASCADE_QUALIFIED))
    final = ref.copy()
    final.positions[5] = final.positions[0] + [1.58, 0.0, 0.0] * np.array([1, 0, 0]) + [0, 1.58, 0]
    object.__setattr__(res, "final_atoms", final)
    out = analyze_cascade(res, ref, cluster_cutoff=3.5, cluster_justification="unit test")
    assert out.summary.n_frenkel_pairs >= 1
    assert out.caveats == ()
