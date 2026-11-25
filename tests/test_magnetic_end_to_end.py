"""Deterministic end-to-end magnetic workflow on synthetic evidence (M10).

Every energy and moment here is synthetic. It exercises the software path from policy selection to
KMC admission and proves nothing about real magnetic energetics, so nothing is called qualified.
"""

from __future__ import annotations

import numpy as np
from ase.build import bulk

from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input
from matsim_agents.science.magnetism.competition import (
    NOT_GLOBAL_GROUND_STATE,
    StateEvaluation,
    plan_competition,
    rank_states,
    run_pending,
)
from matsim_agents.science.magnetism.consistency import (
    MagneticEvidence,
    check_barrier,
    check_formation_energy,
    gate_transition_for_kmc,
)
from matsim_agents.science.magnetism.defects import derive_defect_configuration
from matsim_agents.science.magnetism.policies import configurations_for_policy, select_policy
from matsim_agents.science.magnetism.results import (
    classify_moments,
    parse_qe_magnetic_output,
    result_from_moments,
)

METHOD = {"functional": "pbe", "ecutwfc": 40, "kpts": [2, 2, 2], "code": "synthetic"}
SYNTHETIC_ENERGY = {"ferromagnetic": -100.00, "custom_collinear": -100.02, "nonmagnetic": -99.5}
SYNTHETIC_MOMENT = {"ferromagnetic": 2.2, "custom_collinear": 2.0, "nonmagnetic": 0.0}


def _fe_cr():
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    atoms.symbols[0] = "Cr"
    atoms.symbols[1] = "Cr"
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    return atoms


def _runner(atoms):
    ids = [int(i) for i in atoms.get_array("atom_id")]

    def run(request):
        config = request.config
        magnitude = SYNTHETIC_MOMENT[config.mode]
        moments = [
            0.0 if sym == "Cr" and config.mode == "nonmagnetic" else magnitude
            for sym in atoms.get_chemical_symbols()
        ]
        return StateEvaluation(
            config,
            request.method_hash,
            SYNTHETIC_ENERGY[config.mode],
            len(atoms),
            result_from_moments(moments, ids, converged=True),
            composition=atoms.get_chemical_formula(),
        )

    return run


def test_policy_to_ranking_and_authorization_gate():
    atoms = _fe_cr()
    configs = configurations_for_policy(atoms, "s", select_policy(atoms))
    plan = plan_competition(configs, METHOD)
    assert len(plan.requests) == len(configs) == 3

    import pytest

    with pytest.raises(PermissionError):
        run_pending(plan, _runner(atoms), authorized=False)

    evaluations = run_pending(plan, _runner(atoms), authorized=True)
    report = rank_states(evaluations)
    assert [r.config_id for r in report.ranking][0] == next(
        c.config_id for c in configs if c.mode == "custom_collinear"
    )
    assert report.ranking[0].relative_energy_meV_per_atom == 0.0
    assert report.statement == NOT_GLOBAL_GROUND_STATE
    assert {r.classification for r in report.ranking} >= {"nonmagnetic"}

    again = plan_competition(configs, METHOD, already_done=[r.key for r in plan.requests])
    assert not again.requests and len(again.skipped) == 3


def test_input_generation_uses_the_policy_configuration(tmp_path):
    atoms = _fe_cr()
    configs = configurations_for_policy(atoms, "s", select_policy(atoms))
    custom = next(c for c in configs if c.mode == "custom_collinear")
    # Explicit synthetic conversion (moment / 8); the code refuses to guess valence counts.
    custom = custom.model_copy(
        update={"qe_starting_magnetization": tuple(m / 8.0 for m in custom.initial_moments_muB)}
    )
    settings = QESettings(
        pseudo_dir="/tmp/pp",
        ecutwfc_ry=40.0,
        ecutrho_ry=320.0,
        occupations="smearing",
        kpts=(2, 2, 2),
        pseudopotentials={"Fe": "Fe.pbe.UPF", "Cr": "Cr.pbe.UPF"},
    )
    path = tmp_path / "pw.in"
    write_pw_input(atoms, settings, str(path), magnetic_config=custom)
    text = path.read_text()
    assert "nspin = 2" in text.replace("  ", " ") or "nspin=2" in text.replace(" ", "")
    assert "starting_magnetization(1)" in text and "starting_magnetization(2)" in text


def test_defect_and_barrier_flow_with_synthetic_evidence_is_blocked_from_kmc():
    atoms = _fe_cr()
    configs = configurations_for_policy(atoms, "s", select_policy(atoms))
    parent = next(c for c in configs if c.mode == "ferromagnetic")
    keep = [i for i in range(len(atoms)) if i != 5]
    vacancy_atoms = atoms[keep]
    defect = derive_defect_configuration(
        parent, vacancy_atoms, defect_structure_id="v", defect_kind="vacancy"
    )

    def evidence(label, cfg, atoms_for, quality="software_tested"):
        ids = [int(i) for i in atoms_for.get_array("atom_id")]
        result = result_from_moments([2.2] * len(ids), ids, converged=True)
        return MagneticEvidence(label, "m", cfg, result, quality)

    pristine = evidence("pristine", parent, atoms)
    vac = evidence("vacancy", defect, vacancy_atoms)
    assert classify_moments(vac.result) == "fm_like"
    assert check_formation_energy(vac, pristine).usable

    barrier = check_barrier(vac, [evidence("saddle", defect, vacancy_atoms)], vac)
    assert barrier.usable
    blocked = gate_transition_for_kmc(barrier, [vac])
    assert not blocked.allowed and any("not numerically validated" in r for r in blocked.reasons)
    admitted = gate_transition_for_kmc(barrier, [vac], allow_unqualified=True)
    assert admitted.allowed and admitted.flags


def test_qe_output_parser_classifies_a_synthetic_collapse():
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    atoms.set_array("atom_id", np.arange(len(atoms), dtype=np.int64))
    text = """
     total magnetization       =     0.00 Bohr mag/cell
     absolute magnetization    =     0.00 Bohr mag/cell
     convergence has been achieved in   9 iterations
"""
    result = parse_qe_magnetic_output(text, atoms, spin_polarized=True)
    # Totals alone carry no per-site evidence, so the classifier must not guess an order.
    assert result.local_moments_muB == ()
    assert classify_moments(result) == "insufficient_evidence"
