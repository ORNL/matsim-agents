from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.science.neb.contracts import (
    MigrationMechanism,
    NebResult,
    NebTransition,
    build_correspondence,
    endpoint_from_atoms,
)
from matsim_agents.workflows.active_learning.config import (
    NebEscalationConfig,
    NebKineticsConfig,
)
from matsim_agents.workflows.active_learning.loop import (
    IterationState,
    _apply_neb_escalation,
    _write_neb_kinetics,
)


def _qe_neb_config():
    return QENebConfig(
        n_images=4,
        species=("Cu",),
        pseudo_dir="/tmp/pseudo",
        pseudopotentials={"Cu": "Cu.UPF"},
        ecutwfc_ry=30.0,
        ecutrho_ry=120.0,
        occupations="fixed",
        kpts=(1, 1, 1),
        nspin=1,
    )


def _pathway(uncertainty):
    initial = Atoms("Cu2", positions=[[1, 1, 1], [3, 1, 1]], cell=[8, 8, 8], pbc=True)
    initial.set_array("atom_id", np.array([0, 1]))
    final = initial.copy()
    final.positions[0, 1] += 0.2
    initial_endpoint = endpoint_from_atoms(
        initial, "initial", "initial", parent_id="parent", defect_type="vacancy"
    )
    final_endpoint = endpoint_from_atoms(
        final, "final", "final", parent_id="parent", defect_type="vacancy"
    )
    transition = NebTransition(
        transition_id="transition-1",
        initial=initial_endpoint,
        final=final_endpoint,
        correspondence=build_correspondence(initial_endpoint, final_endpoint),
        mechanism=MigrationMechanism(kind="vacancy_hop", moving_atom_ids=(0,)),
    )
    result = NebResult(
        state="converged_unqualified",
        method_id="mlip-neb",
        n_images=4,
        forward_barrier_eV=0.4,
        reverse_barrier_eV=0.3,
        pathway_uncertainty_eV=uncertainty,
        provenance={"transition_id": transition.transition_id, "backend": "mlip"},
    )
    outcome = SimpleNamespace(
        result=result,
        transition=transition,
        images=(initial, initial.copy(), final.copy(), final),
    )
    return outcome


def test_eligible_converged_mlip_barrier_is_accepted_without_dft(tmp_path):
    policy = NebEscalationConfig(
        policy="mlip_only",
        max_pathway_uncertainty_eV=0.2,
    )
    record, dispatched = _apply_neb_escalation(
        policy,
        _pathway(0.1),
        tmp_path,
        tmp_path / "neb",
        IterationState(iteration=0),
        completed_dft_neb=0,
    )

    assert record["decision"] == "accepted_mlip"
    assert not dispatched


def test_missing_uncertainty_escalates_to_existing_qe_neb_workflow(tmp_path, monkeypatch):
    import matsim_agents.workflows.neb as neb_workflow

    policy = NebEscalationConfig(
        policy="uncertainty_gated",
        max_pathway_uncertainty_eV=0.2,
        max_dft_neb_runs=1,
        dft_neb=_qe_neb_config(),
    )
    dft_result = NebResult(
        state="converged_unqualified",
        method_id="qe-neb",
        n_images=4,
        forward_barrier_eV=0.5,
        reverse_barrier_eV=0.4,
        provenance={"transition_id": "transition-1"},
    )
    calls = []

    def fake_run(pair, config, directory, **kwargs):
        calls.append((pair, config, directory, kwargs))
        return SimpleNamespace(
            plan=SimpleNamespace(status="planned", reasons=(), directory=str(directory)),
            result=dft_result,
        )

    monkeypatch.setattr(neb_workflow, "run_neb_workflow", fake_run)
    iteration_dir = tmp_path / "iteration_0000"
    iteration_dir.mkdir()
    record, dispatched = _apply_neb_escalation(
        policy,
        _pathway(None),
        iteration_dir,
        iteration_dir / "neb",
        IterationState(iteration=0),
        completed_dft_neb=0,
    )

    assert dispatched
    assert record["decision"] == "dft_neb_complete"
    assert record["dft_result"]["method_id"] == "qe-neb"
    assert calls[0][3]["dispatch"] is True
    assert calls[0][0].transition.transition_id == "transition-1"
    assert (iteration_dir / "state.json").is_file()


def test_required_escalation_fails_explicitly_when_budget_is_exhausted(tmp_path):
    policy = NebEscalationConfig(
        policy="uncertainty_gated",
        max_pathway_uncertainty_eV=0.2,
        max_dft_neb_runs=1,
        dft_neb=_qe_neb_config(),
    )
    with pytest.raises(RuntimeError, match="max_dft_neb_runs is exhausted"):
        _apply_neb_escalation(
            policy,
            _pathway(None),
            tmp_path,
            tmp_path / "neb",
            IterationState(iteration=0),
            completed_dft_neb=1,
        )


def test_accepted_mlip_barrier_is_written_as_exploratory_kinetics_evidence(tmp_path):
    outcome = _pathway(0.1)
    target = _write_neb_kinetics(
        SimpleNamespace(
            kinetics=NebKineticsConfig(
                attempt_frequency_hz=1.0e13,
                temperature_K=600.0,
                prefactor_note="user-supplied prefactor",
            )
        ),
        outcome.transition,
        outcome.result,
        tmp_path,
    )
    saved = json.loads(target.read_text())

    assert saved["neb_result"]["provenance"]["backend"] == "mlip"
    assert len(saved["rates"]) == 2
    assert {event["qualification"] for event in saved["events"]} == {"unqualified"}
    assert all(rate["qualification"] == "unqualified" for rate in saved["rates"])
    assert saved["report"]["ready_for_exploratory_kmc"]
