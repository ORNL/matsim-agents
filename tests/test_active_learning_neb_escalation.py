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
from matsim_agents.workflows.active_learning import neb_escalation as al_neb_escalation
from matsim_agents.workflows.active_learning.config import (
    NebEscalationConfig,
    NebKineticsConfig,
)
from matsim_agents.workflows.active_learning.iteration_state import IterationState
from matsim_agents.workflows.active_learning.neb_escalation import (
    _apply_neb_escalation,
    _completed_dft_neb_count,
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
    import matsim_agents.workflows.neb.workflow as neb_workflow

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
        timeout_sec=123.0,
    )

    assert dispatched
    assert record["decision"] == "dft_neb_complete"
    assert record["dft_result"]["method_id"] == "qe-neb"
    assert calls[0][3]["dispatch"] is True
    assert calls[0][3]["timeout_sec"] == 123.0
    assert calls[0][0].transition.transition_id == "transition-1"
    assert (iteration_dir / "state.json").is_file()


def test_unavailable_uncertainty_is_never_treated_as_low_even_when_the_limit_is_huge(tmp_path):
    policy = NebEscalationConfig(policy="mlip_only", max_pathway_uncertainty_eV=1e9)
    record, dispatched = _apply_neb_escalation(
        policy,
        _pathway(None),
        tmp_path,
        tmp_path / "neb",
        IterationState(iteration=0),
        completed_dft_neb=0,
    )

    assert record["decision"] == "not_accepted"
    assert record["escalation"]["reason"] == "uncertainty_unavailable"
    assert not dispatched


def test_neb_image_labels_are_marked_single_points_not_a_dft_mep(tmp_path):
    from matsim_agents.workflows.active_learning.candidates import Candidate
    from matsim_agents.workflows.active_learning.labelling import _record_provenance

    image = Atoms("Cu2", positions=[[1, 1, 1], [3, 1, 1]], cell=[8, 8, 8], pbc=True)
    candidate = Candidate(
        candidate_id="img-1",
        atoms=image.copy(),
        sampling_method="neb",
        model_id="mlip-a",
        provenance={"transition_id": "t1", "label_is_dft_mep": True},
    )
    frame = SimpleNamespace(atoms=image.copy(), source_work_dir=str(tmp_path / "img-1"))
    cfg = SimpleNamespace(loop=SimpleNamespace(out_dir=tmp_path), md=None)

    _record_provenance(cfg, [frame], [candidate], 0)

    assert frame.atoms.info["label_kind"] == "dft_single_point"
    assert frame.atoms.info["label_is_dft_mep"] is False
    assert frame.atoms.info["transition_id"] == "t1"


def test_escalation_record_names_the_policy_reason_and_mlip(tmp_path):
    outcome = _pathway(0.5)
    outcome.result = outcome.result.model_copy(
        update={"provenance": {**outcome.result.provenance, "model_id": "mlip-a"}}
    )
    policy = NebEscalationConfig(policy="mlip_only", max_pathway_uncertainty_eV=0.2)
    record, _ = _apply_neb_escalation(
        policy,
        outcome,
        tmp_path,
        tmp_path / "neb",
        IterationState(iteration=0),
        completed_dft_neb=0,
    )

    assert record["escalation"] == {
        "policy": "mlip_only",
        "mlip_model_id": "mlip-a",
        "reason": "uncertainty_above_limit",
    }


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


def _escalation_policy():
    return NebEscalationConfig(
        policy="always_dft_neb",
        max_pathway_uncertainty_eV=0.2,
        max_dft_neb_runs=2,
        dft_neb=_qe_neb_config(),
    )


def test_constrained_pathway_is_rejected_before_a_dft_run_is_reserved(tmp_path, monkeypatch):
    from ase.constraints import FixAtoms

    import matsim_agents.workflows.neb.workflow as neb_workflow

    monkeypatch.setattr(neb_workflow, "run_neb_workflow", lambda *a, **k: pytest.fail("ran"))
    outcome = _pathway(None)
    outcome.images[1].set_constraint(FixAtoms(indices=[0]))
    root = tmp_path / "run"
    iteration_dir = root / "iteration_0000"
    iteration_dir.mkdir(parents=True)
    state = IterationState(iteration=0)

    with pytest.raises(ValueError, match="constrained images"):
        _apply_neb_escalation(
            _escalation_policy(), outcome, iteration_dir, iteration_dir / "neb", state, 0
        )

    assert state.n_dft_neb_runs == 0
    assert _completed_dft_neb_count(root) == 0


def test_dft_neb_reservation_survives_deletion_of_the_iteration_directory(tmp_path, monkeypatch):
    import shutil

    import matsim_agents.workflows.neb.workflow as neb_workflow

    def interrupted(*args, **kwargs):
        raise RuntimeError("allocation ended")

    monkeypatch.setattr(neb_workflow, "run_neb_workflow", interrupted)
    root = tmp_path / "run"
    iteration_dir = root / "iteration_0000"
    iteration_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="allocation ended"):
        _apply_neb_escalation(
            _escalation_policy(),
            _pathway(None),
            iteration_dir,
            iteration_dir / "neb",
            IterationState(iteration=0),
            0,
        )
    shutil.rmtree(iteration_dir)  # what _scan_resume does to an unfinished iteration

    assert _completed_dft_neb_count(root) == 1


def test_interrupted_reservation_is_added_to_legacy_runs_without_double_counting(tmp_path):
    root = tmp_path / "run"
    for index, count in ((0, 1), (1, 1)):
        directory = root / f"iteration_{index:04d}"
        directory.mkdir(parents=True)
        (directory / "state.json").write_text(
            json.dumps({"iteration": index, "n_dft_neb_runs": count})
        )
    # Iteration 2 reserved a run and was deleted on resume; iteration 1 is ledger-backed.
    (root / "dft_neb_reservations.jsonl").write_text(
        json.dumps({"iteration": 1, "transition_id": "t"})
        + "\n"
        + json.dumps({"iteration": 2, "transition_id": "t"})
        + "\n"
    )

    assert _completed_dft_neb_count(root) == 3


def test_escalation_passes_optimized_images_endpoints_and_dft_environment(tmp_path, monkeypatch):
    import matsim_agents.workflows.neb.workflow as neb_workflow

    calls = []

    def fake_run(pair, config, directory, **kwargs):
        calls.append((pair, kwargs))
        return SimpleNamespace(
            plan=SimpleNamespace(status="planned", reasons=(), directory=str(directory)),
            result=NebResult(
                state="converged_unqualified",
                method_id="qe-neb",
                n_images=4,
                forward_barrier_eV=0.5,
                reverse_barrier_eV=0.4,
                provenance={"transition_id": "transition-1"},
            ),
        )

    monkeypatch.setattr(neb_workflow, "run_neb_workflow", fake_run)
    outcome = _pathway(None)
    relaxed = outcome.images[0].copy()
    relaxed.positions[0, 2] += 0.05
    outcome.initial = relaxed
    root = tmp_path / "run"
    iteration_dir = root / "iteration_0000"
    iteration_dir.mkdir(parents=True)

    record, _ = _apply_neb_escalation(
        _escalation_policy(),
        outcome,
        iteration_dir,
        iteration_dir / "neb",
        IterationState(iteration=0),
        0,
        env={"MATSIM_TEST": "1"},
    )

    pair, kwargs = calls[0]
    assert kwargs["env"] == {"MATSIM_TEST": "1"}
    assert kwargs["images"] == tuple(outcome.images)
    assert np.allclose(pair.initial.positions, relaxed.positions)
    assert record["dft_start_path"] == "mlip_optimized"


@pytest.mark.parametrize(("launcher", "expected_local"), [((), True), (("srun", "-n", "8"), False)])
def test_escalation_environment_is_local_only_without_a_qe_launcher(
    tmp_path, monkeypatch, launcher, expected_local
):

    seen = []

    class _Launch:
        def environment(self, *, local=False):
            seen.append(local)
            return {"LOCAL": str(local)}

    placement = SimpleNamespace(pool=lambda name: SimpleNamespace(launches=[_Launch()]))
    received = []

    def fake_apply(*args, env=None, **kwargs):
        received.append(env)
        return {"decision": "not_configured"}, False

    monkeypatch.setattr(al_neb_escalation, "_apply_neb_escalation", fake_apply)
    policy = _escalation_policy()
    policy = policy.model_copy(
        update={"dft_neb": policy.dft_neb.model_copy(update={"launcher": launcher})}
    )
    cfg = SimpleNamespace(escalation=policy)
    pathway = SimpleNamespace(
        outcome=_pathway(None),
        directory=tmp_path / "neb",
        parameters=SimpleNamespace(kinetics=None),
    )

    al_neb_escalation._escalate_neb_pathways(
        cfg, [pathway], tmp_path, tmp_path, IterationState(iteration=0), 0, placement
    )

    assert seen == [expected_local]
    assert received == [{"LOCAL": str(expected_local)}]


def test_kinetics_evidence_describes_the_calculation_that_produced_the_result(tmp_path):
    from matsim_agents.science.neb.contracts import NebCalculationSpec

    outcome = _pathway(0.1)
    transition = outcome.transition.model_copy(
        update={
            "calculation": NebCalculationSpec(n_images=7, engine="quantum_espresso", method_id="qe")
        }
    )
    target = _write_neb_kinetics(
        SimpleNamespace(
            kinetics=NebKineticsConfig(attempt_frequency_hz=1.0e13, temperature_K=600.0)
        ),
        transition,
        outcome.result,
        tmp_path,
    )
    saved = json.loads(target.read_text())

    assert saved["declared_calculation"]["method_id"] == "qe"
    assert saved["neb_result"]["method_id"] == "mlip-neb"
