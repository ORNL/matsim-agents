from __future__ import annotations

import pickle
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read, write

from matsim_agents.backends.mlip.config import MACEConfig
from matsim_agents.backends.mlip.pool import WorkerOutcome
from matsim_agents.science.neb.contracts import (
    MigrationMechanism,
    NebResult,
    NebTransition,
    build_correspondence,
    endpoint_from_atoms,
)
from matsim_agents.workflows.active_learning.config import (
    AcquisitionConfig,
    NebRefinementStage,
    NebSamplingConfig,
    NEBSamplingStrategy,
    SamplingConfig,
)
from matsim_agents.workflows.active_learning.neb_sampler import (
    NebSamplingOutcome,
    NebSamplingTask,
    NebSamplingWorkerResult,
    sample_neb_candidates,
    sample_neb_candidates_parallel,
)
from matsim_agents.workflows.active_learning.sampling import sample_strategies
from matsim_agents.workflows.active_learning.uncertainty import (
    score_ensemble_pathway,
    score_mc_dropout_pathway,
)


class _ScaledImageEnergy(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = self.scale * float(atoms.positions[0, 0])


class _DropoutImageEnergy(_ScaledImageEnergy):
    def __init__(self, torch):
        super().__init__(1.0)
        self.torch = torch
        self.model = torch.nn.Sequential(torch.nn.Dropout(p=0.5))

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        value = self.torch.tensor([[float(atoms.positions[0, 0])]])
        self.results["energy"] = float(self.model(value).item())


def _endpoints(tmp_path):
    initial = Atoms(
        "Cu2",
        positions=[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    initial.set_array("atom_id", np.array([0, 1]))
    final = initial.copy()
    final.positions[0, 1] += 0.3
    initial_path = tmp_path / "initial.extxyz"
    final_path = tmp_path / "final.extxyz"
    write(initial_path, initial)
    write(final_path, final)
    return initial_path, final_path


def test_ase_neb_emits_shared_candidates_and_barrier_result(tmp_path):
    initial, final = _endpoints(tmp_path)
    config = NebSamplingConfig(
        initial_path=initial,
        final_path=final,
        n_images=4,
        max_steps=2,
        min_distance_A=0.5,
    )
    outcome = sample_neb_candidates(config, EMT(), tmp_path / "neb")

    assert len(outcome.candidates) == 2
    assert all(candidate.sampling_method == "neb" for candidate in outcome.candidates)
    assert all(candidate.md_step is None for candidate in outcome.candidates)
    assert all(candidate.predicted_forces_eV_per_A is not None for candidate in outcome.candidates)
    assert outcome.result.n_images == 4
    assert outcome.result.forward_barrier_eV is not None
    assert outcome.result.reverse_barrier_eV is not None
    assert outcome.result.saddle_image_index in (1, 2)
    assert (tmp_path / "neb" / "neb-images.extxyz").exists()
    assert (tmp_path / "neb" / "initial-to-final-result.json").exists()
    resumed = sample_neb_candidates(config, EMT(), tmp_path / "neb")
    assert resumed.result.optimization_steps >= outcome.result.optimization_steps


def test_independent_neb_pathways_use_configured_worker_pool(tmp_path, monkeypatch):
    import matsim_agents.workflows.active_learning.neb_sampler as neb_sampler

    initial, final = _endpoints(tmp_path)
    strategies = [
        NEBSamplingStrategy(
            method="neb",
            parameters=NebSamplingConfig(
                initial_path=initial,
                final_path=final,
                transition_id=f"path-{index}",
                n_images=4,
                min_distance_A=0.5,
            ),
        )
        for index in range(2)
    ]

    class Pool:
        uses_workers = True

    pool = Pool()
    submitted = []

    def fake_parallel(tasks, worker_pool):
        assert worker_pool is pool
        submitted.extend(tasks)
        return [
            NebSamplingOutcome(
                candidates=[],
                result=NebResult(state="unconverged", method_id="mock", n_images=4),
                images=(),
            )
            for _ in tasks
        ]

    monkeypatch.setattr(neb_sampler, "sample_neb_candidates_parallel", fake_parallel)
    batch = sample_strategies(
        strategies,
        EMT(),
        tmp_path / "batch",
        random_seed=3,
        iteration=0,
        seed_paths=None,
        mlip_config=MACEConfig(model="small"),
        acquisition=None,
        md_pool_factory=lambda: nullcontext(pool),
    )

    assert [task.transition_label for task in submitted] == ["path-0", "path-1"]
    assert len({task.out_dir for task in submitted}) == 2
    assert [path.outcome.result.method_id for path in batch.neb_pathways] == ["mock", "mock"]


def test_neb_worker_task_returns_pickle_safe_result(tmp_path, monkeypatch):
    import matsim_agents.backends.mlip.pool as mlip_pool
    import matsim_agents.workflows.active_learning.md_sampler as md_sampler
    import matsim_agents.workflows.active_learning.neb_sampler as neb_sampler

    initial, final = _endpoints(tmp_path)
    config = NebSamplingConfig(
        initial_path=initial,
        final_path=final,
        n_images=4,
        max_steps=1,
        min_distance_A=0.5,
    )
    monkeypatch.setattr(mlip_pool, "prepare_worker_device", lambda: None)
    monkeypatch.setattr(
        md_sampler,
        "_worker_mlip_calculators",
        lambda *args, **kwargs: (EMT(), []),
    )
    task = NebSamplingTask(
        config_json=config.model_dump_json(),
        mlip_config_json=MACEConfig(model="small").model_dump_json(),
        out_dir=str(tmp_path / "worker-result"),
        transition_label="worker-path",
        iteration=0,
        model_id="mace:small",
        enable_mc_dropout=False,
        uncertainty_mode=None,
        dropout_passes=0,
        dropout_probability=0.1,
        uncertainty_seed=2,
    )

    worker_result = neb_sampler._sample_neb_task(task)
    restored = pickle.loads(pickle.dumps(worker_result))

    assert NebSamplingWorkerResult is type(restored)
    assert len(restored.candidates) == 2

    class LocalPool:
        def map(self, function, tasks):
            return [WorkerOutcome(value=function(item)) for item in tasks]

    outcome = sample_neb_candidates_parallel(
        [replace(task, out_dir=str(tmp_path / "parallel-result"))],
        LocalPool(),
    )[0]
    assert outcome.result.n_images == 4
    assert outcome.images[0].calc is None


def test_ensemble_barrier_uncertainty_uses_per_model_forward_and_reverse_differences():
    images = []
    for energy in (0.0, 2.0, 1.0):
        image = Atoms("Cu", positions=[[energy, 0.0, 0.0]])
        images.append(image)

    score = score_ensemble_pathway(images, [_ScaledImageEnergy(1.0), _ScaledImageEnergy(2.0)])

    assert score.forward_barrier_eV == pytest.approx(1.0)
    assert score.reverse_barrier_eV == pytest.approx(0.5)
    assert score.pathway_eV == pytest.approx(1.0)


def test_mc_dropout_pathway_uncertainty_uses_a_correlated_draw_for_each_path():
    torch = pytest.importorskip("torch")
    images = []
    for energy in (0.0, 2.0, 1.0):
        image = Atoms("Cu", positions=[[energy, 0.0, 0.0]])
        images.append(image)

    score = score_mc_dropout_pathway(images, _DropoutImageEnergy(torch), 32, 0.5, seed=7)

    assert score.forward_barrier_eV > 0.0
    assert score.reverse_barrier_eV == pytest.approx(score.forward_barrier_eV / 2.0)


def test_mc_dropout_configuration_requires_a_real_variance_estimate():
    with pytest.raises(ValueError):
        AcquisitionConfig(mc_dropout_passes=1)
    with pytest.raises(ValueError):
        AcquisitionConfig(mc_dropout_p=0.0)


def test_ci_neb_and_idpp_are_configurable(tmp_path):
    initial, final = _endpoints(tmp_path)
    config = NebSamplingConfig(
        initial_path=initial,
        final_path=final,
        n_images=4,
        interpolation="idpp",
        climbing_image=True,
        max_steps=2,
        min_distance_A=0.5,
    )
    outcome = sample_neb_candidates(config, EMT(), tmp_path / "ci-neb")

    assert outcome.result.climbing_image_used
    assert outcome.candidates


def test_optional_two_stage_neb_runs_climbing_image_refinement(tmp_path):
    initial, final = _endpoints(tmp_path)
    config = NebSamplingConfig(
        initial_path=initial,
        final_path=final,
        n_images=4,
        fmax_eV_per_A=100.0,
        max_steps=2,
        min_distance_A=0.5,
        refinement_stage=NebRefinementStage(fmax_eV_per_A=100.0, max_steps=2),
    )

    outcome = sample_neb_candidates(config, EMT(), tmp_path / "two-stage")

    assert outcome.result.climbing_image_used
    assert outcome.result.provenance["refinement_stage_used"] == "true"
    assert (tmp_path / "two-stage" / "neb-refinement-images.extxyz").is_file()
    assert (tmp_path / "two-stage" / "neb-refinement-state.json").is_file()


def test_endpoint_relaxation_reuses_the_mlip_relaxation_workflow(tmp_path, monkeypatch):
    import matsim_agents.backends.mlip.relaxation as relaxation

    initial, final = _endpoints(tmp_path)
    calls = []

    def fake_relax(args, *, calculators):
        calls.append((args, calculators))
        return SimpleNamespace(converged=True, optimized_structure_path=args.structure_path)

    monkeypatch.setattr(relaxation, "relax_with_mlip", fake_relax)
    config = NebSamplingConfig(
        initial_path=initial,
        final_path=final,
        n_images=4,
        max_steps=1,
        min_distance_A=0.5,
        relax_endpoints=True,
    )
    sample_neb_candidates(config, EMT(), tmp_path / "relaxed", mlip_config=MACEConfig())

    assert len(calls) == 2
    assert calls[0][0].optimizer == "FIRE"
    assert calls[0][1] is calls[1][1]


def test_reused_relaxed_endpoints_must_match_the_active_mlip_provenance(tmp_path):
    initial_path, final_path = _endpoints(tmp_path)
    initial = Atoms(
        "Cu2",
        positions=[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    initial.set_array("atom_id", np.array([0, 1]))
    final = initial.copy()
    final.positions[0, 1] += 0.3
    initial_endpoint = endpoint_from_atoms(
        initial,
        "initial",
        "initial",
        parent_id="parent",
        defect_type="vacancy",
        status="relaxed",
        relaxation_converged=True,
        method="mlip:model-a",
    )
    final_endpoint = endpoint_from_atoms(
        final,
        "final",
        "final",
        parent_id="parent",
        defect_type="vacancy",
        status="relaxed",
        relaxation_converged=True,
        method="mlip:model-a",
    )
    transition = NebTransition(
        transition_id="transition-001",
        initial=initial_endpoint,
        final=final_endpoint,
        correspondence=build_correspondence(initial_endpoint, final_endpoint),
        mechanism=MigrationMechanism(kind="vacancy_hop", moving_atom_ids=(0,)),
    )
    config = NebSamplingConfig(
        initial_path=initial_path,
        final_path=final_path,
        transition=transition,
        n_images=4,
        fmax_eV_per_A=100.0,
        max_steps=1,
        min_distance_A=0.5,
    )

    with pytest.raises(ValueError, match="active MLIP is 'mlip:model-b'"):
        sample_neb_candidates(config, EMT(), tmp_path / "wrong-model", model_id="model-b")

    outcome = sample_neb_candidates(config, EMT(), tmp_path / "matching-model", model_id="model-a")
    assert outcome.result.provenance["initial_endpoint_method"] == "mlip:model-a"


def test_endpoint_constraints_and_magnetic_state_must_match():
    from matsim_agents.workflows.active_learning.neb_sampler import (
        _validate_endpoint_settings_match,
    )

    initial = Atoms("Cu2", positions=[[0, 0, 0], [1, 1, 1]])
    initial.set_array("atom_id", np.array([0, 1]))
    final = initial.copy()
    initial.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match="constraints differ"):
        _validate_endpoint_settings_match(initial, final)

    initial.set_constraint([])
    initial.set_initial_magnetic_moments([1.0, 0.0])
    with pytest.raises(ValueError, match="magnetic configurations differ"):
        _validate_endpoint_settings_match(initial, final)


def test_endpoint_relaxation_rejects_collapsed_typed_transition(tmp_path, monkeypatch):
    import matsim_agents.backends.mlip.relaxation as relaxation

    initial_path, final_path = _endpoints(tmp_path)
    initial, final = read(initial_path), read(final_path)
    initial_endpoint = endpoint_from_atoms(
        initial, "initial", "initial", parent_id="parent", defect_type="vacancy"
    )
    final_endpoint = endpoint_from_atoms(
        final, "final", "final", parent_id="parent", defect_type="vacancy"
    )
    transition = NebTransition(
        transition_id="transition-001",
        initial=initial_endpoint,
        final=final_endpoint,
        correspondence=build_correspondence(initial_endpoint, final_endpoint),
        mechanism=MigrationMechanism(kind="vacancy_hop", moving_atom_ids=(0,)),
    )

    def collapse_endpoints(args, *, calculators):
        return SimpleNamespace(converged=True, optimized_structure_path=initial_path)

    monkeypatch.setattr(relaxation, "relax_with_mlip", collapse_endpoints)
    config = NebSamplingConfig(
        initial_path=initial_path,
        final_path=final_path,
        transition=transition,
        n_images=4,
        max_steps=1,
        min_distance_A=0.5,
        relax_endpoints=True,
    )
    with pytest.raises(ValueError, match="collapsed the declared moving-atom displacement"):
        sample_neb_candidates(
            config,
            EMT(),
            tmp_path / "collapsed",
            mlip_config=MACEConfig(model="small"),
        )


def test_pathway_uncertainty_gate_requires_an_interval_and_ensemble(tmp_path):
    initial, final = _endpoints(tmp_path)
    with pytest.raises(ValueError, match="requires uncertainty_interval_steps"):
        NebSamplingConfig(
            initial_path=initial,
            final_path=final,
            max_pathway_uncertainty_eV=0.2,
        )
    config = NebSamplingConfig(
        initial_path=initial,
        final_path=final,
        n_images=4,
        max_pathway_uncertainty_eV=0.2,
        uncertainty_interval_steps=1,
    )
    with pytest.raises(ValueError, match="requires configured pathway uncertainty"):
        sample_neb_candidates(config, EMT(), tmp_path / "uq", uncertainty_calculators=())


def test_mixed_sampling_configuration_accepts_neb_only_and_combined():
    neb = {
        "method": "neb",
        "parameters": {
            "initial_path": "initial.extxyz",
            "final_path": "final.extxyz",
        },
    }
    config = SamplingConfig(strategies=[neb])
    assert [strategy.method for strategy in config.strategies] == ["neb"]
