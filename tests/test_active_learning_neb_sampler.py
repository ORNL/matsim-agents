from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write

from matsim_agents.backends.mlip.config import MACEConfig
from matsim_agents.workflows.active_learning.config import NebSamplingConfig, SamplingConfig
from matsim_agents.workflows.active_learning.neb_sampler import sample_neb_candidates


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
    with pytest.raises(ValueError, match="requires at least two uncertainty calculators"):
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
