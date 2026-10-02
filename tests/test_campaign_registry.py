from __future__ import annotations

from ase import Atoms
from ase.io import write

from matsim_agents.campaign.registry import (
    CandidateRegistry,
    CandidateSelectionPolicy,
    formula_acquisition_metrics,
    ingest_exploration_result,
    select_dft_refinement_candidates,
    soap_descriptor,
    soap_novelty,
    structure_content_hash,
)
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.seeds import PhaseCandidate
from matsim_agents.discovery.wrapper import CompositionExplorationResult
from matsim_agents.orchestration.state import RelaxationResult


def _write_structure(path, *, displacement: float = 0.0):
    atoms = Atoms(
        ["Nb", "O", "O"],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5 + displacement, 0.5, 0.5],
            [0.25, 0.25, 0.25],
        ],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    write(path, atoms, format="vasp")


def test_registry_retains_lineage_when_seeds_collapse_to_one_family(tmp_path):
    prototype = tmp_path / "prototype.vasp"
    random = tmp_path / "random.vasp"
    relaxed = tmp_path / "relaxed.vasp"
    _write_structure(prototype, displacement=0.02)
    _write_structure(random, displacement=-0.02)
    _write_structure(relaxed)
    composition = parse_composition("NbO2")
    assert composition is not None
    candidates = [
        PhaseCandidate(
            formula="NbO2",
            candidate_id="NbO2-P0000",
            structure_path=str(prototype),
            source="prototype",
            prototype_id="AB2-test",
        ),
        PhaseCandidate(
            formula="NbO2",
            candidate_id="NbO2-R0000",
            structure_path=str(random),
            source="random",
            random_seed=202,
            needs_dft_verification=True,
        ),
    ]
    relaxations = [
        RelaxationResult(
            structure_path=candidate.structure_path,
            optimized_structure_path=str(relaxed),
            trajectory_path="",
            log_csv_path="",
            final_energy_eV=-21.0,
            final_max_force_eV_per_A=0.01,
            num_steps=5,
            converged=True,
        )
        for candidate in candidates
    ]
    exploration = CompositionExplorationResult(
        composition=composition,
        phase_candidates=candidates,
        relaxations=relaxations,
    )
    registry = CandidateRegistry()

    ingest_exploration_result(
        registry,
        exploration,
        iteration=1,
        backend="uma",
        model_identifier="uma-s-1p1",
        uncertainty_by_candidate={"NbO2-P0000": 0.1, "NbO2-R0000": 0.4},
        novelty_reference_paths=[str(prototype)],
    )

    assert len(registry.candidates) == 2
    assert len(registry.relaxed_families) == 1
    family_members = next(iter(registry.relaxed_families.values()))
    assert family_members == ["NbO2-P0000", "NbO2-R0000"]
    assert registry.candidates["NbO2-R0000"].random_seed == 202
    metrics = formula_acquisition_metrics(
        registry,
        "NbO2",
        energy_above_hull_eV_per_atom=0.01,
    )
    assert metrics.predicted_hull_proximity == 0.8
    assert metrics.mlip_uncertainty == 0.25
    assert metrics.structural_diversity == 0.5
    assert metrics.convergence == 1.0

    selected, scores = select_dft_refinement_candidates(
        exploration,
        max_candidates=2,
        policy=CandidateSelectionPolicy(enabled=True, maximum_per_relaxed_family=1),
        uncertainty_by_candidate={"NbO2-P0000": 0.1, "NbO2-R0000": 0.4},
    )
    assert len(selected) == 1
    assert set(scores) == {"NbO2-P0000", "NbO2-R0000"}


def test_registry_soap_species_include_cross_composition_references(tmp_path):
    current = tmp_path / "nb-o.vasp"
    reference = tmp_path / "ta-o.vasp"
    _write_structure(current)
    write(
        reference,
        Atoms(
            ["Ta", "O"],
            scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            cell=[5.0, 5.0, 5.0],
            pbc=True,
        ),
        format="vasp",
    )
    composition = parse_composition("NbO2")
    assert composition is not None
    candidate = PhaseCandidate(formula="NbO2", structure_path=str(current))
    exploration = CompositionExplorationResult(
        composition=composition,
        phase_candidates=[candidate],
        relaxations=[
            RelaxationResult(
                structure_path=str(current),
                optimized_structure_path=str(current),
                trajectory_path="",
                log_csv_path="",
                final_energy_eV=-1.0,
                final_max_force_eV_per_A=0.01,
                num_steps=1,
                converged=True,
            )
        ],
    )

    registry = CandidateRegistry()
    candidate_ids = ingest_exploration_result(
        registry,
        exploration,
        iteration=1,
        novelty_reference_paths=[str(reference)],
    )

    assert registry.evaluations[candidate_ids[0]].structural_novelty is not None


def test_registry_namespaces_retried_candidate_ids(tmp_path):
    seed = tmp_path / "seed.vasp"
    relaxed = tmp_path / "relaxed.vasp"
    retried_relaxed = tmp_path / "retried-relaxed.vasp"
    _write_structure(seed, displacement=0.02)
    _write_structure(relaxed)
    _write_structure(retried_relaxed, displacement=0.0001)
    composition = parse_composition("NbO2")
    assert composition is not None
    candidate = PhaseCandidate(
        formula="NbO2",
        candidate_id="NbO2-P0000",
        structure_path=str(seed),
        source="prototype",
    )
    relaxation = RelaxationResult(
        structure_path=str(seed),
        optimized_structure_path=str(relaxed),
        trajectory_path="",
        log_csv_path="",
        final_energy_eV=-21.0,
        final_max_force_eV_per_A=0.01,
        num_steps=5,
        converged=True,
    )
    exploration = CompositionExplorationResult(
        composition=composition,
        phase_candidates=[candidate],
        relaxations=[relaxation],
    )
    registry = CandidateRegistry()

    first = ingest_exploration_result(registry, exploration, iteration=1)
    relaxation.optimized_structure_path = str(retried_relaxed)
    second = ingest_exploration_result(registry, exploration, iteration=2)

    assert first == ["NbO2-P0000"]
    assert second == ["NbO2-P0000-iter0002"]
    assert set(registry.candidates) == {"NbO2-P0000", "NbO2-P0000-iter0002"}
    assert set(registry.evaluations) == set(registry.candidates)
    assert len(registry.relaxed_families) == 1
    assert (
        registry.candidates[first[0]].relaxed_family_id
        == registry.candidates[second[0]].relaxed_family_id
    )


def test_structure_hash_and_soap_are_translation_invariant(tmp_path):
    first = tmp_path / "first.vasp"
    shifted = tmp_path / "shifted.vasp"
    _write_structure(first)
    atoms = Atoms(
        ["O", "Nb", "O"],
        scaled_positions=[[0.35, 0.35, 0.35], [0.1, 0.1, 0.1], [0.6, 0.6, 0.6]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    write(shifted, atoms, format="vasp")

    assert structure_content_hash(first) == structure_content_hash(shifted)
    first_descriptor = soap_descriptor(first, species=["Nb", "O"])
    shifted_descriptor = soap_descriptor(shifted, species=["Nb", "O"])
    assert soap_novelty(first_descriptor, [shifted_descriptor]) < 1e-12


def test_structure_hash_preserves_nonperiodic_separation(tmp_path):
    first = tmp_path / "first.extxyz"
    separated = tmp_path / "separated.extxyz"
    atoms = Atoms(
        ["O", "O"],
        positions=[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=[True, True, False],
    )
    write(first, atoms)
    atoms.positions[1, 2] += 5.0
    write(separated, atoms)

    assert structure_content_hash(first) != structure_content_hash(separated)
