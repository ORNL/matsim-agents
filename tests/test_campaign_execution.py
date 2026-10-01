from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from matsim_agents.campaign.execution import (
    CampaignDFTRefinementConfig,
    CampaignFormulaExecutionConfig,
    CampaignRetrainingConfig,
    _cross_model_scores,
    _perturbation_robustness,
    _score_relaxed_candidate_uncertainty,
    latest_promoted_model,
    run_formula_with_active_learning,
)
from matsim_agents.campaign.state import CampaignState, FormulaRunRecord
from matsim_agents.campaign.surrogate_hull import (
    evaluate_surrogate_hull,
    reference_coverage,
    surrogate_hull_coverage,
)
from matsim_agents.campaign.unary_references import relax_unary_references
from matsim_agents.discovery.composition import parse_composition
from matsim_agents.discovery.formula import FormulaGenerationPolicy
from matsim_agents.discovery.seeds import PhaseCandidate
from matsim_agents.discovery.stability import ReferenceEnergySet
from matsim_agents.discovery.wrapper import CompositionExplorationResult
from matsim_agents.execution.contracts import EvidenceLevel, WorkflowStatus
from matsim_agents.orchestration.state import RelaxationResult
from matsim_agents.workflows.phase_exploration import (
    PhaseExplorationPolicy,
    PhaseExplorationWorkflowResult,
)
from matsim_agents.workflows.relaxation import (
    RelaxationMode,
    RelaxationStageResult,
    ScientificRelaxationResult,
)


def _config_yaml(tmp_path, pseudopotentials: str) -> str:
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Nb.upf").write_text("Nb", encoding="utf-8")
    (pseudo_dir / "O.upf").write_text("O", encoding="utf-8")
    return f"""
mlip:
  backend: uma
  uma:
    model_name: uma-s-1p1
    task_name: omat
    device: cuda
md:
  seed_source:
    kind: paths
    paths: [/unused/seed.vasp]
  n_steps: 4
  sample_every: 2
acquisition:
  strategy: random
  n_select: 1
dft:
  backend: qe
  qe:
    pw_bin: /unused/pw.x
    pw_wrapper: /unused/wrapper.sh
    pseudo_dir: {pseudo_dir}
    pseudopotentials: {pseudopotentials}
    kpts: [4, 4, 4]
trainer:
  enabled: false
loop:
  n_iterations: 1
  out_dir: /unused/output
"""


def test_surrogate_reference_coverage_requires_pure_element_endpoints():
    elemental_endpoints, subsystems = reference_coverage(["Nb", "NbO", "NbTa", "NbTaO4"])

    assert elemental_endpoints == {"Nb"}
    assert subsystems == {
        frozenset({"Nb", "O"}),
        frozenset({"Nb", "Ta"}),
        frozenset({"Nb", "Ta", "O"}),
    }

    report = surrogate_hull_coverage(["Nb", "NbO"], "NbO", "references.json")
    assert report["provisional"] is True
    assert report["missing_elemental_references"] == ["O"]
    assert report["missing_binary_subsystems"] == []


def test_unary_reference_search_relaxes_and_selects_each_model_endpoint(tmp_path):
    from ase.build import bulk
    from ase.calculators.calculator import Calculator, all_changes

    class VolumeCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": float(atoms.get_volume()),
                "forces": np.zeros((len(atoms), 3)),
            }

    phases = []
    for phase_id, crystal, correction in (
        ("Nb-bcc", "bcc", 0.0),
        ("Nb-bcc-copy", "bcc", 0.0),
        ("Nb-fcc", "fcc", -20.0),
    ):
        path = tmp_path / f"{phase_id}.extxyz"
        write(path, bulk("Nb", crystal, a=3.0))
        phases.append(
            {
                "phase_id": phase_id,
                "formula": "Nb",
                "path": str(path),
                "source": "test",
                "energy_correction_eV_per_atom": correction,
            }
        )

    result = relax_unary_references(
        phases,
        {"Nb", "O"},
        VolumeCalculator(),
        model_identifier="test:model",
        output_dir=tmp_path / "relaxed",
        relax_cell=False,
    )

    assert len(result.phases) == 3
    assert result.generated_counts == {"Nb": 3, "O": 0}
    assert result.converged_counts == {"Nb": 3, "O": 0}
    assert result.unique_converged_counts == {"Nb": 2, "O": 0}
    assert result.selected_endpoints == {"Nb": "Nb-fcc"}
    assert [phase.phase_id for phase in result.phases if phase.selected_endpoint] == ["Nb-fcc"]
    selected = next(phase for phase in result.phases if phase.selected_endpoint)
    assert selected.energy_above_endpoint_eV_per_atom == pytest.approx(0.0)
    assert all(
        phase.energy_above_endpoint_eV_per_atom is not None
        for phase in result.phases
        if phase.converged
    )
    duplicate = next(phase for phase in result.phases if phase.phase_id == "Nb-bcc-copy")
    assert duplicate.duplicate_of == "Nb-bcc"
    assert result.missing_elements == ["O"]
    assert result.provisional is True
    assert (tmp_path / "relaxed" / "unary_reference_search.json").is_file()

    cached = relax_unary_references(
        phases,
        {"Nb", "O"},
        object(),
        model_identifier="test:model",
        output_dir=tmp_path / "relaxed",
        relax_cell=False,
    )
    assert cached == result


def test_surrogate_hull_uses_relaxed_model_specific_unary_endpoints(tmp_path):
    from ase.build import bulk
    from ase.calculators.calculator import Calculator, all_changes

    class ElementCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            energy = sum(-2.0 if symbol == "Nb" else -1.0 for symbol in atoms.symbols)
            self.results = {
                "energy": energy,
                "forces": np.zeros((len(atoms), 3)),
            }

    nb_bcc = tmp_path / "Nb-bcc.extxyz"
    nb_fcc = tmp_path / "Nb-fcc.extxyz"
    ta_bcc = tmp_path / "Ta-bcc.extxyz"
    oxygen = tmp_path / "O2.extxyz"
    write(nb_bcc, bulk("Nb", "bcc", a=3.0))
    write(nb_fcc, bulk("Nb", "fcc", a=3.0))
    write(ta_bcc, bulk("Ta", "bcc", a=3.0))
    write(
        oxygen,
        Atoms("O2", positions=[[0, 0, 0], [1.2, 0, 0]], cell=[12.0] * 3, pbc=True),
    )
    manifest = tmp_path / "references.json"
    manifest.write_text(
        json.dumps(
            {
                "phases": {
                    "Nb-bcc": {
                        "formula": "Nb",
                        "path": str(nb_bcc),
                        "relax_cell": False,
                    },
                    "Nb-fcc": {
                        "formula": "Nb",
                        "path": str(nb_fcc),
                        "relax_cell": False,
                    },
                    "Ta-bcc": {
                        "formula": "Ta",
                        "path": str(ta_bcc),
                        "relax_cell": False,
                    },
                    "O2": {
                        "formula": "O2",
                        "path": str(oxygen),
                        "relax_cell": False,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    labels = [{"structure_path": "target.extxyz", "energy_eV": -3.0}]

    report = evaluate_surrogate_hull(
        manifest,
        "NbO",
        labels,
        ElementCalculator(),
        model_identifier="test:model",
        unary_max_steps=2,
    )

    assert report["missing_elemental_references"] == []
    assert report["unary_reference_search"]["generated_counts"] == {
        "Nb": 2,
        "O": 1,
        "Ta": 1,
    }
    assert report["unary_reference_search"]["converged_counts"] == {
        "Nb": 2,
        "O": 1,
        "Ta": 1,
    }
    assert set(report["unary_reference_search"]["selected_endpoints"]) == {
        "Nb",
        "O",
        "Ta",
    }
    assert report["undercovered_unary_elements"] == ["O"]
    assert report["provisional"] is True
    assert labels[0]["surrogate_formation_energy_eV_per_atom"] == pytest.approx(0.0)
    assert labels[0]["surrogate_energy_above_hull_eV_per_atom"] == pytest.approx(0.0)


def test_relaxed_candidate_uncertainty_uses_registry_candidate_ids(tmp_path, monkeypatch):
    import matsim_agents.campaign.execution as execution
    from matsim_agents.active_learning.config import ALConfig

    config_path = tmp_path / "al.yaml"
    config_path.write_text(_config_yaml(tmp_path, "{Nb: Nb.upf, O: O.upf}"), encoding="utf-8")
    cfg = ALConfig.from_yaml(config_path)
    cfg.acquisition.strategy = "mc_dropout"
    structure = tmp_path / "relaxed.extxyz"
    write(structure, Atoms(["Nb", "O"], positions=[[0, 0, 0], [1, 0, 0]]))
    composition = parse_composition("NbO")
    assert composition is not None
    exploration = CompositionExplorationResult(
        composition=composition,
        phase_candidates=[
            PhaseCandidate(
                formula="NbO",
                structure_path=str(structure),
                candidate_id="NbO-P0042",
            )
        ],
        relaxations=[
            RelaxationResult(
                structure_path=str(structure),
                optimized_structure_path=str(structure),
                trajectory_path="",
                log_csv_path="",
                final_energy_eV=-2.0,
                final_max_force_eV_per_A=0.01,
                num_steps=2,
                converged=True,
            )
        ],
    )
    monkeypatch.setattr(execution, "make_mlip_calculator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        execution,
        "select_candidates",
        lambda candidates, *args, **kwargs: (candidates, np.asarray([0.73])),
    )

    scores = _score_relaxed_candidate_uncertainty(exploration, cfg)

    assert scores == {"NbO-P0042": 0.73}


def test_cross_model_scores_records_ranking_disagreement(tmp_path):
    from ase.calculators.calculator import Calculator, all_changes

    from matsim_agents.active_learning.config import ALConfig

    config_path = tmp_path / "al.yaml"
    config_path.write_text(_config_yaml(tmp_path, "{Nb: Nb.upf, O: O.upf}"), encoding="utf-8")
    cfg = ALConfig.from_yaml(config_path)
    paths = []
    relaxations = []
    for index in range(2):
        path = tmp_path / f"candidate-{index}.extxyz"
        write(path, Atoms(["Nb", "O"], positions=[[0, 0, 0], [1 + index, 0, 0]]))
        paths.append(path)
        relaxations.append(
            RelaxationResult(
                structure_path=str(path),
                optimized_structure_path=str(path),
                trajectory_path="",
                log_csv_path="",
                final_energy_eV=-2.0,
                final_max_force_eV_per_A=0.01,
                num_steps=1,
                converged=True,
            )
        )
    composition = parse_composition("NbO")
    assert composition is not None
    exploration = CompositionExplorationResult(
        composition=composition,
        phase_candidates=[],
        relaxations=relaxations,
    )
    calls = 0

    def calculator_factory(_cfg):
        nonlocal calls
        order = [-2.0, -1.0] if calls == 0 else [-1.0, -2.0]
        calls += 1

        class PositionCalculator(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                index = int(round(atoms.positions[1, 0] - 1))
                self.results = {
                    "energy": order[index],
                    "forces": np.zeros((len(atoms), 3)),
                }

        return PositionCalculator()

    scores = _cross_model_scores(
        exploration,
        [cfg, cfg.model_copy(deep=True)],
        calculator_factory=calculator_factory,
    )

    assert scores["ranking_disagreement"] is True


def test_cross_model_scores_dispatches_validator_to_independent_python(tmp_path, monkeypatch):
    from ase.calculators.calculator import Calculator, all_changes

    import matsim_agents.campaign.execution as execution
    from matsim_agents.active_learning.config import ALConfig

    config_path = tmp_path / "al.yaml"
    config_path.write_text(_config_yaml(tmp_path, "{Nb: Nb.upf, O: O.upf}"), encoding="utf-8")
    cfg = ALConfig.from_yaml(config_path)
    structure = tmp_path / "candidate.extxyz"
    write(structure, Atoms(["Nb", "O"], positions=[[0, 0, 0], [1, 0, 0]]))
    composition = parse_composition("NbO")
    assert composition is not None
    exploration = CompositionExplorationResult(
        composition=composition,
        phase_candidates=[],
        relaxations=[
            RelaxationResult(
                structure_path=str(structure),
                optimized_structure_path=str(structure),
                trajectory_path="",
                log_csv_path="",
                final_energy_eV=-2.0,
                final_max_force_eV_per_A=0.01,
                num_steps=1,
                converged=True,
            )
        ],
    )

    class PrimaryCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": -2.0, "forces": np.zeros((len(atoms), 3))}

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, json.loads(kwargs["input"])))
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "labels": [{"structure_path": str(structure), "energy_per_atom_eV": -1.5}],
                    "ranking": [str(structure)],
                    "surrogate_hull": None,
                }
            )
        )

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    scores = _cross_model_scores(
        exploration,
        [cfg, cfg.model_copy(deep=True)],
        config_paths=[None, config_path],
        validation_python=Path("/independent/mace/python"),
        calculator_factory=lambda _cfg: PrimaryCalculator(),
    )

    assert calls[0][0][0] == "/independent/mace/python"
    assert calls[0][1]["config"] == str(config_path)
    assert len(scores["rankings"]) == 2


def test_cross_model_scores_uses_per_validator_python(tmp_path, monkeypatch):
    import matsim_agents.campaign.execution as execution
    from matsim_agents.active_learning.config import ALConfig

    config_path = tmp_path / "al.yaml"
    config_path.write_text(_config_yaml(tmp_path, "{Nb: Nb.upf, O: O.upf}"), encoding="utf-8")
    cfg = ALConfig.from_yaml(config_path)
    structure = tmp_path / "candidate.extxyz"
    write(structure, Atoms(["Nb", "O"], positions=[[0, 0, 0], [1, 0, 0]]))
    composition = parse_composition("NbO")
    assert composition is not None
    exploration = CompositionExplorationResult(
        composition=composition,
        phase_candidates=[],
        relaxations=[
            RelaxationResult(
                structure_path=str(structure),
                optimized_structure_path=str(structure),
                trajectory_path="",
                log_csv_path="",
                final_energy_eV=-2.0,
                final_max_force_eV_per_A=0.01,
                num_steps=1,
                converged=True,
            )
        ],
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            stdout=json.dumps({"labels": [], "ranking": [], "surrogate_hull": None})
        )

    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    _cross_model_scores(
        exploration,
        [cfg, cfg.model_copy(deep=True), cfg.model_copy(deep=True)],
        config_paths=[None, config_path, config_path],
        validation_pythons=[
            None,
            Path("/independent/mace/python"),
            Path("/independent/hydragnn/python"),
        ],
    )

    assert [call[0] for call in calls] == [
        "/independent/mace/python",
        "/independent/hydragnn/python",
    ]


def test_perturbation_robustness_records_each_seed(tmp_path):
    from matsim_agents.active_learning.config import ALConfig

    config_path = tmp_path / "al.yaml"
    config_path.write_text(_config_yaml(tmp_path, "{Nb: Nb.upf, O: O.upf}"), encoding="utf-8")
    cfg = ALConfig.from_yaml(config_path)
    structure = tmp_path / "relaxed.extxyz"
    atoms = Atoms(["Nb", "O"], positions=[[0, 0, 0], [1, 0, 0]])
    write(structure, atoms)
    composition = parse_composition("NbO")
    assert composition is not None
    exploration = CompositionExplorationResult(
        composition=composition,
        phase_candidates=[],
        relaxations=[
            RelaxationResult(
                structure_path=str(structure),
                optimized_structure_path=str(structure),
                trajectory_path="",
                log_csv_path="",
                final_energy_eV=-2.0,
                final_max_force_eV_per_A=0.01,
                num_steps=1,
                converged=True,
            )
        ],
    )

    def runner(request):
        return RelaxationResult(
            structure_path=request.structure_path,
            optimized_structure_path=request.structure_path,
            trajectory_path="",
            log_csv_path="",
            final_energy_eV=-2.0,
            final_max_force_eV_per_A=0.01,
            num_steps=1,
            converged=True,
        )

    evidence = _perturbation_robustness(
        exploration,
        cfg,
        CampaignFormulaExecutionConfig(
            active_learning_config=config_path,
            phase_policy=PhaseExplorationPolicy(),
            perturbation_trials=3,
            perturbation_seed=17,
        ),
        tmp_path / "robustness",
        relaxation_runner=runner,
    )

    assert [trial["seed"] for trial in evidence["trials"]] == [17, 18, 19]
    assert evidence["robust_fraction"] == 1.0


def test_uma_only_execution_labels_polymorphs_without_dft_or_active_learning(tmp_path):
    config_path = tmp_path / "al.yaml"
    config_path.write_text(_config_yaml(tmp_path, "{Nb: Nb.upf, O: O.upf}"), encoding="utf-8")
    structure = tmp_path / "NbO2.extxyz"
    write(
        structure,
        Atoms(["Nb", "O", "O"], positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
    )
    relaxation = RelaxationResult(
        structure_path=str(structure),
        optimized_structure_path=str(structure),
        trajectory_path="",
        log_csv_path="",
        final_energy_eV=-9.0,
        final_max_force_eV_per_A=0.02,
        num_steps=4,
        converged=True,
    )

    def phase_runner(formula, *, policy, **kwargs):
        assert policy.active_learning is False
        composition = parse_composition(formula)
        assert composition is not None
        return PhaseExplorationWorkflowResult(
            composition=formula,
            initial=CompositionExplorationResult(
                composition=composition,
                phase_candidates=[
                    PhaseCandidate(
                        formula=formula,
                        candidate_id="NbO2-P0001",
                        structure_path=str(structure),
                        num_atoms=3,
                    )
                ],
                relaxations=[relaxation],
            ),
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("UMA-only execution invoked a forbidden DFT/AL path")

    result = run_formula_with_active_learning(
        "NbO2",
        str(tmp_path / "formula"),
        config=CampaignFormulaExecutionConfig(
            active_learning_config=config_path,
            execution_mode="uma_only",
            phase_policy=PhaseExplorationPolicy(active_learning=False),
        ),
        al_runner=forbidden,
        phase_runner=phase_runner,
        relaxation_runner=forbidden,
    )

    evidence = result.active_learning_result
    assert evidence is not None
    assert evidence["execution_mode"] == "uma_only"
    assert evidence["evidence_level"] == "mlip_prediction"
    assert evidence["n_dft_calculations"] == 0
    assert evidence["mlip_labels"] == [
        {
            "candidate_id": "NbO2-P0001",
            "structure_path": str(structure),
            "optimized_structure_path": str(structure),
            "converged": True,
            "final_energy_eV": -9.0,
            "energy_per_atom_eV": -3.0,
            "residual_force_eV_per_A": 0.02,
            "uncertainty": None,
        }
    ]


def test_uma_only_execution_rejects_dft_or_active_learning_configuration(tmp_path):
    config_path = tmp_path / "al.yaml"
    config_path.touch()
    with pytest.raises(ValueError, match="disables active learning"):
        CampaignFormulaExecutionConfig(
            active_learning_config=config_path,
            execution_mode="uma_only",
            phase_policy=PhaseExplorationPolicy(active_learning=True, dft_approved=True),
        )


def test_uma_only_execution_rejects_non_uma_backend(tmp_path, monkeypatch):
    config_path = tmp_path / "al.yaml"
    config_path.touch()
    monkeypatch.setattr(
        "matsim_agents.campaign.execution.ALConfig.from_yaml",
        lambda path: SimpleNamespace(mlip=SimpleNamespace(backend="mace")),
    )

    with pytest.raises(ValueError, match="requires mlip.backend='uma'"):
        run_formula_with_active_learning(
            "NbO2",
            str(tmp_path / "formula"),
            config=CampaignFormulaExecutionConfig(
                active_learning_config=config_path,
                execution_mode="uma_only",
                phase_policy=PhaseExplorationPolicy(active_learning=False),
            ),
            phase_runner=lambda *args, **kwargs: pytest.fail("phase runner was invoked"),
        )


def _vasp_config_yaml(tmp_path) -> str:
    vasp_bin = tmp_path / "vasp_std"
    wrapper = tmp_path / "vasp-wrapper.sh"
    incar = tmp_path / "INCAR.template"
    potcar_dir = tmp_path / "potcars"
    vasp_bin.touch()
    wrapper.touch()
    incar.write_text(
        "ENCUT = 600\nISMEAR = 0\nSIGMA = 0.05\nIBRION = -1\nNSW = 0\n",
        encoding="utf-8",
    )
    potcar_dir.mkdir()
    return f"""
mlip:
    backend: uma
    uma:
        model_name: uma-s-1p1
        task_name: omat
        device: cuda
md:
    seed_source:
        kind: paths
        paths: [/unused/seed.vasp]
    n_steps: 4
    sample_every: 2
acquisition:
    strategy: random
    n_select: 1
dft:
    backend: vasp
    vasp:
        vasp_bin: {vasp_bin}
        vasp_wrapper: {wrapper}
        incar_template: {incar}
        potcar_dir: {potcar_dir}
        nodes_per_job: 2
        ranks_per_node: 4
        threads_per_rank: 8
        extra_incar:
            LMAXMIX: '4'
trainer:
    enabled: false
loop:
    n_iterations: 1
    out_dir: /unused/output
"""


def _empty_result(formula: str, al_result: dict) -> PhaseExplorationWorkflowResult:
    composition = parse_composition(formula)
    assert composition is not None
    return PhaseExplorationWorkflowResult(
        composition=formula,
        initial=CompositionExplorationResult(
            composition=composition,
            phase_candidates=[],
            relaxations=[],
        ),
        active_learning_result=al_result,
        model_promoted=bool(al_result.get("model_promoted")),
    )


def test_formula_execution_rewrites_al_template_and_reports_usage(tmp_path):
    config_path = tmp_path / "al.yaml"
    config_path.write_text(
        _config_yaml(tmp_path, "{Nb: Nb.upf, O: O.upf}"),
        encoding="utf-8",
    )
    observed = {}

    def al_runner(cfg):
        observed["seed_kind"] = cfg.md.seed_source.kind
        observed["compositions"] = cfg.md.seed_source.compositions
        observed["out_dir"] = cfg.loop.out_dir
        state_dir = cfg.loop.out_dir / "iteration_0000"
        state_dir.mkdir(parents=True)
        (state_dir / "state.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "n_dft_converged": 2,
                    "n_dft_failed": 1,
                    "model_promoted": False,
                    "timings_sec": {"total": 1800.0},
                }
            ),
            encoding="utf-8",
        )

    def phase_runner(
        formula,
        *,
        policy,
        output_dir,
        exploration_kwargs,
        active_learning_runner,
    ):
        assert policy.active_learning
        assert exploration_kwargs["mlip_backend"] == "uma"
        assert exploration_kwargs["uma_model_name"] == "uma-s-1p1"
        al_result = active_learning_runner(formula, output_dir, False)
        return _empty_result(formula, al_result)

    result = run_formula_with_active_learning(
        "NbO2",
        str(tmp_path / "formula"),
        config=CampaignFormulaExecutionConfig(
            active_learning_config=config_path,
            phase_policy=PhaseExplorationPolicy(
                active_learning=True,
                dft_approved=True,
            ),
            compute_nodes=2,
        ),
        al_runner=al_runner,
        phase_runner=phase_runner,
    )

    assert observed["seed_kind"] == "compositions"
    assert observed["compositions"] == ["NbO2"]
    assert observed["out_dir"] == tmp_path / "formula" / "active_learning"
    assert result.active_learning_result["n_dft_calculations"] == 3
    assert result.active_learning_result["n_active_learning_iterations"] == 1
    assert result.active_learning_result["node_hours"] == pytest.approx(1.0)
    assert result.active_learning_result["iteration_states"][0]["n_dft_converged"] == 2


def test_formula_execution_refines_with_dft_and_builds_hull_references(tmp_path):
    pytest.importorskip("pymatgen")
    config_path = tmp_path / "al.yaml"
    config_path.write_text(
        _config_yaml(tmp_path, "{Nb: Nb.upf, O: O.upf}"),
        encoding="utf-8",
    )
    structures = {}
    for formula, symbols in {
        "Nb": ["Nb"],
        "O2": ["O", "O"],
        "NbO2": ["Nb", "O", "O"],
    }.items():
        path = tmp_path / f"{formula}.extxyz"
        write(path, Atoms(symbols, positions=[[index, 0, 0] for index in range(len(symbols))]))
        structures[formula] = path

    mlip_relaxation = RelaxationResult(
        structure_path=str(structures["NbO2"]),
        optimized_structure_path=str(structures["NbO2"]),
        trajectory_path="",
        log_csv_path="",
        final_energy_eV=-5.0,
        final_max_force_eV_per_A=0.01,
        num_steps=2,
        converged=True,
    )

    def phase_runner(formula, **kwargs):
        composition = parse_composition(formula)
        assert composition is not None
        return PhaseExplorationWorkflowResult(
            composition=formula,
            initial=CompositionExplorationResult(
                composition=composition,
                phase_candidates=[
                    PhaseCandidate(formula="NbO2", structure_path=str(structures["NbO2"]))
                ],
                relaxations=[mlip_relaxation],
            ),
            active_learning_result={"n_dft_calculations": 1, "node_hours": 0.0},
        )

    energies = {"Nb": -10.0, "O2": -8.0, "NbO2": -20.0}
    observed_settings = []

    def relaxation_runner(cfg):
        assert cfg.mode == RelaxationMode.DFT
        assert cfg.dft is not None
        assert cfg.dft.launcher == [
            "bash",
            "/unused/wrapper.sh",
            "{work_dir}",
            "/unused/pw.x",
            "{input}",
            "1",
            "8",
            "7",
        ]
        observed_settings.append(cfg.dft.settings)
        source = next(
            formula for formula, path in structures.items() if str(path) == cfg.structure_path
        )
        if source == "O2":
            assert cfg.geometry.relax_cell is False
            assert cfg.dft.settings["kpts"] == (1, 1, 1)
            assert cfg.dft.settings["extra_system"]["nspin"] == 2
        return ScientificRelaxationResult(
            run_id=f"relax-{source}",
            run_directory=str(tmp_path / f"run-{source}"),
            mode=RelaxationMode.DFT,
            status=WorkflowStatus.COMPLETE,
            final_structure_path=cfg.structure_path,
            stages=[
                RelaxationStageResult(
                    stage="dft_relaxation",
                    backend="qe",
                    evidence_level=EvidenceLevel.CONVERGED_DFT,
                    input_structure_path=cfg.structure_path,
                    optimized_structure_path=cfg.structure_path,
                    energy_eV=energies[source],
                    max_force_eV_per_A=0.005,
                    steps=3,
                    converged=True,
                )
            ],
        )

    refinement = CampaignDFTRefinementConfig(
        method_signature="qe-test-v1",
        reference_structures={"Nb": structures["Nb"], "O2": structures["O2"]},
        reference_relax_cell={"O2": False},
        reference_settings={
            "O2": {
                "kpts": (1, 1, 1),
                "extra_system": {"nspin": 2, "starting_magnetization(1)": 1.0},
            }
        },
    )
    with pytest.raises(PermissionError, match="requires explicit DFT approval"):
        run_formula_with_active_learning(
            "NbO2",
            str(tmp_path / "formula-unapproved"),
            config=CampaignFormulaExecutionConfig(
                active_learning_config=config_path,
                phase_policy=PhaseExplorationPolicy(active_learning=False),
                dft_refinement=refinement,
            ),
            phase_runner=phase_runner,
            relaxation_runner=relaxation_runner,
        )
    assert observed_settings == []

    result = run_formula_with_active_learning(
        "NbO2",
        str(tmp_path / "formula"),
        config=CampaignFormulaExecutionConfig(
            active_learning_config=config_path,
            phase_policy=PhaseExplorationPolicy(active_learning=False, dft_approved=True),
            dft_refinement=refinement,
        ),
        phase_runner=phase_runner,
        relaxation_runner=relaxation_runner,
    )

    report = result.initial.stability
    assert report is not None
    assert report.ranking_mode == "convex_hull_ranking"
    assert report.reference_set_id == refinement.reference_energies.identifier
    assert report.ground_state.formation_energy_eV_per_atom == pytest.approx(-2 / 3)
    assert result.active_learning_result["n_dft_calculations"] == 4
    assert result.active_learning_result["dft_refinement"]["reference_calculations"] == 2
    assert sum(settings["kpts"] == (4, 4, 4) for settings in observed_settings) == 2

    resumed = run_formula_with_active_learning(
        "NbO2",
        str(tmp_path / "formula-resumed"),
        config=CampaignFormulaExecutionConfig(
            active_learning_config=config_path,
            phase_policy=PhaseExplorationPolicy(active_learning=False, dft_approved=True),
            dft_refinement=refinement,
        ),
        phase_runner=phase_runner,
        relaxation_runner=relaxation_runner,
    )
    assert resumed.active_learning_result["dft_refinement"]["reference_calculations"] == 0
    assert len(observed_settings) == 4


def test_formula_execution_rejects_missing_pseudopotential_mapping(tmp_path):
    config_path = tmp_path / "al.yaml"
    config_path.write_text(
        _config_yaml(tmp_path, "{Nb: Nb.upf}"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lacks elements.*O"):
        run_formula_with_active_learning(
            "NbO2",
            str(tmp_path / "formula"),
            config=CampaignFormulaExecutionConfig(
                active_learning_config=config_path,
                phase_policy=PhaseExplorationPolicy(
                    active_learning=True,
                    dft_approved=True,
                ),
            ),
            al_runner=lambda _: None,
        )


def test_formula_execution_supports_vasp_refinement(tmp_path):
    pytest.importorskip("pymatgen")
    config_path = tmp_path / "al-vasp.yaml"
    config_path.write_text(_vasp_config_yaml(tmp_path), encoding="utf-8")
    structure = tmp_path / "NbO2.extxyz"
    oxygen = tmp_path / "O2.extxyz"
    write(
        structure,
        Atoms(["Nb", "O", "O"], positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
    )
    write(oxygen, Atoms(["O", "O"], positions=[[0, 0, 0], [1.2, 0, 0]]))
    mlip_relaxation = RelaxationResult(
        structure_path=str(structure),
        optimized_structure_path=str(structure),
        trajectory_path="",
        log_csv_path="",
        final_energy_eV=-5.0,
        final_max_force_eV_per_A=0.01,
        num_steps=2,
        converged=True,
    )

    def phase_runner(formula, **kwargs):
        composition = parse_composition(formula)
        assert composition is not None
        return PhaseExplorationWorkflowResult(
            composition=formula,
            initial=CompositionExplorationResult(
                composition=composition,
                phase_candidates=[PhaseCandidate(formula=formula, structure_path=str(structure))],
                relaxations=[mlip_relaxation],
            ),
            active_learning_result={"n_dft_calculations": 1},
        )

    observed = []

    def relaxation_runner(cfg):
        assert cfg.dft is not None
        observed.append(cfg.dft)
        is_reference = cfg.structure_path == str(oxygen)
        optimized = oxygen if is_reference else structure
        return ScientificRelaxationResult(
            run_id="vasp-reference" if is_reference else "vasp-relax",
            run_directory=str(tmp_path / "vasp-run"),
            mode=RelaxationMode.DFT,
            status=WorkflowStatus.COMPLETE,
            final_structure_path=str(optimized),
            stages=[
                RelaxationStageResult(
                    stage="dft_relaxation",
                    backend="vasp",
                    evidence_level=EvidenceLevel.CONVERGED_DFT,
                    input_structure_path=str(optimized),
                    optimized_structure_path=str(optimized),
                    energy_eV=-8.0 if is_reference else -20.0,
                    max_force_eV_per_A=0.005,
                    steps=4,
                    converged=True,
                )
            ],
        )

    with pytest.raises(ValueError, match="reference energies use qe.*refinement uses vasp"):
        run_formula_with_active_learning(
            "NbO2",
            str(tmp_path / "formula-vasp-mismatched"),
            config=CampaignFormulaExecutionConfig(
                active_learning_config=config_path,
                phase_policy=PhaseExplorationPolicy(active_learning=False, dft_approved=True),
                dft_refinement=CampaignDFTRefinementConfig(
                    method_signature="incompatible-test",
                    reference_energies=ReferenceEnergySet(
                        identifier="qe-test",
                        method_signature="incompatible-test",
                        backend="qe",
                        elemental_energies_eV_per_atom={"Nb": -10.0, "O": -4.0},
                    ),
                ),
            ),
            phase_runner=phase_runner,
            relaxation_runner=relaxation_runner,
        )
    with pytest.raises(ValueError, match="lack backend provenance"):
        run_formula_with_active_learning(
            "NbO2",
            str(tmp_path / "formula-vasp-legacy-references"),
            config=CampaignFormulaExecutionConfig(
                active_learning_config=config_path,
                phase_policy=PhaseExplorationPolicy(active_learning=False, dft_approved=True),
                dft_refinement=CampaignDFTRefinementConfig(
                    method_signature="legacy-test",
                    reference_energies=ReferenceEnergySet(
                        identifier="legacy-test",
                        method_signature="legacy-test",
                        elemental_energies_eV_per_atom={"Nb": -10.0, "O": -4.0},
                    ),
                ),
            ),
            phase_runner=phase_runner,
            relaxation_runner=relaxation_runner,
        )

    result = run_formula_with_active_learning(
        "NbO2",
        str(tmp_path / "formula-vasp"),
        config=CampaignFormulaExecutionConfig(
            active_learning_config=config_path,
            phase_policy=PhaseExplorationPolicy(active_learning=False, dft_approved=True),
            dft_refinement=CampaignDFTRefinementConfig(
                method_signature="vasp-pbe-test-v1",
                reference_energies=ReferenceEnergySet(
                    identifier="vasp-test",
                    method_signature="vasp-pbe-test-v1",
                    backend="vasp",
                    elemental_energies_eV_per_atom={"Nb": -10.0},
                ),
                reference_structures={"O2": oxygen},
                reference_relax_cell={"O2": False},
                reference_settings={"O2": {"ispin": 2, "extra_incar": {"MAGMOM": "2*1.0"}}},
            ),
        ),
        phase_runner=phase_runner,
        relaxation_runner=relaxation_runner,
    )

    assert len(observed) == 2
    reference_dft, dft = observed
    assert reference_dft.settings["ispin"] == 2
    assert reference_dft.settings["extra_incar"] == {"LMAXMIX": "4", "MAGMOM": "2*1.0"}
    assert dft.backend == "vasp"
    assert dft.launcher[0:3] == ["bash", str(tmp_path / "vasp-wrapper.sh"), "."]
    assert dft.launcher[3:] == [str(tmp_path / "vasp_std"), "2", "4", "8"]
    assert dft.settings["encut_ev"] == 600
    assert dft.settings["ismear"] == 0
    assert dft.settings["extra_incar"]["LMAXMIX"] == "4"
    assert "IBRION" not in dft.settings["extra_incar"]
    assert result.initial.stability.ranking_mode == "convex_hull_ranking"
    assert result.active_learning_result["dft_refinement"]["candidate_calculations"] == 1


def test_formula_execution_applies_retraining_and_promotion(tmp_path, monkeypatch):
    import matsim_agents.campaign.execution as execution

    config_path = tmp_path / "al.yaml"
    config_path.write_text(
        _config_yaml(tmp_path, "{Nb: Nb.upf, O: O.upf}"),
        encoding="utf-8",
    )
    train_script = tmp_path / "train.py"
    train_script.touch()
    validation_set = tmp_path / "held-out.extxyz"
    validation_set.touch()
    checkpoint = tmp_path / "promoted" / "inference_ckpt.pt"
    observed = {}

    def al_runner(cfg):
        observed["trainer"] = cfg.trainer.model_dump()
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        state_dir = cfg.loop.out_dir / "iteration_0000"
        state_dir.mkdir(parents=True)
        (state_dir / "state.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "n_dft_converged": 2,
                    "n_dft_failed": 0,
                    "model_promoted": True,
                    "new_logdir": str(checkpoint),
                    "candidate_uncertainty": {"md-001": 0.75},
                    "timings_sec": {"total": 60.0},
                }
            ),
            encoding="utf-8",
        )

    def phase_runner(
        formula,
        *,
        policy,
        output_dir,
        exploration_kwargs,
        active_learning_runner,
    ):
        al_result = active_learning_runner(formula, output_dir, True)
        return _empty_result(formula, al_result)

    def capture_cross_model(_exploration, configs, **_kwargs):
        observed["post_promotion_model"] = configs[0].mlip.uma.model_name
        return {"models": {}, "ranking_disagreement": False, "rankings": {}}

    monkeypatch.setattr(execution, "_cross_model_scores", capture_cross_model)

    result = run_formula_with_active_learning(
        "NbO2",
        str(tmp_path / "formula"),
        config=CampaignFormulaExecutionConfig(
            active_learning_config=config_path,
            phase_policy=PhaseExplorationPolicy(
                active_learning=True,
                retrain_mlip=True,
                reevaluate_after_retraining=True,
                dft_approved=True,
                retraining_approved=True,
            ),
            retraining=CampaignRetrainingConfig(
                train_script=train_script,
                epochs=3,
                promote_model=True,
                promotion_approved=True,
                validation_set=validation_set,
            ),
        ),
        al_runner=al_runner,
        phase_runner=phase_runner,
    )

    assert observed["trainer"]["enabled"] is True
    assert observed["trainer"]["epochs_per_iter"] == 3
    assert observed["trainer"]["promote_model"] is True
    assert observed["trainer"]["validation_set"] == validation_set
    assert result.model_promoted is True
    assert result.active_learning_result["exploration_kwargs"] == {
        "uma_model_name": str(checkpoint)
    }
    assert result.active_learning_result["md_candidate_uncertainty"] == {"md-001": 0.75}
    assert result.active_learning_result["candidate_uncertainty"] == {}
    assert observed["post_promotion_model"] == str(checkpoint)


def test_retraining_rejects_unapproved_promotion(tmp_path):
    train_script = tmp_path / "train.py"
    train_script.touch()
    with pytest.raises(ValueError, match="promotion requires explicit approval"):
        CampaignRetrainingConfig(train_script=train_script, promote_model=True)


def test_latest_promoted_model_recovers_checkpoint_from_state(tmp_path):
    campaign = CampaignState(
        campaign_id="resume",
        element_set=["Nb", "O"],
        formula_policy=FormulaGenerationPolicy(
            elements=["Nb", "O"],
            require_charge_balance=False,
        ),
        formula_runs={
            "NbO2": FormulaRunRecord(
                formula="NbO2",
                status=WorkflowStatus.COMPLETE,
                iteration=2,
                model_promoted=True,
                evidence={
                    "iteration_states": [
                        {
                            "model_promoted": True,
                            "new_logdir": str(tmp_path / "inference_ckpt.pt"),
                        }
                    ]
                },
            )
        },
    )

    assert latest_promoted_model(campaign) == str(tmp_path / "inference_ckpt.pt")
