"""Smoke tests that do not require HydraGNN to be installed."""

from matsim_agents.state import MatSimState, RelaxationResult, TaskSpec


def test_state_defaults():
    s = MatSimState()
    assert s.objective == ""
    assert s.results == []
    assert s.iteration == 0


def test_task_spec_defaults():
    t = TaskSpec(structure_path="foo.vasp")
    assert t.kind == "relax"
    assert t.optimizer == "FIRE"


def test_result_roundtrip():
    r = RelaxationResult(
        structure_path="a.vasp",
        optimized_structure_path="a_opt.vasp",
        trajectory_path="a.traj",
        log_csv_path="a.csv",
        final_energy_eV=-1.23,
        final_max_force_eV_per_A=0.01,
        num_steps=42,
        converged=True,
    )
    assert RelaxationResult(**r.model_dump()) == r


def test_graph_compiles():
    from matsim_agents.graph import build_graph

    g = build_graph()
    assert g is not None
