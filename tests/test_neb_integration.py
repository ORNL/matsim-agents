"""Phase B integration on small cells with a fake launcher (no QE, no DFT)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.backends.dft.qe_neb_stages import (
    derive_stage_config,
    load_history,
    prepare_stage,
)
from matsim_agents.cli import app
from matsim_agents.science.defects.bcc import (
    generate_dumbbell_interstitial,
    generate_frenkel_pair,
)
from matsim_agents.science.defects.point_defects import generate_vacancy
from matsim_agents.science.neb.endpoints import (
    generate_frenkel_recombination,
    generate_interstitial_migration,
    generate_vacancy_hop,
    vacancy_neighbors,
)
from matsim_agents.science.neb.serialization import read_pathway
from matsim_agents.science.structures.alloys import AlloySpec, generate_alloy
from matsim_agents.workflows.neb.workflow import ingest_neb_results, run_neb_workflow
from tests.test_neb_workflow import ENERGIES, _cfg, _fake

A = 3.16
PARENT = generate_alloy(AlloySpec(counts={"Mo": 4, "Ta": 4, "W": 8}, repeats=(2, 2, 2), seed=1))


def _hop():
    initial, config = generate_vacancy(PARENT, "p", 0)
    mover = vacancy_neighbors(initial, config)[0]
    return generate_vacancy_hop(initial, config, PARENT, mover)


def _interstitial():
    initial, config = generate_dumbbell_interstitial(PARENT, "p", 0, "W", "110", A)
    target = initial.positions[-1] + np.array([0.0, 0.0, 0.3])
    return generate_interstitial_migration(
        initial, config, {len(PARENT): tuple(target)}, min_distance=1.0
    )


def _frenkel():
    initial, config = generate_frenkel_pair(PARENT, "p", 0, 9, A)
    return generate_frenkel_recombination(initial, config, PARENT)


@pytest.mark.parametrize("make", [_hop, _interstitial])
def test_each_mechanism_dry_runs_and_round_trips_through_dispatch(tmp_path, make):
    pair = make()
    out = tmp_path / "w"
    dry = run_neb_workflow(pair, _cfg(tmp_path), out, min_distance=0.5, max_step=1.5)
    assert dry.plan.status == "planned" and dry.plan.dry_run
    pathway = read_pathway(out / "pathway")
    assert pathway.pathway_id == dry.plan.pathway_id

    live = run_neb_workflow(
        make(),
        _cfg(tmp_path),
        tmp_path / "live",
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(),
        min_distance=0.5,
        max_step=1.5,
    )
    assert live.result.state == "converged_unqualified"
    assert live.result.forward_barrier_eV == pytest.approx(max(ENERGIES) - ENERGIES[0])
    assert live.qualification_blockers


def test_ambiguous_frenkel_displacement_is_rejected_before_any_qe_input(tmp_path):
    out = tmp_path / "w"
    outcome = run_neb_workflow(_frenkel(), _cfg(tmp_path), out, min_distance=0.5, max_step=1.5)
    assert outcome.plan.status == "rejected"
    assert any("ambiguous periodic displacement" in r for r in outcome.plan.reasons)
    assert not list(out.glob("qe/neb.stage*.in"))


def test_stalled_run_restarts_as_a_continue_stage_and_is_ingested(tmp_path):
    out = tmp_path / "w"
    first = run_neb_workflow(
        _hop(),
        _cfg(tmp_path),
        out,
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(converged=False, limit=True),
    )
    assert first.result.state == "unconverged"
    assert ingest_neb_results(out).result.state == "unconverged"
    history = load_history(out / "qe")
    base = QENebConfig.model_validate(history.stages[0].config)
    config = derive_stage_config(base, "continue", nstep_path=400)
    images = read_pathway(out / "pathway").images
    stage = prepare_stage(config, images, kind="continue")
    assert stage.index == len(history.stages) + 1
    assert "restart" in Path(stage.input_path).read_text()
    assert len(load_history(out / "qe").stages) == len(history.stages) + 1


def test_cli_resume_prepares_a_continue_stage_after_a_run(tmp_path):
    out = tmp_path / "w"
    run_neb_workflow(
        _hop(),
        _cfg(tmp_path),
        out,
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(converged=False, limit=True),
    )
    result = CliRunner().invoke(app, ["neb", "resume", str(out), "--nstep-path", "400"])
    assert result.exit_code == 0, result.output
    assert "prepared stage 2" in result.output


def _driver():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neb" / "physical_neb_vacancy.py"
    spec = importlib.util.spec_from_file_location("physical_neb_vacancy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driver_continues_an_unfinished_stage_to_convergence_then_runs_ci(tmp_path):
    calls = []
    stalled, done = _fake(converged=False, limit=True), _fake()

    def launcher(argv, *, cwd, stdout, timeout_sec, env):
        calls.append(timeout_sec)
        return (stalled if len(calls) == 1 else done)(
            argv, cwd=cwd, stdout=stdout, timeout_sec=timeout_sec, env=env
        )

    out = tmp_path / "w"
    first = run_neb_workflow(
        _hop(),
        _cfg(tmp_path),
        out,
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=launcher,
    )
    assert first.result.state == "unconverged"
    driver = _driver()
    from matsim_agents.backends.dft import qe_neb_stages

    original = qe_neb_stages.run_stage
    driver.run_stage = lambda *a, **k: original(*a, **k, launcher=launcher)
    outcome, note = driver.advance_stages(
        out, nstep_path=40, nstep_path_ci=120, max_continues=2, timeout_sec=10.0
    )
    kinds = [s.kind for s in load_history(out / "qe").stages]
    assert kinds[:2] == ["initial", "continue"]
    assert "continue stage(s) run" in note
    assert outcome.result is not None


def test_driver_does_not_continue_when_max_continues_is_zero(tmp_path):
    out = tmp_path / "w"
    run_neb_workflow(
        _hop(),
        _cfg(tmp_path),
        out,
        dispatch=True,
        allow_unrelaxed_endpoints=True,
        launcher=_fake(
            converged=False, limit=True, errors=(0.0, *[0.5] * (len(ENERGIES) - 2), 0.0)
        ),
    )
    driver = _driver()
    outcome, note = driver.advance_stages(
        out, nstep_path=40, nstep_path_ci=120, max_continues=0, timeout_sec=10.0
    )
    assert [s.kind for s in load_history(out / "qe").stages] == ["initial"]
    assert outcome.result.state == "unconverged"
    assert "climbing-image stage not run" in note
