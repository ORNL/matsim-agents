"""run_relaxation's DFT stage end to end, with the DFT code replaced by a recorded run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from ase.build import bulk
from ase.io import read, write

from matsim_agents.workflows.relaxation import (
    DFTBackendConfig,
    RelaxationMode,
    ScientificRelaxationConfig,
    run_relaxation,
)

DATA = Path(__file__).parent / "data" / "dft_outputs"
LAUNCHER = str(DATA / "replay_launcher.py")


def _relax(tmp_path, backend, recorded, *, relax_cell, **dft):
    structure = tmp_path / "Si.extxyz"
    write(structure, bulk("Si", "diamond", a=5.43))
    cfg = ScientificRelaxationConfig(
        structure_path=str(structure),
        mode=RelaxationMode.DFT,
        output_root=str(tmp_path / "runs"),
        dft=DFTBackendConfig(
            backend=backend, launcher=[sys.executable, LAUNCHER, str(DATA / recorded)], **dft
        ),
        dft_approved=True,
        geometry={"relax_cell": relax_cell},
    )
    return run_relaxation(cfg)


@pytest.fixture
def pseudo_dir(tmp_path):
    path = tmp_path / "pseudo"
    path.mkdir()
    (path / "Si.upf").write_text("placeholder\n")
    return path


@pytest.fixture
def potcar_dir(tmp_path):
    path = tmp_path / "potcars"
    (path / "Si").mkdir(parents=True)
    (path / "Si" / "POTCAR").write_text("ENMAX  =  245.345; ENMIN  =  184.009 eV\n")
    return path


def test_converged_qe_relaxation(tmp_path, pseudo_dir):
    result = _relax(
        tmp_path, "qe", "qe/relax_converged", relax_cell=False, pseudo_dir=str(pseudo_dir)
    )
    assert result.status == "complete", result.failure_reason
    (stage,) = result.stages
    assert stage.backend == "qe" and stage.converged
    assert stage.evidence_level == "converged_dft"
    assert stage.steps == 4
    assert stage.energy_eV == pytest.approx(-16.739814 * 13.605693122994)
    assert stage.max_force_eV_per_A is not None
    assert result.final_structure_path == stage.optimized_structure_path
    assert len(read(stage.optimized_structure_path)) == 2
    assert "&CONTROL" in Path(stage.artifacts["input"]).read_text().upper()


def test_dft_relaxation_requires_approval_before_starting_stage(tmp_path, pseudo_dir, monkeypatch):
    structure = tmp_path / "Si.extxyz"
    write(structure, bulk("Si", "diamond", a=5.43))
    cfg = ScientificRelaxationConfig(
        structure_path=str(structure),
        mode=RelaxationMode.DFT,
        output_root=str(tmp_path / "runs"),
        dft=DFTBackendConfig(
            backend="qe",
            launcher=[sys.executable, "unused"],
            pseudo_dir=str(pseudo_dir),
        ),
        dft_approved=False,
    )

    def unexpected_dft_stage(*args, **kwargs):
        pytest.fail("DFT stage must not start without explicit approval")

    monkeypatch.setattr("matsim_agents.workflows.relaxation._dft_stage", unexpected_dft_stage)

    result = run_relaxation(cfg)

    assert result.status == "failed"
    assert result.failure_type == "PermissionError"
    assert "dft_approved=true" in result.failure_reason
    assert result.stages == []


def test_step_limited_qe_relaxation_is_partial(tmp_path, pseudo_dir):
    result = _relax(
        tmp_path, "qe", "qe/relax_step_limited", relax_cell=False, pseudo_dir=str(pseudo_dir)
    )
    assert result.status == "partial"
    (stage,) = result.stages
    assert not stage.converged and stage.steps == 1
    assert stage.evidence_level == "low_fidelity_dft"
    assert "QE did not converge" in stage.failure_reason
    assert not result.validations[0].passed


def test_converged_vasp_cell_relaxation(tmp_path, potcar_dir):
    result = _relax(
        tmp_path, "vasp", "vasp/vc_relax_5steps", relax_cell=True, potcar_dir=str(potcar_dir)
    )
    assert result.status == "complete", result.failure_reason
    (stage,) = result.stages
    assert stage.backend == "vasp" and stage.converged
    assert stage.steps == 5
    assert stage.energy_eV == pytest.approx(-30.11224)
    assert stage.max_force_eV_per_A == pytest.approx(0.00895922)
    work_dir = Path(stage.artifacts["work_dir"])
    assert "ISIF" in (work_dir / "INCAR").read_text()
    assert (work_dir / "POTCAR").read_text().startswith("ENMAX")
    assert len(read(stage.optimized_structure_path)) == 4


@pytest.mark.parametrize(("nelm", "scf_converged"), [(13, True), (4, False)])
def test_vasp_scf_limit_applies_to_the_final_ionic_step(tmp_path, potcar_dir, nelm, scf_converged):
    # The recorded run took [13, 7, 6, 4, 4] SCF iterations per ionic step.
    result = _relax(
        tmp_path,
        "vasp",
        "vasp/vc_relax_5steps",
        relax_cell=True,
        potcar_dir=str(potcar_dir),
        settings={"nelm": nelm},
    )
    assert result.stages[0].scf_converged is scf_converged


@pytest.mark.parametrize("backend", ["qe", "vasp"])
def test_a_launcher_that_never_started_the_code_keeps_its_diagnostic(
    tmp_path, pseudo_dir, potcar_dir, backend
):
    from matsim_agents.execution.recovery import FailureClass, classify_failure

    structure = tmp_path / "Si.extxyz"
    write(structure, bulk("Si", "diamond", a=5.43))
    message = "pw.x not found or not executable: /opt/qe/bin/pw.x"
    cfg = ScientificRelaxationConfig(
        structure_path=str(structure),
        mode=RelaxationMode.DFT,
        output_root=str(tmp_path / "runs"),
        dft=DFTBackendConfig(
            backend=backend,
            launcher=["sh", "-c", f"echo 'ERROR: {message}' >&2; exit 2", "wrapper"],
            pseudo_dir=str(pseudo_dir),
            potcar_dir=str(potcar_dir),
        ),
        dft_approved=True,
        geometry={"relax_cell": False},
    )
    (stage,) = run_relaxation(cfg).stages
    assert stage.failure_reason.endswith(message)
    outcome = {"status": "failed", "return_code": 2, "failure_reason": stage.failure_reason}
    assert classify_failure(outcome) == FailureClass.LAUNCHER


def test_a_code_failure_reason_carries_no_output_tail(tmp_path, pseudo_dir):
    result = _relax(
        tmp_path, "qe", "qe/relax_step_limited", relax_cell=False, pseudo_dir=str(pseudo_dir)
    )
    assert result.stages[0].failure_reason == "QE did not converge (return code 0)"


@pytest.mark.parametrize(
    ("backend", "output", "diagnostic"),
    [
        (
            "qe",
            " %%%%%%%%\n     Error in routine read_namelists (1):\n"
            '     bad line in namelist &electrons: "  mixing_bet = 0.4"\n %%%%%%%%\n\n'
            "     stopping ...\n",
            "Error in routine read_namelists (1): bad line in namelist",
        ),
        (
            "vasp",
            " ---------------------------------------------\n"
            "|     EEEEEEE  RRRRRR   RRRRRR   OOOOOOO  RRRRRR  |\n"
            "|     Error reading item 'NELM' from file INCAR.  |\n"
            " ---------------------------------------------\n"
            "srun: error: nid001: task 0: Exited with exit code 1\n",
            "Error reading item 'NELM' from file INCAR.",
        ),
    ],
)
def test_an_input_error_of_the_code_is_not_retried(
    tmp_path, pseudo_dir, potcar_dir, backend, output, diagnostic
):
    from matsim_agents.execution.recovery import FailureClass, classify_failure

    structure = tmp_path / "Si.extxyz"
    write(structure, bulk("Si", "diamond", a=5.43))
    recorded = tmp_path / "output.txt"
    recorded.write_text(output)
    cfg = ScientificRelaxationConfig(
        structure_path=str(structure),
        mode=RelaxationMode.DFT,
        output_root=str(tmp_path / "runs"),
        dft=DFTBackendConfig(
            backend=backend,
            launcher=["sh", "-c", f"cat {recorded}; exit 1", "wrapper"],
            pseudo_dir=str(pseudo_dir),
            potcar_dir=str(potcar_dir),
        ),
        dft_approved=True,
        geometry={"relax_cell": False},
    )
    (stage,) = run_relaxation(cfg).stages
    assert diagnostic in stage.failure_reason
    outcome = {"status": "failed", "return_code": 1, "failure_reason": stage.failure_reason}
    assert classify_failure(outcome) == FailureClass.APPROVAL_OR_INPUT


@pytest.mark.parametrize(
    ("markers", "converged"),
    [
        (["convergence NOT achieved", "convergence has been achieved"], True),
        (["convergence has been achieved", "convergence NOT achieved"], False),
        (["convergence has been achieved"], True),
        ([], True),
    ],
)
def test_qe_scf_convergence_follows_the_last_cycle(markers, converged):
    from matsim_agents.workflows.relaxation import _qe_scf_converged

    assert _qe_scf_converged("\n".join(f"     {m} in 7 iterations" for m in markers)) is converged


@pytest.mark.parametrize("placed", [True, False])
def test_a_calculation_reports_the_nodes_and_gpus_it_was_placed_on(tmp_path, pseudo_dir, placed):
    from matsim_agents.execution.placement import ProcessLaunch

    structure = tmp_path / "Si.extxyz"
    write(structure, bulk("Si", "diamond", a=5.43))
    recorded = str(DATA / "qe" / "relax_step_limited")
    launcher = [sys.executable, LAUNCHER, recorded]
    launch = None
    if placed:
        launcher += ["{nodes}", "{ranks_per_node}", "{threads_per_rank}"]
        launch = ProcessLaunch(
            hosts=("nid001", "nid002"),
            devices=tuple(range(8)),
            ranks_per_node=8,
            threads_per_rank=7,
            device_kind="rocm-gcd",
            devices_per_gpu=2,
        )
    cfg = ScientificRelaxationConfig(
        structure_path=str(structure),
        mode=RelaxationMode.DFT,
        output_root=str(tmp_path / "runs"),
        dft=DFTBackendConfig(
            backend="qe", launcher=launcher, pseudo_dir=str(pseudo_dir), launch=launch
        ),
        dft_approved=True,
        geometry={"relax_cell": False},
    )
    (stage,) = run_relaxation(cfg).stages
    allocation = (stage.nodes, stage.devices, stage.device_kind, stage.gpus)
    assert allocation == ((2, 16, "rocm-gcd", 8.0) if placed else (None,) * 4)
