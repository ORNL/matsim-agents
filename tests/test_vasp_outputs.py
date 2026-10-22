"""VASP parsers on real VASP outputs (POTCARs are licensed and left out).

relax_1step: Si diamond, ISIF=3; converged at the first ionic step.
vc_relax_5steps: SiC 2H, ISIF=3; "reached required accuracy" after 5 ionic steps.
static: Si diamond single point, NSW=0.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from matsim_agents.backends.dft.vasp_io import parse_vasp_workdir
from matsim_agents.backends.dft.vasp_relax import parse_relax_outputs

VASP = Path(__file__).parent / "data" / "dft_outputs" / "vasp"


@pytest.mark.parametrize(
    ("case", "per_step", "energies", "n_atoms", "wall"),
    [
        ("relax_1step", [11], [-10.84995636], 2, 15.572),
        (
            "vc_relax_5steps",
            [13, 7, 6, 4, 4],
            [-30.10627408, -30.10569869, -30.11218062, -30.11217981, -30.11224],
            4,
            69.494,
        ),
        ("static", [11], [-10.83611547], 2, 11.73),
    ],
)
def test_parse_relax_outputs_on_real_runs(case, per_step, energies, n_atoms, wall):
    result = parse_relax_outputs(VASP / case)
    assert result.converged and result.job_done
    assert result.n_ionic_steps == len(per_step)
    assert result.scf_iterations_per_step == per_step
    assert result.energies_eV == pytest.approx(energies)
    assert result.final_energy_eV == pytest.approx(energies[-1])
    assert len(result.final_atoms) == n_atoms
    assert result.wall_time_sec == pytest.approx(wall)


def test_final_force_and_stress_come_from_the_last_ionic_step():
    result = parse_relax_outputs(VASP / "vc_relax_5steps")
    assert result.final_max_force_eV_per_A == pytest.approx(0.00895922)
    assert result.final_stress_eV_per_A3.shape == (6,)


def test_relaxation_stopped_at_nsw_is_not_converged(tmp_path):
    work = shutil.copytree(VASP / "vc_relax_5steps", tmp_path / "run")
    outcar = work / "OUTCAR"
    outcar.write_text(
        "".join(
            line
            for line in outcar.read_text().splitlines(keepends=True)
            if "reached required accuracy" not in line
        )
    )
    result = parse_relax_outputs(work)
    assert result.job_done and not result.converged
    assert result.n_ionic_steps == 5


def test_nonzero_return_code_is_not_converged():
    assert not parse_relax_outputs(VASP / "relax_1step", return_code=1).converged


@pytest.mark.parametrize(
    ("case", "energy", "n_atoms", "wall"),
    [
        ("static", -10.83611547, 2, 11.73),
        ("vc_relax_5steps", -30.11224, 4, 69.494),
    ],
)
def test_parse_vasp_workdir_on_real_runs(case, energy, n_atoms, wall):
    result = parse_vasp_workdir(VASP / case)
    assert result.converged and result.notes is None
    assert result.energy_eV == pytest.approx(energy)
    assert result.forces_eV_per_A.shape == (n_atoms, 3)
    assert result.stress_eV_per_A3.shape == (6,)
    assert result.n_atoms == n_atoms
    assert result.wall_time_sec == pytest.approx(wall)


def test_killed_run_falls_back_to_oszicar_and_is_not_converged(tmp_path):
    work = shutil.copytree(VASP / "vc_relax_5steps", tmp_path / "run")
    xml = work / "vasprun.xml"
    xml.write_bytes(xml.read_bytes()[: xml.stat().st_size // 2])
    result = parse_vasp_workdir(work)
    assert not result.converged
    assert "vasprun.xml parse failed" in result.notes
    assert result.energy_eV == pytest.approx(-30.11224)
