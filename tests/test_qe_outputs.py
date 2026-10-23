"""parse_pw_stdout on real pw.x 7.5 outputs.

Regenerate them with tests/data/dft_outputs/qe/regenerate.py.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
from pathlib import Path

import pytest

from matsim_agents.backends.dft.qe_relax import parse_pw_stdout

QE = Path(__file__).parent / "data" / "dft_outputs" / "qe"


@pytest.mark.parametrize(
    ("case", "per_step", "total", "converged", "energy_ry"),
    [
        # pw.x: "bfgs converged in 4 scf cycles and 3 bfgs steps".
        ("relax_converged", [9, 5, 6, 3], 23, True, -16.739814),
        # nstep=1: one SCF cycle, then "The maximum number of steps has been reached."
        ("relax_step_limited", [9], 9, False, -16.73742774),
        # "bfgs converged in 7 scf cycles"; the 10-iteration final SCF is not a step.
        ("vc_relax_converged", [9, 5, 6, 8, 6, 5, 5], 54, True, -16.74365136),
        # electron_maxstep=2: "convergence NOT achieved after 2 iterations".
        ("scf_not_converged", [2], 2, False, math.nan),
    ],
)
def test_real_pw_outputs(case, per_step, total, converged, energy_ry):
    result = parse_pw_stdout(str(QE / case / "pw.out"))
    assert result.scf_iterations_per_step == per_step
    assert result.bfgs_steps == len(per_step)
    assert result.scf_iterations_total == total
    assert result.converged is converged
    assert result.job_done
    if math.isnan(energy_ry):
        assert math.isnan(result.final_energy_ry) and result.final_max_force_ry_au is None
    else:
        assert result.final_energy_ry == pytest.approx(energy_ry)
        assert result.final_max_force_ry_au is not None


def test_regenerate_resolves_pw_x_before_touching_fixtures(tmp_path, monkeypatch):
    pytest.importorskip("ase")
    spec = importlib.util.spec_from_file_location("regenerate", QE / "regenerate.py")
    regenerate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regenerate)

    before = {p: p.read_bytes() for p in QE.glob("*/pw.out")}
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="not found"):
        regenerate.main("./bin/pw.x", str(tmp_path))
    assert {p: p.read_bytes() for p in QE.glob("*/pw.out")} == before

    (tmp_path / "bin").mkdir()
    exe = tmp_path / "bin" / "pw.x"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert regenerate.resolve_executable("./bin/pw.x") == str(exe.resolve())
    assert regenerate.resolve_executable("sh") == str(Path(shutil.which("sh")).resolve())
