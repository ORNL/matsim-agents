"""Regenerate the pw.x fixtures in this directory from small Si runs.

Usage (Perlmutter): python regenerate.py PW_X PSEUDO_DIR
for example /global/common/software/nersc9/espresso/7.5-libxc-7.0.0-cpu/bin/pw.x
and a directory holding Si_r.upf. Each case runs serially in a few seconds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ase.build import bulk

from matsim_agents.backends.dft.qe_relax import QESettings, write_pw_input

HERE = Path(__file__).resolve().parent


def resolve_executable(command: str) -> str:
    """Absolute path of ``command``: a path from the current directory, else a PATH lookup."""
    found = shutil.which(command) if os.sep not in command else command
    if not found or not os.access(found, os.X_OK) or Path(found).is_dir():
        raise SystemExit(f"pw.x executable not found: {command}")
    return str(Path(found).resolve())


def main(pw_x: str, pseudo_dir: str) -> None:
    # pw.x runs from each fixture directory, and its pw.out is truncated before launch.
    pw_x = resolve_executable(pw_x)
    atoms = bulk("Si", "diamond", a=5.50)
    atoms.positions[1] += (0.05, -0.03, 0.02)
    base = dict(
        pseudo_dir=pseudo_dir,
        ecutwfc_ry=20.0,
        ecutrho_ry=160.0,
        occupations="fixed",
        kpts=(2, 2, 2),
        pseudopotentials={"Si": "Si_r.upf"},
    )
    cases = {
        "relax_converged": QESettings(calculation="relax", **base),
        "relax_step_limited": QESettings(calculation="relax", nstep=1, **base),
        "vc_relax_converged": QESettings(calculation="vc-relax", **base),
        "scf_not_converged": QESettings(
            calculation="relax", extra_electrons={"electron_maxstep": 2}, **base
        ),
    }
    # pw.x may exit 3 at nstep and 2 when SCF is not converged (7.5 on Perlmutter exits 0).
    allowed = {"relax_step_limited": {0, 3}, "scf_not_converged": {0, 2}}
    # Cray MPICH aborts on a login node unless GPU support is off.
    env = {**os.environ, "MPICH_GPU_SUPPORT_ENABLED": "0", "OMP_NUM_THREADS": "4"}
    for name, settings in cases.items():
        work = HERE / name
        work.mkdir(exist_ok=True)
        write_pw_input(atoms, settings, str(work / "pw.in"))
        with (work / "pw.out").open("w") as out:
            code = subprocess.run([pw_x, "-in", "pw.in"], cwd=work, stdout=out, env=env).returncode
        shutil.rmtree(work / "tmp", ignore_errors=True)
        if code not in allowed.get(name, {0}):
            raise SystemExit(f"{name}: pw.x exited with status {code}; see {work / 'pw.out'}")
        pw_in = work / "pw.in"
        pw_in.write_text(pw_in.read_text().replace(pseudo_dir, "PSEUDO_DIR"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
