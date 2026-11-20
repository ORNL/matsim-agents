"""Replay SCF failures for characterization recovery integration tests.

The VASP iteration-limit line is synthetic; the remaining output is recorded.
"""

import re
import shutil
import sys
from pathlib import Path

backend, fixtures, mode, counter_name = sys.argv[1:5]
fixtures = Path(fixtures)
counter = Path(counter_name)
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
failing = mode == "always" or attempt == 1
if backend == "qe":
    recorded = "scf_not_converged" if failing else "relax_converged"
    sys.stdout.write((fixtures / "qe" / recorded / "pw.out").read_text())
elif backend == "vasp":
    recorded = fixtures / "vasp" / "vc_relax_5steps"
    for name in ("OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR"):
        shutil.copyfile(recorded / name, name)
    if failing:
        nelm = int(re.search(r"^\s*NELM\s*=\s*(\d+)", Path("INCAR").read_text(), re.M)[1])
        with Path("OUTCAR").open("a") as out:
            out.write(f"\n---------------- Iteration 5( {nelm})\n")
    sys.stdout.write((recorded / "vasp.out").read_text())
else:
    raise ValueError(f"Unknown backend: {backend}")
