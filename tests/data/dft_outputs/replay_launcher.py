"""Stands in for pw.x or vasp_std by replaying a recorded run: replay_launcher.py RUN_DIR [...]."""

import shutil
import sys
from pathlib import Path

recorded = Path(sys.argv[1])
if (recorded / "pw.out").is_file():
    sys.stdout.write((recorded / "pw.out").read_text())
else:
    for name in ("OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR"):
        shutil.copyfile(recorded / name, name)
    sys.stdout.write((recorded / "vasp.out").read_text())
