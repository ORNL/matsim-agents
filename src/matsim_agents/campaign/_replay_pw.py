"""Stands in for pw.x by replaying a recorded known-answer run.

    _replay_pw.py REPLAY_DIR WORK_DIR INPUT

The step is the ``relax-<fidelity>-NNN`` component of WORK_DIR nearest to it. The recorded
``REPLAY_DIR/<step>/pw.in`` must equal INPUT, so a replayed output always
belongs to the input the pipeline wrote; the recorded ``pw.out`` goes to stdout.
Standalone (no package imports) so it runs from any interpreter.
"""

import re
import sys
from pathlib import Path

replay, work_dir, input_path = (Path(arg) for arg in sys.argv[1:4])
step = next(
    (part for part in reversed(work_dir.parts) if re.fullmatch(r"relax-\w+-\d{3}", part)), None
)
if step is None:
    sys.exit(f"no relax step in {work_dir}")
recorded = replay / step
if not (recorded / "pw.in").is_file():
    sys.exit(f"no recorded run for {step} in {replay}")
if (recorded / "pw.in").read_text() != input_path.read_text():
    sys.exit(f"{input_path} differs from the recorded {recorded / 'pw.in'}")
sys.stdout.write((recorded / "pw.out").read_text())
