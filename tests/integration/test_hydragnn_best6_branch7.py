"""The real BEST6 checkpoint, loaded through the production path, reproduces recorded energies.

Skipped unless the BEST6 logdir and the pinned HydraGNN checkout (benchmarks/e4_mlip/
frozen-source.json) exist. It runs in a subprocess with only the pinned checkout on the path, so a
stale ``hydragnn`` in the test environment cannot affect it. Energies were computed on CPU in fp64
with the branch-7 head only (rattled supercells, seed 0); a mismatch means the model, the head
selection or the graph construction changed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJ = Path(os.environ.get("PROJ", ROOT.parent))
PIN = json.loads((ROOT / "benchmarks/e4_mlip/frozen-source.json").read_text())["commit"]
FROZEN = Path(os.environ.get("HYDRAGNN_ROOT", PROJ / f"HydraGNN-e4b-frozen-{PIN[:8]}"))
BEST6 = Path(
    os.environ.get(
        "HYDRAGNN_BEST6_LOGDIR",
        PROJ / "HydraGNN/examples/multidataset_hpo_sc26/multidataset_hpo-BEST6-fp64",
    )
)

pytestmark = pytest.mark.skipif(
    not (FROZEN / "hydragnn").is_dir() or not any(BEST6.glob("*.pk")),
    reason="needs the pinned HydraGNN checkout and the BEST6 checkpoint",
)

# Structures are rattled in this order from one generator, so the order is part of the contract.
REFERENCE_EV = {
    "Fe": -27.6009543688,
    "W": -31.4927944456,
    "Cu": -26.0580275884,
    "NaCl": -14.9298135111,
}

SCRIPT = """
import json, sys
import numpy as np
from ase.build import bulk
from matsim_agents.backends.mlip.config import HydraGNNConfig

rng = np.random.default_rng(0)
structures = {
    "Fe": bulk("Fe", "bcc", a=2.87, cubic=True),
    "W": bulk("W", "bcc", a=3.17, cubic=True),
    "Cu": bulk("Cu", "fcc", a=3.61, cubic=True),
    "NaCl": bulk("NaCl", "rocksalt", a=5.64),
}
calculator = HydraGNNConfig(logdir=sys.argv[1], precision="fp64").build_calculator()
out = {"branches": getattr(calculator.model, "num_branches", None),
       "head": calculator.inference_head_name, "energies": {}}
for name, atoms in structures.items():
    atoms = atoms.repeat(2)
    atoms.rattle(0.05, rng=rng)
    atoms.calc = calculator
    out["energies"][name] = atoms.get_potential_energy()
    assert np.isfinite(atoms.get_forces()).all()
print("RESULT" + json.dumps(out))
"""


def test_best6_branch7_energies_match_the_recorded_reference():
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join(
            [str(FROZEN / "examples/multidataset_hpo_sc26"), str(FROZEN), str(ROOT / "src")]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(BEST6)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(x for x in result.stdout.splitlines() if x.startswith("RESULT"))
    out = json.loads(line[len("RESULT") :])
    assert out["branches"] == 1
    assert out["head"] == "OMat24"
    for name, expected in REFERENCE_EV.items():
        assert out["energies"][name] == pytest.approx(expected, abs=1e-5), name
