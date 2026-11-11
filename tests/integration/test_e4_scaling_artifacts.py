"""Opt-in validation of frozen Aurora E4 scaling artifacts."""

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLANS = REPO / "research/paper/e4/aurora-dft-scaling/priority-scaling"

pytestmark = pytest.mark.skipif(
    os.environ.get("MATSIM_RUN_E4_SCALING_ARTIFACT_INTEGRATION") != "1",
    reason="set MATSIM_RUN_E4_SCALING_ARTIFACT_INTEGRATION=1 on Aurora",
)


@pytest.mark.parametrize("manifest_name", ["strong-manifest.json", "weak-manifest.json"])
def test_frozen_aurora_scaling_artifacts_validate(manifest_name):
    # Deferred: the integration CI job does not put the repo root on sys.path.
    from benchmarks.e4.aurora_dft import scaling

    scaling.validate_manifest(PLANS / manifest_name)
