"""The fusion plasma-facing-component example application, loaded for tests."""

import importlib.util
from pathlib import Path

from matsim_agents.science.decision.policy import load_policy

APP = Path(__file__).resolve().parents[1] / "examples" / "fusion_pfc"
PFC_POLICY = APP / "policy.yaml"
POLICY = load_policy(PFC_POLICY)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"fusion_pfc_{name}", APP / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


synthetic = _load("synthetic")
