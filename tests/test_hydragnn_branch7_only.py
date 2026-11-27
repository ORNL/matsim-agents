"""Guard: matsim-agents uses ONLY HydraGNN branch-7 (OMat24); the BranchWeightMLP never returns."""

from __future__ import annotations

import importlib
import re
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from matsim_agents.backends.mlip.config import (
    HYDRAGNN_INFERENCE_HEAD,
    HYDRAGNN_INFERENCE_HEAD_INDEX,
    HydraGNNConfig,
    resolve_hydragnn_inference_head,
)
from matsim_agents.backends.mlip.training import finetune_hydragnn_newhead
from matsim_agents.workflows.active_learning.finetune_eval import HydraGNNFineTune

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"mlp_checkpoint|mlp_precision|BRANCH_MLP_CHECKPOINT|mlp_branch_weights|branch_checkpoint"
    r"|load_fused_stack|branch_routing|branch-routing|unfreeze_backbone|HYDRAGNN_FORCE_BRANCH"
    r"|hydragnn-scratch|training\.finetune_hydragnn\b(?!_)|import finetune_hydragnn\b(?!_)"
)
# Files allowed to name the forbidden tokens because they enforce the ban.
AGENT_INSTRUCTIONS = ("AGENTS.md", ".github/copilot-instructions.md")
ENFORCERS = {
    ".gitignore",
    *AGENT_INSTRUCTIONS,
    "tests/test_hydragnn_branch7_only.py",
    "src/matsim_agents/backends/mlip/config.py",
    "src/matsim_agents/backends/mlip/calculator.py",
    "src/matsim_agents/backends/mlip/hydragnn_energy.py",
    "src/matsim_agents/workflows/active_learning/finetune_eval.py",
}


def test_branch_7_is_omat24():
    assert HYDRAGNN_INFERENCE_HEAD_INDEX == 7
    assert HYDRAGNN_INFERENCE_HEAD == "OMat24"
    for value in (None, 7, "7", "OMat24", "omat24", "branch-7"):
        assert resolve_hydragnn_inference_head(value) == 7


@pytest.mark.parametrize("head", [0, 6, 8, "MPTrj", "branch-0", "Alexandria", True])
def test_other_heads_are_rejected(head):
    with pytest.raises(ValueError):
        resolve_hydragnn_inference_head(head)
    with pytest.raises(ValidationError):
        HydraGNNConfig(logdir="/l", inference_head=head)


def test_config_defaults_to_branch_7():
    cfg = HydraGNNConfig(logdir="/l")
    assert cfg.inference_head == "OMat24"
    assert cfg.fine_tuner == "finetune_hydragnn_newhead"
    assert ":head=OMat24:" in cfg.model_identifier()


@pytest.mark.parametrize(
    "extra",
    [
        {"mlp_checkpoint": "/m.pt"},
        {"mlp_checkpoint": None},
        {"mlp_precision": "fp64"},
        {"fine_tuner": "finetune_hydragnn"},
    ],
)
def test_branch_mlp_config_is_rejected(extra):
    with pytest.raises(ValidationError):
        HydraGNNConfig(logdir="/l", **extra)


@pytest.mark.parametrize(
    "extra",
    [{"weight_threshold": 0.1}, {"unfreeze_backbone": True}, {"strategy": "scratch"}, {"head": 3}],
)
def test_finetune_recipe_is_branch_7_only(extra):
    with pytest.raises(ValidationError):
        HydraGNNFineTune(**extra)
    assert HydraGNNFineTune().head == "OMat24"


def test_newhead_trainer_keeps_pretrained_branch_7():
    assert finetune_hydragnn_newhead.STRATEGIES == ("unfrozen", "frozen")
    with pytest.raises(ValueError):
        finetune_hydragnn_newhead.finetune_hydragnn_newhead(
            "d.extxyz", "out", gfm_logdir="/l", head="MPTrj", seed=0
        )


@pytest.mark.parametrize("module", ["finetune_hydragnn", "branch_routing"])
def test_routed_trainers_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"matsim_agents.backends.mlip.training.{module}")


def test_repository_has_no_branch_mlp_usage():
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
    hits = []
    for name in files:
        if name in ENFORCERS or name.startswith("docs/") or name.endswith((".png", ".pdf")):
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        hits.extend(
            f"{name}:{n}" for n, line in enumerate(text.splitlines(), 1) if FORBIDDEN.search(line)
        )
    assert not hits, "HydraGNN must use only branch-7 (OMat24); forbidden usage:\n" + "\n".join(
        hits
    )


@pytest.mark.parametrize("name", AGENT_INSTRUCTIONS)
def test_agent_instructions_state_the_branch_7_rule(name):
    text = (ROOT / name).read_text()
    assert "branch-7 (OMat24) ONLY" in text
    assert "BranchWeightMLP" in text
