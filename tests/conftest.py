"""Shared pytest fixtures for matsim-agents tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from matsim_agents.state import RelaxationResult, TaskSpec

# ── fake LLM helpers ──────────────────────────────────────────────────────────

def make_fake_llm(*responses: str) -> FakeListChatModel:
    """Return a FakeListChatModel that yields each response in order."""
    return FakeListChatModel(responses=list(responses))


@pytest.fixture
def fake_llm():
    """A simple stub LLM that returns 'OK' for any invocation."""
    return make_fake_llm("OK")


# ── structure & file fixtures ─────────────────────────────────────────────────

_SI_VASP = """\
Si FCC
1.0
  2.715  2.715  0.000
  0.000  2.715  2.715
  2.715  0.000  2.715
Si
2
Direct
  0.000  0.000  0.000
  0.250  0.250  0.250
"""


@pytest.fixture
def si_vasp(tmp_path: Path) -> str:
    """Write a minimal Si VASP structure and return the path."""
    p = tmp_path / "Si.vasp"
    p.write_text(_SI_VASP)
    return str(p)


@pytest.fixture
def fake_relaxation_result(si_vasp: str, tmp_path: Path) -> RelaxationResult:
    """A synthetic RelaxationResult pointing at tmp files."""
    opt = str(tmp_path / "Si_opt.vasp")
    Path(opt).write_text(_SI_VASP)
    return RelaxationResult(
        structure_path=si_vasp,
        optimized_structure_path=opt,
        trajectory_path=str(tmp_path / "Si.traj"),
        log_csv_path=str(tmp_path / "Si.csv"),
        final_energy_eV=-5.432,
        final_max_force_eV_per_A=0.009,
        num_steps=42,
        converged=True,
    )


@pytest.fixture
def fake_task(si_vasp: str) -> TaskSpec:
    return TaskSpec(structure_path=si_vasp, optimizer="FIRE", maxiter=10)


# ── chat config fixture ───────────────────────────────────────────────────────

@pytest.fixture
def discovery_config(tmp_path: Path, si_vasp: str):
    """A DiscoveryChatConfig with all paths pointing at tmp_path."""
    from matsim_agents.chat import DiscoveryChatConfig

    logdir = tmp_path / "logdir"
    logdir.mkdir()
    (logdir / "config.json").write_text("{}")

    return DiscoveryChatConfig(
        logdir=str(logdir),
        mlp_checkpoint=str(tmp_path / "mlp.pt"),
        output_dir=str(tmp_path / "outputs"),
        auto_confirm=True,
    )
