"""Shared pytest fixtures for matsim-agents tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from matsim_agents.state import RelaxationResult, TaskSpec


@pytest.fixture
def elemental_manifest(tmp_path):
    """Synthetic DFT-labelled fixed geometries; not real DFT qualification."""

    def create(energies):
        references = {}
        for element, energy_per_atom in energies.items():
            atoms = Atoms(element + "2")
            path = tmp_path / f"elemental-{element}.extxyz"
            write(path, atoms)
            references[element] = {
                "structure_path": path.name,
                "energy_eV": energy_per_atom * len(atoms),
                "structure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest = tmp_path / "elemental-references.json"
        manifest.write_text(
            json.dumps(
                {
                    "backend": "qe",
                    "method_signature": "synthetic-dft",
                    "references": references,
                }
            )
        )
        return manifest

    return create


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
        hydragnn_branch_mlp_checkpoint=str(tmp_path / "mlp.pt"),
        output_dir=str(tmp_path / "outputs"),
        auto_confirm=True,
    )
