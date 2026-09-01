from __future__ import annotations

import pytest
from ase import Atoms

from matsim_agents.active_learning.trainer import LabelledFrame, append_frames_to_extxyz


def _frame(backend: str) -> LabelledFrame:
    return LabelledFrame(
        atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        energy_eV=-1.0,
        forces_eV_per_A=None,
        stress_eV_per_A3=None,
        source_work_dir="work",
        iteration=0,
        backend=backend,
    )


def test_rejects_mixed_backends_in_one_append(tmp_path) -> None:
    with pytest.raises(ValueError, match="mix DFT energy references"):
        append_frames_to_extxyz([_frame("vasp"), _frame("qe")], tmp_path / "data.extxyz")


def test_rejects_backend_change_across_appends(tmp_path) -> None:
    path = tmp_path / "data.extxyz"
    assert append_frames_to_extxyz([_frame("vasp")], path) == 1
    with pytest.raises(ValueError, match="existing dataset uses 'vasp'"):
        append_frames_to_extxyz([_frame("qe")], path)


def test_allows_same_backend_across_appends(tmp_path) -> None:
    path = tmp_path / "data.extxyz"
    assert append_frames_to_extxyz([_frame("qe")], path) == 1
    assert append_frames_to_extxyz([_frame("qe")], path) == 1
