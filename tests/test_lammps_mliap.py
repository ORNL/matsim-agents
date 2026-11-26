from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

from matsim_agents.backends.external.lammps.mliap.bridge import (
    CalculatorMLIAP,
    UncertaintyGate,
    run_with_gate,
)
from matsim_agents.backends.external.lammps.mliap.parity import (
    lammps_data_text,
    run_lammps_once,
    write_export_record,
)


class FakeLammps:
    def __init__(self, atoms):
        self.atoms = atoms
        self.numpy = SimpleNamespace(extract_atom=lambda name: atoms.positions)
        self.commands = []

    def extract_box(self):
        c = self.atoms.cell.array
        return [0, 0, 0], [c[0][0], c[1][1], c[2][2]], 0.0, 0.0, 0.0, [1, 1, 1], 0

    def command(self, cmd):
        self.commands.append(cmd)


def _data(atoms):
    n = len(atoms)
    return SimpleNamespace(nlocal=n, elems=np.zeros(n, dtype=int), f=np.zeros((n, 3)), energy=None)


def test_compute_forces_adds_calculator_forces_and_energy():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=1)
    obj = CalculatorMLIAP(EMT(), ["Cu"], lmp=FakeLammps(atoms))
    data = _data(atoms)
    obj.compute_forces(data)
    ref = atoms.copy()
    ref.calc = EMT()
    np.testing.assert_allclose(data.f, ref.get_forces(), atol=1e-12)
    assert data.energy == ref.get_potential_energy()


def test_gate_fires_on_disagreement_and_stops_run():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    atoms.rattle(0.1, seed=3)
    gate = UncertaintyGate([EMT(), LennardJones(sigma=2.3, epsilon=0.4)], 1e-3, every=1)
    lmp = FakeLammps(atoms)
    obj = CalculatorMLIAP(EMT(), ["Cu"], lmp=lmp, gate=gate)
    obj.compute_forces(_data(atoms))
    assert gate.triggered and gate.events[0].score_eV_per_A > 1e-3
    assert run_with_gate(lmp, steps=100, chunk=10, obj=obj) == 10


class _NaNForces(EMT):
    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        self.results["forces"] = np.full((len(atoms), 3), np.nan)


def test_gate_rejects_a_non_finite_score():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    gate = UncertaintyGate([EMT(), _NaNForces()], 1e-3, every=1)
    with pytest.raises(FloatingPointError, match="non-finite"):
        gate.check(0, atoms)


def test_run_with_gate_rejects_a_nonpositive_chunk():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    lmp = FakeLammps(atoms)
    obj = CalculatorMLIAP(EMT(), ["Cu"], lmp=lmp)
    with pytest.raises(ValueError, match="chunk"):
        run_with_gate(lmp, steps=10, chunk=0, obj=obj)
    assert lmp.commands == []


def test_parity_evaluation_rejects_a_non_periodic_structure():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.pbc = [True, True, False]
    with pytest.raises(ValueError, match="periodic"):
        run_lammps_once(atoms, EMT(), ["Cu"])


def test_gate_stays_quiet_when_models_agree():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    gate = UncertaintyGate([EMT(), EMT()], 1e-3, every=1)
    obj = CalculatorMLIAP(EMT(), ["Cu"], lmp=FakeLammps(atoms), gate=gate)
    obj.compute_forces(_data(atoms))
    assert not gate.triggered


def test_export_record_hashes_checkpoint_bytes(tmp_path):
    ckpt = tmp_path / "m.model"
    ckpt.write_bytes(b"weights")
    rec = write_export_record(
        tmp_path / "m.json", checkpoint=ckpt, species=["Cu"], short_range=None, parity=None
    )
    assert rec["checkpoint_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert json.loads((tmp_path / "m.json").read_text()) == rec


def test_data_text_header_and_atom_count():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    text = lammps_data_text(atoms, ["Cu"])
    assert "4 atoms" in text and "xlo xhi" in text and text.count("\n1 ") >= 1
