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
    MDStabilityResult,
    ParityResult,
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


def test_data_text_carries_velocities_in_angstrom_per_ps():
    from ase import units

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.set_velocities(np.full((4, 3), 0.01))
    text = lammps_data_text(atoms, ["Cu"])
    body = text.split("Velocities\n\n")[1].splitlines()
    assert len(body) == 4
    assert float(body[0].split()[1]) == pytest.approx(0.01 * 1000.0 * units.fs)
    assert "Velocities" not in lammps_data_text(bulk("Cu", "fcc", a=3.6, cubic=True), ["Cu"])


def test_md_stability_requires_velocities_and_valid_steps():
    from matsim_agents.backends.external.lammps.mliap.parity import (
        md_stability_check,
        run_lammps_md,
    )

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    with pytest.raises(ValueError, match="velocities"):
        md_stability_check(atoms, EMT(), ["Cu"])
    with pytest.raises(ValueError, match="steps"):
        run_lammps_md(atoms, EMT(), ["Cu"], steps=0, timestep_fs=1.0)
    atoms.pbc = [True, True, False]
    with pytest.raises(ValueError, match="periodic"):
        run_lammps_md(atoms, EMT(), ["Cu"], steps=1, timestep_fs=1.0)


class _FakeComm:
    """Two ranks: this one owns atoms 0..n-1 of the second half, the peer owns the first half."""

    def __init__(self, peer, rank):
        self.peer, self.rank = peer, rank

    def allgather(self, mine):
        return [self.peer, mine] if self.rank == 1 else [mine, self.peer]

    def Get_rank(self):  # noqa: N802 - mpi4py API
        return self.rank


@pytest.mark.parametrize("rank", [0, 1])
def test_multi_rank_forces_use_the_whole_system_and_only_rank0_adds_energy(rank):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
    atoms.rattle(0.05, seed=3)
    half = len(atoms) // 2
    mine = slice(half, None) if rank == 1 else slice(0, half)
    other = slice(0, half) if rank == 1 else slice(half, None)

    class _Lmp(FakeLammps):
        def __init__(self):
            super().__init__(atoms[mine])
            self.numpy = SimpleNamespace(
                extract_atom=lambda name: (
                    np.arange(len(atoms))[mine] + 1 if name == "id" else atoms.positions[mine]
                )
            )

    ids = np.arange(len(atoms))
    peer = (ids[other] + 1, atoms.numbers[other], atoms.positions[other])
    obj = CalculatorMLIAP(EMT(), ["Cu"], lmp=_Lmp())
    obj._mpi_comm = lambda: _FakeComm(peer, rank)
    sub = atoms[mine]
    sub.set_cell(atoms.cell)
    data = SimpleNamespace(
        nlocal=len(sub), elems=np.zeros(len(sub), dtype=int), f=np.zeros((len(sub), 3)), energy=None
    )
    obj.compute_forces(data)
    reference = atoms.copy()
    reference.calc = EMT()
    np.testing.assert_allclose(data.f, reference.get_forces()[mine], atol=1e-12)
    expected = reference.get_potential_energy() if rank == 0 else 0.0
    assert data.energy == pytest.approx(expected)


def _stub_checks(monkeypatch, *, parity_ok=True, md_ok=True):
    from matsim_agents.backends.external.lammps.mliap import export

    parity = ParityResult(
        passed=parity_ok,
        n_atoms=4,
        energy_ase_eV=0.0,
        energy_lammps_eV=0.0,
        energy_abs_diff_eV=0.0,
        max_force_abs_diff_eV_per_A=0.0,
        energy_tol_eV=1e-4,
        force_tol_eV_per_A=1e-4,
    )
    md = MDStabilityResult(
        passed=md_ok,
        n_atoms=4,
        steps=2,
        timestep_fs=1.0,
        finite=True,
        atoms_conserved=True,
        max_total_energy_deviation_eV_per_atom=0.0,
        energy_tol_eV_per_atom=1e-3,
        max_position_diff_vs_ase_A=0.0,
        position_tol_A=1e-5,
        max_displacement_A=0.0,
    )
    monkeypatch.setattr(export, "parity_check", lambda *a, **k: parity)
    monkeypatch.setattr(export, "md_stability_check", lambda *a, **k: md)
    return export


@pytest.mark.parametrize(
    ("parity_ok", "md_ok", "approved"),
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_export_records_checkpoint_hash_and_requires_both_checks(
    tmp_path, monkeypatch, parity_ok, md_ok, approved
):
    from matsim_agents.backends.mlip.config import MACEConfig

    export = _stub_checks(monkeypatch, parity_ok=parity_ok, md_ok=md_ok)
    ckpt = tmp_path / "mace_finetuned.model"
    ckpt.write_bytes(b"weights")
    record = export.export_checkpoint_for_lammps(
        MACEConfig(),
        ckpt,
        tmp_path / "out",
        species=["Cu"],
        calculator=EMT(),
    )
    assert record["approved"] is approved
    assert record["checkpoint_sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    saved = json.loads((tmp_path / "out" / "lammps_export.json").read_text())
    assert saved["approved"] is approved
    assert saved["md"] is not None


def test_export_requires_an_existing_checkpoint_file(tmp_path, monkeypatch):
    from matsim_agents.backends.mlip.config import MACEConfig

    export = _stub_checks(monkeypatch)
    with pytest.raises(FileNotFoundError, match="no checkpoint file"):
        export.export_checkpoint_for_lammps(
            MACEConfig(), tmp_path / "missing.model", tmp_path, species=["Cu"], calculator=EMT()
        )


def test_loop_export_is_opt_in_and_never_blocks_promotion(tmp_path, monkeypatch):
    from matsim_agents.backends.external.lammps.mliap import export
    from matsim_agents.workflows.active_learning import loop

    calls = []
    monkeypatch.setattr(
        export,
        "export_checkpoint_for_lammps",
        lambda *a, **k: calls.append(k) or {"approved": True},
    )
    cfg = SimpleNamespace(mlip=object(), trainer=SimpleNamespace(lammps_export_species=[]))
    state = SimpleNamespace(candidate_model_path=str(tmp_path / "m.model"), lammps_export=None)
    loop._export_for_lammps(cfg, state)
    assert calls == [] and state.lammps_export is None

    cfg.trainer.lammps_export_species = ["Cu"]
    loop._export_for_lammps(cfg, state)
    assert calls[0]["species"] == ["Cu"] and state.lammps_export == {"approved": True}

    def boom(*a, **k):
        raise RuntimeError("no lammps")

    monkeypatch.setattr(export, "export_checkpoint_for_lammps", boom)
    loop._export_for_lammps(cfg, state)
    assert state.lammps_export == {"approved": False, "error": "no lammps"}
