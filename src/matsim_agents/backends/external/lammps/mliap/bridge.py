"""ML-IAP unified object that evaluates any ASE MLIP calculator inside LAMMPS."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified as _Base
except ImportError:  # outside a LAMMPS Python install, e.g. unit tests

    class _Base:  # type: ignore[no-redef]
        def __init__(self, interface=None, element_types=None, ndescriptors=None,
                     nparams=None, rcutfac=None):  # fmt: skip
            self.interface = interface
            self.element_types = element_types
            self.ndescriptors = ndescriptors
            self.nparams = nparams
            self.rcutfac = rcutfac


log = logging.getLogger(__name__)

# Pair list is unused; a tiny cutoff keeps ML-IAP neighbor work negligible.
_PLACEHOLDER_RCUTFAC = 0.5


@dataclass
class UncertaintyEvent:
    step: int
    score_eV_per_A: float


@dataclass
class UncertaintyGate:
    """Force-disagreement gate evaluated every `every` force calls."""

    calculators: Sequence
    threshold_eV_per_A: float
    every: int = 10
    events: list[UncertaintyEvent] = field(default_factory=list)

    @property
    def triggered(self) -> bool:
        return bool(self.events)

    def score(self, atoms: Atoms) -> float:
        """RMS over atoms of the per-component force std across the ensemble."""
        if len(self.calculators) < 2:
            raise ValueError("ensemble scoring needs >=2 calculators")
        per_model = []
        for calc in self.calculators:
            copy = atoms.copy()
            copy.calc = calc
            per_model.append(np.asarray(copy.get_forces(), dtype=float))
        std = np.stack(per_model).std(axis=0)
        return float(np.sqrt(np.mean(std**2)))

    def check(self, step: int, atoms: Atoms) -> None:
        score = self.score(atoms)
        if not np.isfinite(score):
            raise FloatingPointError("non-finite ensemble force-disagreement score")
        if score > self.threshold_eV_per_A:
            self.events.append(UncertaintyEvent(step, score))


class CalculatorMLIAP(_Base):
    """Bridge: LAMMPS owns dynamics, the ASE calculator owns the physics.

    Forces are written straight to local atoms, so pressure/virial is unsupported; use NVE/NVT.
    With more than one MPI rank every rank gathers all atoms (``mpi4py`` is required), evaluates
    the whole system and keeps the forces of its own atoms; rank 0 alone reports the energy so the
    LAMMPS sum over ranks counts it once. This is correct but not scalable: the calculator cost
    is paid on every rank.
    """

    def __init__(
        self,
        calculator,
        element_types: Sequence[str],
        lmp=None,
        gate: UncertaintyGate | None = None,
    ):
        super().__init__(
            interface=None,
            element_types=list(element_types),
            ndescriptors=1,
            nparams=1,
            rcutfac=_PLACEHOLDER_RCUTFAC,
        )
        self.calculator = calculator
        self.lmp = lmp
        self.gate = gate
        self.ncalls = 0
        self._z = np.array([atomic_numbers[s] for s in self.element_types])

    def compute_gradients(self, data):
        pass

    def compute_descriptors(self, data):
        pass

    def atoms_from_lammps(self, nlocal: int, elems: np.ndarray) -> Atoms:
        lmp = self.lmp
        if lmp is None:
            raise RuntimeError("attach the lammps instance before running")
        boxlo, boxhi, xy, yz, xz, periodicity, _ = lmp.extract_box()
        cell = np.array(
            [
                [boxhi[0] - boxlo[0], 0.0, 0.0],
                [xy, boxhi[1] - boxlo[1], 0.0],
                [xz, yz, boxhi[2] - boxlo[2]],
            ]
        )
        x = np.array(lmp.numpy.extract_atom("x")[:nlocal], dtype=float)
        return Atoms(
            numbers=self._z[np.asarray(elems)[:nlocal]],
            positions=x,
            cell=cell,
            pbc=[bool(p) for p in periodicity],
        )

    def _mpi_comm(self):
        """The MPI communicator when LAMMPS runs on several ranks, else ``None``."""
        extract = getattr(self.lmp, "extract_setting", None)
        if extract is None or int(extract("world_size")) <= 1:
            return None
        try:
            from mpi4py import MPI
        except ImportError as exc:
            raise RuntimeError(
                "LAMMPS runs on several MPI ranks; the ML-IAP bridge needs mpi4py to gather atoms"
            ) from exc
        return MPI.COMM_WORLD

    def _gathered_atoms(self, comm, nlocal: int, elems: np.ndarray):
        """All atoms of the system ordered by LAMMPS id, plus the rows of this rank's atoms."""
        local = self.atoms_from_lammps(nlocal, elems)
        ids = np.array(self.lmp.numpy.extract_atom("id")[:nlocal], dtype=np.int64)
        parts = comm.allgather((ids, local.numbers, local.positions))
        all_ids = np.concatenate([p[0] for p in parts])
        order = np.argsort(all_ids)
        atoms = Atoms(
            numbers=np.concatenate([p[1] for p in parts])[order],
            positions=np.concatenate([p[2] for p in parts])[order],
            cell=local.cell,
            pbc=local.pbc,
        )
        return atoms, np.searchsorted(all_ids[order], ids)

    def compute_forces(self, data):
        nlocal = int(data.nlocal)
        comm = self._mpi_comm()
        if comm is None:
            atoms = self.atoms_from_lammps(nlocal, data.elems)
            rows = slice(0, nlocal)
        else:
            atoms, rows = self._gathered_atoms(comm, nlocal, data.elems)
        atoms.calc = self.calculator
        forces = np.asarray(atoms.get_forces(), dtype=float)
        energy = float(atoms.get_potential_energy())
        if not (np.isfinite(forces).all() and np.isfinite(energy)):
            raise FloatingPointError("MLIP returned non-finite energy or forces")
        data.f[:nlocal] += forces[rows]
        data.energy = energy if comm is None or comm.Get_rank() == 0 else 0.0
        if self.gate is not None and self.ncalls % self.gate.every == 0:
            self.gate.check(self.ncalls, atoms)
        self.ncalls += 1


def run_with_gate(lmp, steps: int, chunk: int, obj: CalculatorMLIAP) -> int:
    """Run `steps` in chunks; stop early once the gate fires. Returns steps done."""
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    done = 0
    while done < steps:
        n = min(chunk, steps - done)
        lmp.command(f"run {n} pre no post no")
        done += n
        if obj.gate is not None and obj.gate.triggered:
            break
    return done
