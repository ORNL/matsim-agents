"""Radiation cascade protocol: phases, checkpoints and the WP1.11 output contract.

An additional protocol on top of the shared pieces (calculator, structure handling, seeding,
``run_cascade`` integrator), not a second MD framework; thermal MD is untouched. Phases run in
order: ballistic (no thermostat, optional electronic stopping), thermal spike and cooling
(boundary thermostat only if a mask is given). Initial equilibration is the caller's job
(``thermalize`` only draws velocities), because equilibrium MD belongs to the existing thermal
protocol. State is checkpointed after every phase so a restart reproduces the same result.
The result is consumed by WP3 through ``analyze_cascade``. Nothing here qualifies a potential:
the qualification status is carried in, not computed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from matsim_agents.core.hashing import sha256_json
from matsim_agents.core.seeding import numpy_rng
from matsim_agents.science.cascade.dynamics import (
    Boundary,
    CascadeRun,
    ElectronicStopping,
    StepControl,
    run_cascade,
)
from matsim_agents.science.cascade.events import (
    CascadeEvent,
    structure_hash,
    validate_event,
)
from matsim_agents.science.cascade.qualification import Level, Resolution
from matsim_agents.science.cascade.recoil import RecoilInit, initialize_recoil
from matsim_agents.science.defects.survival import (
    SurvivingDefectSummary,
    summarize_surviving_defects,
)
from matsim_agents.science.defects.wigner_seitz import (
    WignerSeitzResult,
    wigner_seitz_analysis,
)


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Phase:
    name: str
    duration_fs: float
    control: StepControl
    use_stopping: bool
    use_boundary: bool


def default_phases(
    ballistic_fs: float = 200.0, spike_fs: float = 2000.0, cool_fs: float = 5000.0
) -> tuple[Phase, ...]:
    ball = StepControl(max_displacement_A=0.02, dt_min_fs=1e-3, dt_max_fs=1.0)
    slow = StepControl(max_displacement_A=0.05, dt_min_fs=1e-2, dt_max_fs=2.0)
    return (
        Phase("ballistic", ballistic_fs, ball, True, False),
        Phase("thermal_spike", spike_fs, slow, True, True),
        Phase("cooling", cool_fs, slow, False, True),
    )


def thermalize(atoms: Atoms, temperature_K: float, seed: int) -> None:
    """Draw Maxwell-Boltzmann velocities with zero net momentum; does not equilibrate."""
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=numpy_rng(seed, "thermal"))
    Stationary(atoms)


@dataclass(frozen=True)
class Snapshot:
    phase: str
    time_fs: float  # time within the phase
    atoms: Atoms
    max_force_eV_per_A: float
    energy_eV: float


@dataclass(frozen=True)
class CascadeResult:
    event: CascadeEvent
    initial_atoms: Atoms
    final_atoms: Atoms
    recoil: RecoilInit
    phases: tuple[str, ...]
    runs: tuple[CascadeRun, ...]
    duration_fs: float
    energy_electronic_eV: float
    energy_thermostat_eV: float
    max_force_eV_per_A: float
    numerically_stable: bool
    qualification: Resolution | None
    provenance: dict = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    snapshots: tuple[Snapshot, ...] = ()


def _save_state(path: Path, atoms: Atoms, phase_index: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        positions=atoms.get_positions(),
        velocities=atoms.get_velocities(),
        meta=json.dumps({**meta, "phase_index": phase_index}),
    )


def _load_state(path: Path, atoms: Atoms) -> int:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    atoms.set_positions(data["positions"])
    atoms.set_velocities(data["velocities"])
    return int(meta["phase_index"])


def run_protocol(
    atoms: Atoms,
    event: CascadeEvent,
    *,
    direction: tuple[float, float, float],
    phases: tuple[Phase, ...],
    stopping: ElectronicStopping | None = None,
    boundary_mask: np.ndarray | None = None,
    boundary_target_K: float | None = None,
    boundary_tau_fs: float = 100.0,
    qualification: Resolution | None = None,
    require_level: Level | None = Level.CASCADE_QUALIFIED,
    convention: str = "replace",
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    max_atom_speed: float | None = None,
    snapshot_every_fs: float | None = None,
) -> CascadeResult:
    """Run the phases in place on ``atoms`` (calculator attached, velocities thermalized)."""
    if atoms.calc is None:
        raise ProtocolError("atoms needs a calculator")
    if not resume:
        validate_event(event, atoms)
    if require_level is not None:
        lvl = qualification.level if qualification is not None else Level.UNASSESSED
        if lvl < require_level:
            raise ProtocolError(
                f"potential is {lvl.name}; {require_level.name} is required for this run "
                "(set require_level=None only for explicit verification runs)"
            )
    if boundary_mask is not None and boundary_target_K is None:
        raise ProtocolError("a boundary mask needs a target temperature")
    initial = atoms.copy()
    ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else None
    start_phase = 0
    recoil: RecoilInit
    if resume:
        if ckpt is None or not (ckpt / "state.npz").exists():
            raise ProtocolError("resume needs an existing checkpoint")
        start_phase = _load_state(ckpt / "state.npz", atoms) + 1
        recoil = RecoilInit(**json.loads((ckpt / "recoil.json").read_text()))
        initial = atoms.copy()
        initial.set_positions(np.load(ckpt / "initial_positions.npy"))
    else:
        recoil = initialize_recoil(
            atoms,
            event.site_index,
            event.energy_eV,
            direction,
            expected_symbol=event.recoil_species.element,
            convention=convention,  # type: ignore[arg-type]
        )
        if ckpt is not None:
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / "recoil.json").write_text(json.dumps(asdict(recoil)))
            np.save(ckpt / "initial_positions.npy", initial.get_positions())
    runs: list[CascadeRun] = []
    snaps: list[Snapshot] = []
    done: list[str] = []
    for k, ph in enumerate(phases):
        if k < start_phase:
            continue
        b = None
        if ph.use_boundary and boundary_mask is not None:
            b = Boundary(boundary_mask, float(boundary_target_K), boundary_tau_fs)  # type: ignore[arg-type]
        last = [-math.inf]

        def grab(a: Atoms, _step: int, t_fs: float, name: str = ph.name, last=last) -> None:
            if snapshot_every_fs is None or t_fs - last[0] < snapshot_every_fs:
                return
            last[0] = t_fs
            c = a.copy()
            f = a.get_forces()
            snaps.append(
                Snapshot(
                    name,
                    float(t_fs),
                    c,
                    float(np.linalg.norm(f, axis=1).max()),
                    float(a.get_potential_energy()),
                )
            )

        run = run_cascade(
            atoms,
            ph.duration_fs,
            control=ph.control,
            stopping=stopping if ph.use_stopping else None,
            boundary=b,
            max_atom_speed=max_atom_speed,
            on_step=grab if snapshot_every_fs is not None else None,
        )
        runs.append(run)
        done.append(ph.name)
        if ckpt is not None:
            _save_state(ckpt / "state.npz", atoms, k, {"phase": ph.name})
    max_f = max((max(r.trace.max_force) for r in runs), default=0.0)
    prov = {
        "event_id": event.event_id,
        "seed": event.seed,
        "initial_structure_hash": structure_hash(initial),
        "phases": [asdict(p) for p in phases],
        "convention": convention,
        "stopping_source": stopping.source if stopping else "none (electronic stopping disabled)",
        "resumed_from_phase": start_phase if resume else None,
        "protocol_hash": sha256_json([asdict(p) for p in phases]),
    }
    return CascadeResult(
        event=event,
        initial_atoms=initial,
        final_atoms=atoms.copy(),
        recoil=recoil,
        phases=tuple(done),
        runs=tuple(runs),
        duration_fs=float(sum(r.duration_fs for r in runs)),
        energy_electronic_eV=float(sum(r.electronic_loss_eV for r in runs)),
        energy_thermostat_eV=float(sum(r.thermostat_loss_eV for r in runs)),
        max_force_eV_per_A=float(max_f),
        numerically_stable=all(r.stable for r in runs),
        qualification=qualification,
        provenance=prov,
        snapshots=tuple(snaps),
    )


@dataclass(frozen=True)
class CascadeAnalysis:
    ws: WignerSeitzResult
    summary: SurvivingDefectSummary
    event_id: str
    qualification_level: str
    caveats: tuple[str, ...]


def analyze_cascade(
    result: CascadeResult,
    reference: Atoms,
    *,
    cluster_cutoff: float,
    cluster_justification: str,
) -> CascadeAnalysis:
    """Wigner-Seitz analysis of the final structure against the pristine ``reference``."""
    ws = wigner_seitz_analysis(reference, result.final_atoms)
    summary = summarize_surviving_defects(
        reference,
        result.final_atoms,
        ws,
        cluster_cutoff=cluster_cutoff,
        cluster_justification=cluster_justification,
        reference_population=float(len(reference)),
        reference_population_definition="atoms in the reference cell",
    )
    caveats = []
    if not result.numerically_stable:
        caveats.append("time-step floor was hit: the integration criterion was violated")
    lvl = result.qualification.level.name if result.qualification else Level.UNASSESSED.name
    if result.qualification is None or result.qualification.level < Level.CASCADE_QUALIFIED:
        caveats.append(f"potential is {lvl}: counts are not qualified cascade statistics")
    return CascadeAnalysis(ws, summary, result.event.event_id, lvl, tuple(caveats))


__all__ = [
    "CascadeAnalysis",
    "CascadeResult",
    "Phase",
    "Snapshot",
    "ProtocolError",
    "analyze_cascade",
    "default_phases",
    "run_protocol",
    "thermalize",
]
