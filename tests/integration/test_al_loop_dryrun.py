"""Integration test: run one iteration of the AL loop with everything mocked.

The point is to exercise the *plumbing* of `run_active_learning` end-to-end
without needing HydraGNN, VASP, or QE. We monkey-patch:

* the HydraGNN calculator factory  → a tiny constant-force calculator
* the seed resolver                → returns a single seed file
* the MD candidate sampler         → returns a deterministic list
* the DFT backend factory          → an in-process backend that fabricates
                                      a converged `DFTResult` per spec
* the trainer                      → a no-op that returns the same logdir

After one iteration we assert that:
* `state.json` was written with `status="complete"`
* `dataset.extxyz` exists and has at least one frame
* the iteration's DFT working directory was created
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms

from matsim_agents.active_learning.candidates import Candidate
from matsim_agents.active_learning.config import (
    AcquisitionConfig,
    ALConfig,
    DFTConfig,
    HydraGNNConfig,
    LoopConfig,
    MACEConfig,
    MDConfig,
    MLIPConfig,
    SeedSourceConfig,
    TrainerConfig,
    UMAConfig,
    VASPConfig,
)
from matsim_agents.active_learning.dft_backend import DFTJobSpec, DFTResult
from matsim_agents.active_learning.evaluate import PromotionDecision, _apply_model_override

# --------------------------------------------------------------------------- #
# Stubs                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class _ConstantForceCalc:
    forces: np.ndarray

    def calculate(self, atoms=None, properties=None, system_changes=None) -> None:
        self.results = {"forces": self.forces, "energy": -1.0}

    implemented_properties = ["forces", "energy"]


class _FakeDFTBackend:
    """Minimal DFTBackend Protocol implementation that fabricates a result."""

    name = "vasp"
    nodes_per_job = 1
    ranks_per_node = 1
    threads_per_rank = 1
    timeout_sec = 60

    def run_one(self, spec: DFTJobSpec) -> DFTResult:
        Path(spec.work_dir).mkdir(parents=True, exist_ok=True)
        n = len(spec.atoms)
        return DFTResult(
            backend=self.name,
            work_dir=spec.work_dir,
            return_code=0,
            converged=True,
            energy_eV=-1.234 * n,
            forces_eV_per_A=np.zeros((n, 3)),
            stress_eV_per_A3=None,
            n_atoms=n,
            wall_time_sec=0.01,
            final_atoms=spec.atoms.copy(),
            notes=None,
        )


def _make_cfg(tmp_path: Path) -> ALConfig:
    """Build a synthetic but pydantic-valid ALConfig (paths exist as stubs)."""
    seed = tmp_path / "seed.vasp"
    seed.write_text("dummy\n")
    vasp_bin = tmp_path / "vasp_std"
    vasp_bin.write_text("#!/bin/bash\n")
    wrapper = tmp_path / "wrap.sh"
    wrapper.write_text("#!/bin/bash\n")
    incar = tmp_path / "INCAR.template"
    incar.write_text("ENCUT = 520\n")
    potcar_dir = tmp_path / "potcars"
    potcar_dir.mkdir()
    si_potcar_dir = potcar_dir / "Si"
    si_potcar_dir.mkdir()
    (si_potcar_dir / "POTCAR").write_text("Si test potential")
    train = tmp_path / "train.py"
    train.write_text("# stub\n")
    logdir = tmp_path / "logdir"
    logdir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    return ALConfig(
        hydragnn=HydraGNNConfig(logdir=logdir),
        md=MDConfig(
            seed_source=SeedSourceConfig(kind="paths", paths=[seed]),
            n_steps=1,
        ),
        acquisition=AcquisitionConfig(
            strategy="random",
            n_select=2,
            diversity_filter=False,
        ),
        dft=DFTConfig(
            backend="vasp",
            vasp=VASPConfig(
                vasp_bin=vasp_bin,
                vasp_wrapper=wrapper,
                incar_template=incar,
                potcar_dir=potcar_dir,
            ),
        ),
        trainer=TrainerConfig(enabled=False, train_script=train),
        loop=LoopConfig(n_iterations=1, out_dir=out_dir, resume=False),
    )


# --------------------------------------------------------------------------- #
# The test                                                                    #
# --------------------------------------------------------------------------- #


def _make_candidate(idx: int) -> Candidate:
    atoms = Atoms(
        symbols=["Si", "Si"],
        positions=[[0.0, 0.0, 0.0], [1.357 + 0.01 * idx, 1.357, 1.357]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    return Candidate(
        candidate_id=f"cand_{idx:03d}",
        atoms=atoms,
        seed_path="/dummy.vasp",
        md_step=idx,
    )


def _patch_runtime(
    loop_mod,
    monkeypatch: pytest.MonkeyPatch,
    *,
    trained_model: Path | None = None,
    promotion_decision: PromotionDecision | None = None,
) -> None:
    monkeypatch.setattr(
        loop_mod,
        "make_mlip_calculator",
        lambda mlip_cfg, **kw: _ConstantForceCalc(forces=np.zeros((2, 3))),
    )
    monkeypatch.setattr(loop_mod, "build_ensemble", lambda hcfg: [])
    monkeypatch.setattr(
        loop_mod,
        "resolve_seed_structures",
        lambda src, out_dir: [Path(path) for path in src.paths],
    )
    monkeypatch.setattr(
        loop_mod,
        "sample_md_candidates",
        lambda md_cfg, calc, out_dir, seed_paths=None: [_make_candidate(i) for i in range(3)],
    )
    monkeypatch.setattr(loop_mod, "make_backend", lambda dft_cfg: _FakeDFTBackend())
    monkeypatch.setattr(
        loop_mod,
        "retrain_hydragnn",
        lambda tcfg, hcfg, dataset_path, iteration, out_logdir: trained_model or hcfg.logdir,
    )
    if promotion_decision is not None:
        monkeypatch.setattr(
            loop_mod,
            "evaluate_promotion_candidate",
            lambda *args, **kwargs: promotion_decision,
        )


def test_one_iteration_dryrun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg(tmp_path)

    # Patch the heavy components inside `loop` (where they are imported).
    import matsim_agents.active_learning.loop as loop_mod

    _patch_runtime(loop_mod, monkeypatch)

    # Run one iteration end-to-end.
    loop_mod.run_active_learning(cfg)

    # ─── Assertions ────────────────────────────────────────────────────────
    out_dir = Path(cfg.loop.out_dir)
    iter_dir = out_dir / "iteration_0000"
    state_file = iter_dir / "state.json"

    assert state_file.is_file(), "Iteration state file was not written"
    import json

    state: dict[str, Any] = json.loads(state_file.read_text())
    assert state["status"] == "complete"
    assert state["iteration"] == 0
    assert state["n_candidates"] == 3
    # n_select=2 with random strategy on 3 candidates -> exactly 2 selected.
    assert state["n_selected"] == 2
    assert len(state["selected_candidate_ids"]) == 2
    assert set(state["candidate_uncertainty"]) == set(state["selected_candidate_ids"])
    assert all(0.0 <= score <= 1.0 for score in state["candidate_uncertainty"].values())
    assert state["n_dft_converged"] == 2
    assert state["n_dft_failed"] == 0
    assert state["dft_backend"] == "vasp"

    # Dataset file written and non-empty.
    dataset = out_dir / "dataset.extxyz"
    assert dataset.is_file()
    assert dataset.stat().st_size > 0

    # Per-candidate DFT work dirs exist.
    dft_dir = iter_dir / "dft"
    assert dft_dir.is_dir()
    assert any(dft_dir.iterdir()), "DFT working directories were not created"


@pytest.mark.parametrize("approved", [True, False])
def test_promotion_decision_controls_model_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approved: bool,
) -> None:
    cfg = _make_cfg(tmp_path)
    validation_set = tmp_path / "held-out.extxyz"
    validation_set.touch()
    cfg.trainer = TrainerConfig(
        enabled=True,
        promote_model=True,
        promotion_approved=True,
        train_script=cfg.trainer.train_script,
        validation_set=validation_set,
    )
    incumbent = cfg.mlip.hydragnn.logdir
    cfg.mlip.hydragnn.checkpoint = "incumbent.pk"
    trained_model = tmp_path / "candidate-model"
    trained_model.mkdir()
    decision = PromotionDecision(
        approved=approved,
        reasons=[] if approved else ["held-out force MAE regressed"],
        candidate_metrics={"force_mae_eV_per_A": 0.1},
        incumbent_metrics={"force_mae_eV_per_A": 0.08},
    )

    import matsim_agents.active_learning.loop as loop_mod

    _patch_runtime(
        loop_mod,
        monkeypatch,
        trained_model=trained_model,
        promotion_decision=decision,
    )
    loop_mod.run_active_learning(cfg)

    import json

    state = json.loads((cfg.loop.out_dir / "iteration_0000" / "state.json").read_text())
    assert state["model_promoted"] is approved
    assert state["promotion_validation"]["approved"] is approved
    expected_model = trained_model if approved else incumbent
    assert cfg.mlip.hydragnn.logdir == expected_model
    assert cfg.mlip.hydragnn.checkpoint == (None if approved else "incumbent.pk")


def test_holdout_split_compares_candidate_without_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.acquisition.n_select = 3
    cfg.trainer = TrainerConfig(
        enabled=True,
        train_script=cfg.trainer.train_script,
        validation_fraction=0.2,
        validation_split_seed=7,
        compare_after_training=True,
    )
    trained_model = tmp_path / "candidate-model"
    trained_model.mkdir()
    decision = PromotionDecision(
        approved=True,
        reasons=[],
        candidate_metrics={"force_mae_eV_per_A": 0.05},
        incumbent_metrics={"force_mae_eV_per_A": 0.1},
    )

    import matsim_agents.active_learning.loop as loop_mod

    _patch_runtime(
        loop_mod,
        monkeypatch,
        trained_model=trained_model,
        promotion_decision=decision,
    )
    loop_mod.run_active_learning(cfg)

    import json

    state = json.loads((cfg.loop.out_dir / "iteration_0000" / "state.json").read_text())
    assert state["n_training_frames"] == 2
    assert state["n_validation_frames"] == 1
    assert state["model_comparison"] == asdict(decision)
    assert state["promotion_validation"] is None
    assert state["model_promoted"] is False


def test_hydragnn_candidate_override_discovers_checkpoint_in_new_logdir(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.mlip.hydragnn.checkpoint = "incumbent.pk"
    candidate = tmp_path / "candidate-model"
    candidate.mkdir()

    _apply_model_override(cfg, str(candidate))

    assert cfg.mlip.hydragnn.logdir == candidate
    assert cfg.mlip.hydragnn.checkpoint is None


def test_resume_retains_last_promoted_logdir_after_rejected_candidate(tmp_path: Path) -> None:
    import json

    from matsim_agents.active_learning.loop import _scan_resume

    promoted = tmp_path / "promoted-model"
    promoted.mkdir()
    for iteration, new_logdir in ((0, str(promoted)), (1, None)):
        iteration_dir = tmp_path / f"iteration_{iteration:04d}"
        iteration_dir.mkdir()
        (iteration_dir / "state.json").write_text(
            json.dumps(
                {
                    "iteration": iteration,
                    "status": "complete",
                    "new_logdir": new_logdir,
                }
            )
        )

    assert _scan_resume(tmp_path) == (2, promoted)


@pytest.mark.parametrize("backend", ["uma", "mace"])
def test_resume_applies_last_promoted_model_for_all_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    import json

    import matsim_agents.active_learning.loop as loop_mod

    cfg = _make_cfg(tmp_path)
    promoted = tmp_path / ("promoted.model" if backend == "mace" else "promoted-uma")
    if backend == "mace":
        promoted.touch()
        cfg.mlip = MLIPConfig(backend="mace", mace=MACEConfig())
    else:
        promoted.mkdir()
        cfg.mlip = MLIPConfig(backend="uma", uma=UMAConfig())
    cfg.loop.resume = True
    iteration_dir = cfg.loop.out_dir / "iteration_0000"
    iteration_dir.mkdir()
    (iteration_dir / "state.json").write_text(
        json.dumps(
            {
                "iteration": 0,
                "status": "complete",
                "new_logdir": str(promoted),
            }
        )
    )
    monkeypatch.setattr(loop_mod, "resolve_seed_structures", lambda *args: [])

    loop_mod.run_active_learning(cfg)

    if backend == "mace":
        assert cfg.mlip.mace is not None
        assert cfg.mlip.mace.family == "checkpoint"
        assert cfg.mlip.mace.model == str(promoted)
    else:
        assert cfg.mlip.uma is not None
        assert cfg.mlip.uma.model_name == str(promoted)
