"""Staged neb.x runs and explicit restarts, with a fake launcher (no QE needed)."""

import hashlib
import json

import numpy as np
import pytest
from ase import Atoms

from matsim_agents.backends.dft.qe_neb import QENebConfig
from matsim_agents.backends.dft.qe_neb_stages import (
    HISTORY_FILE,
    StageError,
    ci_entry_problems,
    derive_stage_config,
    load_history,
    prepare_stage,
    run_stage,
)
from tests.synthetic_upf import write_synthetic_upf

ENERGIES = [-100.0, -99.6, -99.2, -99.7, -100.1]
ENDPOINT_MAX = [-99.0, -99.4, -99.5, -99.7, -100.1]


def _images(n=5):
    out = []
    for k in range(n):
        atoms = Atoms(
            "MoW2",
            positions=[[0.0, 0.0, 0.0], [1.5 + 0.1 * k, 1.5, 1.5], [3.0, 3.0, 3.0]],
            cell=np.eye(3) * 6.0,
            pbc=True,
        )
        atoms.set_array("atom_id", np.array([0, 1, 2]))
        out.append(atoms)
    return out


def _cfg(workdir, **over):
    pseudo = workdir.parent / "pseudos"
    for element in ("Mo", "W"):
        write_synthetic_upf(pseudo, f"{element}.upf", element)
    base = {
        "n_images": 5,
        "species": ("Mo", "W"),
        "pseudo_dir": str(pseudo),
        "pseudopotentials": {"Mo": "Mo.upf", "W": "W.upf"},
        "ecutwfc_ry": 50.0,
        "ecutrho_ry": 400.0,
        "occupations": "smearing",
        "degauss_ry": 0.02,
        "kpts": (2, 2, 2),
        "nspin": 1,
        "nstep_path": 100,
        "workdir": str(workdir),
    }
    base.update(over)
    return QENebConfig(**base)


def _fake(
    energies=ENERGIES,
    errors=(0, 0.03, 0.02, 0.04, 0),
    tail="converged",
    iteration=7,
    ci=None,
    exit_code=0,
):
    """A launcher that writes a QE-style log plus prefix.path / prefix.dat in the cwd."""

    def launch(argv, *, cwd, stdout, timeout_sec, env):
        top = max(energies)
        text = (
            f"\n     ------------------------------ iteration {iteration:3d} "
            "------------------------------\n\n"
            f"\n     activation energy (->) = {top - energies[0]:10.6f} eV\n"
            f"     activation energy (<-) = {top - energies[-1]:10.6f} eV\n\n"
            "     image        energy (eV)        error (eV/A)        frozen\n\n"
        )
        for i, (e, err) in enumerate(zip(energies, errors, strict=True), start=1):
            flag = "T" if i in (1, len(energies)) else "F"
            text += f"     {i:5d}    {e:15.7f}          {err:10.6f}            {flag}\n"
        if ci is not None:
            text += f"\n     climbing image = {ci:2d}\n"
        if tail == "converged":
            text += f"\n     neb: convergence achieved in {iteration:3d} iterations\n"
        elif tail == "limit":
            text += "\n     neb: reached the maximum number of steps\n"
        stdout.write(text)
        from pathlib import Path

        Path(cwd, "neb.path").write_text(f"RESTART INFORMATION\n{iteration}\n{len(argv)}\n")
        rel = [e - energies[0] for e in energies]
        Path(cwd, "neb.dat").write_text(
            "".join(
                f"  {k / 4:16.10f}  {r:16.10f}  {x:16.10f}\n"
                for k, (r, x) in enumerate(zip(rel, errors, strict=True))
            )
        )
        return exit_code

    return launch


def _run_initial(tmp_path, launcher=None, **over):
    config = _cfg(tmp_path / "w", **over)
    images = _images()
    prepare_stage(config, images, kind="initial")
    return config, images, run_stage(config.workdir, launcher=launcher or _fake())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_initial_stage_is_prepared_with_no_ci_and_recorded(tmp_path):
    config = _cfg(tmp_path / "w")
    entry = prepare_stage(config, _images(), kind="initial")
    text = (tmp_path / "w" / "neb.stage1.in").read_text()
    assert entry.index == 1 and entry.status == "prepared" and entry.kind == "initial"
    assert "restart_mode" in text and "'from_scratch'" in text and "no-CI" in text
    history = load_history(config.workdir)
    assert [s.index for s in history.stages] == [1]
    assert entry.input_sha256 == _sha(tmp_path / "w" / "neb.stage1.in")


def test_initial_stage_rejects_restart_or_ci_config(tmp_path):
    bad_restart = _cfg(tmp_path / "w", restart_mode="restart")
    with pytest.raises(StageError, match="from_scratch"):
        prepare_stage(bad_restart, _images(), kind="initial")
    manual = _cfg(tmp_path / "w", ci_scheme="auto")
    with pytest.raises(StageError, match="no-CI"):
        prepare_stage(manual, _images(), kind="initial")
    assert not (tmp_path / "w" / "neb.stage1.in").exists()


def test_initial_stage_refuses_existing_output_and_second_initial(tmp_path):
    config = _cfg(tmp_path / "w")
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "neb.path").write_text("old")
    with pytest.raises(StageError, match="overwritten"):
        prepare_stage(config, _images(), kind="initial")
    (tmp_path / "w" / "neb.path").unlink()
    prepare_stage(config, _images(), kind="initial")
    with pytest.raises(StageError, match="already exists"):
        prepare_stage(config, _images(), kind="initial")


def test_run_stage_records_outcome_and_path_hash(tmp_path):
    config, _, entry = _run_initial(tmp_path)
    work = tmp_path / "w"
    assert entry.status == "run" and entry.outcome == "converged"
    assert entry.iterations_completed == 7 and entry.exit_code == 0
    assert entry.forward_barrier_eV == pytest.approx(0.8)
    assert entry.path_sha256_after == _sha(work / "neb.path")
    assert (work / "neb.stage1.out").is_file() and (work / "neb.stage1.out.record.json").is_file()
    assert load_history(config.workdir).stages[0].outcome == "converged"


def test_run_stage_needs_a_prepared_untampered_stage(tmp_path):
    config = _cfg(tmp_path / "w")
    (tmp_path / "w").mkdir()
    with pytest.raises(StageError, match="no prepared stage"):
        run_stage(config.workdir, launcher=_fake())
    prepare_stage(config, _images(), kind="initial")
    (tmp_path / "w" / "neb.stage1.in").write_text("edited")
    with pytest.raises(StageError, match="changed"):
        run_stage(config.workdir, launcher=_fake())
    assert load_history(config.workdir).stages[0].status == "prepared"


def test_ci_stage_after_convergence_snapshots_and_enables_ci(tmp_path):
    config, images, first = _run_initial(tmp_path)
    work = tmp_path / "w"
    before = {n: _sha(work / n) for n in ("neb.path", "neb.dat", "neb.stage1.out")}
    ci = derive_stage_config(config, "ci", nstep_path=200)
    entry = prepare_stage(ci, images, kind="ci")
    assert entry.index == 2 and entry.kind == "ci"
    assert entry.path_sha256_before == before["neb.path"]
    snap = work / "history" / "stage1"
    for name, digest in before.items():
        assert _sha(snap / name) == digest  # earlier output preserved, hash-verified
        assert _sha(work / name) == digest  # and untouched in place
    text = (work / "neb.stage2.in").read_text()
    assert "'restart'" in text and "auto" in text
    assert (work / "neb.stage1.in").is_file()
    done = run_stage(config.workdir, launcher=_fake(ci=3, iteration=12))
    assert done.outcome == "converged" and done.climbing_images == (3,)
    assert [s.index for s in load_history(config.workdir).stages] == [1, 2]
    assert _sha(work / "neb.stage1.out") == before["neb.stage1.out"]


def test_manual_ci_stage_lists_the_climbing_image(tmp_path):
    config, images, _ = _run_initial(tmp_path)
    ci = derive_stage_config(config, "ci", nstep_path=200, ci_scheme="manual", climbing_images=(3,))
    prepare_stage(ci, images, kind="ci")
    assert "CLIMBING_IMAGES" in (tmp_path / "w" / "neb.stage2.in").read_text()


def test_ci_rejected_when_previous_stage_not_converged_enough(tmp_path):
    loose = _fake(tail="limit", errors=(0, 0.5, 0.4, 0.6, 0), iteration=20)
    config, images, first = _run_initial(tmp_path, launcher=loose, nstep_path=20)
    assert first.outcome == "iteration_limit"
    with pytest.raises(StageError, match="CI entry threshold"):
        prepare_stage(derive_stage_config(config, "ci", nstep_path=50), images, kind="ci")
    assert not (tmp_path / "w" / "neb.stage2.in").exists()
    assert not (tmp_path / "w" / "history").exists()


def test_ci_allowed_after_iteration_limit_with_small_residual(tmp_path):
    near = _fake(tail="limit", errors=(0, 0.08, 0.06, 0.09, 0), iteration=20)
    config, images, first = _run_initial(tmp_path, launcher=near, nstep_path=20)
    assert first.outcome == "iteration_limit" and ci_entry_problems(first) == []
    assert ci_entry_problems(first, threshold=0.05)  # stricter threshold refuses
    prepare_stage(derive_stage_config(config, "ci", nstep_path=50), images, kind="ci")


def test_ci_rejected_when_maximum_is_an_endpoint(tmp_path):
    config, images, first = _run_initial(tmp_path, launcher=_fake(energies=ENDPOINT_MAX))
    assert first.outcome == "converged" and first.forward_barrier_eV is None
    with pytest.raises(StageError, match="endpoint"):
        prepare_stage(derive_stage_config(config, "ci", nstep_path=200), images, kind="ci")


def test_ci_rejected_after_failed_or_incomplete_stage(tmp_path):
    config, images, first = _run_initial(tmp_path, launcher=_fake(tail="none", exit_code=1))
    assert first.outcome == "process_failure"
    with pytest.raises(StageError, match="outcome"):
        prepare_stage(derive_stage_config(config, "ci", nstep_path=200), images, kind="ci")


def test_restart_rejects_missing_or_changed_path_file(tmp_path):
    config, images, _ = _run_initial(tmp_path)
    ci = derive_stage_config(config, "ci", nstep_path=200)
    path = tmp_path / "w" / "neb.path"
    saved = path.read_text()
    path.unlink()
    with pytest.raises(StageError, match="silently restart"):
        prepare_stage(ci, images, kind="ci")
    path.write_text(saved + "tampered\n")
    with pytest.raises(StageError, match="changed since"):
        prepare_stage(ci, images, kind="ci")
    assert not (tmp_path / "w" / "neb.stage2.in").exists()


def test_restart_rejects_incompatible_physics_images_and_step_limit(tmp_path):
    config, images, _ = _run_initial(tmp_path)
    work = tmp_path / "w"
    changed = derive_stage_config(config, "ci", nstep_path=200)
    changed = QENebConfig.model_validate({**changed.model_dump(mode="json"), "ecutwfc_ry": 60.0})
    with pytest.raises(StageError, match="ecutwfc_ry"):
        prepare_stage(changed, images, kind="ci")
    other = _images()
    other[2].positions[1, 0] += 0.01
    with pytest.raises(StageError, match="images differ"):
        prepare_stage(derive_stage_config(config, "ci", nstep_path=200), other, kind="ci")
    with pytest.raises(StageError, match="stop at once"):
        prepare_stage(derive_stage_config(config, "ci", nstep_path=7), images, kind="ci")
    assert not (work / "neb.stage2.in").exists()


def test_continue_after_iteration_limit_keeps_ci_settings(tmp_path):
    slow = _fake(tail="limit", errors=(0, 0.5, 0.4, 0.6, 0), iteration=20)
    config, images, _ = _run_initial(tmp_path, launcher=slow, nstep_path=20)
    entry = prepare_stage(
        derive_stage_config(config, "continue", nstep_path=60), images, kind="continue"
    )
    text = (tmp_path / "w" / "neb.stage2.in").read_text()
    assert entry.kind == "continue" and "'restart'" in text
    assert (tmp_path / "w" / "history" / "stage1" / "neb.path").is_file()


def test_continue_rejected_after_convergence_or_with_ci_change(tmp_path):
    config, images, _ = _run_initial(tmp_path)
    with pytest.raises(StageError, match="cannot continue"):
        prepare_stage(
            derive_stage_config(config, "continue", nstep_path=60), images, kind="continue"
        )
    slow = _fake(tail="limit", errors=(0, 0.5, 0.4, 0.6, 0), iteration=20)
    config2, images2, _ = _run_initial(tmp_path / "again", launcher=slow, nstep_path=20)
    with pytest.raises(StageError, match="keep the CI settings"):
        prepare_stage(derive_stage_config(config2, "ci", nstep_path=60), images2, kind="continue")


def test_cannot_prepare_over_an_unrun_stage_or_without_initial(tmp_path):
    config = _cfg(tmp_path / "w")
    with pytest.raises(StageError, match="no earlier stage"):
        prepare_stage(derive_stage_config(config, "ci", nstep_path=50), _images(), kind="ci")
    prepare_stage(config, _images(), kind="initial")
    with pytest.raises(StageError, match="not been run"):
        prepare_stage(
            derive_stage_config(config, "continue", nstep_path=50), _images(), kind="continue"
        )


def test_restart_requires_restart_mode(tmp_path):
    config, images, _ = _run_initial(tmp_path)
    with pytest.raises(StageError, match="restart_mode"):
        prepare_stage(config, images, kind="ci")


def test_snapshot_never_replaces_different_existing_content(tmp_path):
    config, images, _ = _run_initial(tmp_path)
    snap = tmp_path / "w" / "history" / "stage1"
    snap.mkdir(parents=True)
    (snap / "neb.path").write_text("different")
    with pytest.raises(StageError, match="different content"):
        prepare_stage(derive_stage_config(config, "ci", nstep_path=200), images, kind="ci")
    assert (snap / "neb.path").read_text() == "different"


def test_corrupt_history_is_reported(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / HISTORY_FILE).write_text("{not json")
    with pytest.raises(StageError, match="not a valid stage history"):
        load_history(work)
    (work / HISTORY_FILE).write_text(json.dumps({"schema": "other/1", "stages": []}))
    with pytest.raises(StageError, match="schema"):
        load_history(work)


def test_failed_run_still_records_the_stage(tmp_path):
    _, _, entry = _run_initial(tmp_path, launcher=_fake(tail="none", exit_code=3))
    assert entry.status == "run" and entry.run_status == "exited_nonzero"
    assert entry.outcome == "process_failure" and entry.exit_code == 3
