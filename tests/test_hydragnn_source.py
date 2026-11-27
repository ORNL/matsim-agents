"""The imported HydraGNN must be the clean checkout at the commit the checkpoint records."""

from __future__ import annotations

import json
import subprocess

import pytest

import matsim_agents.backends.mlip.hydragnn_source as source

BEST6_SHA256 = "8532b03296a8ab4cfc3b37beb6d57b2091a50807b155fac93cb1882253d32867"
FROZEN_COMMIT = "81d5b88304f20bb45c840571e2e26c490d1b19da"


def _git(root, *args):
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "HydraGNN"
    (root / "hydragnn").mkdir(parents=True)
    (root / "hydragnn" / "__init__.py").write_text("")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init", "--no-gpg-sign")
    monkeypatch.setattr(source, "imported_hydragnn_root", lambda: root)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return root, commit


def _logdir(tmp_path, commit):
    logdir = tmp_path / "logdir"
    logdir.mkdir()
    (logdir / source.RECORD_NAME).write_text(json.dumps({"commit": commit, "checkout": "X"}))
    return logdir


def test_matching_clean_checkout_passes(tmp_path, checkout):
    _, commit = checkout
    assert source.verify_hydragnn_source(_logdir(tmp_path, commit))["commit"] == commit


def test_other_commit_is_refused_naming_both(tmp_path, checkout):
    _, commit = checkout
    other = "0" * 40
    with pytest.raises(ValueError) as error:
        source.verify_hydragnn_source(_logdir(tmp_path, other))
    assert other in str(error.value) and commit in str(error.value)


def test_dirty_tracked_file_is_refused(tmp_path, checkout):
    root, commit = checkout
    (root / "hydragnn" / "__init__.py").write_text("changed")
    with pytest.raises(ValueError, match="uncommitted changes"):
        source.verify_hydragnn_source(_logdir(tmp_path, commit))


def test_non_git_directory_is_refused(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    (plain / "hydragnn").mkdir(parents=True)
    monkeypatch.setattr(source, "imported_hydragnn_root", lambda: plain)
    with pytest.raises(ValueError, match="not a readable git checkout"):
        source.verify_hydragnn_source(_logdir(tmp_path, "0" * 40))


def test_logdir_without_record_or_registry_entry_is_refused(tmp_path, checkout):
    logdir = tmp_path / "unknown"
    logdir.mkdir()
    (logdir / "model.pk").write_bytes(b"not a registered checkpoint")
    with pytest.raises(FileNotFoundError, match="hydragnn-source.json is missing"):
        source.verify_hydragnn_source(logdir)


def test_malformed_commit_is_refused(tmp_path):
    with pytest.raises(ValueError, match="40-character"):
        source.read_hydragnn_source_record(_logdir(tmp_path, "main"))


def test_registry_pins_best6_to_the_frozen_commit():
    registry = json.loads(source.REGISTRY_PATH.read_text())
    assert registry[BEST6_SHA256]["commit"] == FROZEN_COMMIT
    pin = json.loads(
        (source.REGISTRY_PATH.parents[4] / "benchmarks/e4_mlip/frozen-source.json").read_text()
    )
    assert pin["commit"] == FROZEN_COMMIT


def test_registry_entry_found_by_checkpoint_hash(tmp_path, monkeypatch):
    logdir = tmp_path / "best6"
    logdir.mkdir()
    (logdir / "weights.pk").write_bytes(b"stand-in")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({source._sha256(logdir / "weights.pk"): {"commit": "a" * 40, "checkout": "Z"}})
    )
    monkeypatch.setattr(source, "REGISTRY_PATH", registry)
    assert source.read_hydragnn_source_record(logdir)["commit"] == "a" * 40
