import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest

from matsim_agents.execution.task_board import TaskBoard, TaskBoardError

_BRANCH = "research/numerical-results-refresh"


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_remote(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    repository = tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)
    subprocess.run(
        ["git", "init", "--quiet", "-b", _BRANCH, str(repository)], check=True
    )
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "README.md").write_text("project files stay in place\n", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(repository, "commit", "--quiet", "-m", "Initial project commit")
    _git(repository, "remote", "add", "shared", str(remote))
    _git(repository, "push", "--quiet", "-u", "shared", _BRANCH)
    return repository, remote


def _board(git_remote: tuple[Path, Path]) -> TaskBoard:
    repository, _remote = git_remote
    return TaskBoard(repository, remote="shared")


def _add_task(board: TaskBoard, task_id: str = "task-1") -> None:
    board.add_task(
        task_id,
        title="Test task",
        facility="perlmutter",
        command="echo this is metadata only",
    )


def _set_remote_lease_expired(remote: Path, task_id: str) -> None:
    clone = remote.parent / "lease-editor"
    _git(remote.parent, "clone", "--quiet", str(remote), str(clone))
    _git(clone, "checkout", "--quiet", _BRANCH)
    task_path = clone / ".hpc-coordination" / "tasks" / f"{task_id}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["lease_expires_utc"] = datetime(2000, 1, 1, tzinfo=UTC).isoformat()
    task_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _git(clone, "config", "user.name", "Test User")
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "add", ".hpc-coordination")
    _git(clone, "commit", "--quiet", "-m", "Expire task lease")
    _git(clone, "push", "--quiet", "origin", _BRANCH)


def test_board_uses_current_branch_and_namespaced_records(git_remote):
    board = _board(git_remote)
    board.initialize()
    _add_task(board)

    repository, remote = git_remote
    assert board.branch == _BRANCH
    assert board.list_tasks()[0]["task_id"] == "task-1"
    assert _git(remote, "rev-parse", f"refs/heads/{_BRANCH}")
    tree = _git(remote, "ls-tree", "-r", "--name-only", _BRANCH).splitlines()
    assert ".hpc-coordination/tasks/task-1.json" in tree
    assert "tasks/task-1.json" not in tree
    subprocess.run(
        [
            "git",
            "-C",
            str(remote),
            "cat-file",
            "-e",
            f"{_BRANCH}:.hpc-coordination/tasks/task-1.json",
        ],
        check=True,
    )
    assert (repository / "README.md").read_text(encoding="utf-8") == "project files stay in place\n"


def test_mutation_refuses_to_create_missing_remote_branch(git_remote):
    repository, remote = git_remote
    _git(repository, "checkout", "--quiet", "-b", "unpublished-work")
    board = TaskBoard(repository, remote="shared")

    with pytest.raises(TaskBoardError, match="does not exist"):
        board.initialize()

    assert not _git(remote, "branch", "--list", "unpublished-work")


def test_task_lifecycle_enforces_owner_and_expiring_lease(git_remote):
    board = _board(git_remote)
    _add_task(board)

    board.claim("task-1", worker="worker-a", lease_minutes=30)
    with pytest.raises(TaskBoardError, match="leased by"):
        board.claim("task-1", worker="worker-b")
    with pytest.raises(TaskBoardError, match="not leased"):
        board.heartbeat("task-1", worker="worker-b")

    _set_remote_lease_expired(git_remote[1], "task-1")
    reclaimed = board.claim("task-1", worker="worker-b")
    assert reclaimed["owner"] == "worker-b"
    assert reclaimed["attempts"] == 2
    with pytest.raises(TaskBoardError, match="not leased"):
        board.finish("task-1", worker="worker-a", success=True)

    board.finish("task-1", worker="worker-b", success=True, note="done")
    assert board.list_tasks()[0]["status"] == "completed"


def test_competing_claims_serialize_on_latest_remote_tip(git_remote):
    board = _board(git_remote)
    _add_task(board)
    barrier = Barrier(2)
    boards = [board, _board(git_remote)]

    for candidate in boards:
        checkout_latest = candidate._checkout_latest
        first_checkout = [True]

        def synchronized_checkout(
            root, *, allow_missing, checkout=checkout_latest, first=first_checkout
        ):
            found = checkout(root, allow_missing=allow_missing)
            if first[0]:
                first[0] = False
                barrier.wait(timeout=10)
            return found

        candidate._checkout_latest = synchronized_checkout

    def claim(candidate: TaskBoard, worker: str) -> str:
        try:
            candidate.claim("task-1", worker=worker)
            return "claimed"
        except TaskBoardError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda args: claim(*args),
                zip(boards, ("worker-a", "worker-b"), strict=True),
            )
        )

    assert sorted(outcomes) == ["claimed", "rejected"]
    task = board.list_tasks()[0]
    assert task["status"] == "running"
    assert task["owner"] in {"worker-a", "worker-b"}