"""Git-backed task coordination for independent HPC facility checkouts.

Each mutation is a commit to the checkout's current branch. Non-fast-forward
pushes are retried against the latest branch tip, providing optimistic
serialization between facility checkouts.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_BOARD_DIRECTORY = ".hpc-coordination"
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_BOARD_README = """# HPC Task Coordination Board

This directory is a Git-backed coordination channel on the repository's
current branch. Each task is stored at `tasks/<task-id>.json`; commits are the
audit trail. Facility workers fetch/push the same branch through the configured
Git remote. A claim is protected by an expiring lease; do not run a task unless
the claim succeeded.

Task `command` and `artifact_uri` values are descriptive metadata only. This
board does not dispatch jobs, transfer scientific datasets, or store secrets.
Use shared facility storage or an approved artifact service for large outputs.
"""


class TaskBoardError(RuntimeError):
    """A coordination-board operation could not be completed safely."""


class TaskBoard:
    """Read and mutate task records on the current remote Git branch."""

    def __init__(
        self,
        repository: str | Path = ".",
        *,
        remote: str = "origin",
        branch: str | None = None,
        retries: int = 5,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.remote = remote
        self.remote_url = self._repository_git("remote", "get-url", remote).strip()
        if not self.remote_url:
            raise TaskBoardError(f"Git remote {remote!r} has no URL")
        self.branch = branch or self._repository_git("symbolic-ref", "--short", "HEAD").strip()
        if not self.branch:
            raise TaskBoardError("coordination requires a named current branch")
        self.retries = retries

    def initialize(self) -> None:
        """Create the coordination directory and its usage note if absent."""

        self._mutate_board(lambda root: self._ensure_readme(root))

    def add_task(
        self,
        task_id: str,
        *,
        title: str,
        facility: str,
        command: str,
        artifact_uri: str | None = None,
        details: str | None = None,
    ) -> dict[str, Any]:
        """Publish a pending task. Command text is recorded, never executed."""

        self._validate_task_id(task_id)
        for field_name, value in (("title", title), ("facility", facility), ("command", command)):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        now = self._now()
        task = {
            "schema_version": 1,
            "task_id": task_id,
            "title": title.strip(),
            "facility": facility.strip(),
            "command": command.strip(),
            "artifact_uri": artifact_uri,
            "details": details,
            "status": "pending",
            "owner": None,
            "lease_expires_utc": None,
            "attempts": 0,
            "created_utc": now,
            "updated_utc": now,
            "events": [{"at_utc": now, "actor": "coordinator", "event": "created"}],
        }

        def create(root: Path) -> None:
            self._ensure_readme(root)
            path = self._task_path(root, task_id)
            if path.exists():
                raise TaskBoardError(f"task {task_id!r} already exists")
            self._write_task(path, task)

        self._mutate_board(create)
        return task

    def list_tasks(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Read the current remote snapshot and optionally filter by status."""

        with tempfile.TemporaryDirectory(prefix="matsim-task-board-") as directory:
            root = Path(directory)
            if not self._checkout_latest(root, allow_missing=True):
                return []
            tasks = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((root / _BOARD_DIRECTORY / "tasks").glob("*.json"))
            ]
        if status is not None:
            tasks = [task for task in tasks if task["status"] == status]
        return tasks

    def claim(self, task_id: str, *, worker: str, lease_minutes: int = 60) -> dict[str, Any]:
        """Claim a pending task or reclaim one whose lease has expired."""

        if lease_minutes < 1:
            raise ValueError("lease_minutes must be at least 1")
        return self._change_task(
            task_id,
            worker,
            "claimed",
            lambda task, now: self._claim_update(task, worker, now, lease_minutes),
        )

    def heartbeat(
        self, task_id: str, *, worker: str, lease_minutes: int = 60
    ) -> dict[str, Any]:
        """Extend the active worker's lease."""

        if lease_minutes < 1:
            raise ValueError("lease_minutes must be at least 1")

        def update(task: dict[str, Any], now: datetime) -> None:
            self._require_owner(task, worker, now)
            task["lease_expires_utc"] = (now + timedelta(minutes=lease_minutes)).isoformat()

        return self._change_task(task_id, worker, "heartbeat", update)

    def finish(self, task_id: str, *, worker: str, success: bool, note: str | None = None):
        """Mark the owned, unexpired task completed or failed."""

        event = "completed" if success else "failed"
        status = event

        def update(task: dict[str, Any], now: datetime) -> None:
            self._require_owner(task, worker, now)
            task["status"] = status
            task["lease_expires_utc"] = None
            if note:
                task["result_note"] = note

        return self._change_task(task_id, worker, event, update)

    def _change_task(
        self,
        task_id: str,
        actor: str,
        event: str,
        update: Callable[[dict[str, Any], datetime], None],
    ) -> dict[str, Any]:
        self._validate_task_id(task_id)
        if not actor.strip():
            raise ValueError("worker must not be empty")
        result: dict[str, Any] = {}

        def change(root: Path) -> None:
            nonlocal result
            path = self._task_path(root, task_id)
            if not path.is_file():
                raise TaskBoardError(f"unknown task {task_id!r}")
            task = json.loads(path.read_text(encoding="utf-8"))
            now = datetime.now(UTC)
            update(task, now)
            task["updated_utc"] = now.isoformat()
            task["events"].append(
                {"at_utc": now.isoformat(), "actor": actor, "event": event}
            )
            self._write_task(path, task)
            result = task

        self._mutate_board(change)
        return result

    @staticmethod
    def _claim_update(
        task: dict[str, Any], worker: str, now: datetime, lease_minutes: int
    ) -> None:
        status = task["status"]
        lease = task.get("lease_expires_utc")
        expired = lease is not None and datetime.fromisoformat(lease) <= now
        if status == "running" and not expired:
            raise TaskBoardError(
                f"task {task['task_id']!r} is leased by {task['owner']!r} until {lease}"
            )
        if status not in {"pending", "running"}:
            raise TaskBoardError(f"task {task['task_id']!r} is {status}, not claimable")
        task["status"] = "running"
        task["owner"] = worker
        task["attempts"] += 1
        task["lease_expires_utc"] = (now + timedelta(minutes=lease_minutes)).isoformat()

    @staticmethod
    def _require_owner(task: dict[str, Any], worker: str, now: datetime) -> None:
        if task["status"] != "running" or task.get("owner") != worker:
            raise TaskBoardError(f"task {task['task_id']!r} is not leased by {worker!r}")
        lease = task.get("lease_expires_utc")
        if lease is None or datetime.fromisoformat(lease) <= now:
            raise TaskBoardError(f"lease for task {task['task_id']!r} has expired")

    def _mutate_board(self, mutation: Callable[[Path], None]) -> None:
        last_error: TaskBoardError | None = None
        for _attempt in range(self.retries):
            with tempfile.TemporaryDirectory(prefix="matsim-task-board-") as directory:
                root = Path(directory)
                self._checkout_latest(root, allow_missing=False)
                mutation(root)
                self._git(root, "add", _BOARD_DIRECTORY)
                diff = self._git(root, "diff", "--cached", "--quiet", check=False)
                if diff.returncode == 0:
                    return
                self._git(
                    root,
                    "-c",
                    "user.name=HPC Task Coordinator",
                    "-c",
                    "user.email=hpc-task-coordinator@users.noreply.github.com",
                    "commit",
                    "-m",
                    "Update HPC task board",
                )
                push = self._git(
                    root,
                    "push",
                    "origin",
                    f"HEAD:refs/heads/{self.branch}",
                    check=False,
                )
                if push.returncode == 0:
                    return
                last_error = TaskBoardError("remote update rejected or unavailable; retrying")
        raise last_error or TaskBoardError("could not update the task board")

    def _checkout_latest(self, root: Path, *, allow_missing: bool) -> bool:
        self._git(root, "init", "--quiet", "-b", "board")
        self._git(root, "remote", "add", "origin", self.remote_url)
        refs = self._git(
            root,
            "ls-remote",
            "--exit-code",
            "--heads",
            "origin",
            f"refs/heads/{self.branch}",
            check=False,
        )
        if refs.returncode == 2 and not refs.stdout.strip():
            if allow_missing:
                return False
            raise TaskBoardError(f"coordination branch {self.branch!r} does not exist")
        if refs.returncode != 0:
            raise TaskBoardError("cannot read the coordination branch from the configured remote")
        self._git(root, "fetch", "--quiet", "--depth=1", "origin", self.branch)
        self._git(root, "checkout", "--quiet", "-B", "board", "FETCH_HEAD")
        return True

    @staticmethod
    def _ensure_readme(root: Path) -> None:
        board = root / _BOARD_DIRECTORY
        (board / "tasks").mkdir(parents=True, exist_ok=True)
        readme = board / "README.md"
        if not readme.exists():
            readme.write_text(_BOARD_README, encoding="utf-8")

    @staticmethod
    def _task_path(root: Path, task_id: str) -> Path:
        return root / _BOARD_DIRECTORY / "tasks" / f"{task_id}.json"

    @staticmethod
    def _write_task(path: Path, task: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        if not _TASK_ID.fullmatch(task_id):
            raise ValueError("task_id must be 1-64 safe ASCII letters, digits, '.', '_' or '-'")

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _repository_git(self, *args: str) -> str:
        return self._git(self.repository, *args).stdout

    def _git(
        self,
        cwd: Path,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        if check and result.returncode:
            stderr = result.stderr.replace(self.remote_url, "<remote>").strip()
            raise TaskBoardError(stderr or f"git {args[0]} failed")
        return result


__all__ = ["TaskBoard", "TaskBoardError"]
