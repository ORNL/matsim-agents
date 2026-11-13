"""Reusable spawned workers with parent-enforced task deadlines."""

from __future__ import annotations

import math
import multiprocessing
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any


class TaskDeadlineExceeded(TimeoutError):
    """The worker was killed before its task finished."""


def _serve(connection: Connection, initialize: Callable, execute: Callable) -> None:
    try:
        state = initialize()
        connection.send(("ready", None))
        while True:
            task = connection.recv()
            try:
                result = execute(task, state)
                connection.send(("result", result))
            except Exception:
                connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class DeadlineWorker:
    """Reuse process-local state; discard the entire process after an overdue task."""

    def __init__(self, initialize: Callable, execute: Callable, *, startup_seconds: float = 120):
        self.initialize = initialize
        self.execute = execute
        self.startup_seconds = startup_seconds
        if not math.isfinite(startup_seconds) or startup_seconds <= 0:
            raise ValueError("Worker startup deadline must be finite and positive")
        self.process: BaseProcess | None = None
        self.connection: Connection | None = None

    def _receive(self, seconds: float) -> tuple[str, Any]:
        if self.connection is None:
            raise RuntimeError("Deadline worker has no connection")
        if not self.connection.poll(seconds):
            self.close()
            raise TaskDeadlineExceeded(f"Worker exceeded {seconds:g} seconds")
        try:
            return self.connection.recv()
        except EOFError as error:
            self.close()
            raise RuntimeError("Deadline worker exited without a result") from error

    def run(self, task: Any, *, seconds: float) -> Any:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Task deadline must be finite and positive")
        if self.process is None:
            context = multiprocessing.get_context("spawn")
            self.connection, child = context.Pipe()
            self.process = context.Process(
                target=_serve, args=(child, self.initialize, self.execute)
            )
            self.process.start()
            child.close()
            try:
                kind, detail = self._receive(self.startup_seconds)
            except TaskDeadlineExceeded as error:
                raise RuntimeError("Deadline worker initialization timed out") from error
            if kind != "ready":
                self.close()
                raise RuntimeError(f"Deadline worker initialization failed: {detail}")
        if self.connection is None:
            raise RuntimeError("Deadline worker did not initialize its connection")
        self.connection.send(task)
        kind, detail = self._receive(seconds)
        if kind == "error":
            raise RuntimeError(f"Deadline worker task failed:\n{detail}")
        if kind != "result":
            raise RuntimeError(f"Unexpected deadline worker response: {kind}")
        return detail

    def close(self) -> None:
        if self.process is not None:
            # SIGKILL is necessary: a CUDA/native call can ignore cooperative cancellation.
            if self.process.is_alive():
                self.process.kill()
            self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
