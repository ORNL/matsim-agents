"""Stream native workflow output while retaining an invocation log."""

from __future__ import annotations

import codecs
import os
import signal
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO


def _signal_group(process: subprocess.Popen, requested_signal: signal.Signals) -> None:
    try:
        os.killpg(process.pid, requested_signal)
    except ProcessLookupError:
        if process.poll() is None:
            raise


def run_native_command(
    command: list[str],
    output: TextIO,
    emit: Callable[[str], None],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> int:
    """Preserve interactive stdin, stream output, and stop our local child group on interruption."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"},
        cwd=cwd,
        start_new_session=True,
    )
    assert process.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while chunk := os.read(process.stdout.fileno(), 8192):
            text = decoder.decode(chunk)
            output.write(text)
            output.flush()
            emit(text)
        text = decoder.decode(b"", final=True)
        output.write(text)
        output.flush()
        if text:
            emit(text)
        return process.wait()
    except BaseException:
        _signal_group(process, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)
                process.wait()
        raise
    finally:
        process.stdout.close()
