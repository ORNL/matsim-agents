"""The one place matsim-agents creates random streams.

Every random draw comes from a stream identified by a configured seed and the
purpose of the draw, for example ``numpy_rng(seed, "md", structure_index)``:

* the same seed reproduces every draw;
* streams for different purposes, iterations, or items are independent, so a
  run started at ``seed + 1`` does not replay part of the run at ``seed``;
* adding a stream never shifts existing ones.

A bare non-negative seed (no purpose) is the root stream and equals
``numpy.random.default_rng(seed)`` / ``random.Random(seed)``. Seeds always come
from configuration; nothing outside this module constructs a random generator,
passes a literal seed, or defaults a seed parameter (tests/test_seeding.py).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

import numpy as np

StreamKey = str | int


def stream_seed(seed: int, *stream: StreamKey) -> int:
    """The integer seed of stream ``stream`` of ``seed``.

    Derived seeds are 32-bit (platform independent), which every seeding API accepts,
    including legacy NumPy and ``mace_run_train --seed``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    for key in stream:
        if isinstance(key, bool) or not isinstance(key, str | int):
            raise TypeError(f"stream keys must be strings or integers, got {key!r}")
    if not stream and seed >= 0:
        return seed
    payload = json.dumps([seed, *stream], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def numpy_rng(seed: int, *stream: StreamKey) -> np.random.Generator:
    """NumPy generator for ``stream`` of ``seed``."""
    return np.random.default_rng(stream_seed(seed, *stream))


def python_rng(seed: int, *stream: StreamKey) -> random.Random:
    """Standard-library generator for ``stream`` of ``seed``."""
    return random.Random(stream_seed(seed, *stream))


@contextmanager
def seeded_torch(seed: int, *stream: StreamKey, devices: Iterable[Any]) -> Iterator[None]:
    """Seed PyTorch's CPU generator and those of *devices* for the block, then restore them.

    *devices* are the accelerators the seeded code runs on (``torch.device`` or
    strings such as ``"cuda:1"``); CPU devices in it are ignored. Other
    accelerators are left untouched, so devices reserved for other work get no
    context and keep their random streams. State is restored also on error.
    For code whose randomness comes from torch's global state (e.g. dropout).
    """
    import torch

    by_type: dict[str, set[int]] = {}
    for device in devices:
        device = torch.device(device)
        if device.type == "cpu":
            continue
        backend = getattr(torch, device.type)
        index = device.index if device.index is not None else backend.current_device()
        by_type.setdefault(device.type, set()).add(index)
    value = stream_seed(seed, *stream)
    with ExitStack() as saved:
        saved.enter_context(torch.random.fork_rng(devices=[]))
        for device_type, indices in by_type.items():
            saved.enter_context(
                torch.random.fork_rng(devices=sorted(indices), device_type=device_type)
            )
        torch.random.default_generator.manual_seed(value)
        for device_type, indices in by_type.items():
            backend = getattr(torch, device_type)
            for index in sorted(indices):
                with backend.device(index):
                    backend.manual_seed(value)
        yield


def seed_process(seed: int, *stream: StreamKey) -> None:
    """Seed the global Python, NumPy, and PyTorch generators of this process.

    For training processes whose libraries draw from global state (weight
    initialization, data-loader shuffling, dropout).
    """
    value = stream_seed(seed, *stream)
    random.seed(value)
    np.random.seed(value % 2**32)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(value)
