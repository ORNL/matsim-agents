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
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

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
def seeded_torch(seed: int, *stream: StreamKey) -> Iterator[None]:
    """Seed PyTorch's global generators for the block, then restore them.

    ``torch.manual_seed`` reseeds the CPU, CUDA, and XPU generators, so the state
    of each available accelerator type is saved and restored, also on error.
    For code whose randomness comes from torch's global state (e.g. dropout).
    """
    import torch

    with ExitStack() as saved:
        for device_type in ("cuda", "xpu"):
            backend = getattr(torch, device_type, None)
            if backend is not None and backend.is_available():
                saved.enter_context(
                    torch.random.fork_rng(
                        devices=list(range(backend.device_count())), device_type=device_type
                    )
                )
        saved.enter_context(torch.random.fork_rng(devices=[]))
        torch.manual_seed(stream_seed(seed, *stream))
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
