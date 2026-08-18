"""Fine-tuning cost / timing instrumentation for the AL pipeline.

Captures wall-clock time, GPU-hours, peak GPU memory and (trainable/total)
parameter counts for each fine-tuning run so the paper can report the
*computational cost* of adapting each backend alongside its accuracy.

Two capture paths are supported because the two backends run differently:

* **In-process** (HydraGNN routed-head fine-tune): use :func:`track_cost`,
  which measures wall time and reads ``torch.cuda`` peak-memory counters
  directly.
* **Subprocess** (UMA via the ``fairchem`` CLI): wrap the subprocess in
  :class:`GpuMemorySampler` to sample ``nvidia-smi`` peak memory, and time it
  with :func:`track_cost` (which still records wall time / GPU-hours even
  though torch counters in the parent process stay at zero).

All fields are optional; each caller fills in what it can measure.
"""

from __future__ import annotations

import json
import logging
import shutil
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class CostReport:
    """Serialisable record of one fine-tuning run's computational cost."""

    model_backend: str  # "uma" | "hydragnn"
    dataset_label: str
    base_model: str = ""
    dataset_path: str = ""
    n_train_frames: int = 0
    n_val_frames: int = 0
    epochs: int = 0
    steps: int = 0
    num_gpus: int = 0
    device: str = ""
    trainable_params: int = 0
    total_params: int = 0
    frozen_params: int = 0
    wall_time_s: float = 0.0
    gpu_hours: float = 0.0
    peak_gpu_mem_gb: float = 0.0
    hostname: str = ""
    timestamp: str = ""
    extra: dict = field(default_factory=dict)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        log.info("Wrote cost report -> %s", path)
        return path


def count_parameters(module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts for a torch module."""
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


@contextmanager
def track_cost(report: CostReport, *, reset_cuda_peak: bool = True):
    """Time a code block and record wall time, GPU-hours and torch peak memory.

    ``report.num_gpus`` should be set before/inside the block so GPU-hours can
    be computed. ``peak_gpu_mem_gb`` is only populated from in-process
    ``torch.cuda`` counters; for subprocess training use
    :class:`GpuMemorySampler` and it will overwrite this field if larger.
    """
    torch_cuda = None
    try:  # torch is optional at import time
        import torch

        if torch.cuda.is_available():
            torch_cuda = torch.cuda
    except Exception:  # noqa: BLE001
        torch_cuda = None

    if torch_cuda is not None and reset_cuda_peak:
        import contextlib

        with contextlib.suppress(Exception):
            torch_cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    try:
        yield report
    finally:
        report.wall_time_s = round(time.perf_counter() - start, 3)
        if report.num_gpus:
            report.gpu_hours = round(report.wall_time_s * report.num_gpus / 3600.0, 6)
        if torch_cuda is not None:
            try:
                peak_bytes = max(
                    torch_cuda.max_memory_allocated(i) for i in range(torch_cuda.device_count())
                )
                peak_gb = peak_bytes / (1024**3)
                report.peak_gpu_mem_gb = max(report.peak_gpu_mem_gb, round(peak_gb, 3))
            except Exception:  # noqa: BLE001
                pass
        report.hostname = socket.gethostname()
        report.timestamp = datetime.now(timezone.utc).isoformat()


class GpuMemorySampler:
    """Background sampler of peak GPU memory via ``nvidia-smi``.

    Intended for subprocess-based training (e.g. the UMA ``fairchem`` CLI)
    where in-process ``torch.cuda`` counters cannot see the child's usage. Use
    as a context manager, then read :attr:`peak_gb`. Silently no-ops if
    ``nvidia-smi`` is unavailable.
    """

    def __init__(self, interval_s: float = 5.0, enabled: bool = True):
        self.interval_s = interval_s
        self.peak_gb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # ``nvidia-smi`` reports node-total memory (all processes), so on a
        # shared login node a CPU run would pick up other users' usage. Only
        # sample when explicitly enabled (i.e. an actual GPU training run).
        self._available = enabled and shutil.which("nvidia-smi") is not None

    def _sample_once(self) -> float:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            vals = [float(x.strip()) for x in out.stdout.splitlines() if x.strip()]
            return max(vals) / 1024.0 if vals else 0.0  # MiB -> GiB
        except Exception:  # noqa: BLE001
            return 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_gb = max(self.peak_gb, self._sample_once())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuMemorySampler:
        if self._available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5)
