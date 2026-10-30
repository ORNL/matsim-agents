"""Supply process-group-local ranks to Aurora's TorchComms backend."""

from __future__ import annotations

import inspect
import os
import sys
from threading import RLock
from typing import Any


def install_torchcomms_group_rank_patch(c10d_module: Any) -> None:
    """Set TorchComms rank variables while PyTorch creates each process group."""
    trace = os.environ.get("MATSIM_TORCHCOMMS_TRACE") == "1"
    original = c10d_module._new_process_group_helper
    signature = inspect.signature(original)
    if not {"group_rank", "group_size"}.issubset(signature.parameters):
        raise RuntimeError(
            "Aurora TorchComms compatibility requires PyTorch's process-group "
            "helper to expose group_rank and group_size"
        )

    lock = RLock()
    if trace:
        print(
            f"[torchcomms-patch] installed pid={os.getpid()}",
            file=sys.stderr,
            flush=True,
        )

    def with_group_rank(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        group_rank = bound.arguments.get("group_rank")
        group_size = bound.arguments.get("group_size")
        if group_rank is None or group_size is None:
            return original(*args, **kwargs)

        if trace:
            print(
                "[torchcomms-patch] group "
                f"pid={os.getpid()} rank={group_rank} size={group_size}",
                file=sys.stderr,
                flush=True,
            )
        with lock:
            previous = {
                "TORCHCOMM_RANK": os.environ.get("TORCHCOMM_RANK"),
                "TORCHCOMM_SIZE": os.environ.get("TORCHCOMM_SIZE"),
            }
            os.environ["TORCHCOMM_RANK"] = str(group_rank)
            os.environ["TORCHCOMM_SIZE"] = str(group_size)
            try:
                return original(*args, **kwargs)
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    c10d_module._new_process_group_helper = with_group_rank