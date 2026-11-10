"""Supply process-group-local ranks to Aurora's TorchComms backend."""

from __future__ import annotations

import inspect
import os
import sys
from threading import RLock
from typing import Any


def install_torchcomms_group_rank_patch(
    c10d_module: Any, distributed_module: Any | None = None
) -> None:
    """Apply Aurora's TorchComms process-group compatibility patches."""
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

    def rewrite_xccl_backend(arguments: dict[str, Any], operation: str) -> None:
        backend = arguments.get("backend")
        if backend is not None and str(backend).lower() == "xccl":
            arguments["backend"] = "cpu:gloo,xpu:xccl"
            if trace:
                print(
                    f"[torchcomms-patch] {operation} backend "
                    f"{backend} -> {arguments['backend']}",
                    file=sys.stderr,
                    flush=True,
                )

    original_init = c10d_module.init_process_group
    init_signature = inspect.signature(original_init)

    def with_cpu_backend(*args: Any, **kwargs: Any) -> Any:
        bound = init_signature.bind_partial(*args, **kwargs)
        rewrite_xccl_backend(bound.arguments, "init")
        return original_init(*bound.args, **bound.kwargs)

    c10d_module.init_process_group = with_cpu_backend
    original_new_group = c10d_module.new_group
    new_group_signature = inspect.signature(original_new_group)

    def with_cpu_group_backend(*args: Any, **kwargs: Any) -> Any:
        bound = new_group_signature.bind_partial(*args, **kwargs)
        rewrite_xccl_backend(bound.arguments, "new_group")
        backend = bound.arguments.get("backend")
        if backend is not None and str(backend).lower() == "gloo":
            bound.arguments["backend"] = "cpu:gloo,xpu:xccl"
        return original_new_group(*bound.args, **bound.kwargs)

    c10d_module.new_group = with_cpu_group_backend

    if hasattr(c10d_module, "split_group"):
        original_split_group = c10d_module.split_group
        split_group_signature = inspect.signature(original_split_group)

        def with_cpu_split_backend(*args: Any, **kwargs: Any) -> Any:
            bound = split_group_signature.bind_partial(*args, **kwargs)
            backend = bound.arguments.get("backend")
            if backend is not None:
                backend_text = str(backend).lower()
                backend_parts = [part.strip() for part in backend_text.split(",")]
                if backend_text == "gloo":
                    bound.arguments["backend"] = "cpu:gloo,xpu:xccl"
                    backend_text = "cpu:gloo,xpu:xccl"
                    backend_parts = ["cpu:gloo", "xpu:xccl"]
                requests_xccl = backend_text == "xccl" or "xpu:xccl" in backend_parts
                requests_cpu = any(part.startswith("cpu:") for part in backend_parts)
                if requests_xccl and not requests_cpu:
                    bound.arguments["backend"] = (
                        "cpu:gloo,xpu:xccl" if backend_text == "xccl" else f"cpu:gloo,{backend}"
                    )
                    backend_text = str(bound.arguments["backend"]).lower()
                    backend_parts = [part.strip() for part in backend_text.split(",")]
                if set(backend_parts) == {"cpu:gloo", "xpu:xccl"}:
                    bound.arguments["backend"] = None
                if trace:
                    print(
                        "[torchcomms-patch] split_group backend "
                        f"{backend} -> {bound.arguments.get('backend')}",
                        file=sys.stderr,
                        flush=True,
                    )
            return original_split_group(*bound.args, **bound.kwargs)

        c10d_module.split_group = with_cpu_split_backend
    if distributed_module is not None:
        distributed_module.init_process_group = with_cpu_backend
        distributed_module.new_group = with_cpu_group_backend
        if hasattr(c10d_module, "split_group"):
            distributed_module.split_group = with_cpu_split_backend
