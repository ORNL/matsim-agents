"""Exercise Torch, PyG, and torch_scatter operations on one selected device."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import torch


def require_device(tensor: torch.Tensor, device: torch.device, operation: str) -> None:
    if tensor.device != device:
        raise RuntimeError(f"{operation} returned {tensor.device}, expected {device}")


def qualify(device_name: str) -> dict:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA/ROCm GPU visible; run inside a GPU allocation")
        torch.cuda.set_device(0)
    elif device_name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("No XPU visible; run inside an Aurora compute allocation")
        torch.xpu.set_device(0)
    device = torch.device(f"{device_name}:0") if device_name != "cpu" else torch.device("cpu")

    probe = torch.tensor([2.0, 3.0], device=device)
    torch.testing.assert_close((probe * probe).sum().cpu(), torch.tensor(13.0))

    from torch_geometric.nn import global_add_pool
    from torch_geometric.utils import scatter as pyg_scatter
    from torch_scatter import scatter, scatter_add

    source = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, requires_grad=True)
    index = torch.tensor([0, 1, 0, 1], device=device)
    expected_sum = torch.tensor([4.0, 6.0])
    pyg_sum = pyg_scatter(source, index, dim=0, dim_size=2, reduce="sum")
    torch.testing.assert_close(pyg_sum.cpu(), expected_sum)
    require_device(pyg_sum, device, "PyG scatter sum")
    pyg_sum.sum().backward()
    torch.testing.assert_close(source.grad.cpu(), torch.ones(4))
    require_device(source.grad, device, "PyG scatter gradient")

    features = source.reshape(-1, 1)
    pooled = global_add_pool(features, index)
    torch.testing.assert_close(pooled.cpu(), expected_sum.reshape(-1, 1))
    require_device(pooled, device, "PyG global_add_pool")

    expected_matrix = expected_sum.reshape(-1, 1)
    outputs = {
        "torch_scatter_add": scatter_add(features, index, dim=0, dim_size=2),
        "torch_scatter_sum": scatter(
            features, index, dim=0, dim_size=2, reduce="sum"
        ),
        "torch_scatter_mean": scatter(
            features, index, dim=0, dim_size=2, reduce="mean"
        ),
    }
    for name in ("torch_scatter_add", "torch_scatter_sum"):
        torch.testing.assert_close(outputs[name].cpu(), expected_matrix)
        require_device(outputs[name], device, name)
    torch.testing.assert_close(
        outputs["torch_scatter_mean"].cpu(), torch.tensor([[2.0], [3.0]])
    )
    require_device(outputs["torch_scatter_mean"], device, "torch_scatter_mean")
    source.grad = None
    (outputs["torch_scatter_add"].sum() + outputs["torch_scatter_mean"].sum()).backward()
    torch.testing.assert_close(source.grad.cpu(), torch.full((4,), 1.5))
    require_device(source.grad, device, "torch_scatter reduction gradient")

    if device_name == "cuda":
        torch.cuda.synchronize()
        accelerator = torch.cuda.get_device_name(0)
    elif device_name == "xpu":
        torch.xpu.synchronize()
        accelerator = torch.xpu.get_device_name(0)
    else:
        accelerator = platform.processor() or "CPU"

    return {
        "status": "PASSED",
        "device": str(device),
        "accelerator": accelerator,
        "torch": torch.__version__,
        "torch_geometric": importlib.metadata.version("torch-geometric"),
        "torch_scatter": importlib.metadata.version("torch-scatter"),
        "operations": [
            "torch_arithmetic",
            "pyg_scatter_sum_and_backward",
            "pyg_global_add_pool",
            "torch_scatter_add",
            "torch_scatter_sum",
            "torch_scatter_mean",
            "torch_scatter_reductions_backward",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "xpu"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = qualify(args.device)
    result["host"] = platform.node()
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())