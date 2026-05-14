"""Torchrun-aware HuggingFace smoke loader for Perlmutter multi-node tests.

Launched as one rank per GPU (so 4 ranks/node × N nodes via torchrun under
srun). Uses transformers' built-in tensor-parallel sharding (``tp_plan="auto"``)
which dispatches the model across all ranks via ``torch.distributed`` (NCCL).

Single-node fallback: if ``WORLD_SIZE == 1`` we use ``device_map="auto"`` so
this same loader also works with the existing single-node smoke launcher.

Required env (set by the wrapping shell script):
    MATSIM_HF_MODEL_PATH  — local model directory (offline)
    HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_dist() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank). Initialises ``dist`` if needed."""
    if not _is_distributed():
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(f"[r0] {msg}", flush=True)


def main() -> int:
    model_dir = os.environ.get("MATSIM_HF_MODEL_PATH")
    if not model_dir or not os.path.isdir(model_dir):
        print(f"ERROR: MATSIM_HF_MODEL_PATH missing or not a directory: {model_dir!r}",
              file=sys.stderr, flush=True)
        return 2

    rank, world_size, local_rank = _setup_dist()

    if rank == 0:
        print(f"[r0] torch={torch.__version__}  cuda={torch.cuda.is_available()}  "
              f"world_size={world_size}  local_rank={local_rank}", flush=True)
        print(f"[r0] model_dir={model_dir}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    load_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }

    if world_size > 1:
        # transformers ≥ 4.50 ships an integrated TP planner that builds a
        # device_map across all ranks of the active process group.
        load_kwargs["tp_plan"] = "auto"
    else:
        # Single-node, single-process: shard locally across visible GPUs.
        load_kwargs["device_map"] = "auto"

    _log(rank, f"loading model with kwargs={ {k: str(v) for k, v in load_kwargs.items()} }")
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    _log(rank, f"model loaded in {time.time() - t0:.1f}s")

    # All ranks must run forward; rank 0 alone will print.
    prompt = "What is 2 + 2? Answer in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    _log(rank, "running generate ...")
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )
    _log(rank, f"generate finished in {time.time() - t0:.1f}s")

    if rank == 0:
        text = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        print("=== Response ===", flush=True)
        print(text, flush=True)
        print("================", flush=True)
        print("Smoke test PASSED", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
