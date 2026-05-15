"""mpiexec-aware HuggingFace TP loader for Aurora multi-node jobs.

Launched by ``mpiexec -n (N_NODES * 12)`` (one rank per PVC tile).  Uses
HuggingFace transformers' built-in tensor-parallel planner
(``tp_plan="auto"``) on top of ``torch.distributed`` with the **oneCCL**
backend (XPU).  Single-rank fallback uses ``device_map="auto"`` so the
same loader can be run via ``python`` for a quick sanity check.

Required env (set by the wrapping shell script):
    MATSIM_HF_MODEL_PATH      — local model directory (offline)
    HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
    CCL_KVS_MODE=mpi          — oneCCL bootstraps via MPI

PALS / mpiexec exports the following automatically; we mirror them into
the names torch.distributed expects:
    PALS_RANKID         -> RANK
    PALS_LOCAL_RANKID   -> LOCAL_RANK
    PALS_NODEID         -> (used for diagnostics)
    PALS_LOCAL_SIZE     -> LOCAL_WORLD_SIZE  (== 12 on Aurora)
"""
from __future__ import annotations

import os
import sys
import time

# Intel Extension for PyTorch must be imported BEFORE torch.distributed init
# so the ``ccl`` backend gets registered.
import torch  # noqa: F401  (need the symbol before IPEX imports)
try:
    import intel_extension_for_pytorch  # noqa: F401
    import oneccl_bindings_for_pytorch  # noqa: F401  (registers "ccl" backend)
except Exception as exc:  # pragma: no cover — fail fast with a clear message
    print(f"ERROR: missing Intel/oneCCL bindings: {exc}", file=sys.stderr, flush=True)
    raise

import torch.distributed as dist


# ── env normalisation ────────────────────────────────────────────────────────
def _normalise_env() -> None:
    """Map PALS_* (mpiexec) env vars to the names torch expects."""
    mappings = [
        ("RANK",             ["PMI_RANK", "PALS_RANKID"]),
        ("LOCAL_RANK",       ["PMI_LOCAL_RANK", "PALS_LOCAL_RANKID"]),
        ("WORLD_SIZE",       ["PMI_SIZE", "PALS_NRANKS"]),
        ("LOCAL_WORLD_SIZE", ["PALS_LOCAL_SIZE"]),
    ]
    for dst, candidates in mappings:
        if dst in os.environ:
            continue
        for src in candidates:
            if src in os.environ:
                os.environ[dst] = os.environ[src]
                break


def _setup_dist() -> tuple[int, int, int]:
    _normalise_env()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # One rank per PVC tile.  ZE_AFFINITY_MASK already pins this rank's tile
    # (set in the launcher), so visible-device 0 is always the right device.
    torch.xpu.set_device(0)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="ccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
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
        print(f"[r0] torch={torch.__version__}  xpu_available={torch.xpu.is_available()}  "
              f"world_size={world_size}  local_rank={local_rank}", flush=True)
        print(f"[r0] model_dir={model_dir}", flush=True)
        print(f"[r0] node={os.environ.get('PALS_NODEID', '?')}  "
              f"hostname={os.uname().nodename}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    load_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if world_size > 1:
        # transformers ≥ 4.50: distribute the model across every rank of the
        # active process group via its integrated TP planner.
        load_kwargs["tp_plan"] = "auto"
    else:
        load_kwargs["device_map"] = "auto"

    _log(rank, f"loading model with kwargs={ {k: str(v) for k, v in load_kwargs.items()} }")
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    _log(rank, f"model loaded in {time.time() - t0:.1f}s")

    device = "xpu:0" if torch.xpu.is_available() else "cpu"
    prompt = (
        os.environ.get("MATSIM_MN_PROMPT")
        or "What is 2 + 2? Answer in one sentence."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    _log(rank, "running generate ...")
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("MATSIM_MN_MAX_NEW_TOKENS", "128")),
            do_sample=False,
        )
    _log(rank, f"generate finished in {time.time() - t0:.1f}s")

    if rank == 0:
        text = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print("=== Response ===", flush=True)
        print(text, flush=True)
        print("================", flush=True)
        print("Multi-node TP run PASSED", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
