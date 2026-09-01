#!/bin/bash
# ---------------------------------------------------------------------------
# _rocr_to_hip.sh
#
# Translate ROCR_VISIBLE_DEVICES (which Slurm/srun --gpu-bind sets on
# Frontier) into HIP_VISIBLE_DEVICES (which Ray requires on AMD GPUs).
#
# Usage: prepend to any srun launch line, e.g.
#   srun ... ./_rocr_to_hip.sh "$VENV/bin/python" -m vllm.entrypoints....
#
# Without this, vLLM's lazy `import ray` raises:
#   RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES
# ---------------------------------------------------------------------------
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
  unset ROCR_VISIBLE_DEVICES
fi
exec "$@"
