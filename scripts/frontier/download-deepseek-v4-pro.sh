#!/bin/bash
# Download DeepSeek-V4-Pro (instruct, MoE, 49B active params)
# FP8 weights: ~800 GB download — run with nohup, takes several hours.
#
# Usage:
#   bash run-scripts/download-deepseek-v4-pro.sh
#
# Requires HUGGING_FACE_HUB_TOKEN to be set (model is gated).
# Request access at: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro

HF=/lustre/orion/mat746/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72/bin/hf
MODEL_DIR=/lustre/orion/mat746/proj-shared/models

if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HUGGING_FACE_HUB_TOKEN is not set." >&2
  echo "  export HUGGING_FACE_HUB_TOKEN=hf_..." >&2
  exit 1
fi

nohup $HF download deepseek-ai/DeepSeek-V4-Pro \
  --repo-type model \
  --local-dir "$MODEL_DIR/DeepSeek-V4-Pro" \
  > "$MODEL_DIR/deepseek-v4-pro_download.log" 2>&1 &
echo "DeepSeek-V4-Pro PID: $!"
echo "Log: $MODEL_DIR/deepseek-v4-pro_download.log"
echo "Note: large download, expect several hours."
