#!/bin/bash
HF=/lustre/orion/mat746/proj-shared/HydraGNN/installation_DOE_supercomputers/HydraGNN-Installation-Frontier-ROCm72/hydragnn_venv_rocm72/bin/hf
MODEL_DIR=/lustre/orion/mat746/proj-shared/models

nohup $HF download Qwen/Qwen3.6-27B \
  --repo-type model \
  --local-dir $MODEL_DIR/Qwen3.6-27B \
  > $MODEL_DIR/qwen36-27b_download.log 2>&1 &
echo "Qwen3.6-27B PID: $!"

nohup $HF download Qwen/Qwen3.6-35B-A3B \
  --repo-type model \
  --local-dir $MODEL_DIR/Qwen3.6-35B-A3B \
  > $MODEL_DIR/qwen36-35b-a3b_download.log 2>&1 &
echo "Qwen3.6-35B-A3B PID: $!"

nohup $HF download google/gemma-4-31B-it \
  --repo-type model \
  --local-dir $MODEL_DIR/gemma-4-31B-it \
  > $MODEL_DIR/gemma4-31b_download.log 2>&1 &
echo "gemma-4-31B-it PID: $!"

nohup $HF download google/gemma-4-26B-A4B-it \
  --repo-type model \
  --local-dir $MODEL_DIR/gemma-4-26B-A4B-it \
  > $MODEL_DIR/gemma4-26b-a4b_download.log 2>&1 &
echo "gemma-4-26B-A4B-it PID: $!"

nohup $HF download HuggingFaceTB/SmolLM3-3B \
  --repo-type model \
  --local-dir $MODEL_DIR/SmolLM3-3B \
  > $MODEL_DIR/smollm3-3b_download.log 2>&1 &
echo "SmolLM3-3B PID: $!"
