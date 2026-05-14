# Six-Model Local Benchmark on Frontier

This workflow uses only local model weights and vLLM endpoints (no Ollama).

## 1) Download models locally

Submit:

```bash
sbatch scripts/frontier/download-open-models-frontier.sh
```

Optional subset:

```bash
MODEL_IDS="Qwen/Qwen2.5-14B-Instruct meta-llama/Llama-3.1-8B-Instruct" \
  sbatch scripts/frontier/download-open-models-frontier.sh
```

## 2) Serve models with vLLM

Start one vLLM endpoint per model and export URL env vars expected by the spec:

- `MATSIM_VLLM_QWEN72_BASE_URL`
- `MATSIM_VLLM_QWEN14_BASE_URL`
- `MATSIM_VLLM_LLAMA70_BASE_URL`
- `MATSIM_VLLM_LLAMA8_BASE_URL`
- `MATSIM_VLLM_MIXTRAL_BASE_URL`
- `MATSIM_VLLM_DEEPSEEK32_BASE_URL`

If all models are served by a single endpoint, set only:

- `MATSIM_VLLM_BASE_URL`

## 3) Run benchmark with vLLM-only spec

```bash
BENCHMARK_PROMPT="Search for a Pb-free halide double perovskite candidate and justify stability." \
BENCHMARK_SPEC_FILE="scripts/frontier/six_model_specs.vllm-only.json" \
sbatch scripts/frontier/job-six-model-benchmark-frontier.sh
```

## 4) Artifacts

Outputs are written under:

- `/lustre/orion/mat746/proj-shared/runs/six-model-bench-<jobid>/`
