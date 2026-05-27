#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/src/models/}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/src/models/}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/src/models/}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/src/models/}"

MODEL="${MODEL:-openai/gpt-4o-mini}"
SEED="${SEED:-0}"
EXEC_WORKERS="${EXEC_WORKERS:-64}"
CHECKPOINT_MULTIPLIER="${CHECKPOINT_MULTIPLIER:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-$(( EXEC_WORKERS * CHECKPOINT_MULTIPLIER ))}"
RLM_MAX_DEPTH="${RLM_MAX_DEPTH:-5}"
RLM_MAX_ITERATIONS="${RLM_MAX_ITERATIONS:-8}"
ROOT_DIR="${ROOT_DIR:-src/reasoning_benchmark/debug/rlm_full_gpt4omini_depth5_async_${EXEC_WORKERS}_$(date +%Y%m%d_%H%M%S)}"

uv run --python 3.11 python -m src.reasoning_benchmark.cli \
  --root "$ROOT_DIR" \
  --backend openrouter \
  --model "$MODEL" \
  --seed "$SEED" \
  --hf_dtype bfloat16 \
  --hf_device_map auto \
  --clrs_samples 500 \
  --vllm_tensor_parallel 8 \
  --n 60 \
  --digits 2 4 6 8 10 12 14 16 18 20 \
  --kinds spp bsp edp gcp gcp_d tsp tsp_d ksp msp clrs30 add sub mul lcs rod knap ilp_assign ilp_partition ilp_prod \
  --temperature 0.1 \
  --top_p 0.90 \
  --checkpoint_every "$CHECKPOINT_EVERY" \
  --exec_workers "$EXEC_WORKERS" \
  --only_rlm \
  --rlm_nl \
  --rlm_code \
  --rlm_backend openrouter \
  --rlm_environment docker \
  --rlm_repo_path /tmp/rlm \
  --rlm_max_depth "$RLM_MAX_DEPTH" \
  --rlm_max_iterations "$RLM_MAX_ITERATIONS"
