#!/bin/bash
# Script to run missing model/seed combinations for exps_performance
# Generated: 2026-01-17
#
# Target: Each model should have 3 seeds (0, 1, 2)
#
# Missing experiments (18 total):
#   - anthropic/claude-haiku-4.5: seeds 1, 2
#   - anthropic/claude-opus-4: seeds 1, 2
#   - anthropic/claude-sonnet-4: seeds 0, 1, 2
#   - deepseek/deepseek-chat-v3-0324: seeds 0, 1, 2
#   - deepseek/deepseek-r1: seeds 0, 1, 2
#   - google/gemini-2.0-flash-001: seed 2
#   - openai/gpt-4o-mini: seeds 0, 2
#   - mistralai/ministral-14b-2512: seeds 1, 2
#   - qwen/qwen-2.5-coder-32b-instruct: seed 0

set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# Environment setup
export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/src/models/}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/src/models/}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/src/models/}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/src/models/}"

# Common parameters
COMMON_ARGS="--root src/exps_performance/ \
  --backend openrouter \
  --hf_dtype bfloat16 \
  --hf_device_map auto \
  --clrs_samples 500 \
  --vllm_tensor_parallel 8 \
  --n 60 --digits 2 4 6 8 10 12 14 16 18 20 \
  --kinds spp bsp edp gcp gcp_d tsp tsp_d ksp msp clrs30 add sub mul lcs rod knap ilp_assign ilp_partition ilp_prod \
  --temperature 0.1 --top_p 0.90 \
  --exec_code --batch_size 256 --checkpoint_every 256 --controlled_sim --resume --exec_workers 4"

run_experiment() {
    local model=$1
    local seed=$2
    echo "========================================"
    echo "Running: ${model} seed=${seed}"
    echo "========================================"
    uv run --no-sync python src/exps_performance/main.py \
        ${COMMON_ARGS} \
        --model "${model}" \
        --seed "${seed}"
}

# Claude models
run_experiment "anthropic/claude-haiku-4.5" 1
run_experiment "anthropic/claude-haiku-4.5" 2

run_experiment "anthropic/claude-opus-4" 1
run_experiment "anthropic/claude-opus-4" 2

run_experiment "anthropic/claude-sonnet-4" 0
run_experiment "anthropic/claude-sonnet-4" 1
run_experiment "anthropic/claude-sonnet-4" 2

# DeepSeek models
run_experiment "deepseek/deepseek-chat-v3-0324" 0
run_experiment "deepseek/deepseek-chat-v3-0324" 1
run_experiment "deepseek/deepseek-chat-v3-0324" 2

run_experiment "deepseek/deepseek-r1" 0
run_experiment "deepseek/deepseek-r1" 1
run_experiment "deepseek/deepseek-r1" 2

# Google models
run_experiment "google/gemini-2.0-flash-001" 2

# OpenAI models
run_experiment "openai/gpt-4o-mini" 0
run_experiment "openai/gpt-4o-mini" 2

# Mistral models
run_experiment "mistralai/ministral-14b-2512" 1
run_experiment "mistralai/ministral-14b-2512" 2

# Qwen models
run_experiment "qwen/qwen-2.5-coder-32b-instruct" 0

echo "========================================"
echo "All 18 experiments complete!"
echo "========================================"
