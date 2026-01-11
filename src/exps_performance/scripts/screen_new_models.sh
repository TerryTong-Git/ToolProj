#!/bin/bash
# Screening script for 20 NEW models to identify code > NL trend
# Uses small sample sizes (n=5) for quick screening

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Source environment variables (API keys etc.)
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/src/models/}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/src/models/}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/src/models/}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/src/models/}"

# Screening parameters - small sample for quick evaluation
N_SAMPLES="${N_SAMPLES:-5}"
SEED=0
DIGITS="2 4 6"
KINDS="add sub mul lcs knap rod"
BATCH_SIZE=16

# 20 NEW models to screen (excluding previously tested)
MODELS=(
    # Anthropic - closed, expect strong trend
    "anthropic/claude-sonnet-4"
    "anthropic/claude-opus-4"

    # DeepSeek - strong reasoning models
    "deepseek/deepseek-chat-v3-0324"
    "deepseek/deepseek-r1"
    "deepseek/deepseek-r1-distill-llama-70b"
    "deepseek/deepseek-r1-distill-qwen-14b"

    # Meta Llama - open source baseline
    "meta-llama/llama-3.3-70b-instruct"
    "meta-llama/llama-4-maverick"
    "meta-llama/llama-4-scout"

    # Qwen - strong code models
    "qwen/qwen3-235b-a22b"
    "qwen/qwen3-32b"
    "qwen/qwq-32b"

    # Mistral - varied sizes
    "mistralai/mistral-large-2411"
    "mistralai/mistral-medium-3.1"
    "mistralai/codestral-2508"
    "mistralai/mixtral-8x22b-instruct"

    # Google - closed, expect strong trend
    "google/gemini-2.5-pro"
    "google/gemini-2.0-flash-001"
    "google/gemini-3-pro-preview"

    # OpenAI
    "openai/gpt-4o"
)

echo "=============================================="
echo "Screening 20 NEW models for Code > NL trend"
echo "=============================================="
echo "Parameters:"
echo "  N_SAMPLES: ${N_SAMPLES}"
echo "  SEED: ${SEED}"
echo "  DIGITS: ${DIGITS}"
echo "  KINDS: ${KINDS}"
echo "  Total models: ${#MODELS[@]}"
echo "=============================================="

RESULTS_ROOT="src/exps_performance/results_screening"
mkdir -p "${RESULTS_ROOT}"

# Track successful and failed models
SUCCESSFUL_MODELS=()
FAILED_MODELS=()

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Screening: ${MODEL}"
    echo "========================================"

    # Run experiment
    if uv run --no-sync python src/exps_performance/main.py \
        --root "${RESULTS_ROOT}" \
        --backend openrouter \
        --model "${MODEL}" \
        --n "${N_SAMPLES}" \
        --digits ${DIGITS} \
        --kinds ${KINDS} \
        --exec_code \
        --batch_size "${BATCH_SIZE}" \
        --checkpoint_every "${BATCH_SIZE}" \
        --seed "${SEED}" \
        --controlled_sim \
        --resume \
        --exec_workers 4 \
        --max_tokens 1024 2>&1; then
        SUCCESSFUL_MODELS+=("${MODEL}")
        echo "[SUCCESS] ${MODEL}"
    else
        FAILED_MODELS+=("${MODEL}")
        echo "[FAILED] ${MODEL}"
    fi
done

echo ""
echo "=============================================="
echo "Screening Complete"
echo "=============================================="
echo "Successful models (${#SUCCESSFUL_MODELS[@]}):"
for m in "${SUCCESSFUL_MODELS[@]}"; do
    echo "  - ${m}"
done
echo ""
echo "Failed models (${#FAILED_MODELS[@]}):"
for m in "${FAILED_MODELS[@]}"; do
    echo "  - ${m}"
done
echo ""
echo "Results saved to: ${RESULTS_ROOT}"
echo ""
echo "Next step: Run logistic MI analysis on results"
echo "  uv run --no-sync python -m src.exps_logistic.main \\"
echo "    --results-dir ${RESULTS_ROOT} \\"
echo "    --label gamma --feats tfidf --bits"
