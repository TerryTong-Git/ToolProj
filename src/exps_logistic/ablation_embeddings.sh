#!/bin/bash
# Embedding Model Ablation Study for Logistic Regression MI Estimation
# Uses OpenRouter API for all embedding models
#
# OpenRouter Embedding Models:
#   1. openai/text-embedding-3-small (1536d) - Fast, cost-effective
#   2. openai/text-embedding-3-large (3072d) - Highest quality
#   3. qwen/qwen3-embedding-8b - Multilingual, strong retrieval
#   4. qwen/qwen3-embedding-0.6b - Lightweight Qwen variant
#   5. google/gemini-embedding-001 - Google's embedding model
#   6. mistralai/mistral-embed-2312 - Mistral's embedding model
#
# Top 3 LLMs (by code accuracy/MI from previous experiments):
#   - anthropic/claude-haiku-4.5
#   - mistralai/ministral-14b-2512
#   - openai/gpt-4o-mini

set -e

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable is required"
    echo "Set it with: export OPENROUTER_API_KEY='your-key-here'"
    exit 1
fi

RESULTS_DIR="/nlpgpu/data/terry/ToolProj/src/exps_performance/results"
OUTPUT_DIR="src/exps_logistic/ablation_outputs"
mkdir -p "$OUTPUT_DIR"

# Top 3 LLM models by MI performance
LLMS=(
    "anthropic/claude-haiku-4.5"
    "mistralai/ministral-14b-2512"
    "openai/gpt-4o-mini"
)

# OpenRouter embedding models
EMBED_MODELS=(
    "openai/text-embedding-3-small"
    "openai/text-embedding-3-large"
    "qwen/qwen3-embedding-8b"
    "qwen/qwen3-embedding-0.6b"
    "google/gemini-embedding-001"
    "mistralai/mistral-embed-2312"
)

SEEDS=(0 1 2)
REPS=("code" "nl")

echo "Starting Embedding Ablation Study (OpenRouter)..."
echo "========================================"
echo "Embedding Models: ${EMBED_MODELS[*]}"
echo "LLMs: ${LLMS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Representations: ${REPS[*]}"
echo ""

TOTAL_EXPS=$((${#EMBED_MODELS[@]} * ${#LLMS[@]} * ${#SEEDS[@]} * ${#REPS[@]}))
echo "Total experiments to run: $TOTAL_EXPS"
echo ""

run_experiment() {
    local embed_model=$1
    local llm=$2
    local rep=$3
    local seed=$4

    echo "[$(date '+%H:%M:%S')] Running: embed=$embed_model, llm=$llm, rep=$rep, seed=$seed"

    uv run --no-sync python -m src.exps_logistic.main \
        --results-dir "$RESULTS_DIR" \
        --models "$llm" \
        --rep "$rep" \
        --label gamma \
        --no-cv \
        --bits \
        --feats openrouter \
        --embed-model "$embed_model" \
        --max_iter 20 \
        --C 0.5 \
        --seed "$seed" \
        --kinds-preset extended \
        --batch 64
}

# Run all experiments
COUNTER=0
for embed in "${EMBED_MODELS[@]}"; do
    echo ""
    echo "=== Embedding Model: $embed ==="
    for llm in "${LLMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for rep in "${REPS[@]}"; do
                COUNTER=$((COUNTER + 1))
                echo "[$COUNTER/$TOTAL_EXPS]"
                run_experiment "$embed" "$llm" "$rep" "$seed"
            done
        done
    done
done

echo ""
echo "========================================"
echo "Embedding Ablation Study Complete!"
echo "Total experiments run: $TOTAL_EXPS"
echo ""
echo "To aggregate results, run:"
echo "  uv run python -m src.exps_logistic.aggregate_ablation_results"
