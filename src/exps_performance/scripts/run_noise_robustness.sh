#!/usr/bin/env bash
# Noise Robustness Experiment: Code vs NL
# Experiment 10: Test whether code representation degrades more slowly than NL under noise
#
# Usage:
#   bash src/exps_performance/scripts/run_noise_robustness.sh pilot   # Quick calibration
#   bash src/exps_performance/scripts/run_noise_robustness.sh full    # Full experiment
#   bash src/exps_performance/scripts/run_noise_robustness.sh analyze # Analysis only

set -euo pipefail

# Load environment variables
if [ -f .env ]; then
    source .env
fi

MODE="${1:-pilot}"
MODEL="${2:-anthropic/claude-haiku-4.5}"
SEED="${3:-1}"

echo "============================================"
echo "Noise Robustness Experiment: Code vs NL"
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo "============================================"

case "$MODE" in
    pilot)
        echo "Running PILOT mode for quick calibration..."
        uv run python -m src.exps_performance.noise_runner \
            --model "$MODEL" \
            --backend openrouter \
            --seed "$SEED" \
            --pilot
        ;;

    full)
        echo "Running FULL experiment (14 kinds, 4 digit levels, 7 sigma levels, 6 noise types)..."
        uv run python -m src.exps_performance.noise_runner \
            --model "$MODEL" \
            --backend openrouter \
            --seed "$SEED" \
            --n 10
        ;;

    full-all-seeds)
        echo "Running FULL experiment for seeds 1, 2, 3..."
        for s in 1 2 3; do
            echo "--- Seed $s ---"
            uv run python -m src.exps_performance.noise_runner \
                --model "$MODEL" \
                --backend openrouter \
                --seed "$s" \
                --n 10
        done
        ;;

    validation)
        echo "Running validation with multiple models..."
        MODELS=(
            "openai/gpt-4o-mini"
            "google/gemini-2.5-flash"
        )
        for m in "${MODELS[@]}"; do
            echo "--- Model: $m ---"
            uv run python -m src.exps_performance.noise_runner \
                --model "$m" \
                --backend openrouter \
                --seed "$SEED" \
                --n 10 \
                --kinds add sub binary_search dijkstra gcp \
                --digits_list 2 4 8
        done
        ;;

    analyze)
        echo "Running analysis..."
        RESULTS_DIR="${2:-src/exps_performance/results_noise_v3}"
        uv run python src/exps_performance/analysis_noise_code_vs_nl.py "$RESULTS_DIR"
        ;;

    analyze-pilot)
        echo "Running analysis on pilot results..."
        uv run python src/exps_performance/analysis_noise_code_vs_nl.py src/exps_performance/results_noise_pilot
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 {pilot|full|full-all-seeds|validation|analyze|analyze-pilot} [model] [seed]"
        exit 1
        ;;
esac

echo "Done!"
