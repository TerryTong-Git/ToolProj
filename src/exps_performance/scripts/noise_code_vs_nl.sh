#!/usr/bin/env bash
set -euo pipefail

# Noise experiment: Code Execution (Arm3) vs NL (Arm1) comparison
# Sweeps across multiple models, all algorithmic problem types, all noise types.
# Override defaults via environment variables.

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

UV_CMD="${UV_CMD:-uv}"
BACKEND="${BACKEND:-openrouter}"

# Models (cross-provider)
MODELS=(
    "anthropic/claude-haiku-4.5"
    "anthropic/claude-sonnet-4"
    "openai/gpt-4o-mini"
    "google/gemini-2.5-flash"
)

# Problem types by category
ARITHMETIC="add sub mul"
DP="lcs knap rod"
ILP="ilp_assign ilp_prod ilp_partition"
NPHARD="tsp gcp spp bsp edp msp ksp tspd gcpd"
CLRS="clrs30"

KINDS="${KINDS:-${ARITHMETIC} ${DP} ${ILP} ${NPHARD} ${CLRS}}"

# Experiment parameters
NOISE_TYPES="${NOISE_TYPES:-gaussian uniform numerical textual structural irrelevant}"
NOISE_LEVELS="${NOISE_LEVELS:-0.0 0.5 1.0}"
N_SAMPLES="${N_SAMPLES:-10}"
DIGITS_LIST="${DIGITS_LIST:-2 4}"
ROOT_DIR="${ROOT_DIR:-src/exps_performance/results_noise_code_vs_nl}"
SEEDS=(${SEEDS:-0 1 2})

# Runtime parameters
BATCH_SIZE="${BATCH_SIZE:-16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-16}"
EXEC_WORKERS="${EXEC_WORKERS:-4}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMP="${TEMP:-0.0}"
TOP_P="${TOP_P:-1.0}"

echo "[noise_code_vs_nl.sh] Starting Code vs NL noise experiments"
echo "  models: ${MODELS[*]}"
echo "  kinds: ${KINDS}"
echo "  noise_types: ${NOISE_TYPES}"
echo "  noise_levels: ${NOISE_LEVELS}"
echo "  n_samples: ${N_SAMPLES}"
echo "  seeds: ${SEEDS[*]}"
echo "  output: ${ROOT_DIR}"

mkdir -p "${ROOT_DIR}"

for SEED in "${SEEDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo ""
        echo "========================================"
        echo "Running: model=${MODEL} seed=${SEED}"
        echo "========================================"

        ${UV_CMD} run --no-sync python -m src.exps_performance.noise_runner \
            --backend "${BACKEND}" \
            --model "${MODEL}" \
            --sigma ${NOISE_LEVELS} \
            --noise_types ${NOISE_TYPES} \
            --root "${ROOT_DIR}" \
            --n "${N_SAMPLES}" \
            --digits_list ${DIGITS_LIST} \
            --kinds ${KINDS} \
            --batch_size "${BATCH_SIZE}" \
            --checkpoint_every "${CHECKPOINT_EVERY}" \
            --exec_workers "${EXEC_WORKERS}" \
            --max_tokens "${MAX_TOKENS}" \
            --temperature "${TEMP}" \
            --top_p "${TOP_P}" \
            --seed "${SEED}"
    done
done

echo ""
echo "[noise_code_vs_nl.sh] All experiments complete."
echo "Results saved to: ${ROOT_DIR}"
