#!/bin/bash
# Single model smoke test - verify pipeline works end-to-end with 1 sample
# Uses a cheap model (Qwen 3 32B) with minimal samples

set -e  # Exit on error

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
# Go up to the worktree root (experiment-screen-new-models)
WORKTREE_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
# The actual repo root with source code
REPO_ROOT="/Users/terrytong/Documents/CCG/experiment-screen-new-models"
cd "${REPO_ROOT}"

echo "WORKTREE_ROOT: ${WORKTREE_ROOT}"
echo "REPO_ROOT: ${REPO_ROOT}"

# Source environment variables (API keys etc.)
# Check both current dir and main ToolProj
if [ -f ".env" ]; then
    echo "Loading .env from current directory"
    export $(grep -v '^#' .env | xargs)
elif [ -f "/Users/terrytong/Documents/CCG/ToolProj/.env" ]; then
    echo "Loading .env from main ToolProj"
    export $(grep -v '^#' /Users/terrytong/Documents/CCG/ToolProj/.env | xargs)
fi

# Minimal test parameters
N_SAMPLES=1
SEED=0
DIGITS="2"
KINDS="add"
BATCH_SIZE=1
MODEL="qwen/qwen3-32b"

# Results go to a temp directory for this smoke test
RESULTS_DIR="${REPO_ROOT}/experiments/screen_new_models/results/smoke_test"
mkdir -p "${RESULTS_DIR}"

echo "=============================================="
echo "Single Model Smoke Test"
echo "=============================================="
echo "Model: ${MODEL}"
echo "N_SAMPLES: ${N_SAMPLES}"
echo "DIGITS: ${DIGITS}"
echo "KINDS: ${KINDS}"
echo "Results: ${RESULTS_DIR}"
echo "=============================================="

# Run the performance experiment
uv run --no-sync python src/exps_performance/main.py \
    --root "${RESULTS_DIR}" \
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
    --exec_workers 2 \
    --max_tokens 1024

echo ""
echo "=============================================="
echo "Smoke Test Complete"
echo "=============================================="
echo "Results saved to: ${RESULTS_DIR}"

# Verify results exist
if [ -f "${RESULTS_DIR}/${MODEL//\//-}_seed${SEED}/tb/run_*/res.jsonl" ] 2>/dev/null || [ -n "$(find ${RESULTS_DIR} -name 'res.jsonl' 2>/dev/null)" ]; then
    echo "[SUCCESS] Results file created"
    find "${RESULTS_DIR}" -name 'res.jsonl' -exec wc -l {} \;
else
    echo "[WARNING] No res.jsonl found - checking for partial results"
    ls -la "${RESULTS_DIR}/" || true
fi
