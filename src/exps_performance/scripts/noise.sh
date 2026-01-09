#!/usr/bin/env bash
set -euo pipefail

# Sweep noise experiments with Gaussian & Uniform noise levels and emit plot-ready JSON.
# Override defaults via environment variables: MODEL, BACKEND, NOISE_TYPES, NOISE_LEVELS, ROOT_DIR, N_SAMPLES, DIGITS_LIST, KINDS, SEED, UV_CMD.

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

UV_CMD="${UV_CMD:-uv}"
BACKEND="${BACKEND:-openrouter}"
MODEL="${MODEL:-anthropic/claude-haiku-4.5}"
NOISE_TYPES="${NOISE_TYPES:-gaussian uniform numerical textual structural irrelevant}"
NOISE_LEVELS="${NOISE_LEVELS:-0.0 0.25 0.5 0.75 1.0}"
ROOT_DIR="${ROOT_DIR:-src/exps_performance/results_noise}"
N_SAMPLES="${N_SAMPLES:-4}"
DIGITS_LIST="${DIGITS_LIST:-2 4}"
KINDS="${KINDS:-add sub mul lcs}"
SEED="${SEED:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-8}"
EXEC_WORKERS="${EXEC_WORKERS:-2}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMP="${TEMP:-0.0}"
TOP_P="${TOP_P:-1.0}"

echo "[noise.sh] Running noise experiment"
echo "  model=${MODEL}"
echo "  backend=${BACKEND}"
echo "  noise_types=${NOISE_TYPES}"
echo "  noise_levels=${NOISE_LEVELS}"
echo "  root_dir=${ROOT_DIR}"

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

echo "[noise.sh] Done."
