#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/src/models/}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/src/models/}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/src/models/}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/src/models/}"

SHOT_COUNTS=(0 1 2 3 4 5)
MODEL_SPECS=(
"claude-haiku-4.5|anthropic/claude-haiku-4.5"
"gemini-2.5-flash|google/gemini-2.5-flash"
"mixtral-8x22b-instruct|mistralai/mixtral-8x22b-instruct"
)

SUBSET_FRACTION="${SUBSET_FRACTION:-0.25}"
N_SAMPLES="${N_SAMPLES:-1000000}"
MAX_PER_KIND="${MAX_PER_KIND:-1000000}"
CONCURRENCY="${CONCURRENCY:-32}"
SEED="${SEED:-42}"

for MODEL_SPEC in "${MODEL_SPECS[@]}"; do
  SOURCE_MODEL="${MODEL_SPEC%%|*}"
  MODEL="${MODEL_SPEC##*|}"
  for N_SHOTS in "${SHOT_COUNTS[@]}"; do
    uv run --no-sync python src/exps_functional/run_translation_additivity.py \
      --model "${MODEL}" \
      --source_model "${SOURCE_MODEL}" \
      --subset_fraction "${SUBSET_FRACTION}" \
      --n_shots "${N_SHOTS}" \
      --n_samples "${N_SAMPLES}" \
      --max_per_kind "${MAX_PER_KIND}" \
      --concurrency "${CONCURRENCY}" \
      --seed "${SEED}"
  done
done
