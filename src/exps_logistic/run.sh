#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/src/exps_performance/results}"


MODELS="meta-llama/llama-3.1-405b-instruct"
MODELS_ARG=""
if [ -n "$MODELS" ]; then
    MODELS_ARG="--models $MODELS"
fi

uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep code --device cuda --hf-batch 8  \
    --label gamma --no-cv  --bits --feats hf-cls --embed-model google-bert/bert-base-uncased --max_iter 20 --C 0.5 --kinds-preset extended

uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep nl --device cuda --hf-batch 1  \
    --label gamma --no-cv --bits --feats hf-cls --embed-model google-bert/bert-base-uncased --max_iter 20 --C 0.5 --kinds-preset extended

uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep code --device cuda  \
    --label gamma --no-cv --bits --feats tfidf --kinds-preset extended

uv run --no-sync python -m src.exps_logistic.main --results-dir "$RESULTS_DIR" $MODELS_ARG --rep nl --device cuda  \
    --label gamma --no-cv --bits --feats tfidf --kinds-preset extended

# neulab/codebert-python
