#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/src/models/}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/src/models/}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/src/models/}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/src/models/}"

export MODEL="${MODEL:-openai/gpt-4o-mini}"
export SEED="${SEED:-0}"
export EXEC_WORKERS="${EXEC_WORKERS:-32}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-128}"
export SUBSAMPLE_FRACTION="${SUBSAMPLE_FRACTION:-0.25}"
export RLM_MAX_DEPTH="${RLM_MAX_DEPTH:-5}"
export RLM_MAX_ITERATIONS="${RLM_MAX_ITERATIONS:-8}"
export ROOT_DIR="${ROOT_DIR:-src/exps_performance/debug/rlm_subset25_${MODEL##*/}_async${EXEC_WORKERS}_$(date +%Y%m%d_%H%M%S)}"

uv run --python 3.11 python - <<'PY'
import math
import os
from collections import defaultdict

import src.exps_performance.main as main_mod
from src.exps_performance.dataset import make_dataset as original_make_dataset
from src.exps_performance.main import Args, run


def make_dataset_subset(
    kinds,
    n=3,
    digits_list=None,
    gsm_samples=500,
    clrs_samples=500,
):
    if digits_list is None:
        digits_list = [32]

    full = list(
        original_make_dataset(
            kinds,
            n=n,
            digits_list=digits_list,
            gsm_samples=gsm_samples,
            clrs_samples=clrs_samples,
        )
    )

    fraction = float(os.environ["SUBSAMPLE_FRACTION"])
    buckets = defaultdict(list)
    for q in full:
        buckets[(str(getattr(q, "kind", "")), int(getattr(q, "digits", -1)))].append(q)

    sampled = []
    for key in sorted(buckets):
        items = buckets[key]
        keep = max(1, math.ceil(len(items) * fraction))
        sampled.extend(items[:keep])

    for i, q in enumerate(sampled):
        setattr(q, "original_pos", i)

    print(
        f"Subsampled {len(sampled)} / {len(full)} examples "
        f"across {len(buckets)} (kind, digit) buckets at fraction={fraction:.2f}."
    )
    return sampled


main_mod.make_dataset = make_dataset_subset

args = Args(
    root=os.environ["ROOT_DIR"],
    n=60,
    digits_list=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    kinds=[
        "spp",
        "bsp",
        "edp",
        "gcp",
        "gcp_d",
        "tsp",
        "tsp_d",
        "ksp",
        "msp",
        "clrs30",
        "add",
        "sub",
        "mul",
        "lcs",
        "rod",
        "knap",
        "ilp_assign",
        "ilp_partition",
        "ilp_prod",
    ],
    seed=int(os.environ["SEED"]),
    backend="openrouter",
    model=os.environ["MODEL"],
    hf_dtype="bfloat16",
    hf_device_map="auto",
    clrs_samples=500,
    vllm_tensor_parallel=8,
    temperature=0.1,
    top_p=0.90,
    checkpoint_every=int(os.environ["CHECKPOINT_EVERY"]),
    exec_workers=int(os.environ["EXEC_WORKERS"]),
    only_rlm=True,
    rlm_nl=True,
    rlm_code=True,
    rlm_backend="openrouter",
    rlm_environment="docker",
    rlm_repo_path="/tmp/rlm",
    rlm_max_depth=int(os.environ["RLM_MAX_DEPTH"]),
    rlm_max_iterations=int(os.environ["RLM_MAX_ITERATIONS"]),
)

run(args)
PY
