#!/usr/bin/env bash
set -euo pipefail

PRESETS="${PRESETS:-gpt54_xhigh opus46_max}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-128}"
EXEC_WORKERS="${EXEC_WORKERS:-8}"
SUBSAMPLE_FRACTION="${SUBSAMPLE_FRACTION:-0.25}"
ROOT_PREFIX="${ROOT_PREFIX:-src/reasoning_benchmark/debug/arm123_subset25_reasoning}"

for PRESET in ${PRESETS}; do
  case "${PRESET}" in
    gpt54_xhigh)
      MODEL="openai/gpt-5.4"
      OPENROUTER_REASONING_ENABLED="true"
      OPENROUTER_REASONING_EFFORT="xhigh"
      OPENROUTER_VERBOSITY=""
      ;;
    opus46_max)
      MODEL="anthropic/claude-opus-4.6"
      OPENROUTER_REASONING_ENABLED="true"
      OPENROUTER_REASONING_EFFORT=""
      OPENROUTER_VERBOSITY="max"
      ;;
    *)
      echo "Unknown PRESET=${PRESET}. Supported: gpt54_xhigh opus46_max" >&2
      exit 1
      ;;
  esac

  RUN_ROOT="${ROOT_PREFIX}_${PRESET}_$(date +%Y%m%d_%H%M%S)"

  echo "Running ${PRESET} -> ${MODEL}"
  echo "Results root: ${RUN_ROOT}"

  MODEL="${MODEL}" \
  OPENROUTER_REASONING_ENABLED="${OPENROUTER_REASONING_ENABLED}" \
  OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT}" \
  OPENROUTER_VERBOSITY="${OPENROUTER_VERBOSITY}" \
  RUN_ROOT="${RUN_ROOT}" \
  SEED="${SEED}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
  EXEC_WORKERS="${EXEC_WORKERS}" \
  SUBSAMPLE_FRACTION="${SUBSAMPLE_FRACTION}" \
  uv run --python 3.11 python - <<'PY'
import math
import os
from collections import defaultdict

import src.reasoning_benchmark.runner as main_mod
from src.reasoning_benchmark.task_sets import make_dataset as original_make_dataset
from src.reasoning_benchmark.runner import Args, run


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def make_dataset_subsampled(kinds, n=3, digits_list=None, gsm_samples=500, clrs_samples=500):
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

    print(f"Subsampled {len(sampled)} / {len(full)} examples across {len(buckets)} (kind, digit) buckets.")
    return sampled


main_mod.make_dataset = make_dataset_subsampled

args = Args(
    root=os.environ["RUN_ROOT"],
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
    batch_size=int(os.environ["BATCH_SIZE"]),
    checkpoint_every=int(os.environ["CHECKPOINT_EVERY"]),
    exec_workers=int(os.environ["EXEC_WORKERS"]),
    exec_code=True,
    controlled_sim=False,
    openrouter_reasoning_enabled=_as_bool(os.environ["OPENROUTER_REASONING_ENABLED"]),
    openrouter_reasoning_effort=os.environ["OPENROUTER_REASONING_EFFORT"] or None,
    openrouter_verbosity=os.environ["OPENROUTER_VERBOSITY"] or None,
)

run(args)
PY
done
