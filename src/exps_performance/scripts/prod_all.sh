#!/bin/sh
#SBATCH --job-name=deepseek
#SBATCH --output=log/deepseekOut.txt
#SBATCH --error=log/deepseekErr.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=GA102GL
#SBATCH --time=1-6:00:00

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/src/models/}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/src/models/}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/src/models/}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/src/models/}"

SEEDS=(0 1 2)
MODELS=( #7B Models
# anthropic/claude-haiku-4.5
# qwen/qwen-2.5-coder-32b-instruct
# mistralai/ministral-14b-2512
# meta-llama/llama-3.1-405b-instruct
# openai/gpt-4o-mini
# google/gemini-2.5-flash
mistralai/codestral-2508 
mistralai/mistral-large-2411 
google/gemini-2.0-flash-001 
mistralai/mixtral-8x22b-instruct
)
for SEED in ${SEEDS[@]}; do
  for MODEL in ${MODELS[@]}; do
    uv run --no-sync python src/exps_performance/main.py \
      --root src/exps_performance/ \
      --backend openrouter \
      --model ${MODEL} \
      --hf_dtype bfloat16 \
      --hf_device_map auto \
      --gsm_samples 500 --clrs_samples 500 \
      --vllm_tensor_parallel 8 \
      --n 30 --digits 2 4 6 8 10 12 14 16 18 20  --kinds  spp bsp edp gcp gcp_d tsp tsp_d ksp msp gsm8k clrs30 add sub mul lcs rod knap ilp_assign ilp_partition ilp_prod \
      --exec_code --batch_size 64 --checkpoint_every 64 --seed ${SEED} --controlled_sim --resume --exec_workers 4
  done
done



