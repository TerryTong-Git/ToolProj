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
anthropic/claude-haiku-4.5
qwen/qwen-2.5-coder-32b-instruct
mistralai/ministral-14b-2512
meta-llama/llama-3.1-405b-instruct
openai/gpt-4o-mini
google/gemini-2.5-flash
mistralai/codestral-2508 
mistralai/mistral-large-2411 
google/gemini-2.0-flash-001 
mistralai/mixtral-8x22b-instruct

# allenai/olmo-2-0325-32b-instruct
# anthropic/claude-opus-4
# anthropic/claude-sonnet-4
# deepseek/deepseek-chat-v3-0324
# deepseek/deepseek-r1
# deepseek/deepseek-r1-distill-llama-70b
# deepseek/deepseek-r1-distill-qwen-14b
# google/gemini-2.5-pro
# google/gemini-3-pro-preview
# meta-llama/llama-3-70b-instruct
# meta-llama/llama-3.3-70b-instruct
# meta-llama/llama-4-maverick
# meta-llama/llama-4-scout
# microsoft/phi-4
# microsoft/phi-4-reasoning-plus
# mistralai/codestral-2508
# mistralai/devstral-medium
# mistralai/ministral-14b-2512
# mistralai/mistral-large-2411
# mistralai/mistral-medium-3.1
# mistralai/mixtral-8x22b-instruct
# openai/gpt-4o
# openai/gpt-5-mini
# openai/gpt-5-nano
# openai/gpt-5.1-codex
# openai/gpt-oss-120b
# openai/gpt-oss-20b
# openai/o3-mini
# qwen/qwen-2.5-72b-instruct
# qwen/qwen-2.5-coder-32b-instruct
# qwen/qwen3-235b-a22b
# qwen/qwen3-32b
# qwen/qwen3-coder-30b-a3b-instruct
# qwen/qwq-32b
# x-ai/grok-4-fast
# x-ai/grok-code-fast-1
# z-ai/glm-4.6

)
for SEED in ${SEEDS[@]}; do
  for MODEL in ${MODELS[@]}; do
    uv run --no-sync python src/exps_performance/main.py \
      --root src/exps_performance/ \
      --backend openrouter \
      --model ${MODEL} \
      --hf_dtype bfloat16 \
      --hf_device_map auto \
      --clrs_samples 500 \
      --vllm_tensor_parallel 8 \
      --n 60 --digits 2 4 6 8 10 12 14 16 18 20  --kinds spp bsp edp gcp gcp_d tsp tsp_d ksp msp clrs30 add sub mul lcs rod knap ilp_assign ilp_partition ilp_prod \
      --temperature 0.1 --top_p 0.90 \
      --exec_code --batch_size 256 --checkpoint_every 256 --seed ${SEED} --controlled_sim --resume --exec_workers 4
  done
done



