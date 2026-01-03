#!/bin/sh
#SBATCH --job-name=deepseek
#SBATCH --output=log/deepseekOut.txt
#SBATCH --error=log/deepseekErr.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=GA102GL     
#SBATCH --time=1-6:00:00
export UV_CACHE_DIR="/nlpgpu/data/terry/ToolProj/src/models/"
export HF_HOME="/nlpgpu/data/terry/ToolProj/src/models/"
export HF_DATASETS_CACHE="/nlpgpu/data/terry/ToolProj/src/models/"
export HF_HUB_CACHE="/nlpgpu/data/terry/ToolProj/src/models/"

SEEDS=(1)
MODELS=( #7B Models
# openai/gpt-oss-120b:free
# allenai/olmo-3-32b-think:free
# mistralai/ministral-14b-2512
# meta-llama/llama-3-70b-instruct
# deepseek/deepseek-r1
# openai/gpt-oss-120b
# google/gemini-2.5-flash
# openai/gpt-4o-mini
anthropic/claude-haiku-4.5
# openai/gpt-oss-120b
# openai/gpt-oss-120b
# deepseek/deepseek-r1
# meta-llama/llama-3.1-405b-instruct
# openai/gpt-oss-20b
# qwen/qwen-2.5-coder-32b-instruct
# deepseek/deepseek-r1-distill-llama-70b
# allenai/olmo-2-0325-32b-instruct
# openai/gpt-5-nano
# qwen/qwen-2.5-72b-instruct
# microsoft/phi-4-reasoning-plus
# google/gemini-2.0-flash-lite-001
# meta-llama/llama-4-scout
# anthropic/claude-haiku-4.5
# z-ai/glm-4.6
# openai/gpt-5.1-codex
# google/gemini-2.5-flash
# x-ai/grok-code-fast-1
# openai/gpt-5-mini
# mistralai/devstral-medium

# openai/gpt-5-mini
# meta-llama/llama-3.3-70b-instruct
# microsoft/phi-4
# openai/o3-mini
# google/gemini-2.5-pro
# anthropic/claude-sonnet-4
# qwen/qwen3-coder-30b-a3b-instruct

# allenai/olmo-3-32b-think:free
# z-ai/glm-4.5-air:free
# qwen/qwen3-coder:free
# moonshotai/kimi-k2:free
# qwen/qwen3-235b-a22b:free
# meta-llama/llama-3.3-70b-instruct:free
# nousresearch/hermes-3-llama-3.1-405b:free
# mistralai/mistral-7b-instruct:free
)

for MODEL in ${MODELS[@]}; do
  for SEED in ${SEEDS[@]}; do
    uv run --no-sync python src/exps_performance/main.py \
      --root src/exps_performance/ \
      --backend openrouter \
      --model ${MODEL} \
      --hf_dtype bfloat16 \
      --hf_device_map auto \
      --gsm_samples 500 --clrs_samples 500 \
      --vllm_tensor_parallel 8 \
      --n 25 --digits 2 4 8 16  --kinds  spp bsp edp gcp gcpd tsp tspd ksp msp gsm8k clrs30 add sub mul lcs rod knap ilp_assign ilp_partition ilp_prod \
      --exec_code --batch_size 64 --checkpoint_every 64 --seed ${SEED} --controlled_sim --resume --exec_workers 4
  done
done



