#!/bin/sh
#SBATCH --job-name=gemma
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=GA102GL     
#SBATCH --time=1-6:00:00

SEEDS=(0)
MODELS=( #7B Models
#  "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
#  "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
 "google/gemma-2-9b-it"
)
for MODEL in ${MODELS[@]}; do
  for SEED in ${SEEDS[@]}; do
    pixi run python -m pdb src/exps_performance/main.py \
      --root src/exps_performance/ \
      --backend running \
      --model ${MODEL} \
      --hf_dtype float16 \
      --hf_device_map auto \
      --vllm_tensor_parallel 8 \
      --n 8 --digits 2 4 8 12 --kinds  add tsp ilp_production \
      --exec_code --batch_size 64 --seed ${SEED} --controlled_sim
  done
done


