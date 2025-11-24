#!/bin/sh
#SBATCH --job-name=deepseek
#SBATCH --output=log/deepseekOut.txt
#SBATCH --error=log/deepseekErr.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=GA102GL     
#SBATCH --time=1-6:00:00

SEEDS=(0)
MODELS=( #7B Models
#  "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
 "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# mistralai/Mistral-7B-Instruct-v0.3

)

for MODEL in ${MODELS[@]}; do
  for SEED in ${SEEDS[@]}; do
    pixi run python src/exps_performance/main.py \
      --root src/exps_performance/ \
      --backend vllm \
      --model ${MODEL} \
      --hf_dtype bfloat16 \
      --hf_device_map auto \
      --vllm_tensor_parallel 8 \
      --n 12 --digits 2 4 8 16  --kinds  spp bsp edp gcp gcpd tsp tspd ksp msp gsm8k clrs30 add sub mul lcs rod knap ilp_assign ilp_partition ilp_prod \
      --exec_code --batch_size 64 --seed ${SEED} --controlled_sim
  done
done



