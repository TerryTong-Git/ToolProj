#!/bin/sh
#SBATCH --job-name=llama
#SBATCH --output=log/llamaOut.txt
#SBATCH --error=log/llamaErr.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=GA102GL     
#SBATCH --time=1-6:00:00

SEEDS=(0)
MODELS=( #7B Models
meta-llama/Llama-3.1-8B-Instruct
# "google/gemma-2-9b-it"
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


