#!/bin/sh
#SBATCH --job-name=mistral
#SBATCH --output=log/MistralOut.txt
#SBATCH --error=log/MistralErr.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=GA102GL     
#SBATCH --time=1-6:00:00
export HF_HOME="/nlpgpu/data/terry/ToolProj/src/models/"
export HF_DATASETS_CACHE="/nlpgpu/data/terry/ToolProj/src/models/"
export HF_HUB_CACHE="/nlpgpu/data/terry/ToolProj/src/models/"


SEEDS=(0)
MODELS=( #7B Models
#  "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
#  "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
mistralai/Mistral-Small-24B-Instruct-2501
)
for MODEL in ${MODELS[@]}; do
  for SEED in ${SEEDS[@]}; do
    pixi run python src/exps_performance/main.py \
      --root src/exps_performance/ \
      --backend vllm \
      --model ${MODEL} \
      --hf_dtype float16 \
      --hf_device_map auto \
      --vllm_tensor_parallel 8 \
      --n 25 --digits 2 4 8 16 32 --kinds  spp bsp edp gcp gcpd tsp tspd ksp msp gsm8k clrs30 add sub mul lcs rod knap ilp_assign ilp_partition ilp_prod \
      --exec_code --batch_size 64 --seed ${SEED} --controlled_sim
  done
done


