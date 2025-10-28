#!/bin/bash
#SBATCH --job-name=runmodels        # Job name
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file

#SBATCH -p p_nlp                 # or: interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --time=3:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=tertong@ucdavis.edu    # Email address for notifications

BIGMODELS=( #batch size gonna be bigger here
 "google/gemma-3-12b-it"
 "google/gemma-3-27b-it"
 "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
 "mistralai/Mistral-Small-24B-Instruct-2501" 
)

MODELS=( #7B Models
#  "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
#  "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
#  "google/gemma-2-9b-it"
 "mistralai/Ministral-8B-Instruct-2410"
 "meta-llama/Llama-3.1-8B-Instruct"
)

SEEDS=(0)
for MODEL in ${MODELS[@]}; do
  for SEED in ${SEEDS[@]}; do
    pixi run python cot.py \
      --backend vllm \
      --model ${MODEL} \
      --hf_dtype float16 \
      --hf_device_map auto \
      --vllm_tensor_parallel 8 \
      --n 100 --digits  2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
      --exec_code --batch_size 64 --seed ${SEED}
  done
done




  




