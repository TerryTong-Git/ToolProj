#!/bin/bash
#SBATCH --job-name=lm_eval
#SBATCH --output=output1.txt
#SBATCH --error=error1.txt
#SBATCH -p p_nlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint=GA102GL     
#SBATCH --cpus-per-task=32
#SBATCH --time=1-6:00:00

export HF_HOME="/nlpgpu/data/terry/ToolProj/src/models"
export HF_ALLOW_CODE_EVAL=1
MODELS=(
# marin-community/marin-8b-instruct
# allenai/OLMo-2-1124-7B-Instruct
# allenai/Llama-3.1-Tulu-3-8B
# tiiuae/Falcon3-7B-Instruct
# 01-ai/Yi-1.5-34B-Chat 
# meta-llama/Llama-3.1-8B-Instruct
# meta-llama/Meta-Llama-3-8B-Instruct
# google/gemma-2-9b-it
ibm-granite/granite-3.2-8b-instruct
mistralai/Mistral-Small-Instruct-2409
Qwen/Qwen2.5-32B-Instruct
Qwen/Qwen2.5-14B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
google/gemma-2-27b-it
microsoft/Phi-3-medium-4k-instruct
Qwen/Qwen2.5-Coder-32B-Instruct
 )
for MODEL in ${MODELS[@]}; do
    pixi run lm_eval --model vllm \
        --model_args pretrained=${MODEL},tensor_parallel_size=8,dtype=float16,gpu_memory_utilization=0.9,max_model_len=4096,download_dir=../models \
        --tasks ifeval,bbh,humaneval,mbpp,arc_challenge,arc_easy,mmlu,mmlu_pro \
        --batch_size auto \
        --output_path results/ \
        --confirm_run_unsafe_code
done