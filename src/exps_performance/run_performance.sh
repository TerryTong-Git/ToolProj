

seeds=(0)

# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#     --backend vllm \
#     --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --vllm_tensor_parallel 8 \
#     --n 1000 --digits  2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#     --outdir deepseek_reasoning_actual --exec_code --batch_size 64 --seed ${seed}
# done

# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#     --backend vllm \
#     --model google/gemma-2-9b-it \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --n 1000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#     --outdir gemma_reasoning_actual --exec_code --batch_size 64 --seed ${seed}
# done

# deepseek-ai/DeepSeek-Coder-V2-Instruct

for seed in ${seeds[@]}; do
  pixi run python cot.py \
    --backend vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --hf_dtype float16 \
    --hf_device_map auto \
    --n 1000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
    --outdir llamadeepseek --exec_code --batch_size 64 --seed ${seed}
done

for seed in ${seeds[@]}; do
  pixi run python cot.py \
    --backend vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --hf_dtype float16 \
    --hf_device_map auto \
    --n 1000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
    --outdir deepseekqween32b --exec_code --batch_size 8 --seed ${seed}
done

for seed in ${seeds[@]}; do
  pixi run python cot.py \
    --backend vllm \
    --model mistralai/Mistral-Small-24B-Instruct-2501 \
    --hf_dtype float16 \
    --hf_device_map auto \
    --n 1000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
    --outdir mistralsmall --exec_code --batch_size 8 --seed ${seed}
done


for seed in ${seeds[@]}; do
  pixi run python cot.py \
  --backend vllm \
  --model mistralai/Ministral-8B-Instruct-2410 \
  --hf_dtype float16 \
  --hf_device_map auto \
  --vllm_tensor_parallel 8 \
  --n 1000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
  --outdir execution_ministral_smoke --exec_code --batch_size 64
done

# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#     --backend vllm \
#     --model google/gemma-3-12b-it \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --n 100 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#     --outdir gemma3_12b --exec_code --batch_size 1 --seed ${seed}
# done

# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#     --backend vllm \
#     --model google/gemma-3-27b-it \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --n 100 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#     --outdir gemma3_27b --exec_code --batch_size 1 --seed ${seed}
# done

# 

# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#     --backend vllm \
#     --model google/gemma-2-9b-it \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --n 2000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#     --outdir execution_gemma --exec_code --batch_size 64 --seed ${seed}
# done

# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#   --backend vllm \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --hf_dtype float16 \
#   --hf_device_map auto \
#   --n 2000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#   --outdir execution_llama --exec_code --batch_size 64
# done

# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#   --backend vllm \
#   --model Qwen/Qwen2.5-Coder-7B-Instruct \
#   --hf_dtype float16 \
#   --hf_device_map auto \
#   --vllm_tensor_parallel 1 \
#   --n 2000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#   --outdir out_hf_scale_qwen --exec_code --batch_size 8
# done





  




