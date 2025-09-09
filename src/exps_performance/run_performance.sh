
# pixi run python cot.py \
#   --backend vllm \
#   --model  google/gemma-2-9b-it \
#   --hf_dtype float16 \
#   --hf_device_map auto \
#   --n 10000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#   --outdir out_hf_scale_gemma_rerun --exec_code --batch_size 64

seeds=(1 2 3 4)
for seed in ${seeds[@]}; do
  pixi run python cot.py \
    --backend vllm \
    --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --hf_dtype float16 \
    --hf_device_map auto \
    --n 2000 --digits  2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
    --outdir out_hf_scale_deepseek_working --exec_code --batch_size 64 --seed ${seed}
done

# pixi run python cot.py \
#     --backend vllm \
#     --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --n 2000 --digits  2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#     --outdir out_hf_scale_deepseek_working --exec_code --batch_size 64 --seed 0

# pixi run python cot.py \
#     --backend vllm \
#     --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --n 100 --digits  16 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#     --outdir test --exec_code --batch_size 64 --seed 0

# seeds=(0 1 2 3 4)
# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#     --backend vllm \
#     --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --n 2000 --digits 2 4 8 16 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#     --outdir out_hf_scale_deepseek_code_seeded_nlfix --exec_code --batch_size 64 --seed ${seed}
# done

# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#     --backend vllm \
#     --model google/gemma-2-9b-it \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --n 2000 --digits 2 4 8 16 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#     --outdir out_hf_scale_gemma_seeded_nlfix --exec_code --batch_size 64 --seed ${seed}
# done

# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#     --backend vllm \
#     --vllm_tensor_parallel 6 \
#     --model Qwen/Qwen2.5-Coder-7B-Instruct \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --n 2000 --digits 2 4 8 16 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#     --outdir out_hf_scale_qwen_seeded --exec_code --batch_size 64 --seed ${seed}
# done


# pixi run tensorboard --logdir out_hf/tb

# pixi run python cot.py \
#   --backend vllm \
#   --model deepseek-ai/deepseek-llm-7b-chat \
#   --hf_dtype float16 \
#   --hf_device_map auto \
#   --n 10000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#   --outdir out_hf_scale_deepseek_llm --exec_code --batch_size 64

# pixi run python cot.py \
#   --backend vllm \
#   --model allenai/OLMo-7B-Instruct-hf \
#   --hf_dtype float16 \
#   --hf_device_map auto \
#   --n 10000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#   --outdir out_hf_scale_olmo --exec_code --batch_size 64

# pixi run python cot.py \
#   --backend vllm \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --hf_dtype float16 \
#   --hf_device_map auto \
#   --n 10000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#   --outdir out_hf_scale_meta_llama --exec_code --batch_size 64

# pixi run python cot.py \
#   --backend vllm \
#   --model Qwen/Qwen2.5-Coder-7B-Instruct \
#   --hf_dtype float16 \
#   --hf_device_map auto \
#   --n 10000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
#   --outdir out_hf_scale_qwen --exec_code --batch_size 64

  


