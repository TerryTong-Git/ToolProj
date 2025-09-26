

seeds=(0)

# for seed in ${seeds[@]}; do
#   pixi run python cot.py \
#     --backend vllm \
#     --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
#     --hf_dtype float16 \
#     --hf_device_map auto \
#     --vllm_tensor_parallel 8 \
#     --kinds gsm8k \
#     --outdir gsm8k --exec_code --batch_size 64 --seed ${seed}
# done


for seed in ${seeds[@]}; do
  pixi run python cot.py \
    --backend vllm \
    --model google/gemma-2-9b-it \
    --hf_dtype float16 \
    --hf_device_map auto \
    --kinds gsm8k \
    --outdir gemma_gsm8k --exec_code --batch_size 64 --seed ${seed}
done



for seed in ${seeds[@]}; do
  pixi run python cot.py \
  --backend vllm \
  --model mistralai/Ministral-8B-Instruct-2410 \
  --hf_dtype float16 \
  --hf_device_map auto \
  --vllm_tensor_parallel 8 \
  --kinds gsm8k \
  --outdir ministral_gsm8k --exec_code --batch_size 64
done

for seed in ${seeds[@]}; do
  pixi run python cot.py \
    --backend vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --hf_dtype float16 \
    --hf_device_map auto \
    --kinds gsm8k \
    --outdir qwen_gsm8k --exec_code --batch_size 8 --seed ${seed}
done

for seed in ${seeds[@]}; do
  pixi run python cot.py \
    --backend vllm \
    --model mistralai/Mistral-Small-24B-Instruct-2501 \
    --hf_dtype float16 \
    --hf_device_map auto \
    --kinds gsm8k \
    --outdir mistral_gsm8k --exec_code --batch_size 8 --seed ${seed}
done

for seed in ${seeds[@]}; do
  pixi run python cot.py \
    --backend vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --hf_dtype float16 \
    --hf_device_map auto \
    --n 100 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
    --outdir llama_reasoning --exec_code --batch_size 64 --seed ${seed}
done

  




