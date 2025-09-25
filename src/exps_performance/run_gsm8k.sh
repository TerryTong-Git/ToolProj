

seeds=(0)

for seed in ${seeds[@]}; do
  pixi run python cot.py \
    --backend vllm \
    --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --hf_dtype float16 \
    --hf_device_map auto \
    --vllm_tensor_parallel 8 \
    --kinds gsm8k \
    --outdir gsm8k --exec_code --batch_size 64 --seed ${seed}
done


  




