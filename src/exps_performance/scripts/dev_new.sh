seeds=(0)


for seed in ${seeds[@]}; do
  pixi run python runner.py \
    --backend vllm \
    --model google/gemma-2-9b-it \
    --hf_dtype float16 \
    --hf_device_map auto \
    --kinds nphardeval \
    --exec_code --batch_size 64 --seed ${seed}
done
