seeds=(0)


for seed in ${seeds[@]}; do
  pixi run python main.py \
    --backend vllm \
    --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
    --hf_dtype float16 \
    --hf_device_map auto \
    --kinds clrs30 \
    --exec_code --batch_size 64 --seed ${seed}
done

#eval clrs, inspect some model gens
#give examples in the appendix. Try to have some qualitative observations. 