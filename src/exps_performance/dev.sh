SEEDS=(0)
MODELS=( #7B Models
 "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
#  "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
#  "google/gemma-2-9b-it"
)
for MODEL in ${MODELS[@]}; do
  for SEED in ${SEEDS[@]}; do
    pixi run python cot.py \
      --backend vllm \
      --model ${MODEL} \
      --hf_dtype float16 \
      --hf_device_map auto \
      --vllm_tensor_parallel 8 \
      --n 1000 --digits  2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
      --exec_code --batch_size 64 --seed ${SEED} --controlled_sim
  done
done


