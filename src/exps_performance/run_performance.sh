
pixi run python cot.py \
  --backend vllm \
  --model google/gemma-2-9b-it \
  --hf_dtype float16 \
  --hf_device_map auto \
  --n 10000 --digits 2 4 8 16 32 --kinds add sub mul lcs knap rod ilp_prod ilp_assign ilp_partition \
  --outdir out_hf_scale --exec_code --batch_size 64
  
# pixi run tensorboard --logdir out_hf/tb