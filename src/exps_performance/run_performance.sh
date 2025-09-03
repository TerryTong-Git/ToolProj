
pixi run python cot.py \
  --backend hf \
  --model google/gemma-2-9b-it \
  --hf_dtype auto \
  --hf_device_map auto \
  --n 450 --digits 8 10 12 --kinds add sub mul lcs knap rod ilp_assign ilp_prod ilp_partition \
  --outdir out_hf --exec_code 
  

# pixi run tensorboard --logdir out_hf/tb