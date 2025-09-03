
pixi run python cot.py \
  --backend hf \
  --model google/gemma-2-9b-it \
  --hf_dtype auto \
  --hf_device_map auto \
  --n 10 --digits 8 --kinds lcs knap rod ilp_prod ilp_assign ilp_partition \
  --outdir out_hf_subset --exec_code 
  

# pixi run tensorboard --logdir out_hf/tb