pixi run python -m runner \
  --out_dir ../runs/paired_kl/ \
  --models google/gemma-2-9b-it  \
  --representations nl,code \
  --num_seeds 8 \
  --thetas add,sub \
  --paired_inputs --n_pairs 5 \
  --include_prompt_ll --max_new_tokens 192 --temperature 0.7 --top_p 0.95 --dtype float32 